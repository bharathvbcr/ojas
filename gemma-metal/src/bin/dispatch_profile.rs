//! Per-dispatch GPU timestamp profile of one decode token.
//!
//! ```bash
//! cargo run -p gemma-metal --release --bin dispatch_profile -- --e4b --out DIR
//! cargo run -p gemma-metal --release --bin dispatch_profile -- --mini --out DIR   # synthetic, no weights
//! cargo run -p gemma-metal --release --bin dispatch_profile -- --synthetic-e4b --out DIR   # E4B shapes, random weights
//! cargo run -p gemma-metal --release --bin dispatch_profile -- --model HF_DIR --tokens 8 --rounds 8
//! ```
//!
//! Each token runs in its own command buffer (encode, then `synchronize`), with
//! tessl's opt-in `TESSL_DISPATCH_STAMPS` timestamps on. A stamp ensures prior
//! work has finished, so a dispatch's `dt_ns` is launch + run + drain + barrier
//! + the stamp itself; kernel time and inter-dispatch gaps are not separable.
//! `--generate N` prints N greedy token ids and exits (exactness A/Bs).
//! `--pipelined-only` skips stamps and prints three `bench_decode_tok_s` runs (for
//! barrier / mid-commit probes via env flags). The stamp overhead arm interleaves stamps-off and stamps-on tokens (ABBA).
//!
//! Outputs in `--out`: `dispatch_stamps.tsv` (one row per dispatch per token),
//! `kernel_summary.tsv`, `family_summary.tsv`, `overhead.tsv`; a text summary
//! goes to stdout.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use gemma_metal::config::Gemma4TextConfig;
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel, TokenProfile};
use gemma_metal::quant::QuantScheme;
use gemma_metal::weights::{
    load_from_hf_dir, resolve_default_e4b_mlx_cache, synthetic_mlx_q4_banks, LoadOptions,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("dispatch_profile: {e}");
        std::process::exit(1);
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Coarse kernel family from the kernel name. The exact-name table
/// (`kernel_summary.tsv`) is authoritative; this only groups it.
fn family(name: &str) -> &'static str {
    let n = name.to_ascii_lowercase();
    if n.contains("gemv") || n.contains("gemm") || n.contains("qmv") {
        "gemv"
    } else if n.contains("flash")
        || n.contains("attn")
        || n.contains("attention")
        || n.contains("sdpa")
    {
        "attention"
    } else if n.contains("softcap") || n.contains("argmax") {
        "softcap_argmax"
    } else if n.contains("rms") && n.contains("rope") {
        "norm_rope_fused"
    } else if n.contains("norm") {
        "norm"
    } else if n.contains("rope") {
        "rope"
    } else if n.contains("kv") {
        "kv_append"
    } else if n.contains("ple") {
        "ple"
    } else if n.contains("copy") || n.contains("scale") || n.contains("fill") || n.contains("cast")
    {
        "copy_scale"
    } else if n.contains("gelu")
        || n.contains("silu")
        || n.contains("mul")
        || n.contains("resid")
        || n.contains("add")
    {
        "eltwise"
    } else {
        "other"
    }
}

struct Args {
    mode: Mode,
    out: PathBuf,
    tokens: usize,
    rounds: usize,
    warmup: usize,
    pipelined_steps: usize,
    pipelined_only: bool,
    generate: Option<usize>,
}

enum Mode {
    Mini,
    /// E4B's preset shapes and tensor set with random MLX Q4 weights: the
    /// real dispatch graph and kernel shapes, for timing only.
    SyntheticE4b,
    Real(PathBuf),
}

fn parse_args() -> Result<Args, String> {
    let mut mode = None;
    let mut out = None;
    let mut pipelined_only = false;
    let mut generate = None;
    let (mut tokens, mut rounds, mut warmup, mut pipelined_steps) =
        (8usize, 8usize, 4usize, 16usize);
    let mut it = std::env::args().skip(1);
    let num = |it: &mut std::iter::Skip<std::env::Args>, f: &str| -> Result<usize, String> {
        it.next()
            .ok_or_else(|| format!("{f} needs a value"))?
            .parse::<usize>()
            .map_err(|e| format!("{f}: {e}"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mini" => mode = Some(Mode::Mini),
            "--synthetic-e4b" => mode = Some(Mode::SyntheticE4b),
            "--e4b" => {
                let dir = resolve_default_e4b_mlx_cache().ok_or_else(|| {
                    "no HF snapshot for mlx-community/gemma-4-e4b-it-4bit (searched $HF_HOME/hub, \
                     $HUGGINGFACE_HUB_CACHE, ~/.cache/huggingface/hub); pass --model DIR"
                        .to_string()
                })?;
                mode = Some(Mode::Real(dir));
            }
            "--model" => {
                mode = Some(Mode::Real(PathBuf::from(
                    it.next().ok_or("--model needs a dir")?,
                )))
            }
            "--out" => out = Some(PathBuf::from(it.next().ok_or("--out needs a dir")?)),
            "--tokens" => tokens = num(&mut it, "--tokens")?,
            "--rounds" => rounds = num(&mut it, "--rounds")?,
            "--warmup" => warmup = num(&mut it, "--warmup")?,
            "--pipelined-only" => pipelined_only = true,
            "--generate" => generate = Some(num(&mut it, "--generate")?),
            "--pipelined-steps" => pipelined_steps = num(&mut it, "--pipelined-steps")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if tokens < 5 {
        return Err("--tokens must be >= 5".into());
    }
    Ok(Args {
        mode: mode.ok_or("pass --e4b, --model DIR, --synthetic-e4b or --mini")?,
        out: out.ok_or("pass --out DIR")?,
        tokens,
        rounds,
        warmup,
        pipelined_steps,
        pipelined_only,
        generate,
    })
}

fn load_avg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)
        .map_err(|e| format!("create {}: {e}", args.out.display()))?;

    // The stamp heap exists only for a runtime built with the flag on.
    tessl::ab_flags::set_dispatch_stamps(true);
    println!("load_avg_start: {}", load_avg());
    let (model, label) = match &args.mode {
        Mode::Mini => (
            GpuSynthModel::from_synthetic(
                gemma_metal::forward::SyntheticE4bGraph::mini_parity()
                    .map_err(|e| e.to_string())?,
                QuantScheme::q4_default(),
            )
            .map_err(|e| e.to_string())?,
            "mini_synthetic".to_string(),
        ),
        Mode::SyntheticE4b => {
            let banks = synthetic_mlx_q4_banks(
                Gemma4TextConfig::e4b_preset(),
                LoadOptions {
                    scheme: QuantScheme::q4_mlx_default(),
                    max_seq: 256,
                    ..LoadOptions::default()
                },
                0x5eed,
            )
            .map_err(|e| format!("synthetic E4B banks: {e}"))?;
            (
                GpuSynthModel::from_host_banks(banks).map_err(|e| e.to_string())?,
                "synthetic_e4b (E4B preset shapes, random MLX Q4 weights; timing only)".to_string(),
            )
        }
        Mode::Real(dir) => {
            let banks = load_from_hf_dir(
                dir,
                LoadOptions {
                    scheme: QuantScheme::q4_mlx_default(),
                    max_seq: 256,
                    ..LoadOptions::default()
                },
            )
            .map_err(|e| format!("load weights {}: {e}", dir.display()))?;
            (
                GpuSynthModel::from_host_banks(banks).map_err(|e| e.to_string())?,
                dir.display().to_string(),
            )
        }
    };
    let mut sess = GpuDecodeSession::new(model).map_err(|e| e.to_string())?;
    // Same prompt as `bench --e4b`; the mini graph's vocab is 512, so it gets small ids.
    let prompt: [u32; 4] = if matches!(args.mode, Mode::Mini) {
        [1, 2, 3, 4]
    } else {
        [2, 105, 4368, 1246]
    };
    println!("model: {label}");

    // Warm up: pipelines, pools, clocks. Stamps off, one token per command buffer.
    sess.profile_prime(&prompt).map_err(|e| e.to_string())?;
    for _ in 0..args.warmup {
        sess.profile_decode_token(false)
            .map_err(|e| e.to_string())?;
    }

    if let Some(n) = args.generate {
        // Greedy token ids only, for exactness A/Bs across env flags.
        tessl::ab_flags::set_dispatch_stamps(false);
        let toks = sess.generate(&prompt, n).map_err(|e| e.to_string())?;
        println!("generated: {toks:?}");
        return Ok(());
    }

    if args.pipelined_only {
        // Barrier / commit probes: wall tok/s only, no stamps (mid-commit would drop them).
        tessl::ab_flags::set_dispatch_stamps(false);
        for i in 0..3 {
            let t = sess
                .bench_decode_tok_s(&prompt, args.pipelined_steps)
                .map_err(|e| e.to_string())?;
            println!("pipelined run {i}: {t:.2} tok/s ({:.2} ms/token)", 1e3 / t);
        }
        println!("load_avg_end: {}", load_avg());
        return Ok(());
    }

    // Pipelined baseline before: the existing bench loop (stamps off, no
    // per-token synchronize), for comparison with the one-token-per-CB numbers.
    let tok_s_before = sess
        .bench_decode_tok_s(&prompt, args.pipelined_steps)
        .map_err(|e| e.to_string())?;

    // Overhead arm: ABBA per round, stamps off vs on, same session.
    struct Row {
        stamps: bool,
        t: TokenProfile,
    }
    let mut ovh: Vec<Row> = Vec::new();
    for _ in 0..args.rounds {
        sess.profile_prime(&prompt).map_err(|e| e.to_string())?;
        for &on in &[false, true, true, false] {
            let t = sess.profile_decode_token(on).map_err(|e| e.to_string())?;
            ovh.push(Row { stamps: on, t });
        }
    }
    let mut overhead =
        String::from("stamps\thost_encode_ms\tsync_ms\twall_ms\tdispatches\tbarriers\n");
    for r in &ovh {
        let _ = writeln!(
            overhead,
            "{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{}",
            r.stamps as u8,
            r.t.host_encode_ms,
            r.t.sync_ms,
            r.t.host_encode_ms + r.t.sync_ms,
            r.t.dispatches,
            r.t.barriers
        );
    }
    std::fs::write(args.out.join("overhead.tsv"), &overhead).map_err(|e| e.to_string())?;
    let pick = |on: bool, f: fn(&TokenProfile) -> f64| -> f64 {
        let mut v: Vec<f64> = ovh
            .iter()
            .filter(|r| r.stamps == on)
            .map(|r| f(&r.t))
            .collect();
        median(&mut v)
    };

    // The profile proper: K consecutive tokens, stamps on.
    sess.profile_prime(&prompt).map_err(|e| e.to_string())?;
    let mut toks: Vec<TokenProfile> = Vec::new();
    for _ in 0..args.tokens {
        toks.push(sess.profile_decode_token(true).map_err(|e| e.to_string())?);
    }

    // Pipelined baseline after.
    let tok_s_after = sess
        .bench_decode_tok_s(&prompt, args.pipelined_steps)
        .map_err(|e| e.to_string())?;
    println!("load_avg_end: {}", load_avg());

    // Rows.
    let mut tsv =
        String::from("token\tidx\tlayer\tkernel\tfamily\tbarrier_before\tt_end_ns\tdt_ns\n");
    let mut per_kernel: BTreeMap<String, Vec<f64>> = BTreeMap::new(); // name -> per-token sums (ms)
    let mut per_kernel_n: BTreeMap<String, usize> = BTreeMap::new();
    let mut per_family: BTreeMap<&'static str, (f64, usize)> = BTreeMap::new(); // summed over tokens
    let mut tiny_dt_us: Vec<f64> = Vec::new();
    let mut span_ms: Vec<f64> = Vec::new();
    let mut gemv_ms: Vec<f64> = Vec::new();
    let mut other_ms: Vec<f64> = Vec::new();
    let mut barrier_frac: Vec<f64> = Vec::new();
    let mut n_disp = 0usize;
    for (ti, t) in toks.iter().enumerate() {
        let b = t.stamps.as_ref().ok_or("token without stamps")?;
        if b.overflowed || b.dropped_batches != 0 {
            return Err(format!(
                "token {ti}: stamps incomplete (overflowed={}, dropped_batches={})",
                b.overflowed, b.dropped_batches
            ));
        }
        if b.stamps.len() as u64 != t.dispatches {
            return Err(format!(
                "token {ti}: {} stamps but {} dispatches encoded",
                b.stamps.len(),
                t.dispatches
            ));
        }
        n_disp = b.stamps.len();
        let mut tok_kernel: BTreeMap<&str, f64> = BTreeMap::new();
        let (mut g, mut o) = (0.0f64, 0.0f64);
        let mut with_barrier = 0usize;
        let mut last = b.t0_ns;
        for (i, s) in b.stamps.iter().enumerate() {
            let fam = family(&s.name);
            let _ = writeln!(
                tsv,
                "{ti}\t{i}\t{}\t{}\t{fam}\t{}\t{}\t{}",
                s.layer, s.name, s.barrier_before as u8, s.t_end_ns, s.dt_ns
            );
            let ms = s.dt_ns as f64 / 1e6;
            *tok_kernel.entry(s.name.as_str()).or_default() += ms;
            *per_kernel_n.entry(s.name.clone()).or_default() += 1;
            let e = per_family.entry(fam).or_default();
            e.0 += ms;
            e.1 += 1;
            if fam == "gemv" {
                g += ms;
            } else {
                o += ms;
            }
            if !matches!(fam, "gemv" | "attention") {
                tiny_dt_us.push(s.dt_ns as f64 / 1e3);
            }
            with_barrier += s.barrier_before as usize;
            last = s.t_end_ns;
        }
        for (k, v) in tok_kernel {
            per_kernel.entry(k.to_string()).or_default().push(v);
        }
        span_ms.push((last - b.t0_ns) as f64 / 1e6);
        gemv_ms.push(g);
        other_ms.push(o);
        barrier_frac.push(with_barrier as f64 / b.stamps.len().max(1) as f64);
    }
    std::fs::write(args.out.join("dispatch_stamps.tsv"), &tsv).map_err(|e| e.to_string())?;

    let k = toks.len() as f64;
    let mut kernel_tsv = String::from("kernel\tfamily\tcount_per_token\tms_per_token_median\n");
    for (name, sums) in per_kernel.iter_mut() {
        let _ = writeln!(
            kernel_tsv,
            "{name}\t{}\t{:.1}\t{:.4}",
            family(name),
            per_kernel_n[name] as f64 / k,
            median(sums)
        );
    }
    std::fs::write(args.out.join("kernel_summary.tsv"), &kernel_tsv).map_err(|e| e.to_string())?;
    let mut family_tsv = String::from("family\tcount_per_token\tms_per_token_mean\n");
    for (fam, (ms, n)) in &per_family {
        let _ = writeln!(family_tsv, "{fam}\t{:.1}\t{:.4}", *n as f64 / k, ms / k);
    }
    std::fs::write(args.out.join("family_summary.tsv"), &family_tsv).map_err(|e| e.to_string())?;

    let enc: Vec<f64> = toks.iter().map(|t| t.host_encode_ms).collect();
    let syn: Vec<f64> = toks.iter().map(|t| t.sync_ms).collect();
    let (span_med, gemv_med, other_med) = (
        median(&mut span_ms),
        median(&mut gemv_ms),
        median(&mut other_ms),
    );
    println!();
    println!("=== dispatch profile ({label}) ===");
    let cbs = toks[0]
        .stamps
        .as_ref()
        .map(|b| b.command_buffers)
        .unwrap_or(0);
    println!(
        "tokens={} dispatches/token={n_disp} barriers/token={} commits/token={} command_buffers/token={cbs}",
        toks.len(),
        toks[0].barriers,
        toks[0].commits
    );
    if cbs > 1 {
        println!(
            "WARNING: a token spans {cbs} command buffers; host encode and commit gaps are inside the GPU span, \
             and the first dt of each command buffer is measured from its own origin stamp."
        );
    }
    println!("GPU span (stamps) ms/token, median:   {span_med:.3}");
    println!("  gemv-family ms/token (median):      {gemv_med:.3}");
    println!("  everything else ms/token (median):  {other_med:.3}");
    println!(
        "host encode ms/token (median):        {:.3}",
        median(&mut enc.clone())
    );
    println!(
        "host sync wait ms/token (median):     {:.3}",
        median(&mut syn.clone())
    );
    println!(
        "dispatches preceded by a barrier:     {:.1}%",
        100.0 * median(&mut barrier_frac)
    );
    println!(
        "tiny kernels (non gemv/attention) dt: median {:.2} us over {} dispatches",
        median(&mut tiny_dt_us),
        tiny_dt_us.len()
    );
    println!();
    println!("{family_tsv}");
    println!(
        "stamp overhead (median, {} tokens per arm): sync_ms off {:.3} on {:.3}; encode_ms off {:.3} on {:.3}; wall off {:.3} on {:.3}",
        ovh.iter().filter(|r| r.stamps).count(),
        pick(false, |t| t.sync_ms),
        pick(true, |t| t.sync_ms),
        pick(false, |t| t.host_encode_ms),
        pick(true, |t| t.host_encode_ms),
        pick(false, |t| t.host_encode_ms + t.sync_ms),
        pick(true, |t| t.host_encode_ms + t.sync_ms),
    );
    println!(
        "pipelined bench (stamps off, {} steps, one CB): before {tok_s_before:.2} tok/s ({:.2} ms/token), after {tok_s_after:.2} tok/s ({:.2} ms/token)",
        args.pipelined_steps,
        1e3 / tok_s_before,
        1e3 / tok_s_after
    );
    Ok(())
}
