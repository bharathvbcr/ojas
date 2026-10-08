//! Phase 4 speed harness — synthetic mini + real E4B MLX Q4 when cached.
//!
//! ```bash
//! cargo run -p gemma-metal --release --bin bench
//! cargo run -p gemma-metal --release --bin bench -- --e4b
//! cargo run -p gemma-metal --release --bin bench -- --model /path/to/hf/dir
//! cargo run -p gemma-metal --release --bin bench -- --dflash
//! cargo run -p gemma-metal --release --bin bench -- --dflash-31b
//! # Diagnostic logs ON by default (`[gemma-metal:…]` stderr).
//! # Silence: GEMMA_METAL_LOG=0  |  GEMMA_METAL_INFER_LOG=0 (line-level decode)
//! ```

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use gemma_metal::dflash::{
    generate_with_dflash, generate_with_dflash_host, generate_with_dflash_speed,
    DFlashGpuConditioner, DFlashGpuDraft, HostDFlashDraft, DFLASH_DEFAULT_BLOCK,
};
use gemma_metal::diag;
use gemma_metal::forward::{host_forward_prefill, SyntheticE4bGraph};
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::kernels::{
    flash_attn_swa_h128_prefill, flash_attn_swa_h256_prefill, gemv_quant_host, softcap_argmax,
    GemmaGpu, KernelId,
};
use gemma_metal::quant::{quantize_affine_f32, QuantScheme};
use gemma_metal::step_verify::compare_token_stream;
use gemma_metal::weights::{
    load_from_hf_dir, resolve_default_31b_mlx_cache, resolve_default_dflash_draft_cache,
    resolve_default_e4b_mlx_cache, LoadOptions,
};

fn main() {
    std::process::exit(run());
}

/// Exit code contract (only meaningful when `--thermal` is passed; without it
/// the harness always returns 0 as before):
///   0 — every requested gate passed, or no gate was requested
///   1 — a gate ran and FAILED
///   2 — a gate was requested but could not run (no GPU, no model, too few
///       windows). Never conflated with 0: a check that could not run must not
///       report the same result as a check that ran and passed.
fn run() -> i32 {
    diag::init();
    let argv: Vec<String> = env::args().collect();
    diag::log(
        "bench",
        format_args!("argv={argv:?} version={}", gemma_metal::version()),
    );
    if let Some(rss) = diag::rss_mib() {
        diag::log("bench", format_args!("RSS_start={rss:.1} MiB"));
    }

    println!("gemma-metal Phase 4 speed harness ({})", gemma_metal::version());
    println!("gates.md Phase-0 reference (this host):");
    println!("  E4B Q4 mlx-lm best decode ≈ 75.7 tok/s  (honest lane gate ≥48–60)");
    println!("  Ollama E4B Q4_K_M ≈ 55.8 tok/s");
    println!("  31B Ollama decode ≈ 12.3 tok/s          (product gate ≥15)");
    println!("  31B MLX+DFlash ≈ 37.2 tok/s             (MTP product gate ≥25)");
    println!();

    let mut run_e4b = false;
    let mut run_31b = false;
    let mut run_dflash = false;
    let mut run_dflash_31b = false;
    let mut model_dir: Option<PathBuf> = None;
    let mut thermal = false;
    let mut thermal_cfg = gemma_metal::thermal::ThermalGateConfig::default();
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--e4b" => run_e4b = true,
            "--dflash" => run_dflash = true,
            "--dflash-31b" => {
                run_dflash = true;
                run_dflash_31b = true;
            }
            "--model" => {
                let raw = args.next().map(PathBuf::from);
                match raw.as_ref().and_then(|p| p.to_str()) {
                    Some("31b") | Some("31B") => {
                        run_31b = true;
                        model_dir = resolve_default_31b_mlx_cache();
                    }
                    Some("e4b") | Some("E4B") => {
                        run_e4b = true;
                        model_dir = resolve_default_e4b_mlx_cache();
                    }
                    _ => {
                        model_dir = raw;
                    }
                }
            }
            "--thermal" => thermal = true,
            "--thermal-quick" => {
                thermal = true;
                thermal_cfg = gemma_metal::thermal::ThermalGateConfig::quick();
            }
            "--thermal-secs" => {
                thermal = true;
                thermal_cfg.duration_secs = num_arg(&mut args, "--thermal-secs");
            }
            "--thermal-window" => {
                thermal = true;
                thermal_cfg.window_secs = num_arg(&mut args, "--thermal-window");
            }
            "--thermal-warmup" => {
                thermal = true;
                thermal_cfg.warmup_secs = num_arg(&mut args, "--thermal-warmup");
            }
            "--thermal-ratio" => {
                thermal = true;
                thermal_cfg.min_sustained_ratio = num_arg(&mut args, "--thermal-ratio");
            }
            "--thermal-rolling" => {
                thermal = true;
                thermal_cfg.min_rolling_ratio = num_arg(&mut args, "--thermal-rolling");
            }
            "--thermal-cv" => {
                thermal = true;
                let v = num_arg(&mut args, "--thermal-cv");
                thermal_cfg.max_cv = if v <= 0.0 { None } else { Some(v) };
            }
            "--thermal-floor" => {
                thermal = true;
                thermal_cfg.floor_tok_s = Some(num_arg(&mut args, "--thermal-floor"));
            }
            "--thermal-windows" => {
                thermal = true;
                thermal_cfg.min_windows = count_arg(&mut args, "--thermal-windows");
            }
            "--trace" => gemma_metal::trace::set_cli_mode(gemma_metal::trace::TraceMode::Host),
            "--trace-json" => {
                gemma_metal::trace::set_cli_mode(gemma_metal::trace::TraceMode::Json)
            }
            "--trace-sync" => {
                gemma_metal::trace::set_cli_mode(gemma_metal::trace::TraceMode::Sync)
            }
            "-h" | "--help" => {
                eprintln!(
                    "bench [--e4b] [--dflash] [--dflash-31b] [--model DIR|e4b|31b] [--trace|--trace-json|--trace-sync]"
                );
                eprintln!(
                    "Thermal (sustained-load) gate: --thermal | --thermal-quick"
                );
                eprintln!(
                    "  --thermal-secs S --thermal-window S --thermal-warmup S --thermal-windows N"
                );
                eprintln!(
                    "  --thermal-ratio R (tail/head, default 0.90) --thermal-rolling R (default 0.85)"
                );
                eprintln!(
                    "  --thermal-cv X (dispersion ceiling, default 0.09; <=0 disables)"
                );
                eprintln!(
                    "  --thermal-floor TOK_S (every window must clear it)"
                );
                eprintln!(
                    "  exit 0 pass/not-requested · 1 gate FAILED · 2 gate requested but could not run"
                );
                eprintln!("Logs: GEMMA_METAL_LOG=1 (default ON) | GEMMA_METAL_LOG=0 silence");
                eprintln!("Infer line-log: GEMMA_METAL_INFER_LOG=1 (default ON) | =0 silence");
                eprintln!("Per-op decode rollup: GEMMA_METAL_TRACE=1|json|sync");
                eprintln!(
                    "Burst decode steps: GEMMA_METAL_BENCH_STEPS=N (default 16) — raise to compare"
                );
                eprintln!(
                    "  the burst bench against the sustained gate at the same KV depth"
                );
                return 0;
            }
            _ => {
                diag::log("bench", format_args!("ignoring unknown arg {a}"));
            }
        }
    }
    if model_dir.is_some() && !run_31b {
        // Explicit --model DIR without e4b/31b alias → real-model decode path.
        run_e4b = true;
    }
    if run_31b {
        run_e4b = true; // reuse real-model load/bench path; writer keys off dims/dir
    }
    diag::log(
        "bench",
        format_args!(
            "flags run_e4b={run_e4b} run_31b={run_31b} run_dflash={run_dflash} run_dflash_31b={run_dflash_31b} model_dir={model_dir:?}"
        ),
    );

    // Sustained-load gate reports, aggregated into the process exit code.
    let mut thermal_reports: Vec<gemma_metal::thermal::ThermalReport> = Vec::new();

    // --- Synthetic mini (always) ---
    let model = SyntheticE4bGraph::mini_parity().expect("mini graph");
    let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let _ = host_forward_prefill(&model, &tokens).expect("warmup");
    let t0 = std::time::Instant::now();
    let iters = 20usize;
    for _ in 0..iters {
        let _ = host_forward_prefill(&model, &tokens).expect("fwd");
    }
    let host_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("host synthetic prefill (T=8, 3 layers): {host_ms:.2} ms/iter");

    if let Err(e) = GemmaGpu::new() {
        println!("GPU unavailable — skipping Metal benches: {e}");
        println!("metallib={}", gemma_metal::gemma_metallib_path());
        if thermal {
            println!("thermal gate REQUESTED but could not run (no GPU) — exit 2, not a pass");
            return 2;
        }
        return 0;
    }

    match GpuSynthModel::from_synthetic(
        SyntheticE4bGraph::mini_parity().unwrap(),
        QuantScheme::q4_default(),
    ) {
        Ok(gpu_model) => {
            let mut sess = GpuDecodeSession::new(gpu_model).expect("session");
            let prompt = [1u32, 2, 3, 4];
            match sess.generate(&prompt, 2) {
                Ok(_) => {}
                Err(e) => {
                    println!("GPU Hot synthetic warmup failed (Metal busy?): {e}");
                    println!("(continuing to microbenches / E4B if possible)");
                }
            }
            if sess.model.gpu.rt.pipeline("gemv_q4").is_ok() {
                let steps = 32usize;
                match sess.bench_decode_tok_s(&prompt, steps) {
                    Ok(tok_s) => {
                        println!();
                        println!("=== GPU Hot-bank synthetic decode (honest, mini graph) ===");
                        println!("  graph: vocab=512 hidden=256 layers=3 (SWA+global+consumer)");
                        println!("  decode: {tok_s:.1} tok/s  ({steps} steps after prefill)");
                        if let Ok(ttft_ms) = sess.bench_ttft_ms(&tokens) {
                            println!("  TTFT proxy (prefill T=8): {ttft_ms:.2} ms");
                        }
                        // Gate the mini graph only when no real model was asked
                        // for. Mini's per-step work is dominated by host
                        // dispatch, so on a busy machine it reports SKIPPED
                        // ("too noisy to certify") — true, but not worth
                        // spending the exit code on when a real model is the
                        // subject under test.
                        if thermal && !run_e4b {
                            let meta = serde_json::json!({
                                "model": "mini_synth",
                                "graph": "vocab=512 hidden=256 layers=3",
                                "burst_tok_s_reference": tok_s,
                            });
                            thermal_reports.push(run_thermal_gate(
                                "mini", &mut sess, &prompt, &thermal_cfg, meta,
                            ));
                        }
                    }
                    Err(e) => println!("GPU Hot synthetic decode bench failed: {e}"),
                }
            }
        }
        Err(e) => println!("GPU Hot synthetic decode unavailable: {e}"),
    }

    // --- Real model (E4B or 31B) ---
    if run_e4b {
        let dir = if run_31b {
            model_dir
                .clone()
                .or_else(resolve_default_31b_mlx_cache)
        } else {
            model_dir
                .clone()
                .or_else(resolve_default_e4b_mlx_cache)
        };
        let Some(dir) = dir else {
            let hint = if run_31b {
                "no --model and no HF cache for mlx-community/gemma-4-31b-it-4bit"
            } else {
                "no --model and no HF cache for mlx-community/gemma-4-e4b-it-4bit"
            };
            diag::err_msg("bench", "model resolve", &hint);
            panic!("{hint}");
        };
        diag::log("bench", format_args!("resolved model_dir={}", dir.display()));
        println!();
        println!("=== Real model Q4 (MLX affine) ===");
        println!("  weights: {}", dir.display());
        let load_t0 = std::time::Instant::now();
        let banks = match load_from_hf_dir(
            &dir,
            LoadOptions {
                scheme: QuantScheme::q4_mlx_default(),
                max_seq: 256,
                ..LoadOptions::default()
            },
        ) {
            Ok(b) => b,
            Err(e) => {
                diag::err("bench", "load_from_hf_dir", &e);
                panic!("load weights: {e}");
            }
        };
        let layers = banks.config.num_hidden_layers;
        let hidden = banks.config.hidden_size;
        let vocab = banks.config.vocab_size;
        diag::log(
            "bench",
            format_args!(
                "load stage {:.1}s matrices={} hot={} RSS={:?}",
                load_t0.elapsed().as_secs_f64(),
                banks.matrices.len(),
                diag::fmt_bytes(banks.total_hot_bytes() as u64),
                diag::rss_mib()
            ),
        );
        println!(
            "  loaded {} matrices in {:.1}s; hot≈{:.2} GiB; layers={layers} vocab={vocab} hidden={hidden}",
            banks.matrices.len(),
            load_t0.elapsed().as_secs_f64(),
            banks.total_hot_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        println!(
            "  gaps: PLE Q4 Hot residual on; gate/proj still skipped; mid-commit off by default"
        );

        let upload_t0 = std::time::Instant::now();
        // from_host_banks takes banks by value and drops host residency before return.
        let gpu_model = match GpuSynthModel::from_host_banks(banks) {
            Ok(m) => m,
            Err(e) => {
                diag::err("bench", "Hot upload", &e);
                panic!("upload Hot banks: {e}");
            }
        };
        diag::log(
            "bench",
            format_args!(
                "upload stage {:.1}s (host banks dropped) RSS={:?}",
                upload_t0.elapsed().as_secs_f64(),
                diag::rss_mib()
            ),
        );
        println!(
            "  Hot upload + session prep: {:.1}s",
            upload_t0.elapsed().as_secs_f64()
        );
        let mut sess = match GpuDecodeSession::new(gpu_model) {
            Ok(s) => s,
            Err(e) => {
                diag::err("bench", "GpuDecodeSession::new", &e);
                panic!("session: {e}");
            }
        };

        // Short prompt (BOS + a few ids) — tokenizer not required for speed lane.
        let prompt = [2u32, 105, 4368, 1246]; // bos-ish + arbitrary text ids
        println!("  warmup generate…");
        diag::log("bench", format_args!("warmup generate prompt={prompt:?}"));
        let warm = sess.generate(&prompt, 1);
        match &warm {
            Ok(toks) => {
                diag::log("bench", format_args!("warmup ok seq_len={}", toks.len()));
                println!("  warmup ok, seq_len={}", toks.len());
            }
            Err(e) => {
                diag::err("bench", "warmup generate", e);
                println!("  warmup FAILED: {e}");
                println!("  continuing to timed decode anyway (may also fail)");
            }
        }

        // Decode-step count for the burst bench. Overridable so the burst and
        // sustained numbers can be compared at the same KV depth: a 16-step
        // burst measures decode at pos≈4..20, while a sustained run spends most
        // of its time near `kv_capacity`, and attention cost is not flat in KV
        // length. Comparing the two without this knob compares two different
        // workloads and calls the difference a regression.
        let steps = std::env::var("GEMMA_METAL_BENCH_STEPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(16);
        let ttft_ms = match sess.bench_ttft_ms(&prompt) {
            Ok(ms) => {
                diag::log("bench", format_args!("TTFT={ms:.1} ms prompt_len={}", prompt.len()));
                println!("  TTFT (prefill T={}): {ms:.1} ms", prompt.len());
                Some(ms)
            }
            Err(e) => {
                diag::err("bench", "TTFT", &e);
                println!("  TTFT failed: {e}");
                None
            }
        };
        let tok_s = match sess.bench_decode_tok_s(&prompt, steps) {
            Ok(t) => {
                diag::log("bench", format_args!("decode={t:.2} tok/s steps={steps}"));
                println!("  decode: {t:.2} tok/s  ({steps} steps after prefill)");
                if is_31b_shape(layers, hidden) {
                    println!("  vs Phase-0: 31B Ollama ≈12.3 | MLX+DFlash ≈37 | product ≥15");
                } else {
                    println!(
                        "  vs Phase-0: mlx-lm ≈75.7 | Ollama ≈55.8 | gate ≥48–60"
                    );
                    if t < 48.0 {
                        println!("  verdict: BELOW honest-lane gate (expected for partial/slow v1)");
                    } else {
                        println!("  verdict: at/above lower gate band");
                    }
                }
                Some(t)
            }
            Err(e) => {
                diag::err("bench", "decode bench", &e);
                println!("  decode bench failed: {e}");
                None
            }
        };

        if let Some(rss) = diag::rss_mib() {
            diag::log("bench", format_args!("RSS_end={rss:.1} MiB"));
        }
        write_real_model_result(ttft_ms, tok_s, steps, &dir, layers, hidden, vocab);

        if thermal {
            let label = if is_31b_shape(layers, hidden) { "31b" } else { "e4b" };
            let meta = serde_json::json!({
                "model": model_id_from_dir(&dir, is_31b_shape(layers, hidden)),
                "weights_dir": dir.display().to_string(),
                "layers": layers,
                "hidden": hidden,
                "vocab": vocab,
                "burst_tok_s_reference": tok_s,
            });
            thermal_reports.push(run_thermal_gate(
                label, &mut sess, &prompt, &thermal_cfg, meta,
            ));
        }

        // --- Real-model verify(M) sweep (opt-in: GEMMA_METAL_VERIFY_SWEEP=1) ---
        // The mini sweep measured the M×GEMV FALLBACK (mini cols=256 fails the
        // cols>256 GEMM gate) — its 9.4× at M=8 says nothing about the Q4-GEMM
        // verify E4B/31B actually use. This sweep runs on the loaded Hot session
        // (cols>256 ⇒ real gemm_q4_mlx path) and is the number that decides
        // DDTree / speculative economics (speed_research_frontier Lever A).
        if std::env::var("GEMMA_METAL_VERIFY_SWEEP").ok().as_deref() == Some("1") {
            println!();
            println!("=== Real-model verify(M) sweep (Q4 GEMM path) ===");
            let seeds: [u32; 8] = [3, 4, 5, 6, 7, 8, 9, 10];
            let warmup = 1usize;
            let iters = 4usize;
            let mut rows_json = Vec::new();
            let mut ms_m1: Option<f64> = None;
            for m in 1..=8usize {
                match sess.bench_step_verify_ms(&seeds[..m], warmup, iters) {
                    Ok(ms) => {
                        let ratio = ms_m1.map(|b| ms / b).unwrap_or(1.0);
                        if ms_m1.is_none() {
                            ms_m1 = Some(ms);
                        }
                        println!("  M={m}: {ms:.3} ms/iter  ratio_vs_m1={ratio:.3}");
                        rows_json.push(format!(
                            "{{\"M\": {m}, \"ms_per_iter\": {ms:.4}, \"ratio_vs_m1\": {ratio:.4}}}"
                        ));
                    }
                    Err(e) => {
                        println!("  M={m}: sweep failed: {e}");
                        break;
                    }
                }
            }
            if let Some(b) = ms_m1 {
                let flat = rows_json
                    .last()
                    .map(|_| true)
                    .unwrap_or(false);
                let _ = flat;
                println!(
                    "  verdict: flattens if ratio(M=8) ≪ 8 (MLX NAX class ≈1.6×); \
                     linear ≈8× ⇒ DDTree stays parked (m1={b:.3} ms)"
                );
            }
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let out_dir = manifest.join("bench/results");
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let model_tag = if is_31b_shape(layers, hidden) { "31b" } else { "e4b" };
            let body = format!(
                "{{\n  \"artifact\": \"verify_m_sweep\",\n  \"model\": \"{model_tag}\",\n  \
                 \"layers\": {layers},\n  \"hidden\": {hidden},\n  \
                 \"gemm_path\": \"gemm_q4_mlx (cols>256)\",\n  \
                 \"warmup\": {warmup},\n  \"iters\": {iters},\n  \
                 \"results\": [\n    {}\n  ],\n  \"unix_ts\": {ts}\n}}\n",
                rows_json.join(",\n    ")
            );
            let path = out_dir.join(format!("verify_m_sweep_{model_tag}_{ts}.json"));
            let latest = out_dir.join(format!("verify_m_sweep_{model_tag}_latest.json"));
            if std::fs::write(&path, &body).is_ok() {
                let _ = std::fs::write(&latest, &body);
                println!("  wrote {}", path.display());
            }
        }
    } else {
        println!();
        println!("(skip real model — pass --e4b, --model 31b, or --model DIR)");
    }

    if run_dflash {
        run_dflash_gates(run_dflash_31b);
    }

    // --- Microbenches ---
    let Ok(gpu) = GemmaGpu::new() else {
        return thermal_exit_code(thermal, &thermal_reports);
    };
    println!();
    println!("=== Kernel microbenches ===");
    // Presence check for DFlash draft FA stub.
    match gpu.rt.pipeline(KernelId::FlashAttnSwaH128.entry_name()) {
        Ok(_) => println!("GPU FA SWA h128 (DFlash draft stub): pipeline OK"),
        Err(e) => println!("GPU FA SWA h128 stub missing: {e}"),
    }
    let rows = 2560usize;
    let cols = 2560usize;
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 17) as f32) * 0.01 - 0.05)
        .collect();
    let q = quantize_affine_f32(rows, cols, &data, QuantScheme::q4_default()).unwrap();
    let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.001).collect();
    match gemv_quant_host(&gpu, &q, &x) {
        Ok(_) => {
            let t1 = std::time::Instant::now();
            let g_iters = 50usize;
            for _ in 0..g_iters {
                let _ = gemv_quant_host(&gpu, &q, &x).unwrap();
            }
            let gemv_us = t1.elapsed().as_secs_f64() * 1e6 / g_iters as f64;
            println!("GPU gemv_q4 [{rows}x{cols}] @ M=1: {gemv_us:.1} µs/call");
        }
        Err(e) => println!("GPU gemv_q4 microbench skipped: {e}"),
    }

    let n = 262_144usize;
    let mut logits = vec![0.0f32; n];
    logits[12345] = 3.0;
    let lb = gpu.rt.alloc_buffer(n * 4).unwrap();
    lb.write_f32(&logits);
    if let Err(e) = softcap_argmax(&gpu, &lb, 30.0, n as u32) {
        println!("GPU softcap microbench skipped: {e}");
        return thermal_exit_code(thermal, &thermal_reports);
    }
    let t2 = std::time::Instant::now();
    let s_iters = 20usize;
    for _ in 0..s_iters {
        lb.write_f32(&logits);
        let _ = softcap_argmax(&gpu, &lb, 30.0, n as u32).unwrap();
    }
    let soft_us = t2.elapsed().as_secs_f64() * 1e6 / s_iters as f64;
    println!("GPU softcap_argmax vocab={n}: {soft_us:.1} µs/call");

    // Micro: D=128 FA stub (B=1 T=8 H=4 Hkv=1).
    {
        let b = 1u32;
        let t = 8u32;
        let h = 4u32;
        let hkv = 1u32;
        let d = 128usize;
        let nq = (b * t * h) as usize * d;
        let nkv = (b * t * hkv) as usize * d;
        if let (Ok(qb), Ok(kb), Ok(vb), Ok(ob)) = (
            gpu.rt.alloc_buffer(nq * 4),
            gpu.rt.alloc_buffer(nkv * 4),
            gpu.rt.alloc_buffer(nkv * 4),
            gpu.rt.alloc_buffer(nq * 4),
        ) {
            qb.write_f32(&vec![0.01f32; nq]);
            kb.write_f32(&vec![0.02f32; nkv]);
            vb.write_f32(&vec![0.03f32; nkv]);
            match flash_attn_swa_h128_prefill(&gpu, &qb, &kb, &vb, &ob, b, t, h, hkv, 64, 1.0) {
                Ok(()) => {
                    gpu.synchronize().unwrap();
                    let tfa = std::time::Instant::now();
                    let fa_iters = 40usize;
                    for _ in 0..fa_iters {
                        let _ = flash_attn_swa_h128_prefill(
                            &gpu, &qb, &kb, &vb, &ob, b, t, h, hkv, 64, 1.0,
                        );
                    }
                    gpu.synchronize().unwrap();
                    let fa_us = tfa.elapsed().as_secs_f64() * 1e6 / fa_iters as f64;
                    println!("GPU FA SWA h128 stub prefill B=1 T={t} H={h}: {fa_us:.1} µs/call");
                }
                Err(e) => println!("GPU FA SWA h128 stub microbench skipped: {e}"),
            }
        }
    }

    let b = 1u32;
    let t = 16u32;
    let h = 8u32;
    let hkv = 2u32;
    let d = 256usize;
    let nq = (b * t * h) as usize * d;
    let nkv = (b * t * hkv) as usize * d;
    let qb = gpu.rt.alloc_buffer(nq * 4).unwrap();
    let kb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
    let vb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
    let ob = gpu.rt.alloc_buffer(nq * 4).unwrap();
    qb.write_f32(&vec![0.01f32; nq]);
    kb.write_f32(&vec![0.02f32; nkv]);
    vb.write_f32(&vec![0.03f32; nkv]);
    flash_attn_swa_h256_prefill(&gpu, &qb, &kb, &vb, &ob, b, t, h, hkv, 512, 1.0).unwrap();
    gpu.synchronize().unwrap();
    let t3 = std::time::Instant::now();
    let fa_iters = 30usize;
    for _ in 0..fa_iters {
        flash_attn_swa_h256_prefill(&gpu, &qb, &kb, &vb, &ob, b, t, h, hkv, 512, 1.0).unwrap();
    }
    gpu.synchronize().unwrap();
    let fa_us = t3.elapsed().as_secs_f64() * 1e6 / fa_iters as f64;
    println!("GPU FA SWA h256 prefill B=1 T={t} H={h}: {fa_us:.1} µs/call");

    thermal_exit_code(thermal, &thermal_reports)
}

/// Collapse the sustained-load reports into the process exit code.
///
/// The whole point of the four-valued [`gemma_metal::thermal::Verdict`] is that
/// it survives to the shell: FAIL is 1, and "requested but produced no verdict"
/// is 2. Neither collapses into 0.
fn thermal_exit_code(
    requested: bool,
    reports: &[gemma_metal::thermal::ThermalReport],
) -> i32 {
    if !requested {
        return 0;
    }
    println!();
    println!("=== Sustained-load gate summary ===");
    if reports.is_empty() {
        println!("  thermal gate REQUESTED but no session ran it — exit 2, not a pass");
        return 2;
    }
    for r in reports {
        println!("  {}", r.summary_line());
    }
    if reports.iter().any(|r| !r.passed()) {
        let failed = reports
            .iter()
            .filter(|r| matches!(r.verdict, gemma_metal::thermal::Verdict::Fail(_)))
            .count();
        if failed > 0 {
            println!("  verdict: {failed} gate(s) FAILED — exit 1");
            return 1;
        }
        println!("  verdict: no gate failed, but at least one produced no verdict — exit 2");
        return 2;
    }
    println!("  verdict: all {} gate(s) PASSED — exit 0", reports.len());
    0
}

/// Parse a required numeric flag value. A missing or malformed value exits
/// non-zero rather than silently falling back to a default — a gate configured
/// by a typo is a gate that measures something nobody asked for.
fn num_arg<I: Iterator<Item = String>>(args: &mut I, flag: &str) -> f64 {
    let Some(raw) = args.next() else {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    };
    match raw.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => v,
        Ok(v) => {
            eprintln!("{flag} value {v} is not finite");
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("{flag} value {raw:?} is not a number: {e}");
            std::process::exit(2);
        }
    }
}

/// Parse a required positive-integer flag value. Kept separate from
/// [`num_arg`] so `--thermal-windows 3.7` and `--thermal-windows -1` are named
/// errors rather than a silent truncation to 3 and a saturating cast to 0.
fn count_arg<I: Iterator<Item = String>>(args: &mut I, flag: &str) -> usize {
    let Some(raw) = args.next() else {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            eprintln!("{flag} must be >= 1");
            std::process::exit(2);
        }
        Ok(v) => v,
        Err(e) => {
            eprintln!("{flag} value {raw:?} is not a positive whole number: {e}");
            std::process::exit(2);
        }
    }
}

/// Run the sustained-load gate against a live decode session.
///
/// The unit of work is `n` further decode steps on a session that keeps running
/// — see [`GpuDecodeSession::bench_decode_sustained`]. The session recycles
/// (reset + re-prefill) only when the global KV slot would otherwise fill, so
/// the prefill cost is paid once per `kv_capacity` tokens rather than once per
/// call. That keeps consecutive calls doing near-identical work, which is what
/// makes a change in window rate attributable to the machine.
fn run_thermal_gate(
    label: &str,
    sess: &mut GpuDecodeSession,
    prompt: &[u32],
    cfg: &gemma_metal::thermal::ThermalGateConfig,
    meta: serde_json::Value,
) -> gemma_metal::thermal::ThermalReport {
    use gemma_metal::thermal::run_sustained;

    // Leave room for the prefilled prompt plus the seed step, then a slot of
    // margin so a burst never *reaches* the capacity error.
    let headroom = match sess.kv_capacity() {
        Some(cap) => cap.saturating_sub(prompt.len() + 2).max(1),
        None => 512,
    };
    let mut cfg = cfg.clone();
    cfg.max_batch = Some(match cfg.max_batch {
        Some(existing) => existing.min(headroom),
        None => headroom,
    });

    println!();
    println!("=== Sustained-load (thermal) gate — {label} ===");
    println!(
        "  {:.0}s measured in {:.0}s windows after {:.0}s warmup; gates: tail/head ≥ {:.2}, rolling ≥ {:.2}{}",
        cfg.duration_secs,
        cfg.window_secs,
        cfg.warmup_secs,
        cfg.min_sustained_ratio,
        cfg.min_rolling_ratio,
        cfg.floor_tok_s
            .map(|f| format!(", floor ≥ {f:.1} tok/s"))
            .unwrap_or_default()
    );
    println!(
        "  burst = reset + prefill(T={}) + N steps; KV capacity {:?} ⇒ max N {}",
        prompt.len(),
        sess.kv_capacity(),
        headroom
    );

    let report = run_sustained(&cfg, label, |batch| {
        let want = batch.max(1);
        let done = sess
            .bench_decode_sustained(prompt, want)
            .map_err(|e| e.to_string())?;
        if done != want {
            return Err(format!(
                "sustained decode completed {done} of {want} requested steps"
            ));
        }
        Ok(done)
    });

    println!("  {}", report.summary_line());
    println!("  attribution: {}", report.attribution);
    for w in &report.windows {
        println!(
            "    window {:>2}: {:>6.2}s  {:>6} tok  {:>7.2} tok/s  rss={}  load={}  os={}{}",
            w.index,
            w.secs,
            w.tokens,
            w.tok_s,
            w.rss_mib
                .map(|r| format!("{r:.0} MiB"))
                .unwrap_or_else(|| "n/a".into()),
            w.load_avg
                .map(|l| format!("{l:.2}"))
                .unwrap_or_else(|| "n/a".into()),
            w.thermal_state.map(|s| s.as_str()).unwrap_or("n/a"),
            if w.low_power == Some(true) { " LOW-POWER" } else { "" }
        );
    }
    if report.dropped_partial_windows > 0 {
        println!(
            "    ({} partial trailing window(s) discarded — too short to compare)",
            report.dropped_partial_windows
        );
    }
    write_thermal_result(label, &report, &cfg, meta);
    report
}

/// Persist the gate artifact next to the other `bench/results/*.json`.
fn write_thermal_result(
    label: &str,
    report: &gemma_metal::thermal::ThermalReport,
    cfg: &gemma_metal::thermal::ThermalGateConfig,
    meta: serde_json::Value,
) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest.join("bench/results");
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("thermal: cannot create {}: {e}", out_dir.display());
        return;
    }
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut body = report.to_json(cfg);
    if let (Some(obj), Some(extra)) = (body.as_object_mut(), meta.as_object()) {
        for (k, v) in extra {
            obj.insert(k.clone(), v.clone());
        }
    }
    if let Some(obj) = body.as_object_mut() {
        obj.insert("unix_ts".into(), serde_json::json!(ts));
        obj.insert("host".into(), serde_json::json!(host_tag()));
    }
    let text = match serde_json::to_string_pretty(&body) {
        Ok(t) => t + "\n",
        Err(e) => {
            eprintln!("thermal: serialize failed: {e}");
            return;
        }
    };
    let path = out_dir.join(format!("thermal_gate_{label}_{ts}.json"));
    if let Err(e) = fs::write(&path, &text) {
        eprintln!("thermal: failed to write {}: {e}", path.display());
        return;
    }
    println!("  wrote {}", path.display());
    let latest = out_dir.join(format!("thermal_gate_{label}_latest.json"));
    if let Err(e) = fs::write(&latest, &text) {
        eprintln!("thermal: failed to write {}: {e}", latest.display());
    }
}

/// Best-effort host identifier so a gate artifact records what it ran on.
fn host_tag() -> String {
    let cpu = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-cpu".into());
    let os = std::process::Command::new("sw_vers")
        .args(["-productVersion"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-os".into());
    format!("{cpu} / macOS {os}")
}

fn is_31b_shape(layers: usize, hidden: usize) -> bool {
    // Gemma4-31B: 60 layers × H=5376. E4B: 42 × 2560.
    layers >= 48 || hidden >= 4096
}

fn snapshot_hash_from_dir(dir: &std::path::Path) -> Option<String> {
    // HF hub layout: .../snapshots/<40-hex>/
    let name = dir.file_name()?.to_str()?;
    if name.len() >= 32 && name.chars().all(|c| c.is_ascii_hexdigit()) {
        return Some(name.to_string());
    }
    for anc in dir.ancestors().take(4) {
        if anc.file_name().and_then(|s| s.to_str()) == Some("snapshots") {
            continue;
        }
        if let Some(parent) = anc.parent() {
            if parent.file_name().and_then(|s| s.to_str()) == Some("snapshots") {
                let n = anc.file_name()?.to_str()?;
                if n.len() >= 32 && n.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Some(n.to_string());
                }
            }
        }
    }
    None
}

fn snapshot_mtime_iso(dir: &std::path::Path) -> Option<String> {
    let meta = fs::metadata(dir).ok()?;
    let modified = meta.modified().ok()?;
    let secs = modified.duration_since(UNIX_EPOCH).ok()?.as_secs();
    // UTC YYYY-MM-DD for R8 / artifact hygiene (no chrono dep).
    // Days since Unix epoch → civil date (Howard Hinnant algorithm).
    let z = (secs / 86_400) as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp as i64 + if mp < 10 { 3 } else { -9 };
    let y = y + i64::from(m <= 2);
    Some(format!("{y:04}-{m:02}-{d:02}"))
}

fn model_id_from_dir(dir: &std::path::Path, is_31b: bool) -> String {
    let s = dir.to_string_lossy();
    if s.contains("gemma-4-31b") || s.contains("gemma-4-31B") {
        "mlx-community/gemma-4-31b-it-4bit".into()
    } else if s.contains("gemma-4-e4b") || s.contains("gemma-4-E4B") {
        "mlx-community/gemma-4-e4b-it-4bit".into()
    } else if is_31b {
        "mlx-community/gemma-4-31b-it-4bit".into()
    } else {
        "mlx-community/gemma-4-e4b-it-4bit".into()
    }
}

fn write_real_model_result(
    ttft_ms: Option<f64>,
    tok_s: Option<f64>,
    steps: usize,
    dir: &std::path::Path,
    layers: usize,
    hidden: usize,
    vocab: usize,
) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest.join("bench/results");
    let _ = fs::create_dir_all(&out_dir);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let is_31b = is_31b_shape(layers, hidden);
    let model = model_id_from_dir(dir, is_31b);
    let snap = snapshot_hash_from_dir(dir).unwrap_or_else(|| "unknown".into());
    let snap_date = snapshot_mtime_iso(dir).unwrap_or_else(|| "unknown".into());
    let prefix = if is_31b { "31b" } else { "e4b" };
    let path = out_dir.join(format!("run_{prefix}_gemma_metal_{ts}.json"));
    let notes = if is_31b {
        format!(
            "live dims from session; snapshot={snap} cached={snap_date}; \
             Q4Mlx Hot decode; product ≥15 unmet unless fusion+accept"
        )
    } else {
        format!(
            "live dims from session; snapshot={snap} cached={snap_date}; \
             Q4Mlx Hot decode; vs mlx~75.7 / gate 48-60"
        )
    };
    let body = serde_json::json!({
        "runtime": "gemma-metal",
        "model": model,
        "weights_dir": dir.display().to_string(),
        "snapshot_hash": snap,
        "snapshot_cached_date": snap_date,
        "lane": "honest-partial",
        "layers": layers,
        "hidden": hidden,
        "vocab": vocab,
        "decode_tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "decode_steps": steps,
        "notes": notes,
        "measured_at_unix": ts,
    });
    let text = serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".into());
    if let Err(e) = fs::write(&path, text + "\n") {
        eprintln!("failed to write {}: {e}", path.display());
    } else {
        println!("  wrote {}", path.display());
        let latest = out_dir.join(format!("latest_{prefix}_gemma_metal.json"));
        let _ = fs::copy(&path, &latest);
        println!("  wrote {}", latest.display());
        // Never clobber the other family's latest_* with this run.
        if is_31b {
            let stale = out_dir.join("latest_e4b_gemma_metal.json");
            if let Ok(prev) = fs::read_to_string(&stale) {
                if prev.contains("gemma-4-31b") || prev.contains("\"layers\": 60") {
                    eprintln!(
                        "  note: prior latest_e4b_* looked mislabeled; left untouched (new 31B → latest_31b_*)"
                    );
                }
            }
        }
    }
}

fn mean_accept(accepts: &[gemma_metal::step_verify::BlockAccept]) -> f64 {
    if accepts.is_empty() {
        return 0.0;
    }
    let s: f64 = accepts.iter().map(|a| a.verify.accepted as f64).sum();
    s / accepts.len() as f64
}

fn run_dflash_gates(run_31b: bool) {
    println!();
    println!("=== DFlash parity / tok-s gates ===");
    println!("  GPU draft (Hot Q4 + D=128 FA) · step_verify uses Q4 GEMM + FA(Tq=M) when M>1");

    // --- Synthetic E4B-mini (always when --dflash) ---
    let Ok(host) = SyntheticE4bGraph::mini_parity() else {
        println!("  synthetic mini unavailable — skip");
        return;
    };
    let gpu_model = match GpuSynthModel::from_synthetic(host, QuantScheme::q4_mlx_default()) {
        Ok(m) => m,
        Err(e) => {
            println!("  GPU Hot synthetic unavailable — continue ({e})");
            if run_31b {
                let mut body = serde_json::json!({
                    "runtime": "gemma-metal",
                    "lane": "dflash-parity-gates",
                    "host": "Apple M5 Pro",
                    "date_utc": chrono_like_utc(),
                    "synthetic_mini": {"error": e.to_string()},
                });
                match run_dflash_31b_inner() {
                    Ok(v) => body["real_31b"] = v,
                    Err(err) => {
                        println!("  31B DFlash bench skipped/failed: {err}");
                        body["real_31b"] = serde_json::json!({"error": err});
                    }
                }
                write_dflash_result(&body);
            }
            return;
        }
    };
    if gpu_model.gpu.rt.pipeline(KernelId::GemvQ4Mlx.entry_name()).is_err()
        && gpu_model.gpu.rt.pipeline("gemv_q4").is_err()
    {
        println!("  Metal gemv_q4 / gemv_q4_mlx unavailable — skip");
        return;
    }
    if gpu_model
        .gpu
        .rt
        .pipeline(KernelId::FlashAttnSwaH128.entry_name())
        .is_err()
        || gpu_model
            .gpu
            .rt
            .pipeline(KernelId::MlpSilu.entry_name())
            .is_err()
    {
        println!("  h128 FA / mlp_silu unavailable — skip");
        return;
    }
    let mut sess = match GpuDecodeSession::new(gpu_model) {
        Ok(s) => s,
        Err(e) => {
            println!("  session failed: {e}");
            return;
        }
    };
    let prompt = [3u32, 4, 5, 6];
    let max_new = 16usize;
    let max_ctx = prompt.len() + max_new + 16;

    // Greedy baseline (capture-off tok/s — honest decode speed; hazard skip-auto default).
    let greedy_speed = match sess.generate(&prompt, max_new) {
        Ok(g) => g,
        Err(e) => {
            println!("  greedy generate failed: {e}");
            return;
        }
    };
    sess.model.gpu.synchronize().ok();
    let t0 = std::time::Instant::now();
    let _ = sess.generate(&prompt, max_new);
    sess.model.gpu.synchronize().ok();
    let greedy_secs = t0.elapsed().as_secs_f64();
    let greedy_new = greedy_speed.len().saturating_sub(prompt.len()).max(1);
    let greedy_tps = greedy_new as f64 / greedy_secs;
    let _ = greedy_speed;

    // --- Exactness harness (fresh session + short prompt; always-on barriers) -
    // Prior mini decode in this process can leave schedule state that flips
    // 506↔507 near-ties. Remake the Hot model so exactness matches the unit test.
    tessl::ab_flags::set_hazard_barriers(false);
    println!("  exactness lane: fresh session + always-on barriers (prompt [3,4,5] n=6)");
    let exact_prompt = [3u32, 4, 5];
    let exact_new = 6usize;
    let exact_bs = 3usize;
    let (exact, accepts_exact_mean, exact_tail) = {
        let host = match SyntheticE4bGraph::mini_parity() {
            Ok(h) => h,
            Err(e) => {
                println!("  exactness mini graph failed: {e}");
                return;
            }
        };
        let gpu_model = match GpuSynthModel::from_synthetic(host, QuantScheme::q4_mlx_default()) {
            Ok(m) => m,
            Err(e) => {
                println!("  exactness Hot upload failed: {e}");
                return;
            }
        };
        tessl::ab_flags::set_hazard_barriers(false);
        let mut es = match GpuDecodeSession::new(gpu_model) {
            Ok(s) => s,
            Err(e) => {
                println!("  exactness session failed: {e}");
                return;
            }
        };
        let host_draft_exact = match HostDFlashDraft::synthetic_mini() {
            Ok(d) => d,
            Err(e) => {
                println!("  synthetic draft failed: {e}");
                return;
            }
        };
        es.enable_hidden_capture(host_draft_exact.cfg.target_layer_ids.clone())
            .ok();
        tessl::ab_flags::set_hazard_barriers(false);
        let greedy = match es.generate(&exact_prompt, exact_new) {
            Ok(g) => g,
            Err(e) => {
                println!("  capture-on greedy failed: {e}");
                return;
            }
        };
        tessl::ab_flags::set_hazard_barriers(false);
        let greedy2 = match es.generate(&exact_prompt, exact_new) {
            Ok(g) => g,
            Err(e) => {
                println!("  capture-on greedy #2 failed: {e}");
                return;
            }
        };
        let greedy_stable = greedy == greedy2;
        if !greedy_stable {
            println!(
                "  WARN: capture-on greedy not self-consistent ({:?} vs {:?})",
                &greedy[exact_prompt.len()..],
                &greedy2[exact_prompt.len()..]
            );
        }
        es.disable_hidden_capture();
        tessl::ab_flags::set_hazard_barriers(false);
        let mut draft_exact =
            match DFlashGpuDraft::from_draft(&es.model.gpu, &host_draft_exact, max_ctx) {
                Ok(d) => d,
                Err(e) => {
                    println!("  GPU draft upload failed: {e}");
                    return;
                }
            };
        let cond_exact =
            match DFlashGpuConditioner::from_draft(&es.model.gpu, &host_draft_exact, max_ctx) {
                Ok(c) => c,
                Err(e) => {
                    println!("  conditioner failed: {e}");
                    return;
                }
            };
        es.attach_gpu_conditioner(cond_exact).ok();
        let (dflash_exact_out, accepts_exact) = match generate_with_dflash(
            &mut es,
            &mut draft_exact,
            &exact_prompt,
            exact_new,
            Some(exact_bs),
        ) {
            Ok(v) => v,
            Err(e) => {
                println!("  exactness generate_with_dflash failed: {e}");
                return;
            }
        };
        let n_exact = greedy
            .len()
            .min(dflash_exact_out.len())
            .saturating_sub(exact_prompt.len());
        let greedy_tail = &greedy[exact_prompt.len()..exact_prompt.len() + n_exact];
        let dflash_tail = &dflash_exact_out[exact_prompt.len()..exact_prompt.len() + n_exact];
        let ok = greedy_stable
            && compare_token_stream("mini_dflash_vs_greedy", dflash_tail, greedy_tail)
                .map(|r| r.pass(1e-6, 0.999))
                .unwrap_or(false);
        (
            ok,
            mean_accept(&accepts_exact),
            dflash_tail.to_vec(),
        )
    };
    println!(
        "  mini exactness vs capture-on greedy (bs={exact_bs}, n={}): {}  mean_accept={:.2}",
        exact_tail.len(),
        if exact { "PASS" } else { "FAIL" },
        accepts_exact_mean
    );
    if exact {
        println!("  exact stream new={exact_tail:?}");
    }

    // Throughput sweep on the long-lived session (hazard skip-auto restored).
    tessl::ab_flags::set_hazard_barriers(true);

    // Block-size sweep on GPU draft + verify
    let mut best_bs = DFLASH_DEFAULT_BLOCK;
    let mut best_tps = 0.0f64;
    let mut sweep = Vec::new();
    for bs in [2usize, 3, 4, 5, 6, 8] {
        let host_draft = match HostDFlashDraft::synthetic_mini() {
            Ok(d) => d,
            Err(e) => {
                println!("  synthetic draft failed: {e}");
                return;
            }
        };
        let mut draft = match DFlashGpuDraft::from_draft(&sess.model.gpu, &host_draft, max_ctx) {
            Ok(d) => d,
            Err(e) => {
                println!("  GPU draft upload failed: {e}");
                return;
            }
        };
        let cond = match DFlashGpuConditioner::from_draft(&sess.model.gpu, &host_draft, max_ctx) {
            Ok(c) => c,
            Err(e) => {
                println!("  conditioner failed: {e}");
                return;
            }
        };
        sess.attach_gpu_conditioner(cond).ok();
        // Warm once (speed lane = ambient hazard after steer/capture drop).
        let _ = generate_with_dflash_speed(&mut sess, &mut draft, &prompt, 4, Some(bs));
        // generate teardown drops conditioner — re-attach
        let host_draft = HostDFlashDraft::synthetic_mini().unwrap();
        let mut draft = DFlashGpuDraft::from_draft(&sess.model.gpu, &host_draft, max_ctx).unwrap();
        let cond = DFlashGpuConditioner::from_draft(&sess.model.gpu, &host_draft, max_ctx).unwrap();
        sess.attach_gpu_conditioner(cond).unwrap();
        sess.model.gpu.synchronize().ok();
        let t1 = std::time::Instant::now();
        let (out, accepts) =
            match generate_with_dflash_speed(&mut sess, &mut draft, &prompt, max_new, Some(bs)) {
                Ok(v) => v,
                Err(e) => {
                    println!("  generate_with_dflash_speed bs={bs} failed: {e}");
                    continue;
                }
            };
        sess.model.gpu.synchronize().ok();
        let secs = t1.elapsed().as_secs_f64();
        let n_new = out.len().saturating_sub(prompt.len()).max(1);
        let tps = n_new as f64 / secs;
        let ma = mean_accept(&accepts);
        println!(
            "  mini block={bs}: {tps:.1} tok/s  mean_accept={ma:.2}  new={n_new}"
        );
        sweep.push((bs, tps, ma, n_new));
        if tps > best_tps {
            best_tps = tps;
            best_bs = bs;
        }
    }

    println!("  mini greedy baseline (hazard skip-auto): {greedy_tps:.1} tok/s");
    // Always-on greedy retained for exactness comparison only.
    tessl::ab_flags::set_hazard_barriers(false);
    sess.model.gpu.synchronize().ok();
    let _ = sess.generate(&prompt, max_new);
    sess.model.gpu.synchronize().ok();
    let t_ao = std::time::Instant::now();
    let _ = sess.generate(&prompt, max_new);
    sess.model.gpu.synchronize().ok();
    let greedy_ao_tps = greedy_new as f64 / t_ao.elapsed().as_secs_f64().max(1e-9);
    tessl::ab_flags::set_hazard_barriers(true);
    println!("  mini greedy always-on barriers: {greedy_ao_tps:.1} tok/s");
    println!(
        "  mini block retune: best_bs={best_bs} @ {best_tps:.1} tok/s  (MLX prior was 5)"
    );
    let beat_hazard = best_tps + 1e-6 >= greedy_tps;
    let beat_ao = best_tps + 1e-6 >= greedy_ao_tps;
    // Speed lane uses generate_with_dflash_speed → barrier-matched to hazard greedy.
    if beat_hazard {
        println!(
            "  mini DFlash ≥ hazard greedy ({best_tps:.1} ≥ {greedy_tps:.1})"
        );
    }
    if beat_ao {
        println!(
            "  mini DFlash ≥ always-on greedy ({best_tps:.1} ≥ {greedy_ao_tps:.1})"
        );
    }
    let greedy_tps_gate = greedy_tps; // speed-lane barrier mode

    let mut body = serde_json::json!({
        "runtime": "gemma-metal",
        "lane": "dflash-parity-gates",
        "host": "Apple M5 Pro",
        "date_utc": chrono_like_utc(),
        "synthetic_mini": {
            "greedy_tok_s": greedy_tps,
            "best_block": best_bs,
            "best_dflash_tok_s": best_tps,
            "exact_vs_greedy": exact,
            "exact_prompt": exact_prompt.to_vec(),
            "exact_max_new": exact_new,
            "exact_block": exact_bs,
            "mean_accept_at_exact": accepts_exact_mean,
            "exactness_lane": "always_on_dispatch_barriers",
            "block_sweep": sweep.iter().map(|(bs, tps, ma, n)| {
                serde_json::json!({"block": bs, "tok_s": tps, "mean_accept": ma, "n_new": n})
            }).collect::<Vec<_>>(),
            "notes": "GPU D=128 draft + Q4 GEMM/FA(Tq=M) verify; not product E4B (no DFlash draft for E4B)"
        }
    });

    if run_31b {
        match run_dflash_31b_inner() {
            Ok(v) => {
                body["real_31b"] = v;
            }
            Err(e) => {
                println!("  31B DFlash bench skipped/failed: {e}");
                body["real_31b"] = serde_json::json!({"error": e});
            }
        }
    } else {
        println!("  (skip real 31B — pass --dflash-31b when weights + quiet GPU allow)");
        body["real_31b"] = serde_json::json!(null);
    }

    let exact_31b = body
        .get("real_31b")
        .and_then(|v| v.get("exact_vs_greedy"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Sprint: mean_accept>0 and DFlash ≥ barrier-matched (hazard) greedy.
    let beat_greedy = beat_hazard;
    let accept_speed_done = exact && accepts_exact_mean > 0.0 && beat_greedy;
    let task = if exact || exact_31b {
        "completed"
    } else {
        "pending"
    };
    let accept_task = if accept_speed_done {
        "completed"
    } else {
        "pending"
    };
    body["synthetic_mini"]["greedy_tok_s_hazard"] = serde_json::json!(greedy_tps);
    body["synthetic_mini"]["greedy_tok_s_always_on"] = serde_json::json!(greedy_ao_tps);
    body["synthetic_mini"]["greedy_tok_s"] = serde_json::json!(greedy_tps_gate);
    body["synthetic_mini"]["notes"] = serde_json::json!(
        "Q4Mlx mini + dual act scratch; M×GEMV verify (H=256); steered short-circuit; speed lane=hazard; exactness=always-on"
    );
    body["gate_verdict"] = serde_json::json!({
        "dflash_parity_gates": "dual_scratch_steered_mini_speed_hazard",
        "verify_path": "Q4 GEMM + FA(Tq=M) when M>1 && cols>256; else M×GEMV (act scratch ×VERIFY_MAX_M)",
        "mlx_exact_verify": true,
        "gemma_metal_exact_vs_greedy": exact || exact_31b,
        "product_31b_decode_ge_15": false,
        "product_31b_mtp_ge_25": false,
        "tok_s": {
            "gemma_metal_mini_greedy_hazard": greedy_tps,
            "gemma_metal_mini_greedy_always_on": greedy_ao_tps,
            "gemma_metal_mini_dflash_best": best_tps,
            "mini_dflash_ge_greedy": beat_greedy,
            "mini_dflash_ge_greedy_always_on": beat_ao,
            "mini_dflash_ge_greedy_hazard": beat_hazard
        },
        "exactness": {
            "status": if exact || exact_31b { "PASS" } else { "FAIL_pending" },
            "task": "dflash-exact-parity",
            "task_status": task,
            "definition": "DFlash stream == capture-on greedy under always-on Dispatch barriers (short mini prompt)",
            "mini_pass": exact,
            "real_31b_pass": exact_31b,
            "root_causes_fixed": [
                "Draft/conditioner gemv_bf16_x into plain-Q4 gemv_q4 poisoned shared GPU",
                "Device-side capture (copy_f32) + deferred host assemble after argmax readback",
                "Missing lm_head→softcap RAW barrier under hazard skip-auto (GPU argmax 0 vs host winner)",
                "Exactness lane forces always-on Dispatch barriers so mini near-ties are bit-stable",
                "Capture-on steps force always-on Dispatch barriers (hazard skip-auto collapsed L46/57 absmean→0)",
                "Conditioner fc plain Q8 g64 (was Q4; h_ctx absmean now ≈ MLX 0.0699)"
            ],
            "root_cause_remaining": if exact || exact_31b {
                serde_json::Value::Null
            } else {
                serde_json::json!("exactness still FAIL under always-on barriers")
            },
            "capture_note": "Capture steps force AO; product outer lane stays hazard. Mid-layer copy can still shift ultra-near ties vs capture-off"
        },
        "accept_speed": {
            "task": "dflash-accept-speed",
            "task_status": accept_task,
            "mean_accept_at_exact": accepts_exact_mean,
            "mini_dflash_ge_greedy": beat_greedy,
            "root_causes_fixed": [
                "Synthetic mini draft mask-echo → steer MASK→anchor + steered short-circuit propose",
                "Mini exactness always-on; speed lane restores ambient hazard after capture drop",
                "Dual act-scratch ×VERIFY_MAX_M + gemm_q4_mlx for product cols>256; mini M×GEMV",
                "31B: always-on-while-capture + Q8 conditioner → mean_accept≈1.3 @ bs=5 (sweep)"
            ],
            "root_cause_remaining": if beat_greedy {
                serde_json::Value::Null
            } else {
                serde_json::json!("DFlash still below hazard-matched greedy on speed lane; draft Q4g64 proposals ≠ MLX dense/Q4 stream tokens (native target_next≠531 on short prompt)")
            },
            "notes": "ge_greedy compares speed-lane DFlash vs hazard greedy (barrier-matched)"
        }
    });

    write_dflash_result(&body);
}

fn chrono_like_utc() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Keep ISO-ish without chrono dep: seconds since epoch is enough for artifacts.
    format!("{ts}")
}

fn cksum_f32_row(v: &[f32]) -> serde_json::Value {
    let sum: f32 = v.iter().sum();
    let absmean = if v.is_empty() {
        0.0
    } else {
        v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32
    };
    let first8: Vec<f32> = v.iter().take(8).copied().map(|x| (x * 1e5).round() / 1e5).collect();
    serde_json::json!({
        "sum": (sum * 1e4).round() / 1e4,
        "absmean": (absmean * 1e6).round() / 1e6,
        "first8": first8,
        "shape": [v.len()],
    })
}

/// Block-1 dump vs `golden_intermediates_31b.json` / MLX stream protocol.
///
/// Golden JSON uses `block=[prompt[-1], MASK…]` (pre-sample). Stream (and native
/// generate) use `block=[anchor, MASK…]` after sampling the first token. MLX Q4g64
/// stream proposed ≈ `[602, 236787, 532, 532]`; dense ≈ `[602, 607, 532, 532]`.
fn dump_31b_block1_intermediates(
    sess: &mut GpuDecodeSession,
    draft: &mut DFlashGpuDraft,
    host_draft: &mut HostDFlashDraft,
    prompt: &[u32],
    max_ctx: usize,
) -> Result<serde_json::Value, String> {
    // Dump under always-on so RAW capture copies and softcap/argmax match decode graph.
    let prev_hazard = tessl::ab_flags::hazard_barriers();
    tessl::ab_flags::set_hazard_barriers(false);
    let cond = DFlashGpuConditioner::from_draft(&sess.model.gpu, host_draft, max_ctx)
        .map_err(|e| format!("dump conditioner: {e}"))?;
    sess.attach_gpu_conditioner(cond)
        .map_err(|e| format!("dump attach: {e}"))?;
    draft
        .bind_from_session(&sess.model.gpu, sess)
        .map_err(|e| format!("dump bind: {e}"))?;
    draft.reset_cache();
    sess.reset();

    for &t in &prompt[..prompt.len() - 1] {
        sess.step_prefill(t).map_err(|e| format!("dump prefill: {e}"))?;
    }
    let anchor = sess
        .step(prompt[prompt.len() - 1])
        .map_err(|e| format!("dump step: {e}"))?;

    let (concat, t) = sess
        .captured_concat()
        .map_err(|e| format!("dump concat: {e}"))?;
    let h = sess.model.hidden;
    let n_cap = host_draft.cfg.target_layer_ids.len();
    let last_concat = &concat[(t - 1) * n_cap * h..t * n_cap * h];
    let mut target_hidden_per_layer = Vec::with_capacity(n_cap);
    for i in 0..n_cap {
        let row = &last_concat[i * h..(i + 1) * h];
        target_hidden_per_layer.push(cksum_f32_row(row));
    }

    let fc_out = sess
        .read_conditioner_fc_out()
        .map_err(|e| format!("dump fc_out: {e}"))?;
    let h_ctx = sess
        .read_conditioner_h_ctx()
        .map_err(|e| format!("dump h_ctx: {e}"))?;
    let h_ctx_last = &h_ctx[(t - 1) * h..t * h];
    let h_ctx_len = sess.conditioner_h_ctx_len();
    let ctx_t = h_ctx_len; // first block: full prompt rows

    // Dense host fc→h_ctx (matches MLX golden / unquantized draft.fc).
    let host_h_ctx = host_draft
        .h_ctx_from_capture(&concat, t)
        .map_err(|e| format!("dump host h_ctx: {e}"))?;
    let host_h_ctx_last = &host_h_ctx[(t - 1) * h..t * h];

    let bs = 5usize;
    let mask = host_draft.cfg.mask_token_id;
    let mut block_stream = vec![anchor];
    for _ in 0..bs - 1 {
        block_stream.push(mask);
    }
    let mut block_golden = vec![prompt[prompt.len() - 1]];
    for _ in 0..bs - 1 {
        block_golden.push(mask);
    }

    let h_ctx_buf = sess
        .conditioner_h_ctx_buf()
        .map_err(|e| format!("dump h_ctx buf: {e}"))?;
    draft.reset_cache();
    let proposed_stream = draft
        .propose_block(
            &sess.model.gpu,
            &block_stream,
            h_ctx_buf,
            h_ctx_len,
            ctx_t,
        )
        .map_err(|e| format!("dump gpu propose stream: {e}"))?;
    draft.reset_cache();
    let h_ctx_buf = sess
        .conditioner_h_ctx_buf()
        .map_err(|e| format!("dump h_ctx buf2: {e}"))?;
    let proposed_golden_proto = draft
        .propose_block(
            &sess.model.gpu,
            &block_golden,
            h_ctx_buf,
            h_ctx_len,
            ctx_t,
        )
        .map_err(|e| format!("dump gpu propose golden: {e}"))?;

    // Host dense draft (slow lm_head) — opt-in; localizes GPU Q4 vs dense.
    let proposed_host_dense = if std::env::var("GEMMA_METAL_DUMP_HOST_DENSE")
        .ok()
        .as_deref()
        == Some("1")
    {
        host_draft
            .bind_from_session(sess)
            .map_err(|e| format!("dump host bind: {e}"))?;
        host_draft.reset_cache();
        Some(
            host_draft
                .propose_block(&block_stream, &host_h_ctx, ctx_t)
                .map_err(|e| format!("dump host propose: {e}"))?,
        )
    } else {
        None
    };

    let body = serde_json::json!({
        "input_ids": prompt,
        "target_layer_ids": host_draft.cfg.target_layer_ids,
        "block_size": bs,
        "mask_token_id": mask,
        "embed_scale": sess.model.embed_scale,
        "target_next_argmax": anchor,
        "target_hidden_per_layer": target_hidden_per_layer,
        "fc_out_lastrow": cksum_f32_row(&fc_out),
        "h_ctx_lastrow": cksum_f32_row(h_ctx_last),
        "h_ctx_lastrow_host_dense_fc": cksum_f32_row(host_h_ctx_last),
        "proposed_block_tokens": proposed_stream,
        "proposed_block_tokens_golden_protocol": proposed_golden_proto,
        "proposed_block_tokens_host_dense": proposed_host_dense,
        "mlx_refs": {
            "golden_protocol_dense": [14359, 532, 107, 563],
            "stream_protocol_dense": [602, 607, 532, 532],
            "stream_protocol_q4g64": [602, 236787, 532, 532],
            "target_next_argmax": 531,
            "h_ctx_absmean": 0.069892,
            "fc_out_absmean": 38.93103,
        },
        "draft_attn_offsets_per_layer": (0..5).map(|i| serde_json::json!({
            "S": ctx_t,
            "Lq": bs,
            "q_rope_offset": ctx_t,
            "ctx_rope_offset": 0,
            "is_sliding": i < 4,
            "sliding_window": if i < 4 { serde_json::json!(2048) } else { serde_json::Value::Null },
        })).collect::<Vec<_>>(),
        "note": "proposed_block_tokens = native GPU Q4g64 stream protocol (matches generate_with_dflash). Compare to mlx stream_protocol_q4g64.",
    });

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/results");
    let _ = fs::create_dir_all(&out_dir);
    let path = out_dir.join("native_intermediates_31b.json");
    let text = serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".into());
    fs::write(&path, text + "\n").map_err(|e| format!("write dump: {e}"))?;
    println!("  wrote {}", path.display());
    println!(
        "  dump: anchor={anchor} gpu_stream={proposed_stream:?} gpu_golden_proto={proposed_golden_proto:?} host_dense={proposed_host_dense:?}"
    );
    println!(
        "  dump: h_ctx absmean gpu={:.6} host_fc={:.6} (mlx {:.6})",
        h_ctx_last.iter().map(|x| x.abs()).sum::<f32>() / h as f32,
        host_h_ctx_last.iter().map(|x| x.abs()).sum::<f32>() / h as f32,
        0.069892f32
    );
    tessl::ab_flags::set_hazard_barriers(prev_hazard);
    sess.disable_hidden_capture();
    Ok(body)
}

fn run_dflash_31b_inner() -> Result<serde_json::Value, String> {
    // Correctness: always-on Device barriers by default. Product hazard skip-auto
    // (`GEMMA_METAL_31B_HAZARD=1`) still collapses short-prompt greedy → 236773
    // after NeoX RoPE fix; Lane B owns the hazard fix (F3).
    let use_hazard = std::env::var("GEMMA_METAL_31B_HAZARD").ok().as_deref() == Some("1");
    tessl::ab_flags::set_hazard_barriers(use_hazard);
    let exact_always_on = !use_hazard
        || std::env::var("GEMMA_METAL_31B_EXACT_ALWAYS_ON")
            .ok()
            .as_deref()
            == Some("1");
    println!(
        "  31B barrier lane: {} (set GEMMA_METAL_31B_HAZARD=1 for product skip-auto)",
        if use_hazard { "hazard skip-auto" } else { "always-on" }
    );

    let target = resolve_default_31b_mlx_cache()
        .ok_or_else(|| "no HF cache for mlx-community/gemma-4-31b-it-4bit".to_string())?;
    let draft_dir = resolve_default_dflash_draft_cache()
        .ok_or_else(|| "no HF cache for z-lab/gemma-4-31B-it-DFlash".to_string())?;
    println!("  31B target: {}", target.display());
    println!("  draft:      {}", draft_dir.display());

    let load_t0 = std::time::Instant::now();
    let banks = load_from_hf_dir(
        &target,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 512,
            ..LoadOptions::default()
        },
    )
    .map_err(|e| format!("load 31B: {e}"))?;
    println!(
        "  loaded {} mats in {:.1}s; hot≈{:.2} GiB",
        banks.matrices.len(),
        load_t0.elapsed().as_secs_f64(),
        banks.total_hot_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    // Host banks dropped inside from_host_banks before session KV/scratch alloc.
    let gpu_model =
        GpuSynthModel::from_host_banks(banks).map_err(|e| format!("Hot upload: {e}"))?;
    let mut sess =
        GpuDecodeSession::new(gpu_model).map_err(|e| format!("31b session: {e}"))?;
    // Re-assert barrier lane after GemmaGpu::new (init must not clobber; belt-and-suspenders).
    tessl::ab_flags::set_hazard_barriers(use_hazard);
    let mut host_draft =
        HostDFlashDraft::load_from_dir(&draft_dir).map_err(|e| format!("draft load: {e}"))?;

    // Prefer greet golden prompt for accept measurement — short ids [2,105,4368,1246]
    // MLX-correctly mode-lock to 237076 after token 531, which inflates late accepts
    // and understates early-block draft quality vs chat traffic.
    let prompt: Vec<u32> = {
        let gold_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bench/results/golden_tokens_31b.json");
        let use_greet = std::env::var("GEMMA_METAL_31B_SHORT_PROMPT")
            .ok()
            .as_deref()
            != Some("1");
        if use_greet {
            if let Ok(raw) = fs::read_to_string(&gold_path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
                    if let Some(case) = v["cases"].as_array().and_then(|a| {
                        a.iter().find(|c| c["name"].as_str() == Some("greet"))
                    }) {
                        if let Some(ids) = case["prompt_ids"].as_array() {
                            let p: Vec<u32> = ids
                                .iter()
                                .filter_map(|x| x.as_u64().map(|u| u as u32))
                                .collect();
                            if p.len() >= 4 {
                                println!(
                                    "  31B measure prompt=greet ({} toks; set GEMMA_METAL_31B_SHORT_PROMPT=1 for [2,105,4368,1246])",
                                    p.len()
                                );
                                p
                            } else {
                                vec![2u32, 105, 4368, 1246]
                            }
                        } else {
                            vec![2u32, 105, 4368, 1246]
                        }
                    } else {
                        vec![2u32, 105, 4368, 1246]
                    }
                } else {
                    vec![2u32, 105, 4368, 1246]
                }
            } else {
                vec![2u32, 105, 4368, 1246]
            }
        } else {
            println!("  31B measure prompt=short [2,105,4368,1246]");
            vec![2u32, 105, 4368, 1246]
        }
    };
    let max_new = 24usize;
    let max_ctx = prompt.len() + max_new + 32;
    let mut draft = DFlashGpuDraft::from_draft(&sess.model.gpu, &host_draft, max_ctx)
        .map_err(|e| format!("GPU draft upload: {e}"))?;

    // 31B free-decode without hidden capture collapses (240017/236773) even under
    // always-on; capture (incl. CAPTURE_NOP) yields target_next=531 / greet16 PASS.
    // Health + exactness therefore run capture-on (conditioner attaches capture).
    println!("  31B greedy warm (capture-on)…");
    {
        let cond = DFlashGpuConditioner::from_draft(&sess.model.gpu, &host_draft, max_ctx)
            .map_err(|e| format!("conditioner warm: {e}"))?;
        sess.attach_gpu_conditioner(cond)
            .map_err(|e| format!("attach cond warm: {e}"))?;
    }
    tessl::ab_flags::set_hazard_barriers(use_hazard);
    let _ = sess.generate(&prompt, 2).map_err(|e| format!("warmup: {e}"))?;
    sess.model.gpu.synchronize().ok();
    let t_g = std::time::Instant::now();
    let greedy_speed = sess
        .generate(&prompt, max_new)
        .map_err(|e| format!("greedy: {e}"))?;
    sess.model.gpu.synchronize().ok();
    let greedy_secs = t_g.elapsed().as_secs_f64();
    let greedy_new = greedy_speed.len().saturating_sub(prompt.len()).max(1);
    let greedy_tps = greedy_new as f64 / greedy_secs;
    println!("  31B greedy: {greedy_tps:.2} tok/s  ({greedy_new} new)");
    {
        let tail = &greedy_speed[prompt.len()..];
        let mut u: Vec<u32> = tail.to_vec();
        u.sort_unstable();
        u.dedup();
        println!(
            "  31B greedy unique={} all_same={} first={:?}",
            u.len(),
            u.len() <= 1,
            tail.iter().take(8).copied().collect::<Vec<_>>()
        );
        // Greet gold starts at 100; short prompt MLX next is 531.
        let expect_first = if prompt.as_slice() == [2u32, 105, 4368, 1246] {
            531u32
        } else {
            100u32
        };
        if tail.first().copied() != Some(expect_first) {
            return Err(format!(
                "31B greedy first token {:?} ≠ {expect_first} — refuse D-Flash",
                tail.first()
            ));
        }
        if u.len() <= 1 {
            return Err(format!(
                "31B greedy collapsed (unique={}) — refuse D-Flash measure on unhealthy target",
                u.len()
            ));
        }
    }
    sess.disable_hidden_capture();

    // Exactness baseline must attach conditioner too (deferred post-softcap
    // project). Capture-only vs capture+conditioner still flips early tokens
    // via GPU residency / CB packing even when FC is after argmax.
    // Dump runs LAST — draft propose_block can poison shared GPU scratch.
    if exact_always_on {
        tessl::ab_flags::set_hazard_barriers(false);
        println!("  31B exactness lane: ALWAYS-ON + capture+conditioner");
    } else {
        println!("  31B exactness lane: hazard-ON + capture+conditioner (post-softcap FC)");
    }
    {
        let cond = DFlashGpuConditioner::from_draft(&sess.model.gpu, &host_draft, max_ctx)
            .map_err(|e| format!("conditioner for greedy exact: {e}"))?;
        sess.attach_gpu_conditioner(cond)
            .map_err(|e| format!("attach cond for greedy exact: {e}"))?;
    }
    if exact_always_on {
        tessl::ab_flags::set_hazard_barriers(false);
    }
    let greedy = sess
        .generate(&prompt, max_new)
        .map_err(|e| format!("capture+cond greedy: {e}"))?;
    sess.disable_hidden_capture();
    {
        let tail = &greedy[prompt.len()..];
        let mut u: Vec<u32> = tail.to_vec();
        u.sort_unstable();
        u.dedup();
        println!(
            "  31B capture+cond greedy unique={} all_same={} new={:?}",
            u.len(),
            u.len() <= 1,
            tail
        );
        if u.len() <= 1 {
            return Err(format!(
                "31B capture+cond greedy collapsed (unique={}) — skip exactness/D-Flash",
                u.len()
            ));
        }
    }

    // Block sweep — GPU Q4Mlx draft (default) or host-dense draft
    // (`GEMMA_METAL_31B_HOST_DENSE_DRAFT=1`) for accept ceiling A/B.
    let use_host_dense = std::env::var("GEMMA_METAL_31B_HOST_DENSE_DRAFT")
        .ok()
        .as_deref()
        == Some("1");
    if use_host_dense {
        println!("  31B draft path: host-dense (slow; accept ceiling)");
    } else {
        println!("  31B draft path: GPU Q4Mlx");
    }
    tessl::ab_flags::set_hazard_barriers(use_hazard);
    let mut best_bs = DFLASH_DEFAULT_BLOCK;
    let mut best_tps = 0.0f64;
    let mut sweep = Vec::new();
    let mut first_block_debug = serde_json::Value::Null;
    for &bs in &[3usize, 5] {
        tessl::ab_flags::set_hazard_barriers(use_hazard);
        let cond = DFlashGpuConditioner::from_draft(&sess.model.gpu, &host_draft, max_ctx)
            .map_err(|e| format!("conditioner: {e}"))?;
        sess.attach_gpu_conditioner(cond)
            .map_err(|e| format!("attach cond: {e}"))?;
        if use_host_dense {
            let _ = generate_with_dflash_host(&mut sess, &mut host_draft, &prompt, 4, Some(bs));
            host_draft.reset_cache();
        } else {
            let _ = generate_with_dflash(&mut sess, &mut draft, &prompt, 4, Some(bs));
            draft.reset_cache();
        }
        let cond = DFlashGpuConditioner::from_draft(&sess.model.gpu, &host_draft, max_ctx)
            .map_err(|e| format!("conditioner: {e}"))?;
        sess.attach_gpu_conditioner(cond)
            .map_err(|e| format!("attach cond: {e}"))?;
        sess.model.gpu.synchronize().ok();
        let t1 = std::time::Instant::now();
        let (out, accepts) = if use_host_dense {
            generate_with_dflash_host(&mut sess, &mut host_draft, &prompt, max_new, Some(bs))
                .map_err(|e| format!("dflash-host bs={bs}: {e}"))?
        } else {
            generate_with_dflash(&mut sess, &mut draft, &prompt, max_new, Some(bs))
                .map_err(|e| format!("dflash bs={bs}: {e}"))?
        };
        sess.model.gpu.synchronize().ok();
        let secs = t1.elapsed().as_secs_f64();
        let n_new = out.len().saturating_sub(prompt.len()).max(1);
        let tps = n_new as f64 / secs;
        let ma = mean_accept(&accepts);
        let accept_lens: Vec<usize> = accepts.iter().map(|a| a.verify.accepted).collect();
        println!(
            "  31B block={bs}: {tps:.2} tok/s  mean_accept={ma:.2}  new={n_new}  accepts={accept_lens:?}"
        );
        if bs == 5 {
            first_block_debug = serde_json::json!({
                "accepts": accept_lens,
                "dflash_new": &out[prompt.len()..],
                "embed_scale": sess.model.embed_scale,
                "draft_attn_scale": (128.0f32).powf(-0.5),
                "draft_path": if use_host_dense { "host_dense" } else { "gpu_q4mlx" },
            });
        }
        sweep.push(serde_json::json!({
            "block": bs, "tok_s": tps, "mean_accept": ma, "n_new": n_new,
            "accepts": accept_lens
        }));
        if tps > best_tps {
            best_tps = tps;
            best_bs = bs;
        }
        if use_host_dense {
            host_draft.reset_cache();
        } else {
            draft.reset_cache();
        }
    }

    // Exactness @ best_bs (optional always-on for barrier-matched stream)
    if use_host_dense {
        host_draft.reset_cache();
    } else {
        draft.reset_cache();
    }
    if exact_always_on {
        tessl::ab_flags::set_hazard_barriers(false);
    } else {
        tessl::ab_flags::set_hazard_barriers(true);
    }
    let cond = DFlashGpuConditioner::from_draft(&sess.model.gpu, &host_draft, max_ctx)
        .map_err(|e| format!("conditioner: {e}"))?;
    sess.attach_gpu_conditioner(cond)
        .map_err(|e| format!("attach cond: {e}"))?;
    if exact_always_on {
        tessl::ab_flags::set_hazard_barriers(false);
    }
    let (dflash_out, accepts) = if use_host_dense {
        generate_with_dflash_host(&mut sess, &mut host_draft, &prompt, max_new, Some(best_bs))
            .map_err(|e| format!("dflash-host exactness: {e}"))?
    } else {
        generate_with_dflash(&mut sess, &mut draft, &prompt, max_new, Some(best_bs))
            .map_err(|e| format!("dflash exactness: {e}"))?
    };
    tessl::ab_flags::set_hazard_barriers(use_hazard);
    let n = greedy
        .len()
        .min(dflash_out.len())
        .saturating_sub(prompt.len());
    let greedy_tail = &greedy[prompt.len()..prompt.len() + n];
    let dflash_tail = &dflash_out[prompt.len()..prompt.len() + n];
    let report = compare_token_stream("31b_dflash_vs_greedy", dflash_tail, greedy_tail)
        .map_err(|e| format!("compare: {e}"))?;
    let exact = report.pass(1e-6, 0.999);
    println!(
        "  31B exactness vs capture-on greedy (bs={best_bs}, n={n}): {}  mean_accept={:.2}",
        if exact { "PASS" } else { "FAIL" },
        mean_accept(&accepts)
    );
    println!(
        "  31B vs MLX DFlash ~37 tok/s: custom best={best_tps:.2}  greedy={greedy_tps:.2}"
    );
    println!(
        "  note: verify path = Q4 GEMM + FA(Tq=M); product ≥25 still aspirational vs MLX NAX"
    );

    // Diagnostic dump is OPT-IN (GEMMA_METAL_DUMP_BLOCK1=1 + optional HOST_DENSE).
    // Default off: dump's propose_block + OOM risk previously killed the process
    // (exit 137) before gates JSON was written.
    if std::env::var("GEMMA_METAL_DUMP_BLOCK1").ok().as_deref() == Some("1") {
        let dump_prompt = [2u32, 105, 4368, 1246];
        let dump_ctx = dump_prompt.len() + 32;
        match dump_31b_block1_intermediates(
            &mut sess,
            &mut draft,
            &mut host_draft,
            &dump_prompt,
            dump_ctx.max(max_ctx),
        ) {
            Ok(dump) => {
                println!(
                    "  block1 dump ok; gpu_stream proposed={:?} mlx_q4g64=[602,236787,532,532]",
                    dump.get("proposed_block_tokens")
                );
            }
            Err(e) => println!("  block1 dump FAILED: {e}"),
        }
    }

    let mean_accept_bs5 = sweep
        .iter()
        .find(|s| s.get("block").and_then(|b| b.as_u64()) == Some(5))
        .and_then(|s| s.get("mean_accept").and_then(|m| m.as_f64()))
        .unwrap_or_else(|| mean_accept(&accepts));

    Ok(serde_json::json!({
        "target": target.display().to_string(),
        "draft": draft_dir.display().to_string(),
        "snapshot_hash": snapshot_hash_from_dir(&target),
        "snapshot_cached_date": snapshot_mtime_iso(&target),
        "layers": sess.model.cfg.num_hidden_layers,
        "hidden": sess.model.cfg.hidden_size,
        "vocab": sess.model.cfg.vocab_size,
        "barrier_lane": if use_hazard { "hazard" } else { "always_on" },
        "measure_prompt": if prompt.as_slice() == [2u32, 105, 4368, 1246] {
            "short"
        } else {
            "greet"
        },
        "prompt_len": prompt.len(),
        "greedy_tok_s": greedy_tps,
        "best_block": best_bs,
        "best_dflash_tok_s": best_tps,
        "exact_vs_greedy": exact,
        "mean_accept_at_best": mean_accept(&accepts),
        "mean_accept_at_bs5": mean_accept_bs5,
        "dflash_token_ids_new": dflash_tail,
        "greedy_token_ids_new": greedy_tail,
        "block_sweep": sweep,
        "debug_bs5": first_block_debug,
        "draft_path": if use_host_dense { "host_dense" } else { "gpu_q4mlx" },
        "notes": "greet prompt accept; AO+capture; Q4Mlx draft (or host-dense via env); live post-NeoX RoPE"
    }))
}

fn write_dflash_result(body: &serde_json::Value) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = manifest.join("bench/results");
    let _ = fs::create_dir_all(&out_dir);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = out_dir.join(format!("run_dflash_parity_gates_{ts}.json"));
    let text = serde_json::to_string_pretty(body).unwrap_or_else(|_| "{}".into());
    if let Err(e) = fs::write(&path, text + "\n") {
        eprintln!("failed to write {}: {e}", path.display());
    } else {
        println!("  wrote {}", path.display());
        let latest = out_dir.join("latest_dflash_parity_gates.json");
        let _ = fs::copy(&path, &latest);
        println!("  wrote {}", latest.display());
    }
}
