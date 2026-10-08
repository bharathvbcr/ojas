//! Compare native greedy stream vs `bench/results/golden_tokens_31b.json`.
//! Usage: golden_parity [case_name] [n_tokens]
//!        golden_parity --short-dump   # dump L0/L1 capture vs MLX golden on [2,105,4368,1246]
//! Env: METAL_RUNTIME_HAZARD_BARRIERS=0 recommended for stable compare.

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;
use gemma_metal::weights::{load_from_hf_dir, resolve_default_31b_mlx_cache, LoadOptions};
use serde_json::Value;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().any(|a| a == "--short-dump") {
        return short_dump();
    }
    if argv.iter().any(|a| a == "--free-gen") {
        return free_gen_short();
    }
    let case_name = argv.get(1).cloned().unwrap_or_else(|| "greet".into());
    let n_tokens: usize = argv.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);

    // Always-on before GPU create (GemmaGpu no longer clobbers an explicit set).
    tessl::ab_flags::set_hazard_barriers(false);

    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("bench/results/golden_tokens_31b.json");
    let raw = std::fs::read_to_string(&golden_path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
    let cases = v["cases"].as_array().ok_or("golden: missing cases")?;
    let case = cases
        .iter()
        .find(|c| c["name"].as_str() == Some(case_name.as_str()))
        .ok_or_else(|| format!("case '{case_name}' not found"))?;
    let prompt: Vec<u32> = case["prompt_ids"]
        .as_array()
        .ok_or("prompt_ids")?
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let gold: Vec<u32> = case["greedy_ids"]
        .as_array()
        .ok_or("greedy_ids")?
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let n = n_tokens.min(gold.len());

    let dir = resolve_default_31b_mlx_cache().ok_or("no 31b cache")?;
    eprintln!("case={case_name} n={n} dir={}", dir.display());
    let banks = load_from_hf_dir(
        &dir,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: (prompt.len() + n + 8).max(128),
            ..LoadOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let model = GpuSynthModel::from_host_banks(banks).map_err(|e| e.to_string())?;
    let mut sess = GpuDecodeSession::new(model).map_err(|e| e.to_string())?;
    tessl::ab_flags::set_hazard_barriers(false);
    // 31B free-decode without hidden capture collapses (even under always-on).
    // Capture (incl. CAPTURE_NOP) is required for MLX-matching greedy today.
    let skip_cap = std::env::var("GEMMA_METAL_NO_CAPTURE").ok().as_deref() == Some("1");
    if !skip_cap {
        sess.enable_hidden_capture(vec![0usize, 1, 2, 12, 23, 35, 46, 57])
            .map_err(|e| e.to_string())?;
        eprintln!("greet: hidden capture ON (set GEMMA_METAL_NO_CAPTURE=1 to A/B)");
    }
    let out = sess.generate(&prompt, n).map_err(|e| e.to_string())?;
    let got = &out[prompt.len()..];
    let mut first_mismatch = None;
    for (i, (a, b)) in got.iter().zip(gold.iter()).enumerate() {
        if a != b {
            first_mismatch = Some(i);
            break;
        }
    }
    let match_n = first_mismatch.unwrap_or(got.len().min(n));
    println!("match_prefix={match_n}/{n} first_mismatch={first_mismatch:?}");
    println!("got[:16]={:?}", &got[..got.len().min(16)]);
    println!("gold[:16]={:?}", &gold[..gold.len().min(16)]);
    if match_n < n {
        std::process::exit(1);
    }
    Ok(())
}

fn cksum(row: &[f32]) -> (f32, f32, Vec<f32>) {
    let sum: f32 = row.iter().sum();
    let absmean = row.iter().map(|x| x.abs()).sum::<f32>() / row.len().max(1) as f32;
    let first8: Vec<f32> = row
        .iter()
        .take(8)
        .copied()
        .map(|x| (x * 1e5).round() / 1e5)
        .collect();
    (sum, absmean, first8)
}

/// Free generate on the short prompt (no capture) — contrasts with `--short-dump`.
fn free_gen_short() -> Result<(), String> {
    tessl::ab_flags::set_hazard_barriers(false);
    let prompt = [2u32, 105, 4368, 1246];
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let with_cap = std::env::var("GEMMA_METAL_FREE_GEN_CAPTURE")
        .ok()
        .as_deref()
        == Some("1");
    let dir = resolve_default_31b_mlx_cache().ok_or("no 31b cache")?;
    eprintln!(
        "free-gen n={n} capture={with_cap} dir={}",
        dir.display()
    );
    let banks = load_from_hf_dir(
        &dir,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 64,
            ..LoadOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let model = GpuSynthModel::from_host_banks(banks).map_err(|e| e.to_string())?;
    let mut sess = GpuDecodeSession::new(model).map_err(|e| e.to_string())?;
    // Re-assert after GemmaGpu::new (must stick; prints help catch duplicate-static bugs).
    tessl::ab_flags::set_hazard_barriers(false);
    eprintln!(
        "hazard_skip_auto={} explicitly_set={}",
        tessl::ab_flags::hazard_barriers(),
        tessl::ab_flags::hazard_barriers_explicitly_set()
    );
    if with_cap {
        sess.enable_hidden_capture(vec![0usize, 1, 2, 12, 23, 35, 46, 57])
            .map_err(|e| e.to_string())?;
    }
    // Path A: step-prefill + step (same as short-dump)
    sess.reset();
    for &t in &prompt[..prompt.len() - 1] {
        sess.step_prefill(t).map_err(|e| e.to_string())?;
    }
    let step_next = sess
        .step(prompt[prompt.len() - 1])
        .map_err(|e| e.to_string())?;
    let ls = sess.debug_logits_stats();
    println!("step_path_next={step_next} (expect 531)");
    println!(
        "logits finite={} nan={} min={:.4} max={:.4}",
        ls.finite, ls.nan, ls.min, ls.max
    );

    // Path B: generate()
    let out = sess.generate(&prompt, n).map_err(|e| e.to_string())?;
    let got = &out[prompt.len()..];
    println!("generate_first8={:?}", &got[..got.len().min(8)]);
    let mut u = got.to_vec();
    u.sort_unstable();
    u.dedup();
    println!("generate_unique={} first={}", u.len(), got.first().copied().unwrap_or(0));

    // Path C: host-seeded continuation (rules out FromArgmax chain as sole cause)
    if step_next == 531 && n >= 2 {
        let mut p2 = prompt.to_vec();
        p2.push(531);
        sess.reset();
        for &t in &p2[..p2.len() - 1] {
            sess.step_prefill(t).map_err(|e| e.to_string())?;
        }
        let t2 = sess.step(*p2.last().unwrap()).map_err(|e| e.to_string())?;
        println!("host_seed_second={t2} (generate second was {:?})", got.get(1));
    }

    if step_next != 531 {
        eprintln!("FAIL: step path {step_next} != 531");
        std::process::exit(1);
    }
    Ok(())
}

/// Localize target_next 929≠531: capture L0/L1 after short prompt prefill.
fn short_dump() -> Result<(), String> {
    tessl::ab_flags::set_hazard_barriers(false);
    let prompt = [2u32, 105, 4368, 1246];
    let dir = resolve_default_31b_mlx_cache().ok_or("no 31b cache")?;
    eprintln!("short-dump dir={}", dir.display());
    let banks = load_from_hf_dir(
        &dir,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 64,
            ..LoadOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let model = GpuSynthModel::from_host_banks(banks).map_err(|e| e.to_string())?;
    let mut sess = GpuDecodeSession::new(model).map_err(|e| e.to_string())?;
    let caps = vec![0usize, 1, 2, 12, 23, 35, 46, 57];
    sess.enable_hidden_capture(caps.clone())
        .map_err(|e| e.to_string())?;
    for &t in &prompt[..prompt.len() - 1] {
        sess.step_prefill(t).map_err(|e| e.to_string())?;
    }
    let anchor = sess
        .step(prompt[prompt.len() - 1])
        .map_err(|e| e.to_string())?;
    let (concat, t) = sess.captured_concat().map_err(|e| e.to_string())?;
    let h = sess.model.hidden;
    let n_cap = caps.len();
    println!("target_next_argmax={anchor} (mlx golden 531) t={t}");
    // MLX refs: live dump same snapshot (bench/results/mlx_late_layers.txt).
    let mlx: &[(usize, f32, f32, &[f32])] = &[
        (0, 97.6229, 0.211965, &[-0.04883, -0.02771, 0.00958, 0.12158, -0.0752, -0.09131, 0.20703, -0.09131]),
        (1, -65.4924, 0.167847, &[-0.00094, -0.0027, 0.00032, 0.00748, -0.00238, -0.00693, 0.01416, -0.00589]),
        (2, -53.1320, 0.175145, &[0.002, 0.00015, -0.00275, 0.01141, -0.00082, 0.0024, 0.01129, -0.00064]),
        (12, -65.0047, 0.371683, &[-0.01941, 0.00793, 0.00067, -0.00433, -0.00249, -0.00708, -0.01526, -0.00592]),
        (23, 136.8203, 0.404288, &[0.30469, -0.01111, -0.12158, 0.00824, 0.1582, 0.00099, 0.16016, -0.03174]),
        (35, 353.5899, 1.266726, &[0.69531, 1.38281, -3.32812, -1.61719, -1.92969, -0.01855, 0.63672, -3.46875]),
        (46, 17.1704, 1.255302, &[0.67188, -1.57031, -3.95312, -2.3125, 0.22949, -1.57031, 1.19531, -2.40625]),
        (57, 139.9042, 0.904890, &[2.01562, -1.39062, -1.82812, -0.84766, -0.71875, -1.98438, 1.85938, 0.5]),
    ];
    for (i, &lid) in caps.iter().enumerate() {
        let row = &concat[(t - 1) * n_cap * h + i * h..(t - 1) * n_cap * h + (i + 1) * h];
        let (sum, absmean, first8) = cksum(row);
        let Some(&(_, gsum, gabs, g8)) = mlx.iter().find(|x| x.0 == lid) else { continue };
        let f8_ok = first8.iter().zip(g8.iter()).all(|(a, b)| (a - b).abs() < 0.05 + 0.08 * b.abs());
        let abs_ok = (absmean - gabs).abs() < 0.05 + 0.05 * gabs.abs();
        println!("L{lid}: native sum={sum:.4} absmean={absmean:.6} first8={first8:?}");
        println!("     mlx sum={gsum:.4} absmean={gabs:.6} first8_close={f8_ok} absmean_close={abs_ok}");
    }
    let ls = sess.debug_logits_stats();
    println!("logits finite={} nan={} min={:.4} max={:.4}", ls.finite, ls.nan, ls.min, ls.max);
    if anchor != 531 {
        eprintln!("FAIL: target_next {anchor} != 531");
        std::process::exit(1);
    }
    Ok(())
}
