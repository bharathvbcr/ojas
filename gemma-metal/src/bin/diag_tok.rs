//! Quiet decode health: embed / logits / first tokens.
//! Usage: diag_tok [embed_scale_override] [e4b|31b]
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;
use gemma_metal::weights::{
    load_from_hf_dir, resolve_default_31b_mlx_cache, resolve_default_e4b_mlx_cache, LoadOptions,
};
use std::time::Instant;

fn main() -> Result<(), String> {
    let scale_override: Option<f32> = std::env::args().nth(1).and_then(|s| {
        if s == "e4b" || s == "31b" {
            None
        } else {
            s.parse().ok()
        }
    });
    let model_kind = std::env::args()
        .skip(1)
        .find(|a| a == "e4b" || a == "31b")
        .unwrap_or_else(|| "e4b".into());
    let dir = if model_kind == "31b" {
        resolve_default_31b_mlx_cache().ok_or("no 31b cache")?
    } else {
        resolve_default_e4b_mlx_cache().ok_or("no e4b cache")?
    };
    println!("model={} dir={}", model_kind, dir.display());
    let t0 = Instant::now();
    let banks = load_from_hf_dir(
        &dir,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 128,
            ..LoadOptions::default()
        },
    )
    .map_err(|e| e.to_string())?;
    // Host banks dropped inside from_host_banks before session KV/scratch alloc.
    let mut model = GpuSynthModel::from_host_banks(banks).map_err(|e| e.to_string())?;
    if let Some(s) = scale_override {
        println!("override embed_scale {} -> {}", model.embed_scale, s);
        model.embed_scale = s;
    }
    println!(
        "loaded in {:.1}s hidden={} layers={} embed_scale={:.4} softcap={}",
        t0.elapsed().as_secs_f64(),
        model.hidden,
        model.layers.len(),
        model.embed_scale,
        model.softcap
    );
    let mut sess = GpuDecodeSession::new(model).map_err(|e| e.to_string())?;

    // Hot GEMV smoke: layer0 Q @ ones(hidden) should be non-trivial.
    {
        let gpu = &sess.model.gpu;
        let q = &sess.model.layers[0].q_proj;
        let x = gpu.rt.alloc_buffer((q.cols as usize) * 4).map_err(|e| e.to_string())?;
        let y = gpu.rt.alloc_buffer((q.rows as usize) * 4).map_err(|e| e.to_string())?;
        x.write_f32(&vec![1.0f32; q.cols as usize]);
        q.gemv(gpu, &x, &y).map_err(|e| e.to_string())?;
        gpu.synchronize().map_err(|e| e.to_string())?;
        let got = y.read_f32();
        let mut nan = 0usize;
        let mut max = 0f32;
        let mut nz = 0usize;
        for &v in &got {
            if v.is_nan() {
                nan += 1;
            } else {
                max = max.max(v.abs());
                if v.abs() > 1e-8 {
                    nz += 1;
                }
            }
        }
        println!(
            "hot_gemv_q_ones: rows={} cols={} nz={nz}/{} nan={nan} max_abs={max:.4} layout={:?}",
            q.rows,
            q.cols,
            got.len(),
            q.layout
        );
        let gate = &sess.model.layers[0].gate_proj;
        let gx = gpu.rt.alloc_buffer((gate.cols as usize) * 4).map_err(|e| e.to_string())?;
        let gy = gpu.rt.alloc_buffer((gate.rows as usize) * 4).map_err(|e| e.to_string())?;
        gx.write_f32(&vec![0.1f32; gate.cols as usize]);
        gate.gemv(gpu, &gx, &gy).map_err(|e| e.to_string())?;
        gpu.synchronize().map_err(|e| e.to_string())?;
        let gg = gy.read_f32();
        let gmax = gg.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let gnan = gg.iter().filter(|v| v.is_nan()).count();
        println!(
            "hot_gemv_gate_0.1: rows={} cols={} nan={gnan} max_abs={gmax:.4}",
            gate.rows, gate.cols
        );
    }

    let emb = sess.debug_embed_only(1).map_err(|e| e.to_string())?;
    println!(
        "embed_only: finite={} nan={} min={:.4} max={:.4}",
        emb.finite, emb.nan, emb.min, emb.max
    );

    let prompt = vec![2u32, 150, 2307];
    let gen_n = 16usize;
    let t1 = Instant::now();
    let out = sess.generate(&prompt, gen_n).map_err(|e| e.to_string())?;
    let dt = t1.elapsed().as_secs_f64();
    let new_toks = &out[prompt.len()..];
    let ls = sess.debug_logits_stats();
    let xs = sess.debug_x_stats();
    println!(
        "generated {} new toks in {:.3}s → {:.2} tok/s (wall includes prefill)",
        new_toks.len(),
        dt,
        new_toks.len() as f64 / dt.max(1e-9)
    );
    println!("new tokens: {:?}", new_toks);
    let uniq: std::collections::BTreeSet<_> = new_toks.iter().copied().collect();
    println!(
        "unique={} all_zero={} logits finite={} nan={} min={:.4} max={:.4} argmax={}",
        uniq.len(),
        new_toks.iter().all(|&t| t == 0),
        ls.finite,
        ls.nan,
        ls.min,
        ls.max,
        ls.host_argmax
    );
    println!(
        "resid x: finite={} nan={} min={:.4} max={:.4}",
        xs.finite, xs.nan, xs.min, xs.max
    );
    Ok(())
}
