//! Quiet decode health: embed / logits / first tokens.
//! Usage: diag_tok [embed_scale_override] [e4b|31b] [--model DIR] [--prompt TEXT] [--n N]
//! `--prompt` encodes TEXT with the model's tokenizer (BOS added), greedy-decodes N tokens
//! (default 16) and prints them as text: the readable check that real weights give real words.
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{
    load_from_hf_dir, resolve_default_31b_mlx_cache, resolve_default_e4b_mlx_cache, LoadOptions,
};
use std::path::PathBuf;
use std::time::Instant;

fn flag_value(argv: &[String], name: &str) -> Result<Option<String>, String> {
    match argv.iter().position(|a| a == name) {
        None => Ok(None),
        Some(i) => argv
            .get(i + 1)
            .cloned()
            .map(Some)
            .ok_or_else(|| format!("{name} needs a value")),
    }
}

fn main() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let model_flag = flag_value(&argv, "--model")?.map(PathBuf::from);
    let text_prompt = flag_value(&argv, "--prompt")?;
    let gen_n: usize = match flag_value(&argv, "--n")? {
        Some(s) => s.parse().map_err(|e| format!("--n {s}: {e}"))?,
        None => 16,
    };
    let scale_override: Option<f32> = argv.first().and_then(|s| {
        if s == "e4b" || s == "31b" {
            None
        } else {
            s.parse().ok()
        }
    });
    let model_kind = argv
        .iter()
        .find(|a| *a == "e4b" || *a == "31b")
        .cloned()
        .unwrap_or_else(|| "e4b".into());
    let dir = if let Some(d) = model_flag {
        d
    } else if model_kind == "31b" {
        resolve_default_31b_mlx_cache().ok_or("no 31b cache")?
    } else {
        resolve_default_e4b_mlx_cache().ok_or("no e4b cache")?
    };
    let tokenizer = match &text_prompt {
        Some(_) => {
            let t = GemmaTokenizer::from_hf_dir(&dir).map_err(|e| e.to_string())?;
            if !t.is_ready() {
                return Err(format!("no tokenizer in {}", dir.display()));
            }
            Some(t)
        }
        None => None,
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

    // `--dump-hidden FILE`: every layer's output residual at every prompt position, as JSON
    // `{"argmax": next id, "hidden": [pos][layer][H]}`, for a layer-by-layer diff against a
    // reference dump of the same prompt.
    if let Some(path) = flag_value(&argv, "--dump-hidden")? {
        let prompt = match (&tokenizer, &text_prompt) {
            (Some(t), Some(text)) => {
                let mut ids = t.encode(text, true).map_err(|e| e.to_string())?;
                if ids.first() != Some(&2) {
                    ids.insert(0, 2);
                }
                ids
            }
            _ => vec![2u32, 150, 2307],
        };
        let n_layers = sess.model.layers.len();
        let h = sess.model.hidden;
        sess.enable_hidden_capture((0..n_layers).collect())
            .map_err(|e| e.to_string())?;
        for &t in &prompt[..prompt.len() - 1] {
            sess.step_prefill(t).map_err(|e| e.to_string())?;
        }
        let next = sess.step(prompt[prompt.len() - 1]).map_err(|e| e.to_string())?;
        let (concat, t) = sess.captured_concat().map_err(|e| e.to_string())?;
        let hidden: Vec<Vec<Vec<f32>>> = (0..t)
            .map(|p| {
                (0..n_layers)
                    .map(|l| concat[(p * n_layers + l) * h..(p * n_layers + l + 1) * h].to_vec())
                    .collect()
            })
            .collect();
        let out = serde_json::json!({ "prompt": prompt, "argmax": next, "hidden": hidden });
        std::fs::write(&path, out.to_string()).map_err(|e| format!("{path}: {e}"))?;
        println!("dumped {t} positions x {n_layers} layers to {path}; next argmax {next}");
        return Ok(());
    }

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

    let prompt = match (&tokenizer, &text_prompt) {
        (Some(t), Some(text)) => {
            // tokenizer.json's post-processor does not add Gemma's BOS (id 2); the model needs it.
            let mut ids = t.encode(text, true).map_err(|e| e.to_string())?;
            if ids.first() != Some(&2) {
                ids.insert(0, 2);
            }
            ids
        }
        _ => vec![2u32, 150, 2307],
    };
    println!("prompt ids: {prompt:?}");
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
    if let Some(t) = &tokenizer {
        let text = t.decode(new_toks, false).map_err(|e| e.to_string())?;
        println!("new text: {text:?}");
    }
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
