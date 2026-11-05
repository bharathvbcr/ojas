//! Real E4B weights decode to the same text as mlx_lm.
//!
//! Until 2026-11-04 the E4B graph skipped Gemma4's per-layer input block
//! (`per_layer_input_gate`, `per_layer_projection`, `per_layer_model_projection`,
//! `per_layer_projection_norm` were never loaded) and added a raw table row after
//! attention instead. On real weights it decoded non-language from the first token.
//! The oracle is mlx_lm on the same snapshot: greedy on BOS + "The capital of France is"
//! gives " Paris.\nParis is the capital of France." (first id 9079).
//!
//! Needs the 4.8 GB `mlx-community/gemma-4-e4b-it-4bit` snapshot, so it is ignored by
//! default: `GEMMA_E4B_DIR=/path/to/snapshot cargo test --release --test real_e4b_text -- --ignored`.
//! A missing or unset directory fails the test; it never passes without running.

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::path::PathBuf;

#[test]
#[ignore = "needs the real E4B snapshot in GEMMA_E4B_DIR"]
fn real_e4b_greedy_text_matches_mlx_lm() {
    let dir = PathBuf::from(std::env::var("GEMMA_E4B_DIR").expect("set GEMMA_E4B_DIR"));
    assert!(
        dir.join("model.safetensors").exists(),
        "no model.safetensors in {}",
        dir.display()
    );
    let banks = load_from_hf_dir(
        &dir,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 128,
            ..LoadOptions::default()
        },
    )
    .expect("load E4B");
    let model = GpuSynthModel::from_host_banks(banks).expect("upload E4B");
    assert!(
        model.layers.iter().all(|l| l.ple_block.is_some()) && model.ple_model_proj.is_some(),
        "per-layer input block not loaded"
    );
    let mut sess = GpuDecodeSession::new(model).expect("session");
    // BOS + "The capital of France is" (tokenizer.json does not add BOS itself).
    let prompt = [2u32, 818, 5279, 529, 7001, 563];
    let out = sess.generate(&prompt, 10).expect("generate");
    let new = &out[prompt.len()..];
    assert_eq!(new[0], 9079, "first token differs from mlx_lm (\" Paris\")");
    let tok = GemmaTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let text = tok.decode(new, false).expect("decode");
    assert!(
        text.starts_with(" Paris.\nParis is the capital of France."),
        "greedy text differs from mlx_lm: {text:?}"
    );
}
