//! `step_verify(M)` on real E4B must predict what sequential decode predicts: feed it the
//! greedy continuation and its `next_tokens` must be that continuation shifted by one. Before
//! the per-layer input block was ported to the batched path, it ran a lookup stand-in and
//! single norms there and failed this at every M > 1.
//!
//! `GEMMA_E4B_DIR=... cargo test --release --test real_e4b_verify_exact -- --ignored --nocapture`

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::path::PathBuf;

#[test]
#[ignore = "needs the real E4B snapshot"]
fn verify_block_predicts_the_greedy_continuation() {
    let e4b = PathBuf::from(std::env::var("GEMMA_E4B_DIR").expect("set GEMMA_E4B_DIR"));
    let banks = load_from_hf_dir(
        &e4b,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 256,
            ..LoadOptions::default()
        },
    )
    .expect("load E4B");
    let model = GpuSynthModel::from_host_banks(banks).expect("upload E4B");
    let mut sess = GpuDecodeSession::new(model).expect("session");
    let tok = GemmaTokenizer::from_hf_dir(&e4b).expect("tokenizer");
    let mut failures = Vec::new();
    let mut checked = 0;
    for text in [
        "The capital of France is",
        "Here is a short Python function that reverses a string:",
        "Write a haiku about the ocean.",
    ] {
        let mut prompt = vec![2u32];
        prompt.extend(tok.encode(text, false).expect("encode"));
        let greedy = sess.generate(&prompt, 24).expect("greedy");
        let cont = &greedy[prompt.len()..];
        // Verify blocks at several depths into the continuation.
        for start in [0usize, 7, 15] {
            for m in [2usize, 3, 4, 5] {
                if start + m + 1 > cont.len() {
                    continue;
                }
                sess.reset();
                for &t in &prompt {
                    sess.step(t).expect("prefill");
                }
                for &t in &cont[..start] {
                    sess.step(t).expect("advance");
                }
                let block = &cont[start..start + m];
                let got = sess.step_verify(block).expect("verify").next_tokens;
                let want = &cont[start + 1..start + m + 1];
                checked += 1;
                if got != want {
                    failures.push(format!(
                        "{text:?} start={start} M={m}: got {got:?} want {want:?}"
                    ));
                }
            }
        }
    }
    for f in &failures {
        println!("MISMATCH {f}");
    }
    println!(
        "{} of {checked} verify blocks match sequential greedy",
        checked - failures.len()
    );
    assert!(failures.is_empty());
}
