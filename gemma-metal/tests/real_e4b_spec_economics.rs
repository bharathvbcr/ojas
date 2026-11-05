//! Can speculative decoding beat plain decode on real E4B? The three numbers that decide it,
//! measured before porting the per-layer input block to the batched verify path:
//! - verify(M): `step_verify` over M tokens (the batched path still has the old per-layer
//!   input stand-in, so its tokens are wrong, but its bytes and dispatch graph are within a
//!   few percent of the corrected one: timing only);
//! - draft cost: one `MtpSession::draft_from_hidden` call (host forward of the assistant);
//! - accept rate: `generate_mtp_smoke`, which verifies drafts with the corrected T = 1
//!   `step`; `equal_to_greedy` reports whether its output matches plain greedy decode
//!   (on 2026-11-04 it did not: 0 of 44 drafts accepted, see the README there).
//!
//! `GEMMA_E4B_DIR=... GEMMA_E4B_ASSISTANT_DIR=... cargo test --release \
//!   --test real_e4b_spec_economics -- --ignored --nocapture`

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::mtp::MtpSession;
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::path::PathBuf;
use std::time::Instant;

fn dir(var: &str) -> PathBuf {
    let d = PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")));
    assert!(
        d.join("model.safetensors").exists(),
        "no model.safetensors in {}",
        d.display()
    );
    d
}

#[test]
#[ignore = "needs the real E4B and assistant snapshots"]
fn speculative_decode_economics() {
    let e4b = dir("GEMMA_E4B_DIR");
    let asst = dir("GEMMA_E4B_ASSISTANT_DIR");
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
    let kv = model.kv.clone();
    let mut mtp = MtpSession::from_assistant_dir(&asst, &kv).expect("load assistant");
    let mut sess = GpuDecodeSession::new(model).expect("session");

    // BOS + text (tokenizer.json does not add BOS itself).
    let tok = GemmaTokenizer::from_hf_dir(&e4b).expect("tokenizer");
    let prompts: Vec<Vec<u32>> = [
        "The capital of France is",
        "Here is a short Python function that reverses a string:",
        "Write a haiku about the ocean.",
    ]
    .iter()
    .map(|t| {
        let mut ids = vec![2u32];
        ids.extend(tok.encode(t, false).expect("encode"));
        ids
    })
    .collect();

    // verify(M) after the first prompt's prefill.
    let p0 = &prompts[0];
    let _ = sess.generate(p0, 1).expect("prefill");
    let seeds = [3u32, 4, 5, 6, 7, 8, 9, 10];
    let mut v1 = 0.0;
    for m in 1..=8 {
        let ms = sess
            .bench_step_verify_ms(&seeds[..m], 2, 8)
            .expect("verify bench");
        if m == 1 {
            v1 = ms;
        }
        println!("verify M={m}: {ms:.2} ms ({:.2}x M=1)", ms / v1);
    }

    for (pi, p) in prompts.iter().enumerate() {
        let greedy = sess.generate(p, 64).expect("greedy");
        let t0 = Instant::now();
        let (out, verifies) = sess.generate_mtp_smoke(p, 64, &mut mtp).expect("mtp");
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let drafted: usize = verifies.iter().map(|v| v.drafted).sum();
        let accepted: usize = verifies.iter().map(|v| v.accepted).sum();
        let n = out.len().min(greedy.len());
        let same = out[..n] == greedy[..n];
        println!(
            "prompt {pi}: rounds={} drafted={drafted} accepted={accepted} ({:.1}%) \
             tokens/round={:.2} mtp_wall={wall:.0} ms equal_to_greedy={same}",
            verifies.len(),
            100.0 * accepted as f64 / drafted.max(1) as f64,
            (out.len() - p.len()) as f64 / verifies.len().max(1) as f64,
        );
    }

    // Draft cost: the host assistant forward from one backbone hidden vector.
    let h = vec![0.01f32; mtp.assistant.backbone_hidden_size];
    let _ = mtp.draft_from_hidden(&h).expect("draft");
    let t0 = Instant::now();
    let reps = 20;
    let mut len = 0;
    for _ in 0..reps {
        len = mtp.draft_from_hidden(&h).expect("draft").len();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("draft_from_hidden: {ms:.2} ms per call, {len} tokens per draft");
}
