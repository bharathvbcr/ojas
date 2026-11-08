//! Can speculative decoding beat plain decode on real E4B? The three numbers that decide it:
//! - verify(M): `step_verify` over M tokens (its tokens are checked against sequential decode
//!   by `tests/real_e4b_verify_exact.rs`);
//! - draft cost: one `MtpSession::assistant_step` (host f32 forward of the assistant);
//! - accept rate per draft position: `generate_mtp_smoke`, which verifies drafts with the
//!   T = 1 `step`, so its output must equal plain greedy decode (asserted).
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

    // Accept rate per draft length: P(draft position k accepted | positions < k accepted).
    let greedy: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| sess.generate(p, 64).expect("greedy"))
        .collect();
    for draft_len in 1..=4usize {
        let mut reached = vec![0usize; draft_len];
        let mut hit = vec![0usize; draft_len];
        let (mut rounds, mut new_tokens, mut wall) = (0usize, 0usize, 0.0f64);
        for (pi, p) in prompts.iter().enumerate() {
            let t0 = Instant::now();
            let (out, verifies) = sess
                .generate_mtp_smoke(p, 64, draft_len, &mut mtp)
                .expect("mtp");
            wall += t0.elapsed().as_secs_f64() * 1e3;
            assert_eq!(
                out, greedy[pi],
                "speculative output must equal greedy (prompt {pi})"
            );
            for v in &verifies {
                for k in 0..v.drafted.min(draft_len) {
                    if k > v.accepted {
                        break;
                    }
                    reached[k] += 1;
                    hit[k] += (k < v.accepted) as usize;
                }
            }
            rounds += verifies.len();
            new_tokens += out.len() - p.len();
        }
        let rates: Vec<String> = (0..draft_len)
            .map(|k| format!("{:.2}", hit[k] as f64 / reached[k].max(1) as f64))
            .collect();
        println!(
            "draft_len={draft_len}: accept by position [{}] tokens/round={:.2} rounds={rounds} \
             wall={wall:.0} ms (host drafter, sequential verify)",
            rates.join(", "),
            new_tokens as f64 / rounds.max(1) as f64,
        );
    }

    // Draft cost: one assistant step (host, f32) on the last synced target state.
    let hidden = sess.final_hidden_f32().expect("hidden");
    sess.sync_mtp_cross_kv(&mut mtp).expect("sync");
    let embed = sess.target_embed_row(3).expect("embed");
    let pos = sess.pos();
    let _ = mtp.assistant_step(&embed, &hidden, pos).expect("step");
    let reps = 20;
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = mtp.assistant_step(&embed, &hidden, pos).expect("step");
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("assistant_step: {ms:.2} ms per drafted token (host f32)");
}
