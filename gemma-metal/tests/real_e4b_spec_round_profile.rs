//! Where a speculative round's time goes on real E4B: replays `generate_speculative`'s loop with
//! a host timer around each phase (KV sync to the drafter, drafting, verify, hidden readback,
//! trim), three prompts x 64 tokens, draft lengths 1..=3.
//!
//! `GEMMA_E4B_DIR=... GEMMA_E4B_ASSISTANT_DIR=... cargo test --release \
//!   --test real_e4b_spec_round_profile -- --ignored --nocapture`

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::mtp::MtpSession;
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::path::PathBuf;
use std::time::Instant;

fn dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
}

#[test]
#[ignore = "needs the real E4B and assistant snapshots"]
fn speculative_round_phases() {
    let e4b = dir("GEMMA_E4B_DIR");
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
    let mut mtp =
        MtpSession::from_assistant_dir(dir("GEMMA_E4B_ASSISTANT_DIR"), &kv).expect("assistant");
    let mut sess = GpuDecodeSession::new(model).expect("session");
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
    for k in 1..=3usize {
        // sync, draft, verify, hidden, trim (ms totals), rounds, tokens
        let mut t = [0f64; 5];
        let (mut rounds, mut tokens) = (0usize, 0usize);
        for p in &prompts {
            sess.reset();
            for &x in &p[..p.len() - 1] {
                sess.step(x).expect("prefill");
            }
            let mut next = sess.step(p[p.len() - 1]).expect("first");
            let mut hidden = sess.final_hidden_f32().expect("hidden");
            let mut made = 1usize;
            while made < 64 {
                let c = Instant::now();
                sess.sync_mtp_cross_kv(&mut mtp).expect("sync");
                t[0] += c.elapsed().as_secs_f64() * 1e3;
                let c = Instant::now();
                let pos = sess.pos();
                let drafts = mtp
                    .draft(next, &hidden, pos, k, &mut |x| sess.target_embed_row(x))
                    .expect("draft");
                t[1] += c.elapsed().as_secs_f64() * 1e3;
                let mut block = vec![next];
                block.extend_from_slice(&drafts);
                let c = Instant::now();
                let v = sess.step_verify(&block).expect("verify");
                t[2] += c.elapsed().as_secs_f64() * 1e3;
                let mut n = 0;
                while n < k && drafts[n] == v.next_tokens[n] {
                    n += 1;
                }
                let c = Instant::now();
                hidden = sess.final_hidden_row_f32(n).expect("hidden");
                t[3] += c.elapsed().as_secs_f64() * 1e3;
                let c = Instant::now();
                sess.commit_verify(k + 1, n + 1).expect("commit");
                t[4] += c.elapsed().as_secs_f64() * 1e3;
                next = v.next_tokens[n];
                made += n + 1;
                rounds += 1;
                tokens += n + 1;
            }
        }
        let r = rounds as f64;
        let total: f64 = t.iter().sum();
        println!(
            "k={k}: per round sync {:.2} ms, draft {:.2}, verify {:.2}, hidden {:.2}, trim {:.2}; \
             {:.2} ms/round, {:.2} tokens/round, {:.1} tok/s; verify alone {:.1} tok/s",
            t[0] / r,
            t[1] / r,
            t[2] / r,
            t[3] / r,
            t[4] / r,
            total / r,
            tokens as f64 / r,
            tokens as f64 / total * 1e3,
            tokens as f64 / t[2] * 1e3,
        );
    }
}
