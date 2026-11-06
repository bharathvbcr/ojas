//! Diagnostic for a speculative-vs-greedy divergence: replays `generate_speculative`'s rounds
//! by hand, and at the first round whose accepted tokens leave the greedy stream re-runs the
//! same verify block from a freshly stepped prefix. Same answer from a fresh state means the
//! batched numerics disagree with T = 1 there; a different answer means the loop's state
//! (KV trims, positions) is wrong.
//!
//! `GEMMA_E4B_DIR=... GEMMA_E4B_ASSISTANT_DIR=... cargo test --release \
//!   --test real_e4b_spec_divergence -- --ignored --nocapture`

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::mtp::MtpSession;
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::path::PathBuf;

fn dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
}

#[test]
#[ignore = "needs the real E4B and assistant snapshots"]
fn locate_speculative_divergence() {
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
    let mut prompt = vec![2u32];
    prompt.extend(
        tok.encode("The capital of France is", false)
            .expect("encode"),
    );
    let k = 3;
    let greedy = sess.generate(&prompt, 64).expect("greedy");

    sess.reset();
    for &t in &prompt[..prompt.len() - 1] {
        sess.step(t).expect("prefill");
    }
    let mut next = sess.step(prompt[prompt.len() - 1]).expect("first");
    let mut hidden = sess.final_hidden_f32().expect("hidden");
    let mut out = prompt.clone();
    for round in 0..40 {
        out.push(next);
        assert_eq!(
            out[..],
            greedy[..out.len()],
            "round {round}: `next` already off greedy"
        );
        sess.sync_mtp_cross_kv(&mut mtp).expect("sync");
        let pos = sess.pos();
        let drafts = mtp
            .draft(next, &hidden, pos, k, &mut |t| sess.target_embed_row(t))
            .expect("draft");
        let mut block = vec![next];
        block.extend_from_slice(&drafts);
        let v = sess.step_verify(&block).expect("verify");
        let mut n = 0;
        while n < k && drafts[n] == v.next_tokens[n] {
            n += 1;
        }
        let want = &greedy[out.len()..(out.len() + n + 1).min(greedy.len())];
        let got: Vec<u32> = drafts[..n]
            .iter()
            .copied()
            .chain([v.next_tokens[n]])
            .collect();
        println!(
            "round {round} pos {pos} block {block:?} -> {:?} accept {n} greedy {want:?}",
            v.next_tokens
        );
        if got[..want.len()] != want[..] {
            // Fresh state: step the prefix up to the block, then verify the same block.
            let prefix = out[..out.len() - 1].to_vec();
            sess.reset();
            for &t in &prefix {
                sess.step(t).expect("fresh prefix");
            }
            assert_eq!(sess.pos(), pos, "fresh prefix lands on the same position");
            let fresh = sess.step_verify(&block).expect("fresh verify");
            sess.commit_verify(block.len(), 0).expect("trim");
            let mut seq = Vec::new();
            for &t in &block {
                seq.push(sess.step(t).expect("sequential"));
            }
            println!(
                "DIVERGED at round {round}: loop verify {:?}, fresh verify {:?}, sequential steps {seq:?}",
                v.next_tokens, fresh.next_tokens
            );
            return;
        }
        hidden = sess.final_hidden_row_f32(n).expect("hidden row");
        sess.commit_verify(k + 1, n + 1).expect("commit");
        out.extend_from_slice(&drafts[..n]);
        next = v.next_tokens[n];
        if out.len() + 1 >= greedy.len() {
            break;
        }
    }
    println!("no divergence in the replayed rounds");
}
