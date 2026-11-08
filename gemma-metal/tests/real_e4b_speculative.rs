//! Speculative decode on real E4B (`GpuDecodeSession::generate_speculative`) against the
//! pipelined decode of record (`bench_decode_tok_s`: GPU-chained argmax, no per-token
//! readback), interleaved per repetition in one process, decode only on both sides (the
//! span after the first generated token, prefill excluded), eight prompts x 64 new tokens.
//!
//! Its output must equal plain greedy decode except at an exact tie: two tokens within 0.05
//! in logit, where the batched and T = 1 sums can round either way (the one seen is tied at
//! 27.0 in mlx_lm too).
//!
//! `GEMMA_E4B_DIR=... GEMMA_E4B_ASSISTANT_DIR=... cargo test --release \
//!   --test real_e4b_speculative -- --ignored --nocapture`

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::mtp::MtpSession;
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::path::PathBuf;

fn dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
}

fn spread(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2], v[v.len() - 1])
}

#[test]
#[ignore = "needs the real E4B and assistant snapshots"]
fn speculative_decode_equals_greedy_and_is_timed() {
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
    // Eight prompts of different kinds, so one exact tie (which can flip a prompt onto a branch
    // that drafts better or worse) moves the total by a small share, not a third.
    let prompts: Vec<Vec<u32>> = [
        "The capital of France is",
        "Here is a short Python function that reverses a string:",
        "Write a haiku about the ocean.",
        "Explain in two sentences why the sky is blue.",
        "List the first ten prime numbers:",
        "Translate to French: The weather is nice today and we are going to the park.",
        "A recipe for pancakes needs the following ingredients:",
        "In 1969, the Apollo 11 mission",
    ]
    .iter()
    .map(|t| {
        let mut ids = vec![2u32];
        ids.extend(tok.encode(t, false).expect("encode"));
        ids
    })
    .collect();
    let new = 64;
    let greedy: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| sess.generate(p, new).expect("greedy"))
        .collect();
    let reps = 5;
    for draft_len in [1usize, 2, 3, 4] {
        let (mut spec_rates, mut pipe_rates) = (Vec::new(), Vec::new());
        let mut per_prompt = vec![(0usize, 0usize); prompts.len()];
        let mut ties = 0usize;
        for _ in 0..reps {
            let (mut spec_tok, mut spec_secs, mut pipe_secs) = (0usize, 0.0, 0.0);
            for (pi, p) in prompts.iter().enumerate() {
                let steps = new - 1;
                pipe_secs += steps as f64 / sess.bench_decode_tok_s(p, steps).expect("pipelined");
                let (out, acc, secs) = sess
                    .generate_speculative(p, new, draft_len, &mut mtp)
                    .expect("spec");
                spec_secs += secs;
                spec_tok += out.len() - p.len() - 1;
                per_prompt[pi].0 += out.len() - p.len() - 1;
                per_prompt[pi].1 += acc.len();
                let g = &greedy[pi];
                let n = out.len().min(g.len());
                if let Some(i) = (0..n).find(|&i| out[i] != g[i]) {
                    // Allowed only at a tie: the two tokens' logits after the shared prefix
                    // must be within float noise.
                    sess.reset();
                    for &t in &g[..i] {
                        sess.step(t).expect("prefix");
                    }
                    let l = sess.logits_row_f32(0).expect("logits");
                    let gap = (l[g[i] as usize] - l[out[i] as usize]).abs();
                    assert!(
                        gap < 0.05,
                        "speculative output differs from greedy beyond a tie \
                         (prompt {pi}, k={draft_len}, token {i}, gap {gap})"
                    );
                    ties += 1;
                }
            }
            spec_rates.push(spec_tok as f64 / spec_secs);
            pipe_rates.push((prompts.len() * (new - 1)) as f64 / pipe_secs);
        }
        let ratios: Vec<f64> = spec_rates
            .iter()
            .zip(&pipe_rates)
            .map(|(s, p)| s / p)
            .collect();
        let (s0, s1, s2) = spread(spec_rates);
        let (p0, p1, p2) = spread(pipe_rates);
        let (r0, r1, r2) = spread(ratios);
        let per: Vec<String> = per_prompt
            .iter()
            .map(|(t, r)| format!("{:.2}", *t as f64 / (*r).max(1) as f64))
            .collect();
        println!(
            "draft_len={draft_len}: speculative {s1:.1} tok/s [{s0:.1}, {s2:.1}], pipelined decode \
             {p1:.1} [{p0:.1}, {p2:.1}], paired ratio {r1:.3} [{r0:.3}, {r2:.3}] over {reps} reps; \
             tokens per round by prompt [{}]; {ties} tie divergences",
            per.join(", ")
        );
    }
}
