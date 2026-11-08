//! Dumps the inputs and outputs of `MtpSession::assistant_step` on real E4B state for
//! the transformers oracle (`bench/results/real_e4b_20261104/oracle/mtp_assistant_oracle.py`),
//! which reruns each step with `Gemma4AssistantForCausalLM` in float32 and compares the
//! drafted token and the projected hidden state.
//!
//! `GEMMA_E4B_DIR=... GEMMA_E4B_ASSISTANT_DIR=... MTP_ORACLE_OUT=DIR cargo test --release \
//!   --test real_e4b_mtp_oracle -- --ignored --nocapture`

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::mtp::MtpSession;
use gemma_metal::quant::QuantScheme;
use gemma_metal::tokenizer::GemmaTokenizer;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

fn dir(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var}")))
}

fn write_f32(path: &Path, v: &[f32]) {
    let mut f = std::fs::File::create(path).expect("create");
    for x in v {
        f.write_all(&x.to_le_bytes()).expect("write");
    }
}

#[test]
#[ignore = "needs the real E4B and assistant snapshots"]
fn dump_assistant_steps_for_the_oracle() {
    let e4b = dir("GEMMA_E4B_DIR");
    let out_dir = dir("MTP_ORACLE_OUT");
    std::fs::create_dir_all(&out_dir).expect("out dir");
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
    let mut m =
        MtpSession::from_assistant_dir(dir("GEMMA_E4B_ASSISTANT_DIR"), &kv).expect("assistant");
    let mut sess = GpuDecodeSession::new(model).expect("session");
    let tok = GemmaTokenizer::from_hf_dir(&e4b).expect("tokenizer");

    // The drafter's host-side embedding rows must equal the GPU lookup's.
    for t in [0u32, 2, 9079, 236761, 262_143] {
        let host = sess.target_embed_row(t).expect("host row");
        let gpu = sess.target_embed_row_gpu(t).expect("gpu row");
        let worst = host
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = gpu.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(
            worst <= 1e-6 * scale,
            "embed row {t}: host vs GPU differ by {worst}"
        );
    }
    let mut meta = String::from("[\n");
    let mut case = 0usize;
    for text in [
        "The capital of France is",
        "Here is a short Python function that reverses a string:",
    ] {
        let mut prompt = vec![2u32];
        prompt.extend(tok.encode(text, false).expect("encode"));
        // Step the prompt; `next` is the target's first greedy token, not yet stepped.
        sess.reset();
        for &t in &prompt[..prompt.len() - 1] {
            sess.step(t).expect("prefill");
        }
        let next = sess.step(prompt[prompt.len() - 1]).expect("first token");
        let mut hidden = sess.final_hidden_f32().expect("hidden");
        sess.sync_mtp_cross_kv(&mut m).expect("sync");
        let pos = sess.pos();
        let mut t = next;
        for step in 0..3 {
            let embed = sess.target_embed_row(t).expect("embed");
            let (drafted, projected) = m.assistant_step(&embed, &hidden, pos).expect("step");
            let p = |name: &str| out_dir.join(format!("c{case}_{name}.f32"));
            write_f32(&p("embed"), &embed);
            write_f32(&p("hidden"), &hidden);
            write_f32(&p("projected"), &projected);
            if step == 0 {
                let s = &m.cross_kv.target_sliding;
                let g = &m.cross_kv.target_global;
                write_f32(&p("sk"), &s.k[..s.seq_len * s.heads * s.dim]);
                write_f32(&p("sv"), &s.v[..s.seq_len * s.heads * s.dim]);
                write_f32(&p("gk"), &g.k[..g.seq_len * g.heads * g.dim]);
                write_f32(&p("gv"), &g.v[..g.seq_len * g.heads * g.dim]);
            }
            let s = &m.cross_kv.target_sliding;
            let g = &m.cross_kv.target_global;
            meta += &format!(
                "  {{\"case\": {case}, \"kv_case\": {}, \"text\": {text:?}, \"step\": {step}, \
                 \"input_token\": {t}, \"pos\": {pos}, \"drafted\": {drafted}, \
                 \"sliding\": [{}, {}, {}], \"global\": [{}, {}, {}]}},\n",
                case - step,
                s.seq_len,
                s.heads,
                s.dim,
                g.seq_len,
                g.heads,
                g.dim
            );
            println!("case {case} step {step}: input {t} pos {pos} -> drafted {drafted}");
            hidden = projected;
            t = drafted;
            case += 1;
        }
    }
    meta.truncate(meta.len() - 2);
    meta += "\n]\n";
    std::fs::write(out_dir.join("meta.json"), meta).expect("meta");
}
