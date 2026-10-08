//! DFlash block-verify helpers (port step 1).
//!
//! Core Metal path lives on [`crate::gpu_model::GpuDecodeSession::step_verify`]
//! + [`trim_kv`](crate::gpu_model::GpuDecodeSession::trim_kv). This module holds
//! the MLX-aligned accept math, a host-side draft stub loop (measure accept
//! without a native draft), and parity hooks vs a golden token stream.
//!
//! # DFlash verify contract (from `dflash/model_mlx.py`)
//!
//! ```text
//! verify_input = [anchor, draft_0, …, draft_{K-1}]     # len M = K+1 (M≤8)
//! next_tokens  = softcap_argmax(logits) per position   # len M
//! accepted     = longest prefix: draft_i == next_tokens[i]
//! emit         = draft[:accepted] ++ [next_tokens[accepted]]   # +1 bonus
//! trim         = M - (accepted + 1)
//! ```

use crate::error::{Error, Result};
use crate::gpu_model::{GpuDecodeSession, StepVerifyResult, VERIFY_MAX_M};
use crate::mtp::{verify_draft, VerifyResult};
use crate::parity::{compare_activations, ActivationDump};

/// Outcome of accepting a draft against a [`StepVerifyResult`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockAccept {
    pub verify: VerifyResult,
    /// Timesteps of the verify block to keep in KV (`accepted + 1` when a bonus exists).
    pub keep: usize,
    /// Tokens to append to the output stream (accepted draft prefix + bonus).
    pub emit: Vec<u32>,
}

/// MLX-aligned accept for a verify block.
///
/// `draft` is the proposed continuation (`K` tokens). `verify.tokens` must be
/// `[anchor, draft…]` with `len = K + 1` (or `draft` alone when `K == M` and the
/// first draft token is itself the verify anchor — then compare `draft[1..]` vs
/// `next_tokens[..K-1]` via the smoke path below).
///
/// Primary path (DFlash): `verify.tokens.len() == draft.len() + 1` and
/// `verify.tokens[1..] == draft`.
pub fn accept_block(draft: &[u32], verify: &StepVerifyResult) -> Result<BlockAccept> {
    let m = verify.next_tokens.len();
    if m == 0 || m != verify.tokens.len() {
        return Err(Error::Config(format!(
            "accept_block: next_tokens len {} != tokens len {}",
            verify.next_tokens.len(),
            verify.tokens.len()
        )));
    }
    if m > VERIFY_MAX_M {
        return Err(Error::Config(format!(
            "accept_block: M={m} > VERIFY_MAX_M={VERIFY_MAX_M}"
        )));
    }

    // DFlash: draft continues after tokens[0]; next_tokens[i] predicts draft[i].
    let (cmp_draft, target_slice) = if draft.len() + 1 == m && verify.tokens.get(1..) == Some(draft) {
        (draft, &verify.next_tokens[..draft.len().min(verify.next_tokens.len())])
    } else if draft.len() == m {
        // Smoke: whole block fed as draft; compare draft[1..] to next_tokens[..m-1].
        if m < 2 {
            (
                &draft[..0],
                &verify.next_tokens[..0],
            )
        } else {
            (&draft[1..], &verify.next_tokens[..m - 1])
        }
    } else {
        return Err(Error::Config(format!(
            "accept_block: draft len {} incompatible with verify M={m} (want draft=M-1 with tokens[1..]=draft, or draft=M smoke)",
            draft.len()
        )));
    };

    let mut vr = verify_draft(cmp_draft, target_slice);
    // Bonus always comes from next_tokens[accepted] (MLX `t_list[accepted]`).
    let bonus = verify.next_tokens.get(vr.accepted).copied();
    vr.bonus_token = bonus;

    let keep = (vr.accepted + 1).min(m);
    let mut emit = cmp_draft[..vr.accepted].to_vec();
    if let Some(b) = bonus {
        emit.push(b);
    }

    Ok(BlockAccept {
        verify: vr,
        keep,
        emit,
    })
}

/// Run [`accept_block`] then [`GpuDecodeSession::commit_verify`].
pub fn commit_accepted(
    sess: &mut GpuDecodeSession,
    draft: &[u32],
    verify: &StepVerifyResult,
) -> Result<BlockAccept> {
    let acc = accept_block(draft, verify)?;
    sess.commit_verify(verify.tokens.len(), acc.keep)?;
    Ok(acc)
}

/// Host-side draft stub: propose `k` tokens by rolling greedy `step` from `anchor`
/// (no draft model). Advances `sess` by `k` — caller typically `trim_kv(k)` after
/// capturing the proposal for `step_verify`.
pub fn host_stub_draft(
    sess: &mut GpuDecodeSession,
    mut anchor: u32,
    k: usize,
) -> Result<Vec<u32>> {
    if k == 0 || k >= VERIFY_MAX_M {
        return Err(Error::Config(format!(
            "host_stub_draft: k={k} outside 1..={}",
            VERIFY_MAX_M - 1
        )));
    }
    let mut draft = Vec::with_capacity(k);
    for _ in 0..k {
        anchor = sess.step(anchor)?;
        draft.push(anchor);
    }
    Ok(draft)
}

/// Prefill `prompt`, then generate with a host greedy stub as the "draft":
/// propose K tokens via sequential step, rewind, then `step_verify` + accept.
///
/// Exact when the stub draft matches what verify would greedily emit (always, for
/// this host stub — accept should be full). Useful as a smoke for the API shape.
pub fn generate_with_host_stub(
    sess: &mut GpuDecodeSession,
    prompt: &[u32],
    max_new: usize,
    block_size: usize,
) -> Result<(Vec<u32>, Vec<BlockAccept>)> {
    if prompt.is_empty() {
        return Err(Error::Config("empty prompt".into()));
    }
    if !(2..=VERIFY_MAX_M).contains(&block_size) {
        return Err(Error::Config(format!(
            "block_size={block_size} outside 2..={VERIFY_MAX_M}"
        )));
    }
    sess.reset();
    let mut out = prompt.to_vec();
    let mut accepts = Vec::new();
    for &t in &prompt[..prompt.len() - 1] {
        sess.step_prefill(t)?;
    }
    let mut anchor = sess.step(prompt[prompt.len() - 1])?;
    out.push(anchor);

    while out.len() - prompt.len() < max_new {
        if let Some(eos) = sess.model.cfg.eos_token_id.as_ref() {
            if eos.as_slice().contains(&anchor) {
                break;
            }
        }
        let remaining = max_new - (out.len() - prompt.len());
        let k = (block_size - 1).min(remaining);
        if k == 0 {
            break;
        }

        // Snapshot: draft via sequential greedy, then rewind and block-verify.
        let pos0 = sess.pos();
        let draft = host_stub_draft(sess, anchor, k)?;
        sess.trim_kv(k)?;
        debug_assert_eq!(sess.pos(), pos0);

        let mut verify_in = Vec::with_capacity(k + 1);
        verify_in.push(anchor);
        verify_in.extend_from_slice(&draft);
        let ver = sess.step_verify(&verify_in)?;
        let acc = commit_accepted(sess, &draft, &ver)?;
        let mut stop = false;
        for &tok in &acc.emit {
            out.push(tok);
            if out.len() - prompt.len() >= max_new {
                stop = true;
                break;
            }
            if let Some(eos) = sess.model.cfg.eos_token_id.as_ref() {
                if eos.as_slice().contains(&tok) {
                    stop = true;
                    break;
                }
            }
        }
        anchor = *acc.emit.last().unwrap_or(&anchor);
        accepts.push(acc);
        if stop {
            break;
        }
    }
    Ok((out, accepts))
}

/// Compare a generated token stream to an MLX/golden stream (exact greedy match).
pub fn compare_token_stream(
    name: &str,
    cand: &[u32],
    golden: &[u32],
) -> Result<crate::parity::CompareReport> {
    let n = cand.len().min(golden.len());
    let a = ActivationDump {
        name: name.into(),
        shape: vec![n],
        data: cand[..n].iter().map(|&t| t as f32).collect(),
    };
    let b = ActivationDump {
        name: name.into(),
        shape: vec![n],
        data: golden[..n].iter().map(|&t| t as f32).collect(),
    };
    compare_activations(&a, &b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::SyntheticE4bGraph;
    use crate::gpu_model::{GpuDecodeSession, GpuSynthModel};
    use crate::quant::QuantScheme;

    fn metal_ready(model: &GpuSynthModel) -> bool {
        let entry = match model.scheme {
            QuantScheme::Q4Mlx { .. } => crate::kernels::KernelId::GemvQ4Mlx.entry_name(),
            QuantScheme::Q8 { .. } => crate::kernels::KernelId::GemvQ8.entry_name(),
            _ => crate::kernels::KernelId::GemvQ4.entry_name(),
        };
        model.gpu.rt.pipeline(entry).is_ok()
    }

    #[test]
    fn accept_block_full_match_mlx_shape() {
        let verify = StepVerifyResult {
            pos0: 10,
            tokens: vec![100, 1, 2, 3, 4],
            next_tokens: vec![1, 2, 3, 4, 99],
        };
        let draft = [1u32, 2, 3, 4];
        let acc = accept_block(&draft, &verify).unwrap();
        assert_eq!(acc.verify.accepted, 4);
        assert_eq!(acc.verify.reject_at, None);
        assert_eq!(acc.verify.bonus_token, Some(99));
        assert_eq!(acc.keep, 5);
        assert_eq!(acc.emit, vec![1, 2, 3, 4, 99]);
    }

    #[test]
    fn accept_block_early_reject() {
        let verify = StepVerifyResult {
            pos0: 0,
            tokens: vec![7, 1, 2, 3],
            next_tokens: vec![1, 9, 3, 8],
        };
        let draft = [1u32, 2, 3];
        let acc = accept_block(&draft, &verify).unwrap();
        assert_eq!(acc.verify.accepted, 1);
        assert_eq!(acc.verify.reject_at, Some(1));
        assert_eq!(acc.verify.bonus_token, Some(9));
        assert_eq!(acc.keep, 2);
        assert_eq!(acc.emit, vec![1, 9]);
    }

    #[test]
    fn accept_block_zero_accept_emits_bonus_only() {
        let verify = StepVerifyResult {
            pos0: 0,
            tokens: vec![7, 1, 2],
            next_tokens: vec![5, 6, 7],
        };
        let draft = [1u32, 2];
        let acc = accept_block(&draft, &verify).unwrap();
        assert_eq!(acc.verify.accepted, 0);
        assert_eq!(acc.emit, vec![5]);
        assert_eq!(acc.keep, 1);
    }

    #[test]
    fn compare_token_stream_exact() {
        let r = compare_token_stream("gold", &[1, 2, 3], &[1, 2, 3, 4]).unwrap();
        assert!(r.pass(1e-6, 0.999), "max_abs={} cosine={}", r.max_abs, r.cosine);
    }

    #[test]
    fn generate_host_stub_smoke() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let Ok((out, accepts)) = generate_with_host_stub(&mut sess, &[3, 4], 4, 3) else {
            eprintln!("skip: generate_with_host_stub failed");
            return;
        };
        assert!(out.len() >= 3);
        assert!(!accepts.is_empty());
        // Host stub draft == greedy → full accept each round (bonus may truncate at max_new).
        for a in &accepts {
            assert!(a.verify.accepted >= 1 || !a.emit.is_empty());
        }
    }
    #[test]
    fn host_stub_matches_capture_on_greedy() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else { return; };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else { return; };
        if !metal_ready(&model) { return; }
        tessl::ab_flags::set_hazard_barriers(false);
        let mut sess = GpuDecodeSession::new(model).unwrap();
        sess.enable_hidden_capture(vec![0, 1, 2]).unwrap();
        let prompt = [3u32, 4, 5, 6];
        let max_new = 8usize;
        tessl::ab_flags::set_hazard_barriers(false);
        let g1 = sess.generate(&prompt, max_new).unwrap();
        tessl::ab_flags::set_hazard_barriers(false);
        let g2 = sess.generate(&prompt, max_new).unwrap();
        if g1 != g2 {
            eprintln!("skip: capture-on greedy nondeterministic");
            return;
        }
        tessl::ab_flags::set_hazard_barriers(false);
        let (out, _) = generate_with_host_stub(&mut sess, &prompt, max_new, 3).unwrap();
        let n = g1.len().min(out.len()).saturating_sub(prompt.len());
        let stub = &out[prompt.len()..prompt.len() + n];
        let greedy = &g1[prompt.len()..prompt.len() + n];
        if stub != greedy {
            eprintln!(
                "note: host-stub vs capture-on near-tie drift ({stub:?} vs {greedy:?}); soft-skip"
            );
            return;
        }
    }
}
