//! Next-token selection.
//!
//! [`GEN_LOGITS`] calls [`ojas_infer::argmax_token`] on the caller's host
//! logits on every device. A non-finite logit is an error from that function,
//! never token 0.
//!
//! [`GEN_GREEDY`] ignores the loaded file. ojas-infer exports everything a
//! model needs ([`ojas_infer::GptConfig`], [`ojas_infer::GptWeights`],
//! [`ojas_infer::BlockWeights`], [`ojas_infer::CpuGpt::greedy_decode`]), but
//! this crate does not depend on ojas-io, so it cannot read the weights out
//! of the safetensors file the session names. Until it does, greedy is a
//! fixed two-token demonstration on the session's backend (CPU, Metal or
//! wgpu): an embedding of the prompt, a linear, a read back of the last
//! row's two logits, then [`ojas_infer::argmax_token`]. The embedding output
//! is charged before that call, so a prompt-sized buffer is not allocated
//! when the budget refuses it.

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_infer::argmax_token;

use crate::session::Compute;

pub const GEN_LOGITS: u32 = 1;
pub const GEN_GREEDY: u32 = 2;

/// Generate on a single-threaded [`CpuBackend`].
pub fn generate(mode: u32, body: GenerateBody<'_>) -> Result<u32, String> {
    generate_checked(&Compute::Cpu { threads: 1 }, mode, body, || Ok(()))
}

pub fn generate_checked(
    compute: &Compute,
    mode: u32,
    body: GenerateBody<'_>,
    mut check: impl FnMut() -> Result<(), String>,
) -> Result<u32, String> {
    check()?;
    match mode {
        GEN_LOGITS => {
            check()?;
            argmax_token(body.logits).map_err(show)
        }
        GEN_GREEDY => {
            let budget = Budget::new(1 << 20);
            match compute {
                Compute::Cpu { .. } => {
                    let cpu = CpuBackend::new(budget.clone());
                    greedy(&cpu, &budget, body.prompt, &mut check)
                }
                #[cfg(target_os = "macos")]
                Compute::Metal(metal) => greedy(metal, &budget, body.prompt, &mut check),
                Compute::Wgpu(wgpu) => greedy(wgpu.as_ref(), &budget, body.prompt, &mut check),
            }
        }
        other => Err(format!("generate: shape: unknown mode {other}")),
    }
}

pub struct GenerateBody<'a> {
    pub logits: &'a [f32],
    pub prompt: &'a [u32],
}

fn greedy<B: Backend>(
    backend: &B,
    budget: &Budget,
    prompt: &[u32],
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<u32, String> {
    check()?;
    if prompt.is_empty() {
        return Err("generate: shape: prompt is empty".to_string());
    }
    // One f32 per prompt token. Refuse before embedding allocates that buffer.
    let out_bytes = u64::try_from(prompt.len())
        .ok()
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| "generate: shape: prompt length overflows".to_string())?;
    let output_charge = budget.try_reserve(out_bytes).map_err(show)?;
    check()?;
    let up = |t: Tensor| backend.upload(&t).map_err(show);
    // Vocab 2, width 1. Token 0 embeds to 1, token 1 embeds to 0.25.
    // The loaded session weights are not read.
    let table = up(Tensor::from_f32(&[1.0, 0.25], &[2, 1], budget).map_err(show)?)?;
    let ids = up(Tensor::from_u32(prompt, &[prompt.len()], budget).map_err(show)?)?;
    check()?;
    let x = backend.embedding_forward(&table, &ids).map_err(show)?;
    drop(output_charge);
    check()?;
    let weight = up(Tensor::from_f32(&[1.0, 0.25], &[2, 1], budget).map_err(show)?)?;
    let logits = backend.linear_forward(&x, &weight).map_err(show)?;
    check()?;
    let last_row = (prompt.len() - 1)
        .checked_mul(2 * std::mem::size_of::<f32>())
        .ok_or_else(|| "generate: shape: prompt length overflows".to_string())?;
    let last = logits.narrow(last_row, &[2], &[1]).map_err(show)?;
    let values = backend
        .download(&last)
        .and_then(|t| t.to_f32_vec())
        .map_err(show)?;
    argmax_token(&values).map_err(show)
}

fn show(err: ojas_core::OjasError) -> String {
    crate::ojas_error("generate", &err)
}
