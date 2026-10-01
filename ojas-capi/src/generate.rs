//! Next-token selection.
//!
//! [`GEN_LOGITS`] calls [`ojas_infer::argmax_token`]. A non-finite logit is
//! an error from that function, never token 0.
//!
//! [`GEN_GREEDY`] cannot call [`ojas_infer::CpuGpt::greedy_decode`]: that
//! function is public, and [`ojas_infer::GptWeights`] is public, but the
//! block weight type is not exported, so a model cannot be built from this
//! crate. The public pieces that can be called are [`ojas_infer::embed`] and
//! [`ojas_infer::argmax_token`]. The logits in between are a
//! [`ojas_cpu::CpuBackend`] linear of the embedded prompt.

use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_infer::{argmax_token, embed};

pub const GEN_LOGITS: u32 = 1;
pub const GEN_GREEDY: u32 = 2;

pub fn generate(mode: u32, body: GenerateBody<'_>) -> Result<u32, String> {
    match mode {
        GEN_LOGITS => argmax_token(body.logits).map_err(|err| format!("generate: {err}")),
        GEN_GREEDY => greedy(body.prompt),
        other => Err(format!("generate: shape: unknown mode {other}")),
    }
}

pub struct GenerateBody<'a> {
    pub logits: &'a [f32],
    pub prompt: &'a [u32],
}

fn greedy(prompt: &[u32]) -> Result<u32, String> {
    if prompt.is_empty() {
        return Err("generate: shape: prompt is empty".to_string());
    }
    // Vocab 2, width 1. Token 0 embeds to 1, token 1 embeds to 0.25.
    let table = [1.0f32, 0.25];
    let mut hidden = vec![0.0f32; prompt.len()];
    embed(&table, 1, prompt, &mut hidden).map_err(|err| format!("generate: {err}"))?;
    let budget = Budget::new(1 << 20);
    let cpu = CpuBackend::new(budget.clone());
    let x = Tensor::from_f32(&hidden, &[prompt.len(), 1], &budget)
        .map_err(|err| format!("generate: {err}"))?;
    let weight = Tensor::from_f32(&[1.0, 0.25], &[2, 1], &budget)
        .map_err(|err| format!("generate: {err}"))?;
    let logits = cpu
        .linear_forward(&x, &weight)
        .map_err(|err| format!("generate: {err}"))?;
    let values = logits
        .to_f32_vec()
        .map_err(|err| format!("generate: {err}"))?;
    let start = values
        .len()
        .checked_sub(2)
        .ok_or_else(|| "generate: shape: linear returned fewer than 2 logits".to_string())?;
    argmax_token(&values[start..]).map_err(|err| format!("generate: {err}"))
}
