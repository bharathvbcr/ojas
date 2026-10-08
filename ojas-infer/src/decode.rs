//! The one decode loop, shared by [`crate::CpuGpt`] (host KV cache) and
//! [`crate::DeviceDecoder`] (device KV cache).
//!
//! The prompt is forwarded from the cache's current length. Every emitted
//! token except the last is forwarded, so `N` new tokens need
//! `prompt + N - 1` free positions (`prompt` when `N` is 0). That is checked
//! before any forward: a request that does not fit is
//! [`OjasError::CapacityExceeded`] and the cache is untouched.

use ojas_core::OjasError;

use crate::gpt::argmax_token;
use crate::sample::{sample_token, GenerateConfig, SplitMix64};

/// A model plus its KV cache, forwarded from where the cache ends.
pub(crate) trait Forward {
    /// Positions still free in the cache.
    fn room(&self) -> usize;
    /// The error for a request of `needed` positions that do not fit.
    fn refusal(&self, needed: usize) -> OjasError;
    /// Refuse an id outside the vocabulary.
    fn check_token(&self, op: &'static str, token: u32) -> Result<(), OjasError>;
    /// Forward `tokens` from the cache's end; the last position's logits.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, OjasError>;
    /// [`Self::forward`], then [`argmax_token`] of the logits. A device
    /// decoder overrides it to take the argmax where the logits are.
    fn forward_greedy(&mut self, tokens: &[u32]) -> Result<u32, OjasError> {
        argmax_token(&self.forward(tokens)?)
    }
}

/// Sampled continuation of `prompt`: [`sample_token`] with a [`SplitMix64`]
/// seeded from `cfg.seed`.
pub(crate) fn generate(
    f: &mut impl Forward,
    op: &'static str,
    prompt: &[u32],
    cfg: &GenerateConfig,
) -> Result<Vec<u32>, OjasError> {
    cfg.sampling.validate()?;
    let mut rng = SplitMix64::new(cfg.seed);
    let mut select = |logits: &[f32]| sample_token(logits, &cfg.sampling, &mut rng);
    decode(
        f,
        op,
        prompt,
        cfg.max_new_tokens,
        &cfg.stop_tokens,
        Some(&mut select),
    )
}

/// The decode loop. `select` picks the next id from a logit row; `None`
/// is greedy through [`Forward::forward_greedy`]. A stop token is emitted
/// and ends the loop.
pub(crate) fn decode(
    f: &mut impl Forward,
    op: &'static str,
    prompt: &[u32],
    new_tokens: usize,
    stop_tokens: &[u32],
    mut select: Option<&mut dyn FnMut(&[f32]) -> Result<u32, OjasError>>,
) -> Result<Vec<u32>, OjasError> {
    if prompt.is_empty() {
        return Err(OjasError::Shape {
            op,
            detail: "prompt is empty".into(),
        });
    }
    for &id in prompt.iter().chain(stop_tokens) {
        f.check_token(op, id)?;
    }
    let needed = prompt
        .len()
        .checked_add(new_tokens.saturating_sub(1))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "prompt plus new tokens overflows".into(),
        })?;
    if needed > f.room() {
        return Err(f.refusal(needed));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(new_tokens)
        .map_err(|_| OjasError::OutOfRange {
            op,
            detail: format!("cannot allocate {new_tokens} output ids"),
        })?;
    if new_tokens == 0 {
        // Forwarded for its cache contract: the prompt is appended.
        f.forward(prompt)?;
        return Ok(out);
    }
    let mut step = |tokens: &[u32]| match select.as_mut() {
        Some(select) => select(&f.forward(tokens)?),
        None => f.forward_greedy(tokens),
    };
    let mut next = step(prompt)?;
    loop {
        out.push(next);
        if out.len() == new_tokens || stop_tokens.contains(&next) {
            break;
        }
        next = step(&[next])?;
    }
    Ok(out)
}
