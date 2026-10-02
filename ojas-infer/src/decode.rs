//! The one decode loop, shared by [`crate::CpuGpt`] (host KV cache) and
//! [`crate::DeviceDecoder`] (device KV cache).
//!
//! The prompt is forwarded from the cache's current length. Every emitted
//! token except the last is forwarded, so `N` new tokens need
//! `prompt + N - 1` free positions (`prompt` when `N` is 0). That is checked
//! before any forward: a request that does not fit is
//! [`OjasError::CapacityExceeded`] and the cache is untouched.

use ojas_core::OjasError;

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
    decode(
        f,
        op,
        prompt,
        cfg.max_new_tokens,
        &cfg.stop_tokens,
        |logits| sample_token(logits, &cfg.sampling, &mut rng),
    )
}

/// The decode loop. `select` picks the next id from a logit row; a stop
/// token is emitted and ends the loop.
pub(crate) fn decode(
    f: &mut impl Forward,
    op: &'static str,
    prompt: &[u32],
    new_tokens: usize,
    stop_tokens: &[u32],
    mut select: impl FnMut(&[f32]) -> Result<u32, OjasError>,
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
    let mut logits = f.forward(prompt)?;
    for step in 0..new_tokens {
        let next = select(&logits)?;
        out.push(next);
        if step + 1 == new_tokens || stop_tokens.contains(&next) {
            break;
        }
        logits = f.forward(&[next])?;
    }
    Ok(out)
}
