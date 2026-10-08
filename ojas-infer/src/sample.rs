//! Next-token sampling: temperature, top-k, top-p, seeded.
//!
//! Order, as Hugging Face `generate` applies it: divide by the temperature,
//! keep the `top_k` largest, softmax, keep the smallest most-probable prefix
//! whose mass reaches `top_p`, renormalize, draw. Probabilities are f64.
//!
//! Differences from nanolab `GPT.generate` (model.py:479-490), on purpose:
//! - temperature 0 is greedy. nanolab divides by `max(temperature, 1e-6)`
//!   and always draws.
//! - top-k is strict: candidates are ranked by (logit desc, index asc) and
//!   exactly `k` are kept, so `top_k == 1` is greedy. nanolab's
//!   `logits < v[:, [-1]]` keeps every logit tied with the k-th.
//! - nanolab has no top-p.
//!
//! Logits: `-inf` is a mask (the token cannot be drawn); NaN and `+inf` are
//! [`OjasError::NonFinite`], and so is a row with no finite entry.
//! [`crate::argmax_token`] stays strict and refuses `-inf` too.

use std::cmp::Ordering;

use ojas_core::{exp_exact, OjasError};

/// SplitMix64 with the state as an explicit counter: the same stream as
/// `ojas_data::CounterRng`. ojas-infer does not depend on ojas-data, so the
/// generator is repeated here and a test pins it to that crate's reference
/// values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `temperature == 0` is greedy. `top_k`, when set, is at least 1.
/// `top_p`, when set, is in `(0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
        }
    }

    pub fn validate(&self) -> Result<(), OjasError> {
        let bad = |detail: String| OjasError::OutOfRange {
            op: "SamplingConfig",
            detail,
        };
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return Err(bad(format!(
                "temperature {} is not finite and >= 0",
                self.temperature
            )));
        }
        if self.top_k == Some(0) {
            return Err(bad("top_k is 0".into()));
        }
        if let Some(p) = self.top_p {
            if !(p > 0.0 && p <= 1.0) {
                return Err(bad(format!("top_p {p} is outside (0, 1]")));
            }
        }
        Ok(())
    }
}

/// Settings for [`crate::CpuGpt::generate`]. Generation stops after
/// `max_new_tokens` or right after emitting any of `stop_tokens`, which is
/// included in the output.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateConfig {
    pub sampling: SamplingConfig,
    pub seed: u64,
    pub max_new_tokens: usize,
    pub stop_tokens: Vec<u32>,
}

/// One token id from `logits`. A refused call does not advance `rng`; a
/// greedy call never reads it. The same draw as [`draw`] over
/// [`ranked_candidates`]`(logits, cfg.top_k)`.
pub fn sample_token(
    logits: &[f32],
    cfg: &SamplingConfig,
    rng: &mut SplitMix64,
) -> Result<u32, OjasError> {
    cfg.validate()?;
    check_row(logits)?;
    if cfg.temperature == 0.0 {
        // The leader alone, in one pass with no allocation.
        let best = (0..logits.len())
            .filter(|&i| logits[i].is_finite())
            .reduce(|a, b| if rank(logits, b, a).is_lt() { b } else { a })
            .ok_or(OjasError::NonFinite { op: OP })?;
        return to_id(best);
    }
    draw(&ranked_candidates(logits, cfg.top_k)?, cfg, rng)
}

const OP: &str = "sample_token";

/// Refuse an empty row, and NaN or `+inf` anywhere in it.
fn check_row(logits: &[f32]) -> Result<(), OjasError> {
    if logits.is_empty() {
        return Err(OjasError::Shape {
            op: OP,
            detail: "empty logits".into(),
        });
    }
    if logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return Err(OjasError::NonFinite { op: OP });
    }
    Ok(())
}

/// The candidates of a logit row, `(id, logit)`: its finite entries ranked
/// by (logit desc, index asc), the order [`rank`] defines, and only the
/// first `k` of them when `top_k` is `Some(k)`. The `k` leaders are
/// selected in O(V) and only they are sorted. The refusals of
/// [`sample_token`]: an empty row, NaN or `+inf` anywhere, no finite entry.
pub(crate) fn ranked_candidates(
    logits: &[f32],
    top_k: Option<usize>,
) -> Result<Vec<(u32, f32)>, OjasError> {
    check_row(logits)?;
    let mut ranked: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    if ranked.is_empty() {
        return Err(OjasError::NonFinite { op: OP });
    }
    match top_k {
        Some(k) if k < ranked.len() => {
            ranked.select_nth_unstable_by(k - 1, |&a, &b| rank(logits, a, b));
            ranked.truncate(k);
        }
        _ => {}
    }
    ranked.sort_unstable_by(|&a, &b| rank(logits, a, b));
    ranked
        .into_iter()
        .map(|i| Ok((to_id(i)?, logits[i])))
        .collect()
}

/// One id from `candidates`: finite `(id, logit)` pairs already ranked by
/// (logit desc, index asc) and already cut to `top_k`. Temperature 0 takes
/// the first without reading `rng`. Otherwise: divide by the temperature,
/// softmax, keep the `top_p` prefix, renormalize, draw. No candidates is
/// [`OjasError::NonFinite`]; a refused call does not advance `rng`.
pub(crate) fn draw(
    candidates: &[(u32, f32)],
    cfg: &SamplingConfig,
    rng: &mut SplitMix64,
) -> Result<u32, OjasError> {
    cfg.validate()?;
    let Some(&(leader, _)) = candidates.first() else {
        return Err(OjasError::NonFinite { op: OP });
    };
    if candidates.iter().any(|(_, v)| !v.is_finite()) {
        return Err(OjasError::NonFinite { op: OP });
    }
    if cfg.temperature == 0.0 {
        return Ok(leader);
    }
    // Finite f32 over a positive f32 temperature stays below 2^280 in f64,
    // so every scaled logit is finite, `z - top <= 0`, and the leader
    // contributes exp(0) = 1: `total` is in [1, k]. The exponential is
    // `exp_exact` (the same bits on every platform) of the difference
    // rounded to f32, so a seed draws the same ids everywhere.
    let t = f64::from(cfg.temperature);
    let scaled: Vec<f64> = candidates.iter().map(|&(_, v)| f64::from(v) / t).collect();
    let top = scaled[0];
    let mut probs: Vec<f64> = scaled
        .iter()
        .map(|z| f64::from(exp_exact((z - top) as f32)))
        .collect();
    let total: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= total;
    }
    let mut keep = probs.len();
    if let Some(top_p) = cfg.top_p {
        let target = f64::from(top_p);
        let mut mass = 0.0;
        for (i, p) in probs.iter().enumerate() {
            mass += p;
            if mass >= target {
                keep = i + 1;
                break;
            }
        }
    }
    let kept = &probs[..keep];
    let mass: f64 = kept.iter().sum();
    let u = rng.next_f64() * mass;
    let mut acc = 0.0;
    for (slot, p) in kept.iter().enumerate() {
        acc += p;
        if u < acc {
            return Ok(candidates[slot].0);
        }
    }
    // Rounding left `u` at or above the running sum: the last kept token.
    Ok(candidates[keep - 1].0)
}

/// Candidate order: logit descending (`total_cmp`, so `+0.0` ranks before
/// `-0.0`), then index ascending. Indices are distinct, so no two compare
/// equal and an unstable sort or select gives the one order.
fn rank(logits: &[f32], a: usize, b: usize) -> Ordering {
    logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
}

fn to_id(index: usize) -> Result<u32, OjasError> {
    u32::try_from(index).map_err(|_| OjasError::OutOfRange {
        op: OP,
        detail: "logit index exceeds u32".into(),
    })
}
