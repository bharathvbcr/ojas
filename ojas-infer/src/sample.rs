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
//! One greedy rule: the rank is [`ojas_core::rank_logits`], in which `+0.0`
//! and `-0.0` are one value, so temperature 0 (and `top_k == 1`) takes the
//! lowest index of the largest logit, as [`crate::argmax_token`],
//! `Backend::argmax_rows` and `torch.argmax` do; `Backend::topk_rows` ranks
//! the same way on every backend. Decided 2026-10-10: until then the sampler
//! ranked by `f32::total_cmp`, which put `+0.0` ahead of a `-0.0` at a lower
//! index, so the sampler and the argmax disagreed on that one tie.
//!
//! Logits: `-inf` is a mask (the token cannot be drawn); NaN and `+inf` are
//! [`OjasError::NonFinite`], and so is a row with no finite entry.
//! [`crate::argmax_token`] stays strict and refuses `-inf` too.

use ojas_core::{exp_exact, rank_logits, OjasError};

/// The sampler's generator: ojas's one SplitMix64 (`ojas_kernels`), the same
/// stream as `ojas_data::CounterRng`. `next_f64` is uniform in `[0, 1)`
/// from the top 53 bits.
pub use ojas_kernels::SplitMix64;

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
/// greedy call never reads it. The same draw as `draw` over
/// `ranked_candidates(logits, cfg.top_k)`.
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
            .reduce(|a, b| {
                if rank_logits(logits, b, a).is_lt() {
                    b
                } else {
                    a
                }
            })
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
/// by (logit desc, index asc), the order [`rank_logits`] defines, and only the
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
            ranked.select_nth_unstable_by(k - 1, |&a, &b| rank_logits(logits, a, b));
            ranked.truncate(k);
        }
        _ => {}
    }
    ranked.sort_unstable_by(|&a, &b| rank_logits(logits, a, b));
    ranked
        .into_iter()
        .map(|i| Ok((to_id(i)?, logits[i])))
        .collect()
}

/// [`ranked_candidates`] from one row of `Backend::topk_rows`: `values` and
/// `ids` are its `k` leaders, ranked as [`rank_logits`] ranks them, with `-inf` (a
/// masked logit) after every finite value. The candidates are the finite
/// prefix, so a row with fewer than `k` finite logits gives all of them and
/// one with none is [`OjasError::NonFinite`], as [`ranked_candidates`] has
/// it. A NaN or `+inf` leader (`topk_rows` reports it as a fault) and a
/// finite value after a `-inf` one are refused too.
pub(crate) fn leading_candidates(
    values: &[f32],
    ids: &[u32],
) -> Result<Vec<(u32, f32)>, OjasError> {
    if values.is_empty() || values.len() != ids.len() {
        return Err(OjasError::Shape {
            op: OP,
            detail: format!("{} leader values for {} ids", values.len(), ids.len()),
        });
    }
    if values.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return Err(OjasError::NonFinite { op: OP });
    }
    let finite = values.iter().take_while(|v| v.is_finite()).count();
    if finite == 0 {
        return Err(OjasError::NonFinite { op: OP });
    }
    if values[finite..].iter().any(|v| v.is_finite()) {
        return Err(OjasError::Shape {
            op: OP,
            detail: "a finite leader ranks after a -inf one".into(),
        });
    }
    Ok(ids[..finite]
        .iter()
        .copied()
        .zip(values[..finite].iter().copied())
        .collect())
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

fn to_id(index: usize) -> Result<u32, OjasError> {
    u32::try_from(index).map_err(|_| OjasError::OutOfRange {
        op: OP,
        detail: "logit index exceeds u32".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEG: f32 = f32::NEG_INFINITY;

    /// `-inf` leaders are masks: the finite prefix is the candidates, the
    /// same as `ranked_candidates` over the row they came from.
    #[test]
    fn leading_candidates_drop_trailing_masks() {
        let row = [NEG, 2.0, NEG, -1.0, NEG];
        let want = ranked_candidates(&row, Some(4)).unwrap();
        assert_eq!(want, [(1, 2.0), (3, -1.0)]);
        let got = leading_candidates(&[2.0, -1.0, NEG, NEG], &[1, 3, 0, 2]).unwrap();
        assert_eq!(got, want);
        assert_eq!(
            leading_candidates(&[0.5, -0.0], &[7, 2]).unwrap(),
            [(7, 0.5), (2, -0.0)]
        );
    }

    #[test]
    fn leading_candidates_refusals() {
        let non_finite = |values: &[f32]| {
            let ids: Vec<u32> = (0..values.len() as u32).collect();
            matches!(
                leading_candidates(values, &ids),
                Err(OjasError::NonFinite { op: OP })
            )
        };
        // Every leader masked: the row has no finite logit.
        assert!(non_finite(&[NEG, NEG]));
        assert!(non_finite(&[1.0, f32::NAN]));
        assert!(non_finite(&[f32::INFINITY, 1.0]));
        let shape = |values: &[f32], ids: &[u32]| {
            matches!(
                leading_candidates(values, ids),
                Err(OjasError::Shape { op: OP, .. })
            )
        };
        assert!(shape(&[], &[]));
        assert!(shape(&[1.0], &[0, 1]));
        assert!(shape(&[1.0, NEG, 0.5], &[0, 1, 2]));
    }
}
