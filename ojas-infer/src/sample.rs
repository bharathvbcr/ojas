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

use ojas_core::OjasError;

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
/// greedy call never reads it.
pub fn sample_token(
    logits: &[f32],
    cfg: &SamplingConfig,
    rng: &mut SplitMix64,
) -> Result<u32, OjasError> {
    const OP: &str = "sample_token";
    cfg.validate()?;
    if logits.is_empty() {
        return Err(OjasError::Shape {
            op: OP,
            detail: "empty logits".into(),
        });
    }
    if logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return Err(OjasError::NonFinite { op: OP });
    }
    // Finite candidates ranked by (logit desc, index asc).
    let mut ranked: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    if ranked.is_empty() {
        return Err(OjasError::NonFinite { op: OP });
    }
    ranked.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    if cfg.temperature == 0.0 {
        return to_id(ranked[0]);
    }
    if let Some(k) = cfg.top_k {
        ranked.truncate(k);
    }
    // Finite f32 over a positive f32 temperature stays below 2^280 in f64,
    // so every scaled logit is finite, `z - top <= 0`, and the leader
    // contributes exp(0) = 1: `total` is in [1, k].
    let t = f64::from(cfg.temperature);
    let scaled: Vec<f64> = ranked.iter().map(|&i| f64::from(logits[i]) / t).collect();
    let top = scaled[0];
    let mut probs: Vec<f64> = scaled.iter().map(|z| (z - top).exp()).collect();
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
            return to_id(ranked[slot]);
        }
    }
    // Rounding left `u` at or above the running sum: the last kept token.
    to_id(ranked[keep - 1])
}

fn to_id(index: usize) -> Result<u32, OjasError> {
    u32::try_from(index).map_err(|_| OjasError::OutOfRange {
        op: "sample_token",
        detail: "logit index exceeds u32".into(),
    })
}
