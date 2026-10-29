//! LoRA over [`Graph`]: `y = x W^T + (alpha / r) (x A^T) B^T`.
//!
//! The base weight `W` is frozen ([`Graph::frozen`]); `A` (`[r, in]`) and
//! `B` (`[out, r]`) are the only trainable leaves, two small f32 tensors
//! and ordinary [`Graph::linear`] ops, so the same code trains on a
//! [`ojas_autograd::Tape`] over any backend and runs eagerly on
//! [`crate::Eval`]. Nothing here is specific to one model.
//!
//! Init is PEFT's and the plain-torch reference arm's (`qd_train/lora.py`):
//! `A` is `kaiming_uniform_(a = sqrt(5))`, which for fan-in `in` is
//! `U(-1/sqrt(in), 1/sqrt(in))`; `B` is zero, so the adapter is the
//! identity at init; the scaling is `alpha / r`. Differences from that
//! reference: its draw order is torch's generator stream, which this crate
//! does not reproduce (a parity run loads torch's `A` from a fixture), and
//! it has no dropout path, so [`LoraConfig::dropout`] is PEFT's: the mask
//! multiplies the input of the low-rank branch only, scaled by `1/(1-p)`.
//!
//! The scaling reaches the graph as a constant factor on `B`
//! ([`scaled_b`]), so `dB` carries it by the chain rule. At
//! `alpha == r` the factor is exactly 1 and no op is recorded.
//!
//! A host-side [`LoraLinear`] holds the tensors and folds the adapter into
//! the weight ([`LoraLinear::merge`]) and back ([`LoraLinear::unmerge`]);
//! [`merged_weight_bf16`] is the same fold rounded once to a bf16 tensor for
//! a dense export.

use ojas_core::{bf16_to_f32, f32_to_bf16, Backend, BackendId, Budget, DType, OjasError, Tensor};
use ojas_data::CounterRng;

use crate::graph::Graph;

/// Why a LoRA request was refused.
#[derive(Debug)]
pub enum LoraError {
    /// `r == 0`: an adapter of rank zero trains nothing.
    ZeroRank,
    /// `r` above `min(in, out)`: the update `B A` could not use it.
    RankTooLarge {
        rank: usize,
        in_features: usize,
        out_features: usize,
    },
    /// `alpha` is NaN, infinite or negative.
    BadAlpha(f32),
    /// Dropout probability outside `[0, 1)`.
    BadDropout(f32),
    /// `merge` on an adapter that is already folded into its weight.
    AlreadyMerged,
    /// `unmerge` on an adapter that is not folded into its weight.
    NotMerged,
    /// A tensor does not have the shape or dtype the adapter needs.
    Mismatch(String),
    /// The backend or tensor layer refused.
    Core(OjasError),
}

impl std::fmt::Display for LoraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroRank => write!(f, "lora rank must be at least 1"),
            Self::RankTooLarge {
                rank,
                in_features,
                out_features,
            } => write!(
                f,
                "lora rank {rank} exceeds min(in {in_features}, out {out_features})"
            ),
            Self::BadAlpha(a) => write!(f, "lora alpha {a} is not finite and non-negative"),
            Self::BadDropout(p) => write!(f, "lora dropout {p} is not in [0, 1)"),
            Self::AlreadyMerged => write!(f, "lora adapter is already merged"),
            Self::NotMerged => write!(f, "lora adapter is not merged"),
            Self::Mismatch(detail) => write!(f, "lora: {detail}"),
            Self::Core(e) => write!(f, "lora: {e}"),
        }
    }
}

impl std::error::Error for LoraError {}

impl From<OjasError> for LoraError {
    fn from(e: OjasError) -> Self {
        Self::Core(e)
    }
}

impl From<LoraError> for OjasError {
    fn from(e: LoraError) -> Self {
        match e {
            LoraError::Core(e) => e,
            other => OjasError::OutOfRange {
                op: "lora",
                detail: other.to_string(),
            },
        }
    }
}

/// `r`, `alpha` and dropout of an adapter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoraConfig {
    rank: usize,
    alpha: f32,
    dropout: f32,
}

impl LoraConfig {
    /// `rank >= 1`, `alpha` finite and non-negative (0 is allowed: the
    /// adapter then contributes nothing and trains nothing), `dropout` in
    /// `[0, 1)`.
    pub fn new(rank: usize, alpha: f32, dropout: f32) -> Result<Self, LoraError> {
        if rank == 0 {
            return Err(LoraError::ZeroRank);
        }
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(LoraError::BadAlpha(alpha));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(LoraError::BadDropout(dropout));
        }
        Ok(Self {
            rank,
            alpha,
            dropout,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn dropout(&self) -> f32 {
        self.dropout
    }

    /// `alpha / r`.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Refuse a rank the linear's shape cannot use.
    pub fn check_shape(&self, in_features: usize, out_features: usize) -> Result<(), LoraError> {
        if self.rank > in_features.min(out_features) {
            return Err(LoraError::RankTooLarge {
                rank: self.rank,
                in_features,
                out_features,
            });
        }
        Ok(())
    }
}

/// A source of random words, injectable so a test can fix the dropout
/// stream. [`CounterRng`] is the seeded one.
pub trait Entropy {
    fn next_u64(&mut self) -> u64;
}

impl Entropy for CounterRng {
    fn next_u64(&mut self) -> u64 {
        CounterRng::next_u64(self)
    }
}

/// `2^-53`.
const ULP53: f64 = 1.0 / (1u64 << 53) as f64;

fn unit(rng: &mut impl Entropy) -> f64 {
    (rng.next_u64() >> 11) as f64 * ULP53
}

/// The element count of `dims`, refused if it overflows or its f32 bytes
/// would not fit `budget`, before anything is allocated.
fn element_count(dims: &[usize], budget: &Budget) -> Result<usize, LoraError> {
    let n = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| LoraError::Mismatch(format!("shape {dims:?} overflows")))?;
    let bytes = n
        .checked_mul(4)
        .ok_or_else(|| LoraError::Mismatch(format!("shape {dims:?} overflows in bytes")))?;
    budget.check_room(bytes as u64)?;
    Ok(n)
}

/// `A` `[r, in]` as PEFT initialises it: `kaiming_uniform_(a = sqrt(5))`,
/// i.e. `U(-bound, bound)` with `bound = gain * sqrt(3 / fan_in)`,
/// `gain = sqrt(2 / (1 + a^2)) = sqrt(1/3)`, so `bound = 1 / sqrt(in)`.
pub fn init_a(
    config: &LoraConfig,
    in_features: usize,
    rng: &mut impl Entropy,
    budget: &Budget,
) -> Result<Tensor, LoraError> {
    if in_features == 0 {
        return Err(LoraError::Mismatch("in_features is 0".into()));
    }
    let gain = (2.0 / (1.0 + 5.0f64)).sqrt();
    let bound = gain * (3.0 / in_features as f64).sqrt();
    let n = element_count(&[config.rank, in_features], budget)?;
    let values: Vec<f32> = (0..n)
        .map(|_| ((2.0 * unit(rng) - 1.0) * bound) as f32)
        .collect();
    Ok(Tensor::from_f32(
        &values,
        &[config.rank, in_features],
        budget,
    )?)
}

/// `B` `[out, r]`: zeros.
pub fn init_b(
    config: &LoraConfig,
    out_features: usize,
    budget: &Budget,
) -> Result<Tensor, LoraError> {
    let n = element_count(&[out_features, config.rank], budget)?;
    Ok(Tensor::from_f32(
        &vec![0.0; n],
        &[out_features, config.rank],
        budget,
    )?)
}

/// A dropout mask of `shape`: each element is 0 with probability `p` and
/// `1 / (1 - p)` otherwise, the rule `nn.Dropout` applies in training.
pub fn dropout_mask(
    p: f32,
    shape: &[usize],
    rng: &mut impl Entropy,
    budget: &Budget,
) -> Result<Tensor, LoraError> {
    if !(0.0..1.0).contains(&p) {
        return Err(LoraError::BadDropout(p));
    }
    let n = element_count(shape, budget)?;
    let keep = 1.0 / (1.0 - p);
    let values: Vec<f32> = (0..n)
        .map(|_| if unit(rng) < f64::from(p) { 0.0 } else { keep })
        .collect();
    Ok(Tensor::from_f32(&values, shape, budget)?)
}

/// One adapter: a shared `A` and one `B` per output part. A plain linear
/// has one part. A fused Hugging Face tensor that a model holds as several
/// linears (`in_proj_qkv`, `q_proj`) has one `B` block per part: the output
/// rows of `B` are independent, so the parts are the fused `B` exactly.
#[derive(Clone, Debug)]
pub struct LoraSite<V> {
    pub a: V,
    pub b: Vec<V>,
}

impl LoraSite<Tensor> {
    /// A fresh adapter for parts of `out_parts` output rows each: `A` from
    /// `rng`, every `B` zero.
    pub fn init(
        config: &LoraConfig,
        in_features: usize,
        out_parts: &[usize],
        rng: &mut impl Entropy,
        budget: &Budget,
    ) -> Result<Self, LoraError> {
        let total: usize = out_parts.iter().sum();
        config.check_shape(in_features, total)?;
        let a = init_a(config, in_features, rng, budget)?;
        let b = out_parts
            .iter()
            .map(|&rows| init_b(config, rows, budget))
            .collect::<Result<_, _>>()?;
        Ok(Self { a, b })
    }
}

impl<V> LoraSite<V> {
    /// `f` of `A` and of every `B`, in that order.
    pub fn try_map<W, E>(&self, f: &mut impl FnMut(&V) -> Result<W, E>) -> Result<LoraSite<W>, E> {
        Ok(LoraSite {
            a: f(&self.a)?,
            b: self.b.iter().map(&mut *f).collect::<Result<_, _>>()?,
        })
    }
}

/// `x A^T`, with `mask` (if any) multiplied into `x` first: PEFT's
/// `lora_A(dropout(x))`. Shared by every part of a fused site.
pub fn lora_down<G: Graph>(
    g: &mut G,
    x: &G::V,
    a: &G::V,
    mask: Option<&G::V>,
) -> Result<G::V, OjasError> {
    match mask {
        Some(m) => {
            let dropped = g.mul(x, m)?;
            g.linear(&dropped, a)
        }
        None => g.linear(x, a),
    }
}

/// `B` times `scale`, as a graph node. `scale == 1` is `b` itself;
/// otherwise `scale_like` is a frozen constant of `b`'s shape holding
/// `scale` (see [`scale_constant`]).
pub fn scaled_b<G: Graph>(
    g: &mut G,
    b: &G::V,
    scale: f32,
    scale_like: Option<&G::V>,
) -> Result<G::V, OjasError> {
    if scale == 1.0 {
        return Ok(b.clone());
    }
    let like = scale_like.ok_or_else(|| OjasError::OutOfRange {
        op: "lora::scaled_b",
        detail: "a scaling other than 1 needs a scale constant".into(),
    })?;
    g.mul(b, like)
}

/// A frozen constant of `shape` holding `scale`, recorded outside any
/// checkpointed segment.
pub fn scale_constant<G: Graph>(
    g: &mut G,
    shape: &[usize],
    scale: f32,
    budget: &Budget,
) -> Result<G::V, OjasError> {
    let n: usize = shape.iter().product();
    g.frozen(&Tensor::from_f32(&vec![scale; n], shape, budget)?)
}

/// `h B_eff^T` for the already scaled `b_eff` and `h = x A^T`.
pub fn lora_up<G: Graph>(g: &mut G, h: &G::V, b_eff: &G::V) -> Result<G::V, OjasError> {
    g.linear(h, b_eff)
}

/// The whole LoRA linear `x W^T + (x A^T) B_eff^T` for a plain site.
pub fn lora_linear<G: Graph>(
    g: &mut G,
    x: &G::V,
    w: &G::V,
    a: &G::V,
    b_eff: &G::V,
    mask: Option<&G::V>,
) -> Result<G::V, OjasError> {
    let base = g.linear(x, w)?;
    let h = lora_down(g, x, a, mask)?;
    let delta = lora_up(g, &h, b_eff)?;
    g.add(&base, &delta)
}

/// `[out, in]` of `weight`, after checking `a` (`[r, in]`) and `b`
/// (`[out, r]`, both f32) against `config`.
fn check_operands(
    config: &LoraConfig,
    weight: &Tensor,
    a: &Tensor,
    b: &Tensor,
) -> Result<(usize, usize), LoraError> {
    let &[out, inn] = weight.shape() else {
        return Err(LoraError::Mismatch(format!(
            "weight {:?} is not [out, in]",
            weight.shape()
        )));
    };
    config.check_shape(inn, out)?;
    if a.shape() != [config.rank, inn] || b.shape() != [out, config.rank] {
        return Err(LoraError::Mismatch(format!(
            "A {:?} and B {:?} for W {:?} at rank {}",
            a.shape(),
            b.shape(),
            weight.shape(),
            config.rank
        )));
    }
    for t in [a, b] {
        if t.dtype() != DType::F32 {
            return Err(LoraError::Mismatch("A and B are f32".into()));
        }
    }
    Ok((out, inn))
}

/// The unscaled product `B A` `[out, in]` as a charged host f32 tensor,
/// computed on `backend`. Beside the result only `A^T` (`r * in`) and the
/// backend's product scratch are charged, and both are gone on return.
fn lora_product<B: Backend>(
    backend: &B,
    a: &Tensor,
    b: &Tensor,
    rank: usize,
    inn: usize,
) -> Result<Tensor, LoraError> {
    let budget = backend.budget().clone();
    // B A as `linear(B, A^T)`: `B [out, r]` rows against `A^T^T = A`'s
    // transpose `[in, r]` as the weight.
    let a = a.to_f32_vec()?;
    let mut at = vec![0.0f32; inn * rank];
    for k in 0..rank {
        for j in 0..inn {
            at[j * rank + k] = a[k * inn + j];
        }
    }
    let at = Tensor::from_f32(&at, &[inn, rank], &budget)?;
    let place = |t: &Tensor| -> Result<Tensor, OjasError> {
        if backend.id() == BackendId::Cpu {
            Ok(t.clone())
        } else {
            backend.upload(t)
        }
    };
    let delta = backend.linear_forward(&place(b)?, &place(&at)?)?;
    Ok(delta.to_host(&budget)?)
}

/// `W + (alpha/r) B A` as a bf16 tensor `[out, in]`: the sum in f32, exactly
/// as [`LoraLinear::merge`] forms it (`w + s * d`, `s = alpha / r`, `d` the
/// backend's `B A`), then rounded once to nearest even by
/// [`ojas_core::f32_to_bf16`], the one bf16 conversion of every ojas path.
/// [`LoraLinear::merge`] itself keeps the f32 sum; this is the export form.
///
/// `weight` is f32 or bf16 (a bf16 weight is widened exactly), `a` and `b`
/// f32, all on the host. Nothing is mutated. `alpha == 0` gives `weight`
/// itself for a bf16 weight and its single rounding for an f32 one, except
/// that a stored `-0.0` becomes `+0.0` (`-0.0 + 0.0` in the f32 sum, as in
/// [`LoraLinear::merge`]); rank 0 is
/// refused by [`LoraConfig::new`]. A NaN or infinite weight, adapter or
/// product, or a finite sum that rounds past the largest bf16, is
/// [`OjasError::NonFinite`] and no tensor is returned. Charged: `A^T`, the
/// backend's scratch and the f32 product `[out, in]` while the result
/// `[out, in]` bf16 is written, so the peak is one f32 weight plus half a
/// weight plus the small terms; no f32 merged weight is built.
pub fn merged_weight_bf16<B: Backend>(
    config: &LoraConfig,
    weight: &Tensor,
    a: &Tensor,
    b: &Tensor,
    backend: &B,
) -> Result<Tensor, LoraError> {
    const OP: &str = "lora::merged_weight_bf16";
    let (out, inn) = check_operands(config, weight, a, b)?;
    if !matches!(weight.dtype(), DType::F32 | DType::Bf16) {
        return Err(LoraError::Mismatch(format!(
            "weight is {:?}, not f32 or bf16",
            weight.dtype()
        )));
    }
    let budget = backend.budget().clone();
    let delta = lora_product(backend, a, b, config.rank, inn)?;
    let delta = delta.f32_slice()?;
    let mut merged = Tensor::zeros(&[out, inn], DType::Bf16, &budget)?;
    let sum = merged.bf16_slice_mut()?;
    if delta.len() != sum.len() {
        return Err(LoraError::Mismatch(format!(
            "B A has {} elements, the weight {}",
            delta.len(),
            sum.len()
        )));
    }
    let s = config.scaling();
    match weight.dtype() {
        DType::Bf16 => {
            let w = weight.bf16_slice()?;
            for ((m, &w), &d) in sum.iter_mut().zip(w).zip(delta) {
                *m = f32_to_bf16(bf16_to_f32(w) + s * d);
            }
        }
        _ => {
            let w = weight.f32_slice()?;
            for ((m, &w), &d) in sum.iter_mut().zip(w).zip(delta) {
                *m = f32_to_bf16(w + s * d);
            }
        }
    }
    // A bf16 is finite exactly when its exponent field is not all ones; NaN
    // and infinity keep that field through the rounding.
    if sum.iter().any(|&bits| bits & 0x7F80 == 0x7F80) {
        return Err(LoraError::Core(OjasError::NonFinite { op: OP }));
    }
    Ok(merged)
}

/// A LoRA linear's host tensors and its fold into the weight.
///
/// `weight` is what a graph should read as `W`: the original while the
/// adapter is separate, `W + (alpha/r) B A` (f32) while merged. The
/// original is kept as stored (any dtype) while merged and put back by
/// [`LoraLinear::unmerge`], so the round trip is exact by construction and
/// does not rely on `(W + d) - d == W`.
#[derive(Clone, Debug)]
pub struct LoraLinear {
    config: LoraConfig,
    a: Tensor,
    b: Tensor,
    weight: Tensor,
    original: Option<Tensor>,
}

impl LoraLinear {
    /// `weight` `[out, in]` (f32 or bf16), `a` `[r, in]` and `b` `[out, r]`
    /// f32, all on the host.
    pub fn new(
        config: LoraConfig,
        weight: Tensor,
        a: Tensor,
        b: Tensor,
    ) -> Result<Self, LoraError> {
        check_operands(&config, &weight, &a, &b)?;
        Ok(Self {
            config,
            a,
            b,
            weight,
            original: None,
        })
    }

    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    pub fn a(&self) -> &Tensor {
        &self.a
    }

    pub fn b(&self) -> &Tensor {
        &self.b
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn is_merged(&self) -> bool {
        self.original.is_some()
    }

    /// Fold `(alpha/r) B A` into the weight in f32: `W' = W + s (B A)`, the
    /// product on `backend` and the add on the host, each rounded once. A
    /// bf16 weight is widened first; its stored bits are kept for
    /// [`LoraLinear::unmerge`]. Merging twice is [`LoraError::AlreadyMerged`].
    pub fn merge<B: Backend>(&mut self, backend: &B) -> Result<(), LoraError> {
        if self.original.is_some() {
            return Err(LoraError::AlreadyMerged);
        }
        let budget = backend.budget().clone();
        let &[out, inn] = self.weight.shape() else {
            return Err(LoraError::Mismatch("weight is not [out, in]".into()));
        };
        // The product stays in its charged host tensor and the sum is written
        // straight into the charged result, so no weight-sized buffer is
        // held outside the budget.
        let delta = lora_product(backend, &self.a, &self.b, self.config.rank, inn)?;
        let delta = delta.f32_slice()?;
        let s = self.config.scaling();
        let mut merged = Tensor::zeros(&[out, inn], DType::F32, &budget)?;
        let sum = merged.f32_slice_mut()?;
        if delta.len() != sum.len() {
            return Err(LoraError::Mismatch(format!(
                "B A has {} elements, the weight {}",
                delta.len(),
                sum.len()
            )));
        }
        match self.weight.dtype() {
            DType::Bf16 => {
                let w = self.weight.bf16_slice()?;
                for ((m, &w), &d) in sum.iter_mut().zip(w).zip(delta) {
                    *m = bf16_to_f32(w) + s * d;
                }
            }
            _ => {
                let w = self.weight.f32_slice()?;
                for ((m, &w), &d) in sum.iter_mut().zip(w).zip(delta) {
                    *m = w + s * d;
                }
            }
        }
        if sum.iter().any(|v| !v.is_finite()) {
            return Err(LoraError::Core(OjasError::NonFinite { op: "lora::merge" }));
        }
        self.original = Some(std::mem::replace(&mut self.weight, merged));
        Ok(())
    }

    /// [`merged_weight_bf16`] of this layer's original weight (the stored
    /// one while merged), leaving the layer as it is.
    pub fn merged_weight_bf16<B: Backend>(&self, backend: &B) -> Result<Tensor, LoraError> {
        let base = self.original.as_ref().unwrap_or(&self.weight);
        merged_weight_bf16(&self.config, base, &self.a, &self.b, backend)
    }

    /// Put the original weight back, bit for bit. Without a prior merge it
    /// is [`LoraError::NotMerged`].
    pub fn unmerge(&mut self) -> Result<(), LoraError> {
        let original = self.original.take().ok_or(LoraError::NotMerged)?;
        self.weight = original;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_refuses_what_cannot_train() {
        assert!(matches!(
            LoraConfig::new(0, 16.0, 0.0),
            Err(LoraError::ZeroRank)
        ));
        assert!(matches!(
            LoraConfig::new(4, f32::NAN, 0.0),
            Err(LoraError::BadAlpha(_))
        ));
        assert!(matches!(
            LoraConfig::new(4, -1.0, 0.0),
            Err(LoraError::BadAlpha(_))
        ));
        assert!(matches!(
            LoraConfig::new(4, 1.0, 1.0),
            Err(LoraError::BadDropout(_))
        ));
        let c = LoraConfig::new(8, 0.0, 0.0).unwrap();
        assert_eq!(c.scaling(), 0.0);
        assert!(matches!(
            c.check_shape(8, 7),
            Err(LoraError::RankTooLarge { .. })
        ));
        c.check_shape(8, 8).unwrap();
    }

    #[test]
    fn a_is_inside_the_kaiming_bound_and_b_is_zero() {
        let budget = Budget::new(1 << 24);
        let c = LoraConfig::new(16, 16.0, 0.0).unwrap();
        let mut rng = CounterRng::new(7);
        let a = init_a(&c, 256, &mut rng, &budget).unwrap();
        let bound = 1.0 / 256f32.sqrt();
        let v = a.to_f32_vec().unwrap();
        assert!(v.iter().all(|x| x.abs() <= bound));
        assert!(v.iter().any(|x| x.abs() > 0.9 * bound));
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.05 * bound);
        let b = init_b(&c, 32, &budget).unwrap();
        assert!(b.to_f32_vec().unwrap().iter().all(|x| x.to_bits() == 0));
    }

    #[test]
    fn a_fixed_seed_gives_the_same_mask_and_another_differs() {
        let budget = Budget::new(1 << 20);
        let m = |seed| {
            let mut rng = CounterRng::new(seed);
            dropout_mask(0.25, &[64, 8], &mut rng, &budget)
                .unwrap()
                .to_f32_vec()
                .unwrap()
        };
        assert_eq!(m(3), m(3));
        assert_ne!(m(3), m(4));
        let v = m(3);
        assert!(v.iter().all(|&x| x == 0.0 || x == 1.0 / 0.75));
        assert!(v.contains(&0.0));
        let none = {
            let mut rng = CounterRng::new(3);
            dropout_mask(0.0, &[4], &mut rng, &budget)
                .unwrap()
                .to_f32_vec()
                .unwrap()
        };
        assert_eq!(none, vec![1.0; 4]);
    }
}
