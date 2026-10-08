//! Torch `clip_grad_norm_` and single-tensor AdamW, plus Muon NS5 in f32 or bf16.
//!
//! Clip coefficient is `max_norm / (total_norm + CLIP_GRAD_NORM_EPS)` with
//! [`ojas_core::CLIP_GRAD_NORM_EPS`] = `1e-6`, then `min(1, coefficient)`.
//! The epsilon stays in the denominator; the scale is not `max_norm / total_norm`.
//! When the clamped coefficient is 1, the gradient bits are left unchanged.
//!
//! AdamW (torch decoupled; the step's scalars are formed in f64 by
//! [`ojas_core::check_adamw`]). Under [`Numerics::Exact`] every element
//! operation is f64 and rounds to f32 once at the store. Under
//! [`Numerics::Fast`] the scalars are rounded to f32 once and every element
//! operation is f32, as torch's single-tensor AdamW is on f32 tensors and as
//! the Metal kernel is with the same scalars. The steps:
//! 1. If weight decay is not exactly 0, `p *= 1 - lr * weight_decay` first.
//!    Weight decay 0 skips that multiply.
//! 2. Moments. The first moment uses torch `lerp`, which switches at weight 0.5.
//! 3. `denom = sqrt(v) / sqrt(1 - beta2^step) + eps`, with `eps` outside the square root.
//! 4. `p += -lr / (1 - beta1^step) * m / denom`.
//!
//! `beta^step` is computed in f64 by [`ojas_core::pow_u64`] (binary exponentiation),
//! not `f32` pow, inside [`ojas_core::check_adamw`], which also advances the step.
//!
//! Newton-Schulz on CPU stores the iterate in f32. Under
//! [`Ns5Precision::Bf16`] that iterate holds bf16 values and every
//! intermediate is rounded as stock nanolab's `X = G.bfloat16()` rounds it
//! (see `newton_schulz`). Coefficients are the f64
//! literals 3.4445, -4.7750, 2.0315. Frobenius epsilon is 1e-7, added to the
//! norm, not inside the square root. The step scalars `1 - lr * wd` and
//! `-lr * max(1, rows/cols)^0.5` are formed in f64 and rounded to f32 once,
//! as nanolab's Python floats are. The Nesterov blend and the parameter
//! update are one fused `mul_add` per value, as torch's `add(.., alpha=..)`
//! is; the momentum buffer and the decay are rounded op by op.
//!
//! Configs outside torch's accepted ranges (negative `lr`, `weight_decay`, or
//! Muon `momentum`) are [`OjasError::OutOfRange`], matching the Metal path.

use ojas_core::{
    check_adamw, require_ns5, round_f32_to_bf16, AdamWConfig, MuonNs5Config, Ns5Precision,
    Numerics, OjasError, Tensor, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS,
};

use std::sync::Mutex;

use crate::gemm::{gemm_out, gram_out, scratch as gemm_scratch, whole_call, Mat};
use crate::linalg::transpose;
use crate::pool::Exec;
use crate::pool::{self, scoped};
use crate::validate::{all_finite, f32_values, nonfinite, nonfinite_first, product, shape};

/// Values per task of the in-place element passes ([`scale_in_place`],
/// [`adamw_in_place`]). Each value is computed alone, so the cut changes no
/// bit.
const ELEM_BLOCK: usize = 1 << 17;

/// Values per block of the Fast norm's fixed partition.
const NORM_BLOCK: usize = 1 << 20;

/// Partial sums per Fast norm block, combined in lane order.
const NORM_LANES: usize = 8;

fn norm_of(op: &'static str, sum_sq: f64) -> Result<f32, OjasError> {
    let norm = sum_sq.sqrt() as f32;
    if !norm.is_finite() {
        return Err(nonfinite(op));
    }
    Ok(norm)
}

/// `acc + sum v²` in f64, ascending.
fn sum_sq_ascending(acc: f64, values: impl Iterator<Item = f32>) -> f64 {
    values.fold(acc, |sum, v| sum + f64::from(v) * f64::from(v))
}

/// Sum of squares in f64: an f32 sum overflows near 1.8e19 per element while
/// the norm itself is still a finite f32. Ascending from the first value of
/// the first part.
pub(crate) fn total_norm(op: &'static str, parts: &[Vec<f32>]) -> Result<f32, OjasError> {
    norm_of(
        op,
        parts
            .iter()
            .fold(0.0, |acc, part| sum_sq_ascending(acc, part.iter().copied())),
    )
}

/// Global norm of gradients read in place.
///
/// Exact: [`total_norm`]'s ascending f64 sum, the same bits, on the calling
/// thread. Fast: each gradient is cut into [`NORM_BLOCK`]-value blocks, a
/// block's f64 sum of squares is [`NORM_LANES`] lane sums combined in lane
/// order, and the block sums are added in order on the caller. The cut
/// depends only on the gradients' lengths, so the bits do not depend on the
/// thread count. The f64 sum of squares of finite f32 values cannot
/// overflow, so a non-finite norm means a non-finite gradient value:
/// [`OjasError::NonFinite`], which is how the norm pass doubles as the scan.
pub(crate) fn grad_norm(
    op: &'static str,
    exec: Exec<'_>,
    parts: &[&[f32]],
) -> Result<f32, OjasError> {
    let value = |v: &f32| *v;
    match exec.numerics {
        Numerics::Exact => norm_of(
            op,
            parts.iter().fold(0.0, |acc, part| {
                sum_sq_ascending(acc, part.iter().map(value))
            }),
        ),
        Numerics::Fast => {
            let blocks: Vec<&[f32]> = parts.iter().flat_map(|p| p.chunks(NORM_BLOCK)).collect();
            let sums = scoped::map(exec, blocks.len(), |i| {
                let (chunks, rest) = blocks[i].as_chunks::<NORM_LANES>();
                let mut acc = [0.0f64; NORM_LANES];
                for chunk in chunks {
                    for (a, word) in acc.iter_mut().zip(chunk) {
                        let v = f64::from(value(word));
                        *a += v * v;
                    }
                }
                let lanes = acc.iter().fold(0.0f64, |s, &v| s + v);
                Ok(sum_sq_ascending(lanes, rest.iter().map(value)))
            })?;
            norm_of(op, sums.iter().fold(0.0f64, |s, &v| s + v))
        }
    }
}

/// `grads[i] *= scale` in place for every gradient, all or nothing, with no
/// buffer and no charge.
///
/// Every gradient is proven writable ([`Tensor::ensure_writable_f32`])
/// before the first one changes, so a shared or device gradient is refused
/// with nothing scaled. Each gradient's blocks are then scaled where they
/// are, on the pool. `scale` is in `[0, 1)` ([`ojas_core::clip_scale`]) and
/// the values are finite (the norm pass scanned them), so every product is
/// finite, and a [`scoped::fill`] task that does not panic is never refused:
/// no gradient is left half scaled.
pub(crate) fn scale_in_place(
    exec: Exec<'_>,
    grads: &mut [Tensor],
    scale: f32,
) -> Result<(), OjasError> {
    for grad in grads.iter_mut() {
        let len = grad.num_elements()?;
        grad.ensure_writable_f32(len)?;
    }
    for grad in grads.iter_mut() {
        scoped::fill(exec, grad.f32_slice_mut()?, ELEM_BLOCK, |_, chunk| {
            for value in chunk {
                *value *= scale;
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// The scalars of one AdamW step, checked by [`check_adamw`], the one owner
/// every backend shares, so the f64 bits of `1 - beta^step` match.
#[derive(Clone, Copy)]
pub(crate) struct AdamCoeffs {
    beta1: f64,
    beta2: f64,
    one_minus_b1: f64,
    one_minus_b2: f64,
    step_size: f64,
    bc2_sqrt: f64,
    eps: f64,
    decay: f64,
    decays: bool,
}

impl AdamCoeffs {
    pub(crate) fn new(config: AdamWConfig, step_before: u64) -> Result<Self, OjasError> {
        let (_, bc1, bc2) = check_adamw(config, step_before)?;
        Ok(Self {
            beta1: config.beta1,
            beta2: config.beta2,
            one_minus_b1: 1.0 - config.beta1,
            one_minus_b2: 1.0 - config.beta2,
            step_size: config.lr / bc1,
            bc2_sqrt: bc2.sqrt(),
            eps: config.eps,
            decay: 1.0 - config.lr * config.weight_decay,
            decays: config.weight_decay != 0.0,
        })
    }

    /// The new first moment in f64. Torch `lerp(m, g, 1 - beta1)` switches
    /// formula at weight 0.5; `LOW` is `1 - beta1 < 0.5`.
    #[inline(always)]
    fn moment1<const LOW: bool>(&self, g: f64, m: f64) -> f64 {
        if LOW {
            m + self.one_minus_b1 * (g - m)
        } else {
            g - (g - m) * self.beta1
        }
    }

    #[inline(always)]
    fn moment2(&self, g: f64, v: f64) -> f64 {
        self.beta2 * v + self.one_minus_b2 * g * g
    }

    /// One element: `(param, moment1, moment2, finite)`. `finite` is false
    /// when a stored value or an intermediate (`m`, `v`, `denom`, `delta`) is
    /// not finite. Branch-free per element so a loop of it vectorizes;
    /// `DECAYS` (weight decay not 0) and `LOW` are uniform over a step.
    #[inline(always)]
    fn step<const DECAYS: bool, const LOW: bool>(
        &self,
        p: f32,
        g: f32,
        m: f32,
        v: f32,
    ) -> (f32, f32, f32, bool) {
        let g = f64::from(g);
        let m = self.moment1::<LOW>(g, f64::from(m));
        let v = self.moment2(g, f64::from(v));
        let denom = v.sqrt() / self.bc2_sqrt + self.eps;
        let delta = (-self.step_size) * m / denom;
        let mut q = f64::from(p);
        if DECAYS {
            q *= self.decay;
        }
        q += delta;
        // Without decay a zero step keeps the stored bits (`-0.0 + 0.0`
        // would flip a negative zero).
        let new_p = if !DECAYS && delta == 0.0 { p } else { q as f32 };
        let (m32, v32) = (m as f32, v as f32);
        let finite = m.is_finite()
            & v.is_finite()
            & denom.is_finite()
            & delta.is_finite()
            & new_p.is_finite()
            & m32.is_finite()
            & v32.is_finite();
        (new_p, m32, v32, finite)
    }

    /// [`AdamCoeffs::step`] for one element with the step's flags, or `None`
    /// when it is not finite.
    fn update(&self, p: f32, g: f32, m: f32, v: f32) -> Option<(f32, f32, f32)> {
        let (np, nm, nv, finite) = match (self.decays, self.low()) {
            (true, true) => self.step::<true, true>(p, g, m, v),
            (true, false) => self.step::<true, false>(p, g, m, v),
            (false, true) => self.step::<false, true>(p, g, m, v),
            (false, false) => self.step::<false, false>(p, g, m, v),
        };
        finite.then_some((np, nm, nv))
    }

    fn low(&self) -> bool {
        self.one_minus_b1 < 0.5
    }

    /// The same scalars rounded to f32 once, for [`Numerics::Fast`].
    fn fast(&self) -> AdamCoeffs32 {
        AdamCoeffs32 {
            beta1: self.beta1 as f32,
            beta2: self.beta2 as f32,
            one_minus_b1: self.one_minus_b1 as f32,
            one_minus_b2: self.one_minus_b2 as f32,
            step_size: self.step_size as f32,
            bc2_sqrt: self.bc2_sqrt as f32,
            eps: self.eps as f32,
            decay: self.decay as f32,
        }
    }
}

/// [`AdamCoeffs`] rounded to f32, for the [`Numerics::Fast`] element
/// arithmetic. f32 square root and division run four lanes wide, where the
/// f64 ones run two; at 50304x768 the f64 update pass was about two thirds of
/// the step's samples.
#[derive(Clone, Copy)]
struct AdamCoeffs32 {
    beta1: f32,
    beta2: f32,
    one_minus_b1: f32,
    one_minus_b2: f32,
    step_size: f32,
    bc2_sqrt: f32,
    eps: f32,
    decay: f32,
}

/// The element arithmetic of one AdamW step: f64 ([`AdamCoeffs`], Exact) or
/// f32 ([`AdamCoeffs32`], Fast). Both passes of [`adamw_in_place`] use one
/// implementation, so the values the second pass stores are the bits the
/// first pass found finite.
trait AdamMath: Sync {
    /// `(param, moment1, moment2, finite)` for one element.
    fn elem<const DECAYS: bool, const LOW: bool>(
        &self,
        p: f32,
        g: f32,
        m: f32,
        v: f32,
    ) -> (f32, f32, f32, bool);
}

impl AdamMath for AdamCoeffs {
    #[inline(always)]
    fn elem<const DECAYS: bool, const LOW: bool>(
        &self,
        p: f32,
        g: f32,
        m: f32,
        v: f32,
    ) -> (f32, f32, f32, bool) {
        self.step::<DECAYS, LOW>(p, g, m, v)
    }
}

impl AdamMath for AdamCoeffs32 {
    /// [`AdamCoeffs::step`] in f32: the same formulas and the same finite
    /// checks, with no rounding at the store because nothing is wider.
    #[inline(always)]
    fn elem<const DECAYS: bool, const LOW: bool>(
        &self,
        p: f32,
        g: f32,
        m: f32,
        v: f32,
    ) -> (f32, f32, f32, bool) {
        let m = self.m1::<LOW>(g, m);
        let v = self.m2(g, v);
        let denom = v.sqrt() / self.bc2_sqrt + self.eps;
        let delta = (-self.step_size) * m / denom;
        let mut q = p;
        if DECAYS {
            q *= self.decay;
        }
        q += delta;
        // As in the f64 step: without decay a zero step keeps the stored bits.
        let new_p = if !DECAYS && delta == 0.0 { p } else { q };
        let finite = f32_finite(m)
            & f32_finite(v)
            & f32_finite(denom)
            & f32_finite(delta)
            & f32_finite(new_p);
        (new_p, m, v, finite)
    }
}

/// `x.is_finite()`: exponent not all ones. Same booleans as `is_finite`,
/// including zeros and subnormals.
#[inline(always)]
fn f32_finite(x: f32) -> bool {
    (x.to_bits() & 0x7fff_ffff) < 0x7f80_0000
}

impl AdamCoeffs32 {
    /// [`AdamCoeffs::moment1`] in f32.
    #[inline(always)]
    fn m1<const LOW: bool>(&self, g: f32, m: f32) -> f32 {
        if LOW {
            m + self.one_minus_b1 * (g - m)
        } else {
            g - (g - m) * self.beta1
        }
    }

    /// [`AdamCoeffs::moment2`] in f32.
    #[inline(always)]
    fn m2(&self, g: f32, v: f32) -> f32 {
        self.beta2 * v + self.one_minus_b2 * g * g
    }
}

/// Block `b` (`n` values) of the [`ELEM_BLOCK`] cut of `values`.
fn elem_block<'w>(
    op: &'static str,
    values: &'w [f32],
    b: usize,
    n: usize,
) -> Result<&'w [f32], OjasError> {
    let start = b * ELEM_BLOCK;
    values
        .get(start..start + n)
        .ok_or_else(|| shape(op, "element block exceeds a tensor"))
}

/// Pass 1 of [`adamw_in_place`] over one block: whether every element's
/// step is finite. Nothing is stored. A counted index loop, not a zip:
/// the vectorizer turns it into contiguous groups and a scalar remainder.
/// `&=` evaluates every element. The four slices are the same block; a
/// shorter one panics rather than leaving a tail unchecked.
fn adam_check<C: AdamMath, const DECAYS: bool, const LOW: bool>(
    c: &C,
    p: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
) -> bool {
    let n = block_len(p, g, m, v);
    let mut ok = true;
    for i in 0..n {
        ok &= c.elem::<DECAYS, LOW>(p[i], g[i], m[i], v[i]).3;
    }
    ok
}

/// Pass 2 of [`adamw_in_place`] over one block: every element's step,
/// stored in place. Element `i` reads only element `i` of each input.
fn adam_store<C: AdamMath, const DECAYS: bool, const LOW: bool>(
    c: &C,
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
) {
    let n = block_len(p, g, m, v);
    for i in 0..n {
        let (np, nm, nv, _) = c.elem::<DECAYS, LOW>(p[i], g[i], m[i], v[i]);
        p[i] = np;
        m[i] = nm;
        v[i] = nv;
    }
}

/// Length of one AdamW block. The four slices are cut to the same `n` by
/// the caller. A shorter slice is a bug: stopping at the shortest would
/// store a prefix and report success.
fn block_len(p: &[f32], g: &[f32], m: &[f32], v: &[f32]) -> usize {
    debug_assert_eq!(p.len(), g.len());
    debug_assert_eq!(p.len(), m.len());
    debug_assert_eq!(p.len(), v.len());
    p.len()
}

/// `(param, moment1, moment2)` after one step.
type AdamState = (Vec<f32>, Vec<f32>, Vec<f32>);

/// AdamW on host vectors, for [`crate::HybridOptimizer`]: the element
/// arithmetic of [`adamw_in_place`], on the calling thread.
pub(crate) fn adamw(
    param: &[f32],
    grad: &[f32],
    moment1: &[f32],
    moment2: &[f32],
    step_before: u64,
    config: AdamWConfig,
) -> Result<AdamState, OjasError> {
    const OP: &str = "adamw_step";
    let coeffs = AdamCoeffs::new(config, step_before)?;
    let len = param.len();
    if grad.len() != len || moment1.len() != len || moment2.len() != len {
        return Err(shape(OP, "adamw tensors differ in length"));
    }
    let (mut ps, mut ms, mut vs) = (
        Vec::with_capacity(len),
        Vec::with_capacity(len),
        Vec::with_capacity(len),
    );
    for i in 0..len {
        let (p, m, v) = coeffs
            .update(param[i], grad[i], moment1[i], moment2[i])
            .ok_or_else(|| nonfinite(OP))?;
        ps.push(p);
        ms.push(m);
        vs.push(v);
    }
    Ok((ps, ms, vs))
}

/// One AdamW step written into `param`, `moment1` and `moment2` in place,
/// all or nothing, with no buffer and no charge.
///
/// The caller has checked the four layouts ([`crate::validate::f32_layouts`]).
/// The three targets are proven writable first
/// ([`Tensor::ensure_writable_f32`]: a shared or device allocation is
/// refused before anything is computed). Pass 1 computes every element's
/// step on the pool and only checks that it is finite. A non-finite input
/// always gives a non-finite result (a NaN or infinite gradient or moment
/// reaches `m` or `v`; a parameter reaches the stored value), so that pass
/// is also the inputs' NaN scan, and a refusal before it goes through
/// [`nonfinite_first`], which keeps the scan-first order. Pass 2 recomputes
/// each element with the same arithmetic, so it stores exactly the values
/// pass 1 found finite, and writes the parameter and both moments in place.
/// Element `i` reads only element `i` of each input, so a store cannot
/// change another element's step.
///
/// The element arithmetic is f64 under [`Numerics::Exact`] and f32 under
/// [`Numerics::Fast`] ([`AdamMath`]). Each element is computed alone, so the
/// bits do not depend on the thread count under either.
pub(crate) fn adamw_in_place(
    op: &'static str,
    exec: Exec<'_>,
    param: &mut Tensor,
    grad: &Tensor,
    moment1: &mut Tensor,
    moment2: &mut Tensor,
    coeffs: AdamCoeffs,
) -> Result<(), OjasError> {
    let len = param.num_elements()?;
    let claimed = param
        .ensure_writable_f32(len)
        .and_then(|()| moment1.ensure_writable_f32(len))
        .and_then(|()| moment2.ensure_writable_f32(len));
    if let Err(err) = claimed {
        return Err(nonfinite_first(
            op,
            exec,
            &[param, grad, moment1, moment2],
            err,
        ));
    }
    let flags = (coeffs.decays, coeffs.low());
    match exec.numerics {
        Numerics::Exact => adam_passes(op, exec, param, grad, moment1, moment2, &coeffs, flags),
        Numerics::Fast => adam_passes(
            op,
            exec,
            param,
            grad,
            moment1,
            moment2,
            &coeffs.fast(),
            flags,
        ),
    }
}

/// One pass-2 block of the parameter and both moments, claimed once by the
/// task that stores it.
type AdamBlock<'a> = Mutex<Option<(&'a mut [f32], &'a mut [f32], &'a mut [f32])>>;

/// The two passes of [`adamw_in_place`] with the element arithmetic `c`.
/// `flags` is `(decays, low)`.
#[allow(clippy::too_many_arguments)]
fn adam_passes<C: AdamMath>(
    op: &'static str,
    exec: Exec<'_>,
    param: &mut Tensor,
    grad: &Tensor,
    moment1: &mut Tensor,
    moment2: &mut Tensor,
    c: &C,
    flags: (bool, bool),
) -> Result<(), OjasError> {
    let g = f32_values(op, grad)?;
    {
        let (p, m, v) = (
            f32_values(op, param)?,
            f32_values(op, moment1)?,
            f32_values(op, moment2)?,
        );
        let finite = scoped::map(exec, p.len().div_ceil(ELEM_BLOCK), |b| {
            let n = ELEM_BLOCK.min(p.len() - b * ELEM_BLOCK);
            let (p, g, m, v) = (
                elem_block(op, p, b, n)?,
                elem_block(op, g, b, n)?,
                elem_block(op, m, b, n)?,
                elem_block(op, v, b, n)?,
            );
            Ok(match flags {
                (true, true) => adam_check::<C, true, true>(c, p, g, m, v),
                (true, false) => adam_check::<C, true, false>(c, p, g, m, v),
                (false, true) => adam_check::<C, false, true>(c, p, g, m, v),
                (false, false) => adam_check::<C, false, false>(c, p, g, m, v),
            })
        })?;
        if !finite.into_iter().all(|ok| ok) {
            return Err(nonfinite(op));
        }
    }
    let (p, m, v) = (
        param.f32_slice_mut()?,
        moment1.f32_slice_mut()?,
        moment2.f32_slice_mut()?,
    );
    let blocks: Vec<AdamBlock<'_>> = p
        .chunks_mut(ELEM_BLOCK)
        .zip(m.chunks_mut(ELEM_BLOCK))
        .zip(v.chunks_mut(ELEM_BLOCK))
        .map(|((p, m), v)| Mutex::new(Some((p, m, v))))
        .collect();
    scoped::map(exec, blocks.len(), |b| {
        let (p, m, v) = blocks[b]
            .lock()
            .map_err(|_| shape(op, "adamw block lock poisoned"))?
            .take()
            .ok_or_else(|| shape(op, "adamw block claimed twice"))?;
        let g = elem_block(op, g, b, p.len())?;
        match flags {
            (true, true) => adam_store::<C, true, true>(c, p, g, m, v),
            (true, false) => adam_store::<C, true, false>(c, p, g, m, v),
            (false, true) => adam_store::<C, false, true>(c, p, g, m, v),
            (false, false) => adam_store::<C, false, false>(c, p, g, m, v),
        }
        Ok(())
    })?;
    Ok(())
}

/// `f(a[i], b[i])` for every `i`, in row chunks on scoped threads, into one
/// new vector.
fn zip_map<F>(exec: Exec<'_>, a: &[f32], b: &[f32], f: F) -> Result<Vec<f32>, OjasError>
where
    F: Fn(f32, f32) -> f32 + Sync,
{
    let len = a.len().min(b.len());
    let mut out = vec![0.0f32; len];
    scoped::rows_into(exec, &mut out, len, 1, |range, dst| {
        for ((slot, &x), &y) in dst.iter_mut().zip(&a[range.clone()]).zip(&b[range]) {
            *slot = f(x, y);
        }
        Ok(())
    })?;
    Ok(out)
}

/// `b[i] = f(a[i], b[i])` in place, in [`zip_map`]'s cut.
fn zip_into<F>(exec: Exec<'_>, a: &[f32], b: &mut [f32], f: F) -> Result<(), OjasError>
where
    F: Fn(f32, f32) -> f32 + Sync,
{
    let len = a.len().min(b.len());
    scoped::rows_into(exec, &mut b[..len], len, 1, |range, dst| {
        for (slot, &x) in dst.iter_mut().zip(&a[range]) {
            *slot = f(x, *slot);
        }
        Ok(())
    })?;
    Ok(())
}

fn non_negative(op: &'static str, scalars: &[(&str, f64)]) -> Result<(), OjasError> {
    match scalars.iter().find(|(_, value)| *value < 0.0) {
        Some((name, value)) => Err(OjasError::OutOfRange {
            op,
            detail: format!("{name} {value} is negative"),
        }),
        None => Ok(()),
    }
}

/// The Muon config and step count every caller checks before it charges or
/// writes: NS5 steps, finite scalars, and no negative `lr`, `momentum` or
/// `weight_decay`.
pub(crate) fn check_muon(config: MuonNs5Config) -> Result<(), OjasError> {
    const OP: &str = "muon_ns5_step";
    require_ns5(5)?;
    if !config.lr.is_finite() || !config.momentum.is_finite() || !config.weight_decay.is_finite() {
        return Err(nonfinite(OP));
    }
    non_negative(
        OP,
        &[
            ("lr", config.lr),
            ("momentum", config.momentum),
            ("weight_decay", config.weight_decay),
        ],
    )
}

pub(crate) fn muon_ns5(
    exec: Exec<'_>,
    param: &[f32],
    grad: &[f32],
    momentum: &[f32],
    rows: usize,
    cols: usize,
    config: MuonNs5Config,
) -> Result<(Option<Vec<f32>>, Vec<f32>), OjasError> {
    const OP: &str = "muon_ns5_step";
    check_muon(config)?;
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "muon matrix length overflows".to_string(),
        })?;
    let lens = [param.len(), grad.len(), momentum.len()];
    if lens != [len; 3] {
        return Err(shape(OP, "muon tensors differ in length"));
    }
    let mom = config.momentum as f32;
    let buf = zip_map(exec, momentum, grad, |m, g| mom * m + g)?;
    if !all_finite(&buf) {
        return Err(nonfinite(OP));
    }
    // torch's `g.add(buf, alpha=mom)` and `p.add_(o, alpha=-lr * scale)` are
    // one fused multiply-add per value (`a + alpha * b` as `fmadd`), so
    // both are `mul_add` here, under Exact too. The momentum buffer
    // (`buf.mul_(mom).add_(g)`) and the decay (`p.mul_(1 - lr * wd)`) are
    // separate ops, each rounded.
    let update: Vec<f32> = if config.nesterov {
        zip_map(exec, grad, &buf, |g, b| mom.mul_add(b, g))?
    } else {
        buf.to_vec()
    };
    let ortho = newton_schulz(exec, update, rows, cols, config.ns5)?;
    if cols == 0 {
        return Err(shape(OP, "empty tensor"));
    }
    let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
    let alpha = (-config.lr * scale) as f32;
    let all_zero = ortho.iter().all(|v| *v == 0.0);
    // `None`: the parameter keeps its bits (no decay and a zero update).
    let new_p = if config.weight_decay == 0.0 && all_zero {
        None
    } else if config.weight_decay != 0.0 {
        let decay = (1.0 - config.lr * config.weight_decay) as f32;
        Some(zip_map(exec, param, &ortho, |p, o| {
            alpha.mul_add(o, p * decay)
        })?)
    } else {
        Some(zip_map(exec, param, &ortho, |p, o| alpha.mul_add(o, p))?)
    };
    drop(ortho);
    if new_p.as_deref().is_some_and(|p| !all_finite(p)) {
        return Err(nonfinite(OP));
    }
    Ok((new_p, buf))
}

/// Row bands of a Muon `A @ A` or `B @ X` on macOS, at any thread count.
/// Six is the pool size the band split was measured at
/// (`docs/bench-cpu-vs-torch.md`).
const MUON_ROW_BANDS: usize = 6;

/// `A · B` for one Newton-Schulz product.
///
/// On macOS a Fast product is one `cblas_sgemm`. That call runs on the
/// calling thread: `BLASGetThreading` is multi-threaded, and
/// `VECLIB_MAXIMUM_THREADS` does not raise it, but a live thread sample
/// still sees only the caller (measured on the Muon shapes, including a
/// 4096 cube), so the product is cut into row bands, each one whole
/// Accelerate call, and the pool's threads do the bands.
///
/// The cut depends on the shape only, never on the thread count: one thread
/// runs the same bands in turn. A band's bits are not always those rows of
/// the uncut call. They matched on an M5 Pro, but on a GitHub M1 runner and
/// under Rosetta x86_64 the 768×768 step moved by up to 1.7e-8 between one
/// call and six bands. `A @ A` and `B @ X` take [`MUON_ROW_BANDS`] bands.
/// The tall 2048×768 step does not call this function for those two
/// products. `X @ Xᵀ` is not a call here: it is [`gram_out`], one
/// `cblas_ssyrk` (until 2026-10-08 it was two `cblas_sgemm` bands here when
/// `k < 2m`, which `ssyrk` beat at every Muon shape). A
/// band that would miss the whole-call cutoff is not split: the packed
/// kernel is a different result. A product Accelerate would not take for a
/// single row stays one call.
///
/// Off macOS a whole call is `ojas_simd::sgemm_tile` over the plan's tiles,
/// which already spreads one product over the pool's threads, and its bits
/// do not depend on the tiling. Bands there would nest a second set of
/// spawned threads, each packing into its own buffer that
/// [`crate::gemm::scratch`] (and so [`muon_scratch`]) does not count, so
/// every product stays one [`crate::gemm::gemm`] call.
fn ns_gemm_out(exec: Exec<'_>, a: Mat<'_>, b: Mat<'_>, c: &mut [f32]) -> Result<(), OjasError> {
    const OP: &str = "muon_ns5_step";
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let bands = if !cfg!(target_os = "macos") || !whole_call(exec.numerics, 1, k, n) {
        1
    } else if m >= MUON_ROW_BANDS {
        MUON_ROW_BANDS
    } else {
        1
    };
    if bands <= 1 {
        return gemm_out(OP, exec, &a, &b, c);
    }
    let len = product(OP, &[m, n])?;
    if c.len() != len {
        return Err(shape(OP, "muon gemm output length differs"));
    }
    let rows = pool::ranges(m, bands);
    let lens: Vec<usize> = rows.iter().map(|band| band.len() * n).collect();
    let parts = scoped::cut(c, &lens)?;
    let jobs: Vec<_> = rows.into_iter().zip(parts).collect();
    scoped::fill_parts(exec, jobs, |_, (band, part)| {
        let start = band
            .start
            .checked_mul(a.rs)
            .ok_or_else(|| shape(OP, "muon gemm band start overflows"))?;
        if start > a.data.len() {
            return Err(shape(OP, "muon gemm band starts past its matrix"));
        }
        let a_band = Mat {
            data: &a.data[start..],
            rows: band.len(),
            cols: k,
            rs: a.rs,
            cs: a.cs,
        };
        gemm_out(OP, exec, &a_band, &b, part)
    })?;
    Ok(())
}

/// Round every value to bf16 where it is ([`Ns5Precision::Bf16`]), in
/// [`ELEM_BLOCK`] pieces on the pool. Each value is rounded alone, so the
/// cut changes no bit.
fn round_bf16_in_place(exec: Exec<'_>, values: &mut [f32]) -> Result<(), OjasError> {
    scoped::fill(exec, values, ELEM_BLOCK, |_, chunk| {
        for value in chunk {
            *value = round_f32_to_bf16(*value);
        }
        Ok(())
    })?;
    Ok(())
}

/// Five Newton-Schulz steps on `update` (`rows x cols`, row-major).
///
/// Under [`Ns5Precision::Bf16`] the iterate holds bf16 values widened to
/// f32, and every op rounds its result as torch's eager bf16 op does:
/// `X = G.bfloat16()`, the norm, `norm + eps`, the division, each GEMM
/// output (an f32 accumulation of bf16 operands, whose products are exact
/// in f32), each scalar multiple and each sum. The GEMMs and the band split
/// are the f32 path's; only the accumulation order inside a GEMM can differ
/// from torch's, and then by at most one bf16 rounding of that output.
fn newton_schulz(
    exec: Exec<'_>,
    mut update: Vec<f32>,
    rows: usize,
    cols: usize,
    precision: Ns5Precision,
) -> Result<Vec<f32>, OjasError> {
    const OP: &str = "muon_ns5_step";
    let bf16 = precision == Ns5Precision::Bf16;
    if bf16 {
        round_bf16_in_place(exec, &mut update)?;
    }
    // Tall 2048×768 only: the bytes stay row-major 2048×768. The iterate is
    // that buffer read as 768×2048 (row stride 1, column stride 768), which
    // Apple cblas accepts as `CblasTrans` with leading dimension 768. `B @ X`
    // is stored as `Xᵀ @ Bᵀ`, one `cblas_sgemm` into that same orientation,
    // so neither transpose allocates. Square 768 and tall 3072×768 still copy.
    //
    // The gate is a measured shape, not an oversight. On 2026-10-08 (M5 Pro,
    // 6 threads, `bench/results/2026-10-08-cpu-hot-paths/muon-gate.txt`)
    // this view and `one_cblas` below opened to every tall matrix were
    // slower than the transpose and six bands in all three interleaved
    // rounds: 3072×768 by 1.2-2.8×, Qwen3.5's 6144×2048 by 1.5-2.9×. A view
    // runs `B @ X` and `A @ A` as single calls on the calling thread, which
    // loses to six bands once the product is large. At 2048×768 itself the
    // view and the general path were within that run's noise (load 23-51).
    let view = rows == 2048 && cols == 768;
    let mut transposed = false;
    let (mut x, r, c) = if view {
        (update, cols, rows)
    } else if rows > cols {
        transposed = true;
        let x = transpose(OP, &update, rows, cols)?;
        drop(update);
        (x, cols, rows)
    } else {
        (update, rows, cols)
    };
    let mut sum_sq = 0.0f64;
    if view {
        // Same add order as the row-major transpose the copy used to build.
        for j in 0..cols {
            for i in 0..rows {
                let v = f64::from(x[i * cols + j]);
                sum_sq += v * v;
            }
        }
    } else {
        for value in &x {
            sum_sq += f64::from(*value) * f64::from(*value);
        }
    }
    let norm = sum_sq.sqrt() as f32;
    let denom = if bf16 {
        round_f32_to_bf16(round_f32_to_bf16(norm) + (MUON_NS_EPS as f32))
    } else {
        norm + (MUON_NS_EPS as f32)
    };
    if !(denom.is_finite() && denom != 0.0) {
        return Err(nonfinite(OP));
    }
    for value in &mut x {
        *value /= denom;
    }
    if bf16 {
        round_bf16_in_place(exec, &mut x)?;
    }
    let a = MUON_NS5_A as f32;
    let b_coef = MUON_NS5_B as f32;
    let c_coef = MUON_NS5_C as f32;
    // Tall 2048×768 only: the iterate is 768×2048. `A @ A` and `B @ X` are
    // each one `cblas_sgemm`. Square 768 and tall 3072×768 stay on
    // `ns_gemm_out`'s band loop. At every shape `X @ Xᵀ` is [`gram_out`].
    // (Only the view reaches 768×2048 here: a transposed copy of that shape
    // would be a 2048×768 input, which `view` takes first.)
    let one_cblas = view;
    // The five products and combinations of a step reuse three buffers made
    // once per call (before 2026-10-07 each step allocated five): `A` (r×r),
    // `A²` (r×r), which `B = b A + c A²` then overwrites, and `B X` (the
    // iterate's size), which `a X + B X` then overwrites before it trades
    // places with `X`. Each value is the same single expression as before,
    // so no bit changes.
    let sq = product(OP, &[r, r])?;
    let mut am = vec![0.0f32; sq];
    let mut a2 = vec![0.0f32; sq];
    let mut bx = vec![0.0f32; x.len()];
    for _ in 0..5 {
        let xm = if view {
            Mat::row_major(&x, c, r).t()
        } else {
            Mat::row_major(&x, r, c)
        };
        gram_out(OP, exec, &xm, &mut am)?;
        if bf16 {
            round_bf16_in_place(exec, &mut am)?;
        }
        let a_mat = Mat::row_major(&am, r, r);
        if one_cblas {
            gemm_out(OP, exec, &a_mat, &a_mat, &mut a2)?;
        } else {
            ns_gemm_out(exec, a_mat, a_mat, &mut a2)?;
        }
        if bf16 {
            round_bf16_in_place(exec, &mut a2)?;
            zip_into(exec, &am, &mut a2, |am, a2v| {
                let r = round_f32_to_bf16;
                r(r(b_coef * am) + r(c_coef * a2v))
            })?;
        } else {
            zip_into(exec, &am, &mut a2, |am, a2v| b_coef * am + c_coef * a2v)?;
        }
        let b_mat = &a2;
        if view {
            let sm = Mat::row_major(&x, c, r);
            let bt = Mat::row_major(b_mat, r, r).t();
            gemm_out(OP, exec, &sm, &bt, &mut bx)?;
        } else if one_cblas {
            let bm = Mat::row_major(b_mat, r, r);
            gemm_out(OP, exec, &bm, &xm, &mut bx)?;
        } else {
            let bm = Mat::row_major(b_mat, r, r);
            ns_gemm_out(exec, bm, xm, &mut bx)?;
        }
        if bf16 {
            round_bf16_in_place(exec, &mut bx)?;
            zip_into(exec, &x, &mut bx, |xv, bxv| {
                let r = round_f32_to_bf16;
                r(r(a * xv) + bxv)
            })?;
        } else {
            zip_into(exec, &x, &mut bx, |xv, bxv| a * xv + bxv)?;
        }
        if !all_finite(&bx) {
            return Err(nonfinite(OP));
        }
        std::mem::swap(&mut x, &mut bx);
    }
    drop((am, a2, bx));
    if transposed {
        x = transpose(OP, &x, r, c)?;
    }
    Ok(x)
}

/// Most f32 values [`muon_ns5`] holds at once besides its three inputs
/// (param, grad, momentum), for a `rows x cols` matrix under `exec`.
///
/// With `L = rows * cols`, `r = min(rows, cols)`, `c = max(rows, cols)`,
/// `R = r * r`, and `S(m, k, n)` the packing and tile floats
/// [`crate::gemm::scratch`] counts for an `m x k` by `k x n` product. A
/// [`zip_map`] of `n` values holds its task parts and the joined result,
/// `2n`. `buf` (`L`) lives from the first step to the end:
/// - `buf`: parts and result, `2L`;
/// - Nesterov `update`: `buf`, parts, result, `3L` (else `buf.to_vec()`, `2L`);
/// - a tall matrix is transposed into the iterate before `update` drops, `3L`;
/// - each Newton-Schulz step, with `buf`, the iterate `x` and the three
///   buffers the steps reuse (`B x`, `L`; `A` and `A²`, `2R`) live, `3L + 2R`,
///   plus a product's packing: `A = x xᵀ` `S(r, c, r)`, `A²` `S(r, r, r)`,
///   `B x` `S(r, r, c)`; `B` and `a x + B x` are written over `A²` and `B x`;
/// - the tall result transposed back, `3L`;
/// - the new parameter: `buf`, the orthogonalized update, parts, result, `4L`.
///
/// The largest of these is the bound.
pub(crate) fn muon_scratch(
    op: &'static str,
    exec: Exec<'_>,
    rows: usize,
    cols: usize,
) -> Result<usize, OjasError> {
    let (r, c) = (rows.min(cols), rows.max(cols));
    let l = product(op, &[rows, cols])?;
    let sq = product(op, &[r, r])?;
    let times = |n: usize, k: usize| n.checked_mul(k).ok_or_else(|| scratch_overflow(op));
    let sum = |terms: &[usize]| {
        terms
            .iter()
            .try_fold(0usize, |acc, &t| acc.checked_add(t))
            .ok_or_else(|| scratch_overflow(op))
    };
    let (l3, l4) = (times(l, 3)?, times(l, 4)?);
    let phases = [
        l3,
        sum(&[l3, times(sq, 2)?, gemm_scratch(op, exec, r, c, r)?])?,
        sum(&[l3, times(sq, 2)?, gemm_scratch(op, exec, r, r, r)?])?,
        sum(&[l3, times(sq, 2)?, gemm_scratch(op, exec, r, r, c)?])?,
        l4,
    ];
    Ok(phases.into_iter().max().unwrap_or(0))
}

fn scratch_overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "optimizer scratch length overflows".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ojas_core::{Ns5Precision, Numerics};

    use super::newton_schulz;
    use crate::gemm::whole_calls;
    use crate::pool::{Exec, Pool};

    /// A square 768 Fast Newton-Schulz run on one thread (every band inline,
    /// so this thread sees every whole call) makes 13 Accelerate calls a
    /// step: `X @ Xᵀ` as one `cblas_ssyrk`, then six bands each of `A @ A`
    /// and `B @ X`. Before 2026-10-08 `X @ Xᵀ` was two `cblas_sgemm` bands
    /// at this shape (`k < 2m`), 14 a step and 70 a run.
    #[cfg(target_os = "macos")]
    #[test]
    fn square_768_gram_product_is_one_accelerate_call_a_step() {
        let pool = Arc::new(Pool::new(1).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let n = 768;
        let mut state = 7u64;
        let update: Vec<f32> = (0..n * n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        let before = whole_calls();
        let out = newton_schulz(exec, update, n, n, Ns5Precision::F32).unwrap();
        let after = whole_calls();
        assert_eq!(out.len(), n * n);
        assert_eq!((after.0 - before.0, after.1 - before.1), (5 * 13, 0));
    }
}
