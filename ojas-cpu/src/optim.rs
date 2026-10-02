//! Torch `clip_grad_norm_` and single-tensor AdamW, plus Muon NS5 in f32.
//!
//! Clip coefficient is `max_norm / (total_norm + CLIP_GRAD_NORM_EPS)` with
//! [`ojas_core::CLIP_GRAD_NORM_EPS`] = `1e-6`, then `min(1, coefficient)`.
//! The epsilon stays in the denominator; the scale is not `max_norm / total_norm`.
//! When the clamped coefficient is 1, the gradient bits are left unchanged.
//!
//! AdamW (torch decoupled, scalars in f64):
//! 1. If weight decay is not exactly 0, `p *= 1 - lr * weight_decay` first.
//!    Weight decay 0 skips that multiply.
//! 2. Moments. The first moment uses torch `lerp`, which switches at weight 0.5.
//! 3. `denom = sqrt(v) / sqrt(1 - beta2^step) + eps`, with `eps` outside the square root.
//! 4. `p += -lr / (1 - beta1^step) * m / denom`.
//!
//! `beta^step` is computed in f64 by [`ojas_core::pow_u64`] (binary exponentiation),
//! not `f32` pow, inside [`ojas_core::check_adamw`], which also advances the step.
//!
//! Newton-Schulz on CPU stores the iterate in f32. Coefficients are the f64
//! literals 3.4445, -4.7750, 2.0315. Frobenius epsilon is 1e-7, added to the
//! norm, not inside the square root. The step scalars `1 - lr * wd` and
//! `-lr * max(1, rows/cols)^0.5` are formed in f64 and rounded to f32 once,
//! as nanolab's Python floats are.
//!
//! Configs outside torch's accepted ranges (negative `lr`, `weight_decay`, or
//! Muon `momentum`) are [`OjasError::OutOfRange`], matching the Metal path.

use ojas_core::{
    check_adamw, require_ns5, AdamWConfig, Budget, MuonNs5Config, Numerics, OjasError, Scratch,
    Tensor, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS,
};

use std::sync::Arc;

use crate::gemm::{gemm, scratch as gemm_scratch, Mat};
use crate::linalg::transpose;
use crate::pool::scoped;
use crate::pool::Exec;
use crate::validate::{all_finite, f32_words, nonfinite, nonfinite_first, product, shape};

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

/// Global norm of gradients given as native-endian words read in place.
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
    parts: &[&[[u8; 4]]],
) -> Result<f32, OjasError> {
    let value = |word: &[u8; 4]| f32::from_ne_bytes(*word);
    match exec.numerics {
        Numerics::Exact => norm_of(
            op,
            parts.iter().fold(0.0, |acc, part| {
                sum_sq_ascending(acc, part.iter().map(value))
            }),
        ),
        Numerics::Fast => {
            let blocks: Vec<&[[u8; 4]]> = parts.iter().flat_map(|p| p.chunks(NORM_BLOCK)).collect();
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

/// `grads[i] *= scale` in place for every gradient, all or nothing.
///
/// One buffer as long as the largest gradient is charged first; then every
/// gradient is proven writable ([`Tensor::ensure_writable_f32`]) before the
/// first one changes. Each gradient's scaled values are computed into the
/// buffer from its bytes, in parallel, and written with
/// [`Tensor::write_f32`]. `scale` is in `[0, 1)` ([`ojas_core::clip_scale`])
/// and the values are finite, so every product is finite and no write after
/// the claim can be refused.
pub(crate) fn scale_in_place(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    grads: &mut [Tensor],
    scale: f32,
) -> Result<(), OjasError> {
    let mut largest = 0usize;
    for grad in grads.iter() {
        largest = largest.max(grad.num_elements()?);
    }
    let mut buf = Scratch::<f32>::try_alloc(largest, budget)?;
    for grad in grads.iter_mut() {
        let len = grad.num_elements()?;
        grad.ensure_writable_f32(len)?;
    }
    for grad in grads.iter_mut() {
        let len = grad.num_elements()?;
        let dst = buf
            .as_mut_slice()
            .get_mut(..len)
            .ok_or_else(|| shape(op, "gradient longer than the scale buffer"))?;
        let src = f32_words(op, grad)?;
        scoped::fill(exec, dst, ELEM_BLOCK, |b, chunk| {
            let base = b * ELEM_BLOCK;
            for (d, word) in chunk.iter_mut().zip(&src[base..]) {
                *d = f32::from_ne_bytes(*word) * scale;
            }
            Ok(())
        })?;
        grad.write_f32(&buf.as_slice()[..len])?;
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
}

/// Block `b` (`n` values) of the [`ELEM_BLOCK`] cut of `words`.
fn elem_block<'w>(
    op: &'static str,
    words: &'w [[u8; 4]],
    b: usize,
    n: usize,
) -> Result<&'w [[u8; 4]], OjasError> {
    let start = b * ELEM_BLOCK;
    words
        .get(start..start + n)
        .ok_or_else(|| shape(op, "element block exceeds a tensor"))
}

/// Native-endian f32 words as values.
#[inline(always)]
fn val(word: &[u8; 4]) -> f32 {
    f32::from_ne_bytes(*word)
}

/// Pass 1 of [`adamw_in_place`] over one block: the new parameters into
/// `out`, and whether every element's step was finite.
fn adam_params<const DECAYS: bool, const LOW: bool>(
    c: &AdamCoeffs,
    out: &mut [f32],
    p: &[[u8; 4]],
    g: &[[u8; 4]],
    m: &[[u8; 4]],
    v: &[[u8; 4]],
) -> bool {
    let mut ok = true;
    for ((((slot, p), g), m), v) in out.iter_mut().zip(p).zip(g).zip(m).zip(v) {
        let (np, _, _, finite) = c.step::<DECAYS, LOW>(val(p), val(g), val(m), val(v));
        *slot = np;
        ok &= finite;
    }
    ok
}

fn adam_moment1<const LOW: bool>(c: &AdamCoeffs, out: &mut [f32], g: &[[u8; 4]], m: &[[u8; 4]]) {
    for ((slot, g), m) in out.iter_mut().zip(g).zip(m) {
        *slot = c.moment1::<LOW>(f64::from(val(g)), f64::from(val(m))) as f32;
    }
}

fn adam_moment2(c: &AdamCoeffs, out: &mut [f32], g: &[[u8; 4]], v: &[[u8; 4]]) {
    for ((slot, g), v) in out.iter_mut().zip(g).zip(v) {
        *slot = c.moment2(f64::from(val(g)), f64::from(val(v))) as f32;
    }
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

/// One AdamW step written into `param`, `moment1` and `moment2` in place.
///
/// The tensors are read in place; the caller has checked their layouts
/// ([`crate::validate::f32_layouts`]). One buffer of `len` values is
/// charged; then all three targets are proven writable. The first parallel
/// pass computes every element's whole update into the buffer as the new
/// parameter and refuses a non-finite result before anything is written. A
/// non-finite input always gives a non-finite result (a NaN or infinite
/// gradient or moment reaches `m` or `v`; a parameter reaches the stored
/// value), so that pass is also the inputs' NaN scan; a refusal before it
/// goes through [`nonfinite_first`], which keeps the scan-first order.
/// The parameter is written, then the buffer is refilled with the new first
/// moment and written, then with the second. The moments depend only on the
/// old moments and the gradient, so the parameter's write does not change
/// them, and recomputing them repeats the first pass's values exactly; those
/// were finite, so no write after the first can be refused.
///
/// Each element is the reference arithmetic in f64, whatever the numerics
/// contract and thread count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn adamw_in_place(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    param: &mut Tensor,
    grad: &Tensor,
    moment1: &mut Tensor,
    moment2: &mut Tensor,
    coeffs: AdamCoeffs,
) -> Result<(), OjasError> {
    let len = param.num_elements()?;
    let claimed = Scratch::<f32>::try_alloc(len, budget).and_then(|buf| {
        param.ensure_writable_f32(len)?;
        moment1.ensure_writable_f32(len)?;
        moment2.ensure_writable_f32(len)?;
        Ok(buf)
    });
    let mut buf = match claimed {
        Ok(buf) => buf,
        Err(err) => {
            return Err(nonfinite_first(
                op,
                exec,
                &[param, grad, moment1, moment2],
                err,
            ))
        }
    };
    {
        let (p, g, m, v) = (
            f32_words(op, param)?,
            f32_words(op, grad)?,
            f32_words(op, moment1)?,
            f32_words(op, moment2)?,
        );
        let flags = (coeffs.decays, coeffs.low());
        scoped::fill(exec, buf.as_mut_slice(), ELEM_BLOCK, |b, out| {
            let n = out.len();
            let (p, g, m, v) = (
                elem_block(op, p, b, n)?,
                elem_block(op, g, b, n)?,
                elem_block(op, m, b, n)?,
                elem_block(op, v, b, n)?,
            );
            let finite = match flags {
                (true, true) => adam_params::<true, true>(&coeffs, out, p, g, m, v),
                (true, false) => adam_params::<true, false>(&coeffs, out, p, g, m, v),
                (false, true) => adam_params::<false, true>(&coeffs, out, p, g, m, v),
                (false, false) => adam_params::<false, false>(&coeffs, out, p, g, m, v),
            };
            if finite {
                Ok(())
            } else {
                Err(nonfinite(op))
            }
        })?;
    }
    param.write_f32(buf.as_slice())?;
    {
        let (g, m) = (f32_words(op, grad)?, f32_words(op, moment1)?);
        let low = coeffs.low();
        scoped::fill(exec, buf.as_mut_slice(), ELEM_BLOCK, |b, out| {
            let n = out.len();
            let (g, m) = (elem_block(op, g, b, n)?, elem_block(op, m, b, n)?);
            if low {
                adam_moment1::<true>(&coeffs, out, g, m);
            } else {
                adam_moment1::<false>(&coeffs, out, g, m);
            }
            Ok(())
        })?;
    }
    moment1.write_f32(buf.as_slice())?;
    {
        let (g, v) = (f32_words(op, grad)?, f32_words(op, moment2)?);
        scoped::fill(exec, buf.as_mut_slice(), ELEM_BLOCK, |b, out| {
            let n = out.len();
            adam_moment2(
                &coeffs,
                out,
                elem_block(op, g, b, n)?,
                elem_block(op, v, b, n)?,
            );
            Ok(())
        })?;
    }
    moment2.write_f32(buf.as_slice())?;
    Ok(())
}

/// `f(a[i], b[i])` for every `i`, in element chunks on the pool.
fn zip_map<F>(
    exec: Exec<'_>,
    a: Arc<Vec<f32>>,
    b: Arc<Vec<f32>>,
    f: F,
) -> Result<Vec<f32>, OjasError>
where
    F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
{
    let len = a.len().min(b.len());
    exec.rows(len, 1, move |range| {
        Ok(range.map(|i| f(a[i], b[i])).collect())
    })
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
    param: Vec<f32>,
    grad: Vec<f32>,
    momentum: Vec<f32>,
    rows: usize,
    cols: usize,
    config: MuonNs5Config,
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    const OP: &str = "muon_ns5_step";
    check_muon(config)?;
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "muon matrix length overflows".to_string(),
        })?;
    if param.len() != len || grad.len() != len || momentum.len() != len {
        return Err(shape(OP, "muon tensors differ in length"));
    }
    let mom = config.momentum as f32;
    let grad = Arc::new(grad);
    let buf = Arc::new(zip_map(
        exec,
        Arc::new(momentum),
        Arc::clone(&grad),
        move |m, g| mom * m + g,
    )?);
    if !all_finite(&buf) {
        return Err(nonfinite(OP));
    }
    let update: Vec<f32> = if config.nesterov {
        zip_map(exec, grad, Arc::clone(&buf), move |g, b| g + mom * b)?
    } else {
        buf.to_vec()
    };
    let ortho = newton_schulz(exec, update, rows, cols)?;
    if cols == 0 {
        return Err(shape(OP, "empty tensor"));
    }
    let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
    let alpha = (-config.lr * scale) as f32;
    let all_zero = ortho.iter().all(|v| *v == 0.0);
    let new_p = if config.weight_decay == 0.0 && all_zero {
        param
    } else if config.weight_decay != 0.0 {
        let decay = (1.0 - config.lr * config.weight_decay) as f32;
        zip_map(exec, Arc::new(param), Arc::new(ortho), move |p, o| {
            p * decay + alpha * o
        })?
    } else {
        zip_map(exec, Arc::new(param), Arc::new(ortho), move |p, o| {
            p + alpha * o
        })?
    };
    if !all_finite(&new_p) {
        return Err(nonfinite(OP));
    }
    let buf = Arc::try_unwrap(buf).unwrap_or_else(|shared| shared.to_vec());
    Ok((new_p, buf))
}

fn newton_schulz(
    exec: Exec<'_>,
    update: Vec<f32>,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, OjasError> {
    const OP: &str = "muon_ns5_step";
    let mut transposed = false;
    let (mut x, r, c) = if rows > cols {
        transposed = true;
        let x = transpose(OP, &update, rows, cols)?;
        drop(update);
        (x, cols, rows)
    } else {
        (update, rows, cols)
    };
    let mut sum_sq = 0.0f64;
    for value in &x {
        sum_sq += f64::from(*value) * f64::from(*value);
    }
    let norm = sum_sq.sqrt() as f32;
    let denom = norm + (MUON_NS_EPS as f32);
    if !(denom.is_finite() && denom != 0.0) {
        return Err(nonfinite(OP));
    }
    for value in &mut x {
        *value /= denom;
    }
    let a = MUON_NS5_A as f32;
    let b_coef = MUON_NS5_B as f32;
    let c_coef = MUON_NS5_C as f32;
    for _ in 0..5 {
        let xm = Mat::row_major(Arc::new(x), r, c);
        let a_mat = Mat::row_major(Arc::new(gemm(OP, exec, &xm, &xm.t())?), r, r);
        let a2 = gemm(OP, exec, &a_mat, &a_mat)?;
        let b_mat = zip_map(
            exec,
            Arc::clone(&a_mat.data),
            Arc::new(a2),
            move |am, a2v| b_coef * am + c_coef * a2v,
        )?;
        let bx = gemm(OP, exec, &Mat::row_major(Arc::new(b_mat), r, r), &xm)?;
        let next = zip_map(exec, Arc::clone(&xm.data), Arc::new(bx), move |xv, bxv| {
            a * xv + bxv
        })?;
        if !all_finite(&next) {
            return Err(nonfinite(OP));
        }
        x = next;
    }
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
/// - each Newton-Schulz step, with `buf` and the iterate `x` (`2L`) live:
///   `A = x xᵀ`: `+ R + S(r, c, r)`;
///   `A²`: `+ 2R + S(r, r, r)`;
///   `B = b A + c A²`: `A`, `A²`, parts, result, `+ 4R`;
///   `B x`: `A`, `B`, result, `+ 2R + L + S(r, r, c)`;
///   `a x + B x`: `A`, `B x`, parts, result, `+ R + 3L`;
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
    let (l2, l3, l4) = (times(l, 2)?, times(l, 3)?, times(l, 4)?);
    let phases = [
        l3,
        sum(&[l2, sq, gemm_scratch(op, exec, r, c, r)?])?,
        sum(&[l2, times(sq, 2)?, gemm_scratch(op, exec, r, r, r)?])?,
        sum(&[l2, times(sq, 4)?])?,
        sum(&[l3, times(sq, 2)?, gemm_scratch(op, exec, r, r, c)?])?,
        sum(&[l2, sq, l3])?,
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
