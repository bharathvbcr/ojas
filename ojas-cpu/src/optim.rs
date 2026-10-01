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
    check_adamw, require_ns5, AdamWConfig, MuonNs5Config, OjasError, MUON_NS5_A, MUON_NS5_B,
    MUON_NS5_C, MUON_NS_EPS,
};

use std::sync::Arc;

use crate::gemm::{gemm, Mat};
use crate::linalg::transpose;
use crate::pool::Exec;
use crate::validate::{all_finite, nonfinite, shape};

/// Sum of squares in f64: an f32 sum overflows near 1.8e19 per element while
/// the norm itself is still a finite f32.
pub(crate) fn total_norm(op: &'static str, parts: &[Vec<f32>]) -> Result<f32, OjasError> {
    let mut sum_sq = 0.0f64;
    for part in parts {
        for &value in part {
            sum_sq += f64::from(value) * f64::from(value);
        }
    }
    let norm = sum_sq.sqrt() as f32;
    if !norm.is_finite() {
        return Err(nonfinite(op));
    }
    Ok(norm)
}

/// `(param, moment1, moment2)` after one step.
type AdamState = (Vec<f32>, Vec<f32>, Vec<f32>);

/// Elements are independent, so they run in chunks on the pool; each
/// element's arithmetic is unchanged.
pub(crate) fn adamw(
    exec: Exec<'_>,
    param: Vec<f32>,
    grad: Vec<f32>,
    moment1: Vec<f32>,
    moment2: Vec<f32>,
    step_before: u64,
    config: AdamWConfig,
) -> Result<AdamState, OjasError> {
    const OP: &str = "adamw_step";
    // Config, step counter and both bias corrections, checked by the one
    // owner every backend shares, so the f64 bits of `1 - beta^step` match.
    let (_, bc1, bc2) = check_adamw(config, step_before)?;
    if param.len() != grad.len() || param.len() != moment1.len() || param.len() != moment2.len() {
        return Err(shape(OP, "adamw tensors differ in length"));
    }
    let step_size = config.lr / bc1;
    let one_minus_b1 = 1.0 - config.beta1;
    let one_minus_b2 = 1.0 - config.beta2;
    let decay = 1.0 - config.lr * config.weight_decay;
    let len = param.len();
    let inputs = Arc::new((param, grad, moment1, moment2));
    let parts = exec.chunks(len, ADAM_MIN_CHUNK, move |range| {
        let (param, grad, moment1, moment2) = &*inputs;
        let mut new_p = Vec::with_capacity(range.len());
        let mut new_m = Vec::with_capacity(range.len());
        let mut new_v = Vec::with_capacity(range.len());
        for i in range {
            let g = f64::from(grad[i]);
            let mut p = f64::from(param[i]);
            let mut m = f64::from(moment1[i]);
            let mut v = f64::from(moment2[i]);
            if config.weight_decay != 0.0 {
                p *= decay;
            }
            // Torch lerp(self, other, weight) switches formula at weight 0.5.
            // Here weight is (1 - beta1) and other is the gradient.
            if one_minus_b1 < 0.5 {
                m += one_minus_b1 * (g - m);
            } else {
                m = g - (g - m) * config.beta1;
            }
            v = config.beta2 * v + one_minus_b2 * g * g;
            let denom = v.sqrt() / bc2.sqrt() + config.eps;
            let delta = (-step_size) * m / denom;
            if !(m.is_finite() && v.is_finite() && denom.is_finite() && delta.is_finite()) {
                return Err(nonfinite(OP));
            }
            if config.weight_decay == 0.0 && delta == 0.0 {
                new_p.push(param[i]);
            } else {
                p += delta;
                if !p.is_finite() {
                    return Err(nonfinite(OP));
                }
                let stored = p as f32;
                if !stored.is_finite() {
                    return Err(nonfinite(OP));
                }
                new_p.push(stored);
            }
            let stored_m = m as f32;
            let stored_v = v as f32;
            if !(stored_m.is_finite() && stored_v.is_finite()) {
                return Err(nonfinite(OP));
            }
            new_m.push(stored_m);
            new_v.push(stored_v);
        }
        Ok((new_p, new_m, new_v))
    })?;
    let mut chunks = Vec::with_capacity(parts.len());
    for part in parts {
        chunks.push(part?);
    }
    if chunks.len() == 1 {
        if let Some(only) = chunks.pop() {
            return Ok(only);
        }
    }
    let (mut ps, mut ms, mut vs) = (
        Vec::with_capacity(chunks.len()),
        Vec::with_capacity(chunks.len()),
        Vec::with_capacity(chunks.len()),
    );
    for (p, m, v) in chunks {
        ps.push(p);
        ms.push(m);
        vs.push(v);
    }
    Ok((join(ps, len), join(ms, len), join(vs, len)))
}

/// `chunks` in order as one vector of `len` values, each chunk freed as soon
/// as it is copied.
fn join(chunks: Vec<Vec<f32>>, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for chunk in chunks {
        out.extend_from_slice(&chunk);
    }
    out
}

/// AdamW elements per pool task; each costs a few f64 divides and a sqrt.
const ADAM_MIN_CHUNK: usize = 1 << 14;

/// Most f32 values [`adamw`] holds at once besides its four inputs.
///
/// The tasks' results are three vectors of `len` values in all. One chunk
/// comes back as they are. Several are joined one result at a time
/// ([`join`]): the first joined vector is allocated while every chunk is
/// still live, `3 len + len`; each later one replaces the chunks it
/// consumed. So `4 len`.
pub(crate) fn adamw_scratch(op: &'static str, len: usize) -> Result<usize, OjasError> {
    len.checked_mul(4).ok_or_else(|| scratch_overflow(op))
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
    )?;
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
