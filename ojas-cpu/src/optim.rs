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
//! `beta^step` is computed in f64 by binary exponentiation, not `f32` pow.
//! The stored step is incremented with [`ojas_core::next_step`].
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
    next_step, require_ns5, AdamWConfig, MuonNs5Config, OjasError, CLIP_GRAD_NORM_EPS, MUON_NS5_A,
    MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS,
};

use crate::linalg::{matmul, transpose};
use crate::validate::{get, nonfinite, shape};

pub(crate) fn clip_scale(max_norm: f32, total_norm: f32) -> Result<f32, OjasError> {
    if !max_norm.is_finite() {
        return Err(nonfinite("clip_grad_norm"));
    }
    if max_norm < 0.0 {
        return Err(OjasError::OutOfRange {
            op: "clip_grad_norm",
            detail: format!("max_norm {max_norm} is negative"),
        });
    }
    if !total_norm.is_finite() {
        return Err(nonfinite("clip_grad_norm"));
    }
    let coef = max_norm / (total_norm + CLIP_GRAD_NORM_EPS);
    if !coef.is_finite() {
        return Err(nonfinite("clip_grad_norm"));
    }
    Ok(coef.min(1.0))
}

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

pub(crate) fn adamw(
    param: &[f32],
    grad: &[f32],
    moment1: &[f32],
    moment2: &[f32],
    step_before: u64,
    config: AdamWConfig,
) -> Result<AdamState, OjasError> {
    const OP: &str = "adamw_step";
    let step = next_step(step_before)?;
    check_adam(config)?;
    if param.len() != grad.len() || param.len() != moment1.len() || param.len() != moment2.len() {
        return Err(shape(OP, "adamw tensors differ in length"));
    }
    let bc1 = 1.0 - pow_u64(config.beta1, step);
    let bc2 = 1.0 - pow_u64(config.beta2, step);
    if !(bc1.is_finite() && bc2.is_finite() && bc1 > 0.0 && bc2 >= 0.0) {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: "adamw bias correction is not a positive finite value".to_string(),
        });
    }
    let step_size = config.lr / bc1;
    let one_minus_b1 = 1.0 - config.beta1;
    let one_minus_b2 = 1.0 - config.beta2;
    let decay = 1.0 - config.lr * config.weight_decay;
    let mut new_p = Vec::with_capacity(param.len());
    let mut new_m = Vec::with_capacity(param.len());
    let mut new_v = Vec::with_capacity(param.len());
    for i in 0..param.len() {
        let g = f64::from(get(OP, grad, i)?);
        let mut p = f64::from(get(OP, param, i)?);
        let mut m = f64::from(get(OP, moment1, i)?);
        let mut v = f64::from(get(OP, moment2, i)?);
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
}

fn check_adam(config: AdamWConfig) -> Result<(), OjasError> {
    const OP: &str = "adamw_step";
    let scalars = [
        config.lr,
        config.beta1,
        config.beta2,
        config.eps,
        config.weight_decay,
    ];
    if scalars.iter().any(|value| !value.is_finite()) {
        return Err(nonfinite(OP));
    }
    if !(config.beta1 >= 0.0 && config.beta1 < 1.0 && config.beta2 >= 0.0 && config.beta2 < 1.0) {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: "adamw beta must be in [0, 1)".to_string(),
        });
    }
    if config.eps <= 0.0 {
        return Err(OjasError::OutOfRange {
            op: OP,
            detail: "adamw eps must be > 0".to_string(),
        });
    }
    non_negative(
        OP,
        &[("lr", config.lr), ("weight_decay", config.weight_decay)],
    )
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

/// f64 binary exponentiation. A huge step underflows to 0 instead of overflowing an f32 pow.
pub(crate) fn pow_u64(base: f64, mut exp: u64) -> f64 {
    let mut result = 1.0f64;
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result *= b;
        }
        exp >>= 1;
        if exp > 0 {
            b *= b;
        }
    }
    result
}

pub(crate) fn muon_ns5(
    param: &[f32],
    grad: &[f32],
    momentum: &[f32],
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
    let buf: Vec<f32> = momentum
        .iter()
        .zip(grad)
        .map(|(m, g)| mom * m + g)
        .collect();
    if buf.iter().any(|v| !v.is_finite()) {
        return Err(nonfinite(OP));
    }
    let update: Vec<f32> = if config.nesterov {
        grad.iter().zip(&buf).map(|(g, b)| g + mom * b).collect()
    } else {
        buf.clone()
    };
    let ortho = newton_schulz(&update, rows, cols)?;
    if cols == 0 {
        return Err(shape(OP, "empty tensor"));
    }
    let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
    let alpha = (-config.lr * scale) as f32;
    let all_zero = ortho.iter().all(|v| *v == 0.0);
    let mut new_p = param.to_vec();
    if !(config.weight_decay == 0.0 && all_zero) {
        if config.weight_decay != 0.0 {
            let decay = (1.0 - config.lr * config.weight_decay) as f32;
            for p in &mut new_p {
                *p *= decay;
            }
        }
        for (p, o) in new_p.iter_mut().zip(&ortho) {
            *p += alpha * o;
        }
    }
    if new_p.iter().any(|v| !v.is_finite()) {
        return Err(nonfinite(OP));
    }
    Ok((new_p, buf))
}

fn newton_schulz(update: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, OjasError> {
    const OP: &str = "muon_ns5_step";
    let mut transposed = false;
    let (mut x, r, c) = if rows > cols {
        transposed = true;
        (transpose(OP, update, rows, cols)?, cols, rows)
    } else {
        (update.to_vec(), rows, cols)
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
        let xt = transpose(OP, &x, r, c)?;
        let a_mat = matmul(OP, &x, &xt, r, c, r)?;
        let a2 = matmul(OP, &a_mat, &a_mat, r, r, r)?;
        let mut b_mat = vec![0.0f32; r * r];
        for i in 0..b_mat.len() {
            b_mat[i] = b_coef * a_mat[i] + c_coef * a2[i];
        }
        let bx = matmul(OP, &b_mat, &x, r, r, c)?;
        for i in 0..x.len() {
            x[i] = a * x[i] + bx[i];
            if !x[i].is_finite() {
                return Err(nonfinite(OP));
            }
        }
    }
    if transposed {
        x = transpose(OP, &x, r, c)?;
    }
    Ok(x)
}
