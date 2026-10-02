//! AdamW on the host, for parameters that live outside the step provider.
//!
//! Lappi's span head is two `[H, H]` projections and two `[H]` vectors in f32 on the host
//! (`AUDIT/ojas-training-2026-10-01/fable-advice.md` Q1 row 8); tessl has no head, so its
//! optimizer step cannot be the provider's. It is the SAME optimizer as the tower's: one
//! `torch.optim.AdamW` over both in the PyTorch trainer (`python/qd_train/optim.py:402-405`),
//! the same betas, eps, weight decay, learning rate, step count and clip coefficient. This is
//! that update for one f32 tensor, and [`crate::mock::ToyProvider`] uses it for its parameters
//! too.
//!
//! **What is reproduced, line by line** (`torch/optim/adam.py`, `_single_tensor_adam`, torch
//! 2.12.1 as installed in `/Users/bharath/.venvs/ml`, `decoupled_weight_decay=True`, `amsgrad`,
//! `maximize`, `capturable` and `differentiable` off), each Python scalar entering the f32
//! kernel as f32, which is what ATen does with a wrapped Python number:
//!
//! ```text
//! param.mul_(1 - lr * weight_decay)                      # skipped when weight_decay == 0
//! exp_avg.lerp_(grad, 1 - beta1)                         # weight < 0.5: fma(w, g - m, m)
//! exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
//! bias_correction1 = 1 - beta1 ** step                   # f64
//! bias_correction2 = 1 - beta2 ** step                   # f64
//! step_size = lr / bias_correction1                      # f64
//! bias_correction2_sqrt = bias_correction2 ** 0.5        # f64
//! denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
//! param.addcdiv_(exp_avg, denom, value=-step_size)       # p + (-step_size * m) / denom
//! ```
//!
//! The gradient enters already multiplied by the clip coefficient in f32, as
//! `clip_grad_norm_`'s in-place `mul_` leaves it. ATen's vectorised and scalar-tail kernels can
//! differ in whether a multiply-add is fused, so bit equality with torch is not claimed;
//! `tests/adamw_oracle.rs` holds this to tessl's AdamW bound (1e-6 of each tensor's max) against
//! values `tools/qd_train_oracle_trainer.py` dumped from torch itself, and reports how many
//! elements are bit-identical.

use crate::recipe::EntryHyper;
use crate::step::{AdamWHyper, StepError};

/// The two moments of one tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Moments {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Moments {
    pub fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }
}

/// One entry's optimizer state: its moments and its own AdamW step count. torch keeps
/// `state["step"]` per parameter and advances it only on a step where that parameter has a
/// gradient, so an entry's count can lag the model's.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryState {
    pub moments: Moments,
    pub steps: u64,
}

impl EntryState {
    pub fn zeros(n: usize) -> Self {
        Self {
            moments: Moments::zeros(n),
            steps: 0,
        }
    }
}

/// One entry's AdamW step, as `torch.optim.AdamW` takes it over one parameter.
///
/// - `grad` `None` is torch's `p.grad is None` (after `zero_grad(set_to_none=True)`, a parameter
///   no loss reached this step). The entry is skipped: no decay, no moment update, and no step
///   count. Returns `false`.
/// - Otherwise the entry's own count advances and is the bias-correction step. The entry's rate
///   is `lr * row.lr_scale`, and its decoupled decay is `1 - (lr * row.lr_scale) *
///   row.weight_decay`: torch's group `lr` is already scaled, and `param.mul_(1 - lr *
///   weight_decay)` reads it. Eps and betas are the row's own. `grad_scale` is the clip
///   coefficient. Returns `true`.
pub fn adamw_entry_step(
    param: &mut [f32],
    grad: Option<&[f32]>,
    state: &mut EntryState,
    row: &EntryHyper,
    lr: f64,
    grad_scale: f64,
) -> Result<bool, StepError> {
    let Some(grad) = grad else {
        return Ok(false);
    };
    let t = state
        .steps
        .checked_add(1)
        .ok_or_else(|| StepError::Contract(format!("{}: the step count overflows", row.name)))?;
    let hyper = AdamWHyper {
        lr,
        beta1: row.beta1,
        beta2: row.beta2,
        eps: row.eps,
        grad_scale,
    };
    adamw_update(param, grad, &mut state.moments, &hyper, lr * row.lr_scale, row.weight_decay, t)?;
    state.steps = t;
    Ok(true)
}

/// One AdamW update of `param` in place. `lr` is this tensor's learning rate (the schedule's
/// rate times its group's scale); `hyper.lr` is not read. `step` is the 1-based count.
pub fn adamw_update(
    param: &mut [f32],
    grad: &[f32],
    moments: &mut Moments,
    hyper: &AdamWHyper,
    lr: f64,
    weight_decay: f64,
    step: u64,
) -> Result<(), StepError> {
    hyper.validate()?;
    if !(lr.is_finite() && lr > 0.0) {
        return Err(StepError::Contract(format!("lr {lr} must be finite and > 0")));
    }
    if !(weight_decay.is_finite() && weight_decay >= 0.0) {
        return Err(StepError::Contract(format!("weight decay {weight_decay} must be finite and >= 0")));
    }
    if step == 0 {
        return Err(StepError::Contract("step 0 zeroes the bias correction; steps count from 1".into()));
    }
    let n = param.len();
    if grad.len() != n || moments.m.len() != n || moments.v.len() != n {
        return Err(StepError::Contract(format!(
            "param {n}, grad {}, moments {}/{}: one tensor, four lengths",
            grad.len(),
            moments.m.len(),
            moments.v.len()
        )));
    }
    // torch keeps `step` as a float tensor and `_get_value` hands Python a float, so the bias
    // corrections are C `pow(beta, (double)step)` -- `powf`, not `powi`'s repeated products.
    if step > (1u64 << 53) {
        return Err(StepError::Contract(format!("step {step} is not exact in f64")));
    }
    let t = step as f64;
    let scale = hyper.grad_scale as f32;
    let decay = (1.0 - lr * weight_decay) as f32;
    let w1 = (1.0 - hyper.beta1) as f32;
    let b2 = hyper.beta2 as f32;
    let w2 = (1.0 - hyper.beta2) as f32;
    let bc1 = 1.0 - hyper.beta1.powf(t);
    let bc2 = 1.0 - hyper.beta2.powf(t);
    let step_size = lr / bc1;
    let bc2_sqrt = bc2.powf(0.5) as f32;
    let eps = hyper.eps as f32;
    let neg_step = (-step_size) as f32;
    for i in 0..n {
        let g = grad[i] * scale;
        let mut p = param[i];
        if weight_decay != 0.0 {
            p *= decay;
        }
        let m0 = moments.m[i];
        // torch's lerp for a weight below 0.5: start + weight * (end - start), fused.
        let m = if w1.abs() < 0.5 {
            w1.mul_add(g - m0, m0)
        } else {
            (w1 - 1.0).mul_add(g - m0, g)
        };
        let v = moments.v[i] * b2 + w2 * g * g;
        let denom = v.sqrt() / bc2_sqrt + eps;
        p += neg_step * m / denom;
        moments.m[i] = m;
        moments.v[i] = v;
        param[i] = p;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyper() -> AdamWHyper {
        AdamWHyper {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_scale: 1.0,
        }
    }

    #[test]
    fn the_first_step_moves_each_weight_by_about_lr_against_its_gradient() {
        // At t = 1, m = 0.1 g and v = 0.001 g^2, so m / (sqrt(v / 0.001) + eps) = 0.1 sign(g),
        // and step_size = lr / 0.1: the update is -lr * sign(g) up to eps.
        let mut p = vec![1.0f32, -2.0, 0.5];
        let g = [0.3f32, -4.0, 1e-3];
        let mut mo = Moments::zeros(3);
        adamw_update(&mut p, &g, &mut mo, &hyper(), 1e-3, 0.0, 1).unwrap();
        for (got, (p0, gi)) in p.iter().zip([1.0f32, -2.0, 0.5].iter().zip(g)) {
            let want = p0 - 1e-3 * gi.signum();
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn weight_decay_is_decoupled_and_scaled_by_the_tensors_lr() {
        let mut p = vec![2.0f32];
        let mut mo = Moments::zeros(1);
        // A zero gradient: only the decay moves the weight, by the factor (1 - lr * wd).
        adamw_update(&mut p, &[0.0], &mut mo, &hyper(), 0.5, 0.01, 1).unwrap();
        assert_eq!(p[0], 2.0 * (1.0 - 0.5f64 * 0.01) as f32);
    }

    #[test]
    fn refuses_step_zero_mismatched_lengths_and_bad_scalars() {
        let mut p = vec![0.0f32; 2];
        let mut mo = Moments::zeros(2);
        assert!(adamw_update(&mut p, &[0.0; 2], &mut mo, &hyper(), 1e-3, 0.01, 0).is_err());
        assert!(adamw_update(&mut p, &[0.0; 3], &mut mo, &hyper(), 1e-3, 0.01, 1).is_err());
        assert!(adamw_update(&mut p, &[0.0; 2], &mut mo, &hyper(), f64::NAN, 0.01, 1).is_err());
        assert!(adamw_update(&mut p, &[0.0; 2], &mut mo, &hyper(), 1e-3, -1.0, 1).is_err());
        let mut h = hyper();
        h.grad_scale = 0.0;
        assert!(adamw_update(&mut p, &[0.0; 2], &mut mo, &h, 1e-3, 0.01, 1).is_err());
        assert_eq!(p, vec![0.0; 2], "nothing moved on a refusal");
    }
}
