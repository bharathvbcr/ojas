//! AdamW on the host, for parameters that live outside the step provider.
//!
//! Lappi's span head is two `[H, H]` projections and two `[H]` vectors in f32 on the host
//! (`AUDIT/ojas-training-2026-10-01/fable-advice.md` Q1 row 8); tessl has no head, so its
//! optimizer step cannot be the provider's. It is the SAME optimizer as the tower's: one
//! `torch.optim.AdamW` over both in the PyTorch trainer (`python/qd_train/optim.py:402-405`),
//! the same betas, eps, weight decay, learning rate, step count and clip coefficient. This is
//! that update for one f32 tensor, and [`super::mock::ToyProvider`] uses it for its parameters
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

use ojas_core::{AdamWConfig, AdamWScalars};

use super::recipe::EntryHyper;
use super::step::{AdamWHyper, StepError};

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
    adamw_update(
        param,
        grad,
        &mut state.moments,
        &hyper,
        lr * row.lr_scale,
        row.weight_decay,
        t,
    )?;
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
        return Err(StepError::Contract(format!(
            "lr {lr} must be finite and > 0"
        )));
    }
    if !(weight_decay.is_finite() && weight_decay >= 0.0) {
        return Err(StepError::Contract(format!(
            "weight decay {weight_decay} must be finite and >= 0"
        )));
    }
    if step == 0 {
        return Err(StepError::Contract(
            "step 0 zeroes the bias correction; steps count from 1".into(),
        ));
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
        return Err(StepError::Contract(format!(
            "step {step} is not exact in f64"
        )));
    }
    let t = step as f64;
    let scale = hyper.grad_scale as f32;
    let bc1 = 1.0 - hyper.beta1.powf(t);
    let bc2 = 1.0 - hyper.beta2.powf(t);
    // ojas-core forms every other scalar from torch's libm bias corrections.
    let scalars = AdamWScalars::from_bias_corrections(
        AdamWConfig {
            lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            weight_decay,
        },
        step,
        bc1,
        bc2,
    );
    let decay = scalars.decay as f32;
    let w1 = scalars.one_minus_b1 as f32;
    let b2 = scalars.beta2 as f32;
    let w2 = scalars.one_minus_b2 as f32;
    let bc2_sqrt = scalars.bc2_sqrt as f32;
    let eps = scalars.eps as f32;
    let neg_step = (-scalars.step_size) as f32;
    // One element's update, evaluated by the refusal pass and by the write pass alike, so the
    // value that is checked is the value that is stored.
    let element = |raw_grad: f32, param: f32, m0: f32, v0: f32| {
        let g = raw_grad * scale;
        let mut p = param;
        if weight_decay != 0.0 {
            p *= decay;
        }
        // torch's lerp for a weight below 0.5: start + weight * (end - start), fused.
        let m = if w1.abs() < 0.5 {
            w1.mul_add(g - m0, m0)
        } else {
            (w1 - 1.0).mul_add(g - m0, g)
        };
        let g2 = g * g;
        let v = v0 * b2 + w2 * g * g;
        let denom = v.sqrt() / bc2_sqrt + eps;
        p += neg_step * m / denom;
        (g, g2, m, v, p)
    };
    // Refuse before any write: an error found at element k must not leave elements 0..k
    // updated. A finite gradient whose square overflows f32 (|g| above ~1.8e19, for example
    // from a clip bound that large) makes the second moment +inf, and the next step would
    // divide by it.
    for i in 0..n {
        let (g, g2, m, v, _) = element(grad[i], param[i], moments.m[i], moments.v[i]);
        for (quantity, value) in [("g * g", g2), ("first moment", m), ("second moment", v)] {
            if !value.is_finite() {
                return Err(StepError::Contract(format!(
                    "element {i}: {quantity} is {value} (scaled gradient {g}); nothing was written"
                )));
            }
        }
    }
    for i in 0..n {
        let (_, _, m, v, p) = element(grad[i], param[i], moments.m[i], moments.v[i]);
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

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|v| v.to_bits()).collect()
    }

    /// Runs one update and returns the (param, m, v) bit patterns, or the refusal.
    fn run(
        p: &[f32],
        g: &[f32],
        mo: &Moments,
        h: &AdamWHyper,
    ) -> (Result<(), StepError>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let (mut p, mut mo) = (p.to_vec(), mo.clone());
        let r = adamw_update(&mut p, g, &mut mo, h, 1e-3, 0.01, 3);
        (r, bits(&p), bits(&mo.m), bits(&mo.v))
    }

    fn state() -> (Vec<f32>, Moments) {
        (
            vec![0.5, -1.25, 2.0],
            Moments {
                m: vec![0.01, -0.02, 0.03],
                v: vec![1e-4, 2e-4, 3e-4],
            },
        )
    }

    /// A refusal must leave `param`, `m` and `v` exactly as they came in, whichever element
    /// overflowed.
    fn assert_refused_untouched(g: &[f32], h: &AdamWHyper, what: &str) {
        let (p0, mo0) = state();
        let (r, p, m, v) = run(&p0, g, &mo0, h);
        let e = r.expect_err("a non-finite moment must be refused");
        assert!(e.to_string().contains(what), "{e}");
        assert_eq!(p, bits(&p0), "param written on a refusal");
        assert_eq!(m, bits(&mo0.m), "m written on a refusal");
        assert_eq!(v, bits(&mo0.v), "v written on a refusal");
    }

    #[test]
    fn a_finite_gradient_whose_square_overflows_is_refused_and_writes_nothing() {
        // 1e20 is a finite f32, and 1e20 * 1e20 is +inf.
        assert!(1e20f32.is_finite() && !(1e20f32 * 1e20f32).is_finite());
        assert_refused_untouched(&[1e20, 0.1, 0.1], &hyper(), "element 0");
    }

    #[test]
    fn an_overflow_at_the_last_element_leaves_the_earlier_elements_unwritten() {
        assert_refused_untouched(&[0.1, -0.2, -1e20], &hyper(), "element 2");
    }

    #[test]
    fn a_scale_that_overflows_a_finite_gradient_is_refused() {
        let mut h = hyper();
        h.grad_scale = 1e10;
        assert_refused_untouched(&[0.1, 1e10, 0.1], &h, "element 1");
    }

    #[test]
    fn a_non_finite_gradient_is_refused_and_writes_nothing() {
        assert_refused_untouched(&[0.1, f32::NAN, 0.1], &hyper(), "element 1");
        assert_refused_untouched(&[f32::INFINITY, 0.1, 0.1], &hyper(), "element 0");
    }

    #[test]
    fn an_empty_tensor_is_still_ok() {
        let mut mo = Moments::zeros(0);
        adamw_update(&mut [], &[], &mut mo, &hyper(), 1e-3, 0.01, 1).unwrap();
    }

    /// The update loop exactly as aff9179 had it, before the refusal pass existed.
    fn reference_aff9179(p: &mut [f32], g: &[f32], mo: &mut Moments, h: &AdamWHyper, step: u64) {
        let (lr, wd) = (1e-3f64, 0.01f64);
        let t = step as f64;
        let bc1 = 1.0 - h.beta1.powf(t);
        let bc2 = 1.0 - h.beta2.powf(t);
        let sc = AdamWScalars::from_bias_corrections(
            AdamWConfig {
                lr,
                beta1: h.beta1,
                beta2: h.beta2,
                eps: h.eps,
                weight_decay: wd,
            },
            step,
            bc1,
            bc2,
        );
        let (decay, w1, b2, w2) = (
            sc.decay as f32,
            sc.one_minus_b1 as f32,
            sc.beta2 as f32,
            sc.one_minus_b2 as f32,
        );
        let (bc2_sqrt, eps, neg_step) = (sc.bc2_sqrt as f32, sc.eps as f32, (-sc.step_size) as f32);
        for i in 0..p.len() {
            let gi = g[i] * h.grad_scale as f32;
            let mut pi = p[i] * decay;
            let m0 = mo.m[i];
            let m = if w1.abs() < 0.5 {
                w1.mul_add(gi - m0, m0)
            } else {
                (w1 - 1.0).mul_add(gi - m0, gi)
            };
            let v = mo.v[i] * b2 + w2 * gi * gi;
            let denom = v.sqrt() / bc2_sqrt + eps;
            pi += neg_step * m / denom;
            mo.m[i] = m;
            mo.v[i] = v;
            p[i] = pi;
        }
    }

    /// A finite update is bit-identical to the pre-refusal loop on both lerp branches:
    /// beta1 = 0.9 gives `w1 = 0.1` (below 0.5), beta1 = 0.3 gives `w1 = 0.7`. The first
    /// branch is also pinned to element bits recorded from aff9179.
    #[test]
    fn a_finite_update_keeps_its_bits_on_both_lerp_branches() {
        let g = [0.3f32, -4.0, 1e-3];
        for beta1 in [0.9, 0.3] {
            let mut h = hyper();
            h.beta1 = beta1;
            let (p0, mo0) = state();
            let (r, p, m, v) = run(&p0, &g, &mo0, &h);
            r.unwrap();
            let (mut rp, mut rmo) = (p0.clone(), mo0.clone());
            reference_aff9179(&mut rp, &g, &mut rmo, &h, 3);
            assert_eq!(p, bits(&rp), "param, beta1 {beta1}");
            assert_eq!(m, bits(&rmo.m), "m, beta1 {beta1}");
            assert_eq!(v, bits(&rmo.v), "v, beta1 {beta1}");
            if beta1 == 0.9 {
                let got: Vec<u32> = p.into_iter().chain(m).chain(v).collect();
                assert_eq!(got, PINNED_LOW_W1);
            }
        }
    }

    const PINNED_LOW_W1: [u32; 9] = [
        1056945257, 3214928346, 1073739003, 1025490551, 3201696793, 1021182161, 960962533,
        1015330162, 966598960,
    ];
}
