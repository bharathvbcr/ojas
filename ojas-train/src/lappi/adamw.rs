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

/// The step count and hyper-parameters one entry's update runs with.
fn entry_inputs(
    state: &EntryState,
    row: &EntryHyper,
    lr: f64,
    grad_scale: f64,
) -> Result<(u64, AdamWHyper), StepError> {
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
    Ok((t, hyper))
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
///
/// A refusal writes nothing and leaves the count where it was. A caller that steps several
/// entries and must not leave some of them stepped asks [`adamw_entry_check`] of every entry
/// first: it applies this function's own refusal rule.
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
    let (t, hyper) = entry_inputs(state, row, lr, grad_scale)?;
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

/// Whether [`adamw_entry_step`] with these arguments would succeed, without writing anything.
/// `Ok(())` is also the answer for a `None` gradient, which is skipped, not refused. The refusal
/// is the one the step would return: this runs the same validation and the same scan.
pub fn adamw_entry_check(
    param: &[f32],
    grad: Option<&[f32]>,
    state: &EntryState,
    row: &EntryHyper,
    lr: f64,
    grad_scale: f64,
) -> Result<(), StepError> {
    let Some(grad) = grad else {
        return Ok(());
    };
    let (t, hyper) = entry_inputs(state, row, lr, grad_scale)?;
    adamw_check(
        param,
        grad,
        &state.moments,
        &hyper,
        lr * row.lr_scale,
        row.weight_decay,
        t,
    )
}

/// The validated scalars of one update, and the lengths they were validated against.
struct Prepared {
    kernel: Kernel,
    decaying: bool,
}

/// Every refusal that does not depend on the element values: the scalars, the step, the lengths.
fn prepare(
    param: &[f32],
    grad: &[f32],
    moments: &Moments,
    hyper: &AdamWHyper,
    lr: f64,
    weight_decay: f64,
    step: u64,
) -> Result<Prepared, StepError> {
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
    Ok(Prepared {
        kernel: kernel(hyper, lr, weight_decay, step),
        decaying: weight_decay != 0.0,
    })
}

/// Refuses, before any write, an update that would store a non-finite value: an error found at
/// element k must not leave elements 0..k updated. The scan runs the very pass that then stores,
/// minus the stores, so what is checked is exactly what is stored: the first moment, the second
/// moment and the parameter. A finite gradient whose scaled square overflows f32 in `w2 * g * g`
/// (|g| above ~1.8e20 at beta2 = 0.999) stores a +inf second moment, and the next step divides by
/// it. A gradient whose bare `g * g` is +inf is NOT refused when `(w2 * g) * g` is finite: that is
/// the value the loop stores, and it stays unchanged.
fn scan_refusal(
    p: &Prepared,
    param: &[f32],
    grad: &[f32],
    m: &[f32],
    v: &[f32],
) -> Result<(), StepError> {
    if scan_pass(&p.kernel, p.decaying, param, grad, m, v) != 0 {
        return Err(first_refusal(&p.kernel, p.decaying, param, grad, m, v));
    }
    Ok(())
}

/// Whether [`adamw_update`] with these arguments would succeed, without writing anything: the
/// same validation, the same scan, the same refusal. It is the one place that rule lives, so a
/// caller that steps several tensors (a provider's bank, a head) can ask every tensor first and
/// then apply, instead of leaving the earlier tensors stepped when a later one refuses.
pub fn adamw_check(
    param: &[f32],
    grad: &[f32],
    moments: &Moments,
    hyper: &AdamWHyper,
    lr: f64,
    weight_decay: f64,
    step: u64,
) -> Result<(), StepError> {
    let p = prepare(param, grad, moments, hyper, lr, weight_decay, step)?;
    scan_refusal(&p, param, grad, &moments.m, &moments.v)
}

/// One AdamW update of `param` in place. `lr` is this tensor's learning rate (the schedule's
/// rate times its group's scale); `hyper.lr` is not read. `step` is the 1-based count.
///
/// A refusal writes nothing: `param` and both moments are as they came in. See [`adamw_check`]
/// for asking without applying.
pub fn adamw_update(
    param: &mut [f32],
    grad: &[f32],
    moments: &mut Moments,
    hyper: &AdamWHyper,
    lr: f64,
    weight_decay: f64,
    step: u64,
) -> Result<(), StepError> {
    let p = prepare(param, grad, moments, hyper, lr, weight_decay, step)?;
    let n = param.len();
    // Re-slicing to `n` up front lets the compiler drop the per-element bounds checks.
    let (param, grad) = (&mut param[..n], &grad[..n]);
    let (m, v) = (&mut moments.m[..n], &mut moments.v[..n]);
    scan_refusal(&p, param, grad, m, v)?;
    write_pass::<false>(&p.kernel, p.decaying, param, grad, m, v);
    Ok(())
}

/// The f32 scalars of one update, formed once.
#[derive(Clone, Copy)]
struct Kernel {
    scale: f32,
    decay: f32,
    w1: f32,
    b2: f32,
    w2: f32,
    bc2_sqrt: f32,
    eps: f32,
    neg_step: f32,
}

/// The scalars of one update, from validated inputs.
fn kernel(hyper: &AdamWHyper, lr: f64, weight_decay: f64, step: u64) -> Kernel {
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
    Kernel {
        scale,
        decay,
        w1,
        b2,
        w2,
        bc2_sqrt,
        eps,
        neg_step,
    }
}

/// Bits of the mask [`scan_pass`] returns.
const BAD_FIRST_MOMENT: u8 = 1;
const BAD_SECOND_MOMENT: u8 = 2;
const BAD_PARAM: u8 = 4;

/// One element's update: the new `(param, first moment, second moment)`. Every pass shares this
/// one body, so what the scan checks is what the write pass stores. The lerp branch (torch's
/// weight below 0.5 or not) and the decay are loop-invariant, so each of the four combinations
/// is its own instantiation, chosen once per pass.
#[inline(always)]
fn element<const DECAY: bool, const LOW_LERP: bool>(
    k: &Kernel,
    p: f32,
    g: f32,
    m0: f32,
    v0: f32,
) -> (f32, f32, f32) {
    let g = g * k.scale;
    let mut p = p;
    if DECAY {
        p *= k.decay;
    }
    // torch's lerp for a weight below 0.5: start + weight * (end - start), fused.
    let mi = if LOW_LERP {
        k.w1.mul_add(g - m0, m0)
    } else {
        (k.w1 - 1.0).mul_add(g - m0, g)
    };
    let vi = v0 * k.b2 + k.w2 * g * g;
    let denom = vi.sqrt() / k.bc2_sqrt + k.eps;
    p += k.neg_step * mi / denom;
    (p, mi, vi)
}

/// Which stored quantities of one element's update are non-finite, as a mask of the `BAD_*` bits.
#[inline(always)]
fn bad_bits(p: f32, mi: f32, vi: f32) -> [bool; 3] {
    [!mi.is_finite(), !vi.is_finite(), !p.is_finite()]
}

fn mask(bad: [bool; 3]) -> u8 {
    u8::from(bad[0]) * BAD_FIRST_MOMENT
        + u8::from(bad[1]) * BAD_SECOND_MOMENT
        + u8::from(bad[2]) * BAD_PARAM
}

/// Stores nothing and returns the OR over all elements of which stored quantities would be
/// non-finite. It has no early exit, so it vectorises like the write pass.
#[inline(always)]
fn scan_pass(k: &Kernel, decaying: bool, param: &[f32], grad: &[f32], m: &[f32], v: &[f32]) -> u8 {
    #[inline(always)]
    fn run<const DECAY: bool, const LOW_LERP: bool>(
        k: &Kernel,
        param: &[f32],
        grad: &[f32],
        m: &[f32],
        v: &[f32],
    ) -> u8 {
        let n = param.len();
        let (grad, m, v) = (&grad[..n], &m[..n], &v[..n]);
        let mut bad = [false; 3];
        for i in 0..n {
            let (p, mi, vi) = element::<DECAY, LOW_LERP>(k, param[i], grad[i], m[i], v[i]);
            let b = bad_bits(p, mi, vi);
            bad = [bad[0] | b[0], bad[1] | b[1], bad[2] | b[2]];
        }
        mask(bad)
    }
    match (decaying, k.w1.abs() < 0.5) {
        (true, true) => run::<true, true>(k, param, grad, m, v),
        (true, false) => run::<true, false>(k, param, grad, m, v),
        (false, true) => run::<false, true>(k, param, grad, m, v),
        (false, false) => run::<false, false>(k, param, grad, m, v),
    }
}

/// Stores each element's new `param`, `m` and `v`. With `CHECK` it also returns what
/// [`scan_pass`] would have (after the stores: the fused single-pass shape, which cannot refuse
/// before writing and is used only to measure what that would save); otherwise 0.
#[inline(always)]
fn write_pass<const CHECK: bool>(
    k: &Kernel,
    decaying: bool,
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
) -> u8 {
    #[inline(always)]
    fn run<const CHECK: bool, const DECAY: bool, const LOW_LERP: bool>(
        k: &Kernel,
        param: &mut [f32],
        grad: &[f32],
        m: &mut [f32],
        v: &mut [f32],
    ) -> u8 {
        let n = param.len();
        let (grad, m, v) = (&grad[..n], &mut m[..n], &mut v[..n]);
        let mut bad = [false; 3];
        for i in 0..n {
            let (p, mi, vi) = element::<DECAY, LOW_LERP>(k, param[i], grad[i], m[i], v[i]);
            if CHECK {
                let b = bad_bits(p, mi, vi);
                bad = [bad[0] | b[0], bad[1] | b[1], bad[2] | b[2]];
            }
            m[i] = mi;
            v[i] = vi;
            param[i] = p;
        }
        mask(bad)
    }
    match (decaying, k.w1.abs() < 0.5) {
        (true, true) => run::<CHECK, true, true>(k, param, grad, m, v),
        (true, false) => run::<CHECK, true, false>(k, param, grad, m, v),
        (false, true) => run::<CHECK, false, true>(k, param, grad, m, v),
        (false, false) => run::<CHECK, false, false>(k, param, grad, m, v),
    }
}

/// The refusal for the first element whose update stores a non-finite value. Cold: it runs
/// only after the scan tripped, and re-runs that same scan one element at a time.
fn first_refusal(
    k: &Kernel,
    decaying: bool,
    param: &[f32],
    grad: &[f32],
    m: &[f32],
    v: &[f32],
) -> StepError {
    for i in 0..param.len() {
        let mask = scan_pass(
            k,
            decaying,
            &param[i..=i],
            &grad[i..=i],
            &m[i..=i],
            &v[i..=i],
        );
        if mask != 0 {
            let names: Vec<&str> = [
                (BAD_FIRST_MOMENT, "first moment"),
                (BAD_SECOND_MOMENT, "second moment"),
                (BAD_PARAM, "parameter"),
            ]
            .into_iter()
            .filter(|(bit, _)| mask & bit != 0)
            .map(|(_, name)| name)
            .collect();
            return StepError::Contract(format!(
                "element {i}: the updated {} would be non-finite (gradient {}, first moment {}, \
                 second moment {}); nothing was written",
                names.join(" and "),
                grad[i],
                m[i],
                v[i]
            ));
        }
    }
    StepError::Contract("a non-finite update was detected; nothing was written".into())
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
    fn a_finite_gradient_whose_stored_second_moment_overflows_is_refused_and_writes_nothing() {
        // 1e21 is a finite f32; the stored `(w2 * g) * g` is about 1e39 and overflows.
        assert!(1e21f32.is_finite() && !((0.001f32 * 1e21) * 1e21).is_finite());
        assert_refused_untouched(&[1e21, 0.1, 0.1], &hyper(), "element 0");
    }

    /// `g * g` is +inf for 3e19, but the stored `(w2 * g) * g` is about 9e35: aff9179 stored
    /// finite values and returned Ok, so the result must not change.
    #[test]
    fn a_gradient_whose_bare_square_overflows_but_whose_stored_moment_does_not_is_updated() {
        let g = [0.1f32, 3e19, -0.2];
        assert!(!(g[1] * g[1]).is_finite() && ((0.001f32 * g[1]) * g[1]).is_finite());
        let h = hyper();
        let (p0, mo0) = state();
        let (r, p, m, v) = run(&p0, &g, &mo0, &h);
        r.expect("a finite stored update must be accepted");
        let (mut rp, mut rmo) = (p0.clone(), mo0.clone());
        reference_aff9179(&mut rp, &g, &mut rmo, &h, 3);
        assert_eq!(p, bits(&rp));
        assert_eq!(m, bits(&rmo.m));
        assert_eq!(v, bits(&rmo.v));
        assert!(rmo.v.iter().all(|x| x.is_finite()) && rp.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn a_corrupt_second_moment_on_input_is_refused_and_writes_nothing() {
        for bad in [f32::INFINITY, f32::NAN] {
            let (p0, mut mo0) = state();
            mo0.v[1] = bad;
            let (r, p, m, v) = run(&p0, &[0.1, 0.1, 0.1], &mo0, &hyper());
            let e = r.expect_err("a corrupt moment must be refused");
            assert!(e.to_string().contains("element 1"), "{e}");
            assert_eq!(p, bits(&p0));
            assert_eq!(m, bits(&mo0.m));
            assert_eq!(v, bits(&mo0.v));
        }
    }

    /// A huge first moment over a zero second moment divides by `eps` and stores a +inf
    /// parameter. aff9179 returned Ok there; the stored parameter is as much a corruption as the
    /// moment.
    #[test]
    fn a_non_finite_stored_parameter_is_refused_and_writes_nothing() {
        let (p0, mut mo0) = state();
        mo0.m[2] = 3e38;
        mo0.v[2] = 0.0;
        let (mut rp, mut rmo) = (p0.clone(), mo0.clone());
        reference_aff9179(&mut rp, &[0.1, 0.1, 0.0], &mut rmo, &hyper(), 3);
        assert!(
            rp[2].is_infinite(),
            "the aff9179 loop stores an infinite parameter here"
        );
        let (r, p, m, v) = run(&p0, &[0.1, 0.1, 0.0], &mo0, &hyper());
        let e = r.expect_err("an infinite parameter must be refused");
        assert!(
            e.to_string().contains("element 2") && e.to_string().contains("parameter"),
            "{e}"
        );
        assert_eq!(p, bits(&p0));
        assert_eq!(m, bits(&mo0.m));
        assert_eq!(v, bits(&mo0.v));
    }

    #[test]
    fn an_overflow_at_the_last_element_leaves_the_earlier_elements_unwritten() {
        assert_refused_untouched(&[0.1, -0.2, -1e21], &hyper(), "element 2");
    }

    #[test]
    fn a_scale_that_overflows_a_finite_gradient_is_refused() {
        let mut h = hyper();
        h.grad_scale = 1e10;
        // 1e11 * 1e10 = 1e21 after the scale.
        assert_refused_untouched(&[0.1, 1e11, 0.1], &h, "element 1");
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
        // Opaque scalars, as they are to `adamw_update` (a caller's runtime values): with
        // literals the compiler folds the lerp and decay branches and the baseline is not the
        // function aff9179 shipped.
        use std::hint::black_box;
        let (lr, wd) = (black_box(1e-3f64), black_box(0.01f64));
        let h = &AdamWHyper {
            lr: black_box(h.lr),
            beta1: black_box(h.beta1),
            beta2: black_box(h.beta2),
            eps: black_box(h.eps),
            grad_scale: black_box(h.grad_scale),
        };
        let step = black_box(step);
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
            let mut pi = p[i];
            if wd != 0.0 {
                pi *= decay;
            }
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

    #[test]
    fn adamw_check_agrees_with_adamw_update_and_writes_nothing() {
        let h = hyper();
        // Accepted: the check says Ok and moves nothing; the update then succeeds.
        let (p0, mo0) = state();
        let g = [0.3f32, -4.0, 1e-3];
        adamw_check(&p0, &g, &mo0, &h, 1e-3, 0.01, 3).unwrap();
        assert!(run(&p0, &g, &mo0, &h).0.is_ok());
        // Refused: the same refusal text as the update, for every kind of refusal.
        let (mut bad_m, mut bad_p) = (mo0.clone(), mo0.clone());
        bad_m.v[1] = f32::NAN;
        bad_p.m[2] = 3e38;
        bad_p.v[2] = 0.0;
        let cases: [(&[f32], &Moments, f64, f64, u64); 6] = [
            (&[1e21, 0.1, 0.1], &mo0, 1e-3, 0.01, 3),
            (&[0.1, 0.1, 0.1], &bad_m, 1e-3, 0.01, 3),
            (&[0.1, 0.1, 0.0], &bad_p, 1e-3, 0.01, 3),
            (&[0.1; 3], &mo0, 0.0, 0.01, 3),
            (&[0.1; 3], &mo0, 1e-3, -1.0, 3),
            (&[0.1; 3], &mo0, 1e-3, 0.01, 0),
        ];
        for (g, mo, lr, wd, step) in cases {
            let (mut p, mut mo_u) = (p0.clone(), mo.clone());
            let upd = adamw_update(&mut p, g, &mut mo_u, &h, lr, wd, step).unwrap_err();
            let chk = adamw_check(&p0, g, mo, &h, lr, wd, step).unwrap_err();
            assert_eq!(upd.to_string(), chk.to_string());
        }
    }

    fn row(lr_scale: f64) -> EntryHyper {
        EntryHyper {
            name: "head".into(),
            lr_scale,
            weight_decay: 0.01,
            eps: 1e-8,
            beta1: 0.9,
            beta2: 0.999,
        }
    }

    #[test]
    fn an_entry_check_is_the_entry_steps_verdict_without_its_effects() {
        // A skipped (None) gradient and a finite one pass; an overflowing, a zero-scale and a
        // step-count-overflow entry are refused, and the check never moves the state.
        let ok_grad = [0.1f32, 0.2];
        let big = [1e21f32, 0.0];
        let st = EntryState::zeros(2);
        let p = [0.5f32, 0.5];
        adamw_entry_check(&p, None, &st, &row(1.0), 1e-3, 1.0).unwrap();
        adamw_entry_check(&p, Some(&ok_grad), &st, &row(1.0), 1e-3, 1.0).unwrap();
        adamw_entry_check(&p, Some(&big), &st, &row(1.0), 1e-3, 1.0).unwrap_err();
        adamw_entry_check(&p, Some(&ok_grad), &st, &row(0.0), 1e-3, 1.0).unwrap_err();
        let mut max = EntryState::zeros(2);
        max.steps = u64::MAX;
        adamw_entry_check(&p, Some(&ok_grad), &max, &row(1.0), 1e-3, 1.0).unwrap_err();
        assert_eq!(st, EntryState::zeros(2));
        // Entry step refuses the same inputs, writing nothing and not counting.
        for (grad, r) in [(&big, row(1.0)), (&ok_grad, row(0.0))] {
            let (mut pp, mut s) = (p, EntryState::zeros(2));
            adamw_entry_step(&mut pp, Some(grad), &mut s, &r, 1e-3, 1.0).unwrap_err();
            assert_eq!((pp, s), (p, EntryState::zeros(2)));
        }
        // An entry with no elements passes.
        adamw_entry_check(&[], Some(&[]), &EntryState::zeros(0), &row(1.0), 1e-3, 1.0).unwrap();
    }

    /// Cost of the refusal pass, split: the aff9179 loop, the scan alone, the write pass alone,
    /// the full two-pass `adamw_update` and the fused single-pass shape (write and check in one
    /// loop, which cannot refuse before writing). Interleaved, min of 5, at 2^16, 2^20 and 2^24
    /// elements; the same config is measured twice (`full` and `full'`) as the noise floor. Run with
    /// `cargo test --release -p ojas-train --lib adamw_refusal_pass_cost -- --ignored --nocapture --test-threads=1`.
    #[test]
    #[ignore = "timing; run by hand with --test-threads=1"]
    fn adamw_refusal_pass_cost() {
        use std::time::Instant;
        let h = hyper();
        let k = kernel(&h, 1e-3, 0.01, 3);
        for n in [1usize << 16, 1 << 20, 1 << 24] {
            let g: Vec<f32> = (0..n).map(|i| ((i % 977) as f32 - 488.0) * 1e-3).collect();
            let p0: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
            let mut st: Vec<(Vec<f32>, Moments)> =
                (0..6).map(|_| (p0.clone(), Moments::zeros(n))).collect();
            // 0 two-pass, 1 two-pass again (noise), 2 aff9179 loop, 3 scan alone (param/moments
            // unchanged), 4 write alone, 5 fused single pass.
            let mut t = [f64::MAX; 6];
            for _ in 0..5 {
                let s = Instant::now();
                let (p, m) = &mut st[0];
                adamw_update(p, &g, m, &h, 1e-3, 0.01, 3).unwrap();
                t[0] = t[0].min(s.elapsed().as_secs_f64());
                let s = Instant::now();
                let (p, m) = &mut st[1];
                adamw_update(p, &g, m, &h, 1e-3, 0.01, 3).unwrap();
                t[1] = t[1].min(s.elapsed().as_secs_f64());
                let s = Instant::now();
                let (p, m) = &mut st[2];
                reference_aff9179(p, &g, m, &h, 3);
                t[2] = t[2].min(s.elapsed().as_secs_f64());
                let s = Instant::now();
                let (p, m) = &mut st[3];
                assert_eq!(scan_pass(&k, true, p, &g, &m.m, &m.v), 0);
                t[3] = t[3].min(s.elapsed().as_secs_f64());
                let s = Instant::now();
                let (p, m) = &mut st[4];
                write_pass::<false>(&k, true, p, &g, &mut m.m, &mut m.v);
                t[4] = t[4].min(s.elapsed().as_secs_f64());
                let s = Instant::now();
                let (p, m) = &mut st[5];
                assert_eq!(write_pass::<true>(&k, true, p, &g, &mut m.m, &mut m.v), 0);
                t[5] = t[5].min(s.elapsed().as_secs_f64());
            }
            assert_eq!(bits(&st[0].0), bits(&st[2].0), "update == aff9179 loop");
            assert_eq!(bits(&st[4].0), bits(&st[2].0), "write pass == aff9179 loop");
            assert_eq!(bits(&st[5].0), bits(&st[2].0), "fused == aff9179 loop");
            let ms = t.map(|x| x * 1e3);
            println!(
                "adamw n=2^{} min-of-5 ms: two-pass {:.3}, two-pass again {:.3} (noise {:+.1}%), \
                 aff9179 loop {:.3}, scan alone {:.3}, write alone {:.3}, fused single-pass {:.3} \
                 (fused/two-pass {:.3})",
                n.trailing_zeros(),
                ms[0],
                ms[1],
                (ms[1] / ms[0] - 1.0) * 100.0,
                ms[2],
                ms[3],
                ms[4],
                ms[5],
                ms[5] / ms[0].min(ms[1])
            );
        }
    }
}
