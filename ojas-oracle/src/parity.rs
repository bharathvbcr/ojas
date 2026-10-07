//! The parity gates later lanes are held to, against the torch fixtures.
//!
//! A model under test implements [`ParityModel`]; the runners pick the
//! fixture, the batch and the tolerance, so a caller cannot loosen a gate by
//! accident. The tolerances are pinned here and by `tests/parity_gates.rs`.
//!
//! ojas-model (framework items 8 and 9) enables these by implementing
//! [`ParityModel`] for its `Eval`/`Trainer` in its own tests; see
//! `ojas-oracle/README.md`, "Enabling the model parity tests".

use std::fmt;

use ojas_core::OjasError;

use crate::golden::{
    tiny_forward, tiny_grads, tiny_init, tiny_trace, GradsAt, Ns5, TensorSet, TokenBatch,
    TrainSetup,
};

/// Forward loss: |Δ| / |torch| (framework item 13).
pub const FORWARD_LOSS_REL_TOL: f64 = 1e-5;
/// Forward logits, normwise over the whole `[B, T, V]` tensor. Not an item-13
/// number: it localises a loss failure and holds whenever the loss gate does.
pub const LOGITS_NORMWISE_REL_TOL: f64 = 1e-5;
/// Gradients: per parameter, ‖g − g_torch‖₂ / ‖g_torch‖₂ (item 13). Where
/// torch's gradient is exactly zero, the model's must be exactly zero.
pub const GRAD_NORMWISE_REL_TOL: f64 = 1e-4;
/// 5-step trace, per-step mean micro-loss: |Δ| in nats (item 13; §10 uses
/// |Δ| for losses).
pub const TRACE_LOSS_ABS_TOL: f64 = 1e-4;
/// 5-step trace, parameters after the last step: per tensor, normwise
/// relative, exact zero where torch's is zero (item 13).
pub const TRACE_PARAM_NORMWISE_REL_TOL: f64 = 1e-4;
/// The bf16-NS5 trace's parameters after the last step, normwise relative.
/// Not item 13's 1e-4: stock bf16 Newton-Schulz is discontinuous in its
/// input. In torch alone, moving a 64x192 gradient by 1e-6 relative flips
/// the bf16 rounding of 2 of its values and moves nanolab's NS5 output by
/// 1.9e-2 normwise; the f32 iteration moves by 2e-6. A model whose
/// gradients match torch's to [`GRAD_NORMWISE_REL_TOL`] rather than bit for
/// bit lands that far from torch on a zero-initialized matrix (measured
/// 1.87e-2 on `blocks.0.ffn.down.weight`). This bound catches a wrong update,
/// not the precision; the loss gate, [`TRACE_LOSS_ABS_TOL`], is what tells
/// a bf16 iteration from an f32 one on this trace.
pub const TRACE_BF16_PARAM_NORMWISE_REL_TOL: f64 = 5e-2;
/// framework-design.md §10, CI on CPU: the 40-step curve, |Δ| per step.
pub const CURVE_EARLY_ABS_TOL: f64 = 2e-4;
/// Steps `1..=CURVE_EARLY_STEPS` use the early tolerance.
pub const CURVE_EARLY_STEPS: usize = 10;
pub const CURVE_LATE_ABS_TOL: f64 = 2e-3;
/// §10: the 40-step curve must end at least this far below ln V.
pub const CURVE_MIN_DROP_NATS: f64 = 1.0;
/// Schedule multipliers: nanolab computes `peak * m / peak` in f64.
pub const LR_MULT_REL_TOL: f64 = 1e-12;

/// A gate that did not hold.
#[derive(Clone, Debug, PartialEq)]
pub struct ParityFailure {
    pub check: String,
    pub detail: String,
}

impl fmt::Display for ParityFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parity {}: {}", self.check, self.detail)
    }
}

impl std::error::Error for ParityFailure {}

fn fail(check: &str, detail: impl Into<String>) -> ParityFailure {
    ParityFailure {
        check: check.to_string(),
        detail: detail.into(),
    }
}

fn from_ojas(check: &str) -> impl Fn(OjasError) -> ParityFailure + '_ {
    move |e| fail(check, format!("model error: {e}"))
}

/// ‖got − want‖₂ / ‖want‖₂ in f64. `Ok(0.0)` when both are exactly zero;
/// `Ok(f64::INFINITY)` when `want` is zero and `got` is not. Non-finite
/// values in `got` and length mismatches are failures.
pub fn normwise_rel(check: &str, got: &[f32], want: &[f32]) -> Result<f64, ParityFailure> {
    if got.len() != want.len() {
        return Err(fail(
            check,
            format!("{} values, torch has {}", got.len(), want.len()),
        ));
    }
    if let Some(i) = got.iter().position(|v| !v.is_finite()) {
        return Err(fail(check, format!("non-finite value at {i}")));
    }
    let (mut diff, mut norm) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = f64::from(g) - f64::from(w);
        diff += d * d;
        norm += f64::from(w) * f64::from(w);
    }
    if norm == 0.0 {
        return Ok(if diff == 0.0 { 0.0 } else { f64::INFINITY });
    }
    Ok((diff / norm).sqrt())
}

/// Every tensor of `want` must be in `got` with the same shape and within
/// `tol` normwise. `got` may hold extra tensors only if they are listed in
/// `allowed_extra` and are exactly zero (a gradient torch leaves as `None`).
/// Returns the worst relative error.
pub fn check_tensors(
    check: &str,
    got: &TensorSet,
    want: &TensorSet,
    allowed_extra: &[String],
    tol: f64,
) -> Result<f64, ParityFailure> {
    let mut worst = 0.0f64;
    for w in want.iter() {
        let g = got
            .get(&w.name)
            .ok_or_else(|| fail(check, format!("missing {}", w.name)))?;
        if g.shape != w.shape {
            return Err(fail(
                check,
                format!("{}: shape {:?}, torch {:?}", w.name, g.shape, w.shape),
            ));
        }
        let rel = normwise_rel(&format!("{check} {}", w.name), &g.data, &w.data)?;
        if rel > tol {
            return Err(fail(
                check,
                format!("{}: normwise relative error {rel:e} > {tol:e}", w.name),
            ));
        }
        worst = worst.max(rel);
    }
    for g in got.iter() {
        if want.get(&g.name).is_some() {
            continue;
        }
        if !allowed_extra.contains(&g.name) {
            return Err(fail(check, format!("unexpected tensor {}", g.name)));
        }
        if g.data.iter().any(|&v| v != 0.0) {
            return Err(fail(
                check,
                format!(
                    "{} has no gradient in torch but a non-zero one here",
                    g.name
                ),
            ));
        }
    }
    Ok(worst)
}

/// What a model must provide for the gates. Implemented by the model lane's
/// test adapter; ojas-oracle does not depend on any model crate.
pub trait ParityModel {
    /// Loss (mean cross-entropy over all `B·T` targets) and the logits
    /// `[B, T, V]` for `batch`, from `params` (nanolab names, §2 shapes).
    fn forward(
        &mut self,
        params: &TensorSet,
        batch: &TokenBatch,
    ) -> Result<(f32, Vec<f32>), OjasError>;

    /// d(loss · seed)/dθ for `batch` from `params`, one tensor per parameter
    /// under its nanolab name.
    fn grads(
        &mut self,
        params: &TensorSet,
        batch: &TokenBatch,
        seed: f32,
    ) -> Result<TensorSet, OjasError>;

    /// Train `steps` steps from `params` with `setup` (optimizers, schedule,
    /// clip, and batches from ojas-data's `BatchSampler` over
    /// `setup.token_bin`). Returns the per-step mean micro-loss and the final
    /// parameters.
    fn train(
        &mut self,
        params: &TensorSet,
        setup: &TrainSetup,
        steps: usize,
    ) -> Result<(Vec<f64>, TensorSet), OjasError>;
}

/// What a passing runner measured.
#[derive(Clone, Debug, PartialEq)]
pub struct ParityReport {
    /// Worst relative or absolute error, per the gate's definition.
    pub worst: f64,
    pub tolerance: f64,
}

fn load(check: &str) -> impl Fn(OjasError) -> ParityFailure + '_ {
    move |e| fail(check, format!("fixture: {e}"))
}

/// (b) Forward loss within [`FORWARD_LOSS_REL_TOL`] and logits within
/// [`LOGITS_NORMWISE_REL_TOL`], from the init on the fixture batch.
pub fn forward_parity<M: ParityModel + ?Sized>(m: &mut M) -> Result<ParityReport, ParityFailure> {
    const C: &str = "forward";
    let init = tiny_init().map_err(load(C))?;
    let fx = tiny_forward().map_err(load(C))?;
    let (loss, logits) = m.forward(&init.params, &fx.batch).map_err(from_ojas(C))?;
    if !loss.is_finite() {
        return Err(fail(C, "loss is not finite"));
    }
    let rel = (f64::from(loss) - f64::from(fx.loss)).abs() / f64::from(fx.loss).abs();
    if rel > FORWARD_LOSS_REL_TOL {
        return Err(fail(
            C,
            format!("loss {loss} vs torch {}: relative {rel:e}", fx.loss),
        ));
    }
    let lrel = normwise_rel("forward logits", &logits, &fx.logits)?;
    if lrel > LOGITS_NORMWISE_REL_TOL {
        return Err(fail(C, format!("logits normwise relative {lrel:e}")));
    }
    Ok(ParityReport {
        worst: rel,
        tolerance: FORWARD_LOSS_REL_TOL,
    })
}

/// (c) Every gradient within [`GRAD_NORMWISE_REL_TOL`], at seed 1/K.
pub fn grads_parity<M: ParityModel + ?Sized>(
    m: &mut M,
    at: GradsAt,
) -> Result<ParityReport, ParityFailure> {
    const C: &str = "grads";
    let fx = tiny_grads(at).map_err(load(C))?;
    let params = match at {
        GradsAt::Init => tiny_init().map_err(load(C))?.params,
        GradsAt::Step5 => tiny_trace(Ns5::F32).map_err(load(C))?.params,
    };
    let batch = tiny_forward().map_err(load(C))?.batch;
    let got = m
        .grads(&params, &batch, fx.loss_seed)
        .map_err(from_ojas(C))?;
    let worst = check_tensors(C, &got, &fx.grads, &fx.no_grad, GRAD_NORMWISE_REL_TOL)?;
    Ok(ParityReport {
        worst,
        tolerance: GRAD_NORMWISE_REL_TOL,
    })
}

/// (d) The f32-NS5 trace's first `params_after_step` (5) steps: per-step
/// mean loss within [`TRACE_LOSS_ABS_TOL`], final parameters within
/// [`TRACE_PARAM_NORMWISE_REL_TOL`].
pub fn trace_parity<M: ParityModel + ?Sized>(m: &mut M) -> Result<ParityReport, ParityFailure> {
    trace_parity_ns5(m, Ns5::F32)
}

/// (d) against the trace of either NS5 precision. [`Ns5::Bf16`] is stock
/// nanolab; the model is told which one through [`TrainSetup::ns5`]. The loss
/// gate is the same for both; the bf16 trace's parameters are held to
/// [`TRACE_BF16_PARAM_NORMWISE_REL_TOL`].
pub fn trace_parity_ns5<M: ParityModel + ?Sized>(
    m: &mut M,
    ns5: Ns5,
) -> Result<ParityReport, ParityFailure> {
    const C: &str = "trace";
    let init = tiny_init().map_err(load(C))?;
    let fx = tiny_trace(ns5).map_err(load(C))?;
    let n = fx.params_after_step;
    let (losses, params) = m.train(&init.params, &fx.train, n).map_err(from_ojas(C))?;
    if losses.len() != n {
        return Err(fail(C, format!("{} losses for {n} steps", losses.len())));
    }
    let mut worst = 0.0f64;
    for (step, (&got, &want)) in losses.iter().zip(&fx.mean_loss).enumerate() {
        let d = (got - want).abs();
        if d.is_nan() || d > TRACE_LOSS_ABS_TOL {
            return Err(fail(
                C,
                format!("step {}: loss {got} vs torch {want} (|Δ| {d:e})", step + 1),
            ));
        }
        worst = worst.max(d);
    }
    let param_tol = match ns5 {
        Ns5::F32 => TRACE_PARAM_NORMWISE_REL_TOL,
        Ns5::Bf16 => TRACE_BF16_PARAM_NORMWISE_REL_TOL,
    };
    check_tensors(C, &params, &fx.params, &[], param_tol)?;
    Ok(ParityReport {
        worst,
        tolerance: TRACE_LOSS_ABS_TOL,
    })
}

/// §10 CI curve: all 40 steps of the f32-NS5 trace, |Δ| within
/// [`CURVE_EARLY_ABS_TOL`] for steps 1–10 and [`CURVE_LATE_ABS_TOL`] after,
/// and the last loss at least [`CURVE_MIN_DROP_NATS`] below ln V.
pub fn curve_parity<M: ParityModel + ?Sized>(m: &mut M) -> Result<ParityReport, ParityFailure> {
    curve_parity_ns5(m, Ns5::F32)
}

/// The §10 curve gate against the trace of either NS5 precision.
pub fn curve_parity_ns5<M: ParityModel + ?Sized>(
    m: &mut M,
    ns5: Ns5,
) -> Result<ParityReport, ParityFailure> {
    const C: &str = "curve";
    let init = tiny_init().map_err(load(C))?;
    let fx = tiny_trace(ns5).map_err(load(C))?;
    let (losses, _) = m
        .train(&init.params, &fx.train, fx.steps)
        .map_err(from_ojas(C))?;
    check_curve(&losses, &fx.mean_loss, init.spec.vocab)
}

/// The §10 curve gate on recorded losses.
pub fn check_curve(got: &[f64], want: &[f64], vocab: usize) -> Result<ParityReport, ParityFailure> {
    const C: &str = "curve";
    if got.len() != want.len() {
        return Err(fail(
            C,
            format!("{} losses, torch has {}", got.len(), want.len()),
        ));
    }
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = if i < CURVE_EARLY_STEPS {
            CURVE_EARLY_ABS_TOL
        } else {
            CURVE_LATE_ABS_TOL
        };
        let d = (g - w).abs();
        if d.is_nan() || d > tol {
            return Err(fail(C, format!("step {}: |Δ| {d:e} > {tol:e}", i + 1)));
        }
        worst = worst.max(d);
    }
    let last = *got.last().ok_or_else(|| fail(C, "no losses"))?;
    let drop = (vocab as f64).ln() - last;
    if drop.is_nan() || drop < CURVE_MIN_DROP_NATS {
        return Err(fail(
            C,
            format!("final loss {last} is only {drop} nats below ln V"),
        ));
    }
    Ok(ParityReport {
        worst,
        tolerance: CURVE_LATE_ABS_TOL,
    })
}
