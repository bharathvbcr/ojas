//! The parity gates later lanes are held to, against the torch fixtures.
//!
//! A model under test implements [`ParityModel`]; the runners pick the
//! fixture, the batch and the tolerance, so a caller cannot loosen a gate by
//! accident. The tolerances are pinned here and by `tests/parity_gates.rs`.
//!
//! ojas-model (framework items 8 and 9) enables these by implementing
//! [`ParityModel`] for its `Eval`/`Trainer` in its own tests; see
//! `ojas-oracle/README.md`, "Enabling the model parity tests".
//!
//! The constants below are the gates of `CpuBackend` (Exact), which sums
//! sequentially like torch's CPU kernels. A device backend sums in another
//! order, so it is held to a [`Gates`] of its own, passed to the `_with`
//! runners and stated, with its reason, where the backend is tested; the
//! plain runners always use [`Gates::EXACT`].

use std::fmt;

use ojas_core::OjasError;

use crate::golden::{
    tiny_autocast_gap, tiny_forward, tiny_grads, tiny_init, tiny_trace, GradsAt, Ns5, TensorSet,
    TokenBatch, TrainSetup,
};
use crate::spec::{tiny_spec, ModelSpec};
use crate::variants::VariantFixture;

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
/// The bf16 tier's loss-divergence bound, pre-registered 2026-10-10
/// (gp-bf16-compute-tier): over the 40-step f32-NS5 trace, a model's
/// autocast-bf16 run must stay within
/// `max(CURVE_LATE_ABS_TOL, AUTOCAST_GAP_FACTOR * max_t gap_t)` nats of its
/// own f32 run at every step, where `gap_t` is torch's autocast-bf16 vs f32
/// gap on the same trace (`autocast_gap.json`). The max over the trace, not
/// the per-step gap, because torch's gap crosses zero. Each run uses its
/// own f32 Newton-Schulz, so the gap is the autocast tier's alone. torch's
/// rounding scope differs from ojas's (torch autocast lowers matmul-class
/// ops and keeps their outputs bf16; ojas also rounds the operands of the
/// ops listed in `autocast.rs`), so the model's bf16 run is not gated on
/// torch's bf16 run, only reported against it.
pub const AUTOCAST_GAP_FACTOR: f64 = 2.0;

/// The tolerances one backend is held to. [`Gates::EXACT`] is the constants
/// above, which `CpuBackend` (Exact) meets; a device backend passes its own,
/// every field of which must be positive and finite
/// ([`Gates::validate`], checked by every `_with` runner).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    /// Forward loss, relative ([`FORWARD_LOSS_REL_TOL`]).
    pub forward_loss_rel: f64,
    /// Forward logits, normwise relative ([`LOGITS_NORMWISE_REL_TOL`]).
    pub logits_normwise_rel: f64,
    /// Gradients, per parameter, normwise relative ([`GRAD_NORMWISE_REL_TOL`]).
    pub grad_normwise_rel: f64,
    /// 5-step trace, per-step mean loss, absolute ([`TRACE_LOSS_ABS_TOL`]).
    pub trace_loss_abs: f64,
    /// 5-step f32-NS5 trace, parameters, normwise relative
    /// ([`TRACE_PARAM_NORMWISE_REL_TOL`]).
    pub trace_param_normwise_rel: f64,
    /// 5-step bf16-NS5 trace, parameters
    /// ([`TRACE_BF16_PARAM_NORMWISE_REL_TOL`]).
    pub trace_bf16_param_normwise_rel: f64,
    /// 40-step curve, steps `1..=CURVE_EARLY_STEPS`, absolute
    /// ([`CURVE_EARLY_ABS_TOL`]).
    pub curve_early_abs: f64,
    /// 40-step curve, later steps, absolute ([`CURVE_LATE_ABS_TOL`]).
    pub curve_late_abs: f64,
}

impl Gates {
    /// The gates of `CpuBackend` (Exact).
    pub const EXACT: Gates = Gates {
        forward_loss_rel: FORWARD_LOSS_REL_TOL,
        logits_normwise_rel: LOGITS_NORMWISE_REL_TOL,
        grad_normwise_rel: GRAD_NORMWISE_REL_TOL,
        trace_loss_abs: TRACE_LOSS_ABS_TOL,
        trace_param_normwise_rel: TRACE_PARAM_NORMWISE_REL_TOL,
        trace_bf16_param_normwise_rel: TRACE_BF16_PARAM_NORMWISE_REL_TOL,
        curve_early_abs: CURVE_EARLY_ABS_TOL,
        curve_late_abs: CURVE_LATE_ABS_TOL,
    };

    /// Every tolerance positive and finite: a NaN or zero gate would pass or
    /// fail everything for the wrong reason.
    pub fn validate(&self) -> Result<(), ParityFailure> {
        let fields = [
            ("forward_loss_rel", self.forward_loss_rel),
            ("logits_normwise_rel", self.logits_normwise_rel),
            ("grad_normwise_rel", self.grad_normwise_rel),
            ("trace_loss_abs", self.trace_loss_abs),
            ("trace_param_normwise_rel", self.trace_param_normwise_rel),
            (
                "trace_bf16_param_normwise_rel",
                self.trace_bf16_param_normwise_rel,
            ),
            ("curve_early_abs", self.curve_early_abs),
            ("curve_late_abs", self.curve_late_abs),
        ];
        for (name, v) in fields {
            if !(v.is_finite() && v > 0.0) {
                return Err(fail(
                    "gates",
                    format!("{name} = {v} is not positive and finite"),
                ));
            }
        }
        Ok(())
    }
}

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
/// `Ok(f64::INFINITY)` when `want` is zero and `got` is not. A length
/// mismatch, or a non-finite value in `got` or in `want`, is a failure.
/// A NaN reference used to return `Ok(NaN)`, and `NaN > tol` is false, so
/// the tensor gate accepted it.
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
    if let Some(i) = want.iter().position(|v| !v.is_finite()) {
        return Err(fail(check, format!("non-finite reference at {i}")));
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
/// `tol` normwise. A non-finite value in either tensor fails. `got` may hold
/// extra tensors only if they are listed in `allowed_extra` and are exactly
/// zero (a gradient torch leaves as `None`). Returns the worst relative error.
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
        // `!(rel <= tol)` rejects NaN. `rel > tol` does not: a NaN comparison is false.
        if !(rel <= tol) {
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
    /// Select the model shape the next calls run. A runner calls this with
    /// the fixture's spec before anything else, so a model is never left to
    /// assume the shape. The default accepts only the tiny spec and refuses
    /// every other, so an adapter that has not been written for another
    /// shape fails loudly instead of running the wrong model.
    fn configure(&mut self, spec: &ModelSpec) -> Result<(), OjasError> {
        if *spec == tiny_spec() {
            Ok(())
        } else {
            Err(OjasError::Unsupported {
                op: "ParityModel::configure",
                detail: format!("this adapter runs only the tiny spec, not {spec:?}"),
            })
        }
    }

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

/// What a passing forward gate measured.
#[derive(Clone, Debug, PartialEq)]
pub struct ForwardReport {
    /// |loss - torch| / |torch|.
    pub loss_rel: f64,
    /// Normwise relative error of the logits.
    pub logits_normwise: f64,
}

/// (b) Forward loss within [`FORWARD_LOSS_REL_TOL`] and logits within
/// [`LOGITS_NORMWISE_REL_TOL`], from the init on the fixture batch.
pub fn forward_parity<M: ParityModel + ?Sized>(m: &mut M) -> Result<ParityReport, ParityFailure> {
    let r = forward_parity_with(m, &Gates::EXACT)?;
    Ok(ParityReport {
        worst: r.loss_rel,
        tolerance: FORWARD_LOSS_REL_TOL,
    })
}

/// [`forward_parity`] under `gates`.
pub fn forward_parity_with<M: ParityModel + ?Sized>(
    m: &mut M,
    gates: &Gates,
) -> Result<ForwardReport, ParityFailure> {
    const C: &str = "forward";
    gates.validate()?;
    let init = tiny_init().map_err(load(C))?;
    let fx = tiny_forward().map_err(load(C))?;
    m.configure(&init.spec).map_err(from_ojas(C))?;
    let (loss, logits) = m.forward(&init.params, &fx.batch).map_err(from_ojas(C))?;
    check_forward(C, loss, &logits, fx.loss, &fx.logits, gates)
}

/// A model's loss and logits against torch's, under `gates`.
fn check_forward(
    check: &str,
    loss: f32,
    logits: &[f32],
    want_loss: f32,
    want_logits: &[f32],
    gates: &Gates,
) -> Result<ForwardReport, ParityFailure> {
    if !loss.is_finite() {
        return Err(fail(check, "loss is not finite"));
    }
    if !want_loss.is_finite() {
        return Err(fail(check, "torch loss is not finite"));
    }
    // Identical zeros are a match. 0/0 is NaN, and `!(NaN <= tol)` would
    // reject a model that agrees with torch on a zero loss. A zero torch
    // loss with a non-zero model loss stays a non-finite ratio and fails below.
    let rel = if loss == 0.0 && want_loss == 0.0 {
        0.0
    } else {
        (f64::from(loss) - f64::from(want_loss)).abs() / f64::from(want_loss).abs()
    };
    if !(rel <= gates.forward_loss_rel) {
        return Err(fail(
            check,
            format!(
                "loss {loss} vs torch {want_loss}: relative {rel:e} > {:e}",
                gates.forward_loss_rel
            ),
        ));
    }
    let lrel = normwise_rel(&format!("{check} logits"), logits, want_logits)?;
    if !(lrel <= gates.logits_normwise_rel) {
        return Err(fail(
            check,
            format!(
                "logits normwise relative {lrel:e} > {:e}",
                gates.logits_normwise_rel
            ),
        ));
    }
    Ok(ForwardReport {
        loss_rel: rel,
        logits_normwise: lrel,
    })
}

/// (b) on a [`VariantFixture`]: the model is configured with the fixture's
/// spec, then run on the fixture's parameters and batch.
pub fn variant_forward_parity<M: ParityModel + ?Sized>(
    m: &mut M,
    fx: &VariantFixture,
    gates: &Gates,
) -> Result<ForwardReport, ParityFailure> {
    let check = format!("forward {}", fx.variant.name());
    gates.validate()?;
    m.configure(&fx.spec).map_err(from_ojas(&check))?;
    let (loss, logits) = m
        .forward(&fx.params, &fx.batch)
        .map_err(from_ojas(&check))?;
    check_forward(&check, loss, &logits, fx.loss, &fx.logits, gates)
}

/// (c) on a [`VariantFixture`]: every gradient within
/// `gates.grad_normwise_rel`, at the fixture's loss seed.
pub fn variant_grads_parity<M: ParityModel + ?Sized>(
    m: &mut M,
    fx: &VariantFixture,
    gates: &Gates,
) -> Result<ParityReport, ParityFailure> {
    let check = format!("grads {}", fx.variant.name());
    gates.validate()?;
    m.configure(&fx.spec).map_err(from_ojas(&check))?;
    let got = m
        .grads(&fx.params, &fx.batch, fx.loss_seed)
        .map_err(from_ojas(&check))?;
    let worst = check_tensors(
        &check,
        &got,
        &fx.grads,
        &fx.no_grad,
        gates.grad_normwise_rel,
    )?;
    Ok(ParityReport {
        worst,
        tolerance: gates.grad_normwise_rel,
    })
}

/// (c) Every gradient within [`GRAD_NORMWISE_REL_TOL`], at seed 1/K.
pub fn grads_parity<M: ParityModel + ?Sized>(
    m: &mut M,
    at: GradsAt,
) -> Result<ParityReport, ParityFailure> {
    grads_parity_with(m, at, &Gates::EXACT)
}

/// [`grads_parity`] under `gates`.
pub fn grads_parity_with<M: ParityModel + ?Sized>(
    m: &mut M,
    at: GradsAt,
    gates: &Gates,
) -> Result<ParityReport, ParityFailure> {
    const C: &str = "grads";
    gates.validate()?;
    let fx = tiny_grads(at).map_err(load(C))?;
    let params = match at {
        GradsAt::Init => tiny_init().map_err(load(C))?.params,
        GradsAt::Step5 => tiny_trace(Ns5::F32).map_err(load(C))?.params,
    };
    let batch = tiny_forward().map_err(load(C))?.batch;
    m.configure(&tiny_spec()).map_err(from_ojas(C))?;
    let got = m
        .grads(&params, &batch, fx.loss_seed)
        .map_err(from_ojas(C))?;
    let worst = check_tensors(C, &got, &fx.grads, &fx.no_grad, gates.grad_normwise_rel)?;
    Ok(ParityReport {
        worst,
        tolerance: gates.grad_normwise_rel,
    })
}

/// (d) The f32-NS5 trace's first `params_after_step` (5) steps: per-step
/// mean loss within [`TRACE_LOSS_ABS_TOL`], final parameters within
/// [`TRACE_PARAM_NORMWISE_REL_TOL`].
pub fn trace_parity<M: ParityModel + ?Sized>(m: &mut M) -> Result<ParityReport, ParityFailure> {
    trace_parity_ns5(m, Ns5::F32)
}

/// What a passing trace gate measured.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceReport {
    /// Worst per-step |mean loss - torch|, in nats.
    pub loss_abs: f64,
    /// Worst per-tensor normwise relative error of the final parameters.
    pub param_normwise: f64,
}

/// (d) against the trace of either NS5 precision. [`Ns5::Bf16`] is stock
/// nanolab; the model is told which one through [`TrainSetup::ns5`]. The loss
/// gate is the same for both; the bf16 trace's parameters are held to
/// [`TRACE_BF16_PARAM_NORMWISE_REL_TOL`].
pub fn trace_parity_ns5<M: ParityModel + ?Sized>(
    m: &mut M,
    ns5: Ns5,
) -> Result<ParityReport, ParityFailure> {
    let r = trace_parity_ns5_with(m, ns5, &Gates::EXACT)?;
    Ok(ParityReport {
        worst: r.loss_abs,
        tolerance: TRACE_LOSS_ABS_TOL,
    })
}

/// [`trace_parity_ns5`] under `gates`.
pub fn trace_parity_ns5_with<M: ParityModel + ?Sized>(
    m: &mut M,
    ns5: Ns5,
    gates: &Gates,
) -> Result<TraceReport, ParityFailure> {
    const C: &str = "trace";
    gates.validate()?;
    let init = tiny_init().map_err(load(C))?;
    let fx = tiny_trace(ns5).map_err(load(C))?;
    let n = fx.params_after_step;
    m.configure(&init.spec).map_err(from_ojas(C))?;
    let (losses, params) = m.train(&init.params, &fx.train, n).map_err(from_ojas(C))?;
    if losses.len() != n {
        return Err(fail(C, format!("{} losses for {n} steps", losses.len())));
    }
    let mut worst = 0.0f64;
    for (step, (&got, &want)) in losses.iter().zip(&fx.mean_loss).enumerate() {
        let d = (got - want).abs();
        if d.is_nan() || d > gates.trace_loss_abs {
            return Err(fail(
                C,
                format!(
                    "step {}: loss {got} vs torch {want} (|Δ| {d:e} > {:e})",
                    step + 1,
                    gates.trace_loss_abs
                ),
            ));
        }
        worst = worst.max(d);
    }
    let param_tol = match ns5 {
        Ns5::F32 => gates.trace_param_normwise_rel,
        Ns5::Bf16 => gates.trace_bf16_param_normwise_rel,
    };
    let param_normwise = check_tensors(C, &params, &fx.params, &[], param_tol)?;
    Ok(TraceReport {
        loss_abs: worst,
        param_normwise,
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
    curve_parity_ns5_with(m, ns5, &Gates::EXACT)
}

/// [`curve_parity_ns5`] under `gates`.
pub fn curve_parity_ns5_with<M: ParityModel + ?Sized>(
    m: &mut M,
    ns5: Ns5,
    gates: &Gates,
) -> Result<ParityReport, ParityFailure> {
    const C: &str = "curve";
    gates.validate()?;
    let init = tiny_init().map_err(load(C))?;
    let fx = tiny_trace(ns5).map_err(load(C))?;
    m.configure(&init.spec).map_err(from_ojas(C))?;
    let (losses, _) = m
        .train(&init.params, &fx.train, fx.steps)
        .map_err(from_ojas(C))?;
    check_curve_with(&losses, &fx.mean_loss, init.spec.vocab, gates)
}

/// The §10 curve gate on recorded losses.
pub fn check_curve(got: &[f64], want: &[f64], vocab: usize) -> Result<ParityReport, ParityFailure> {
    check_curve_with(got, want, vocab, &Gates::EXACT)
}

/// [`check_curve`] under `gates`.
pub fn check_curve_with(
    got: &[f64],
    want: &[f64],
    vocab: usize,
    gates: &Gates,
) -> Result<ParityReport, ParityFailure> {
    const C: &str = "curve";
    gates.validate()?;
    if got.len() != want.len() {
        return Err(fail(
            C,
            format!("{} losses, torch has {}", got.len(), want.len()),
        ));
    }
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let tol = if i < CURVE_EARLY_STEPS {
            gates.curve_early_abs
        } else {
            gates.curve_late_abs
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
        tolerance: gates.curve_late_abs,
    })
}

/// `max(CURVE_LATE_ABS_TOL, AUTOCAST_GAP_FACTOR * max_t gap_t)` over
/// torch's autocast gap ([`AUTOCAST_GAP_FACTOR`]).
pub fn autocast_divergence_bound() -> Result<f64, ParityFailure> {
    const CHECK: &str = "autocast divergence bound";
    let gap = tiny_autocast_gap().map_err(from_ojas(CHECK))?;
    let worst = gap.gaps().into_iter().fold(0.0f64, f64::max);
    Ok(CURVE_LATE_ABS_TOL.max(AUTOCAST_GAP_FACTOR * worst))
}

/// What [`check_autocast_divergence`] measured.
#[derive(Clone, Debug, PartialEq)]
pub struct AutocastReport {
    /// Worst `|bf16 - f32|` of the model's two runs, and its bound.
    pub worst: f64,
    pub bound: f64,
    /// torch's own worst gap on the trace.
    pub torch_worst: f64,
    /// Worst `|model bf16 - torch bf16|`, reported, not gated.
    pub vs_torch_bf16: f64,
}

/// A model's 40-step f32 and autocast-bf16 losses (per-step means, on the
/// f32-NS5 trace's setup) against the pre-registered bound
/// ([`AUTOCAST_GAP_FACTOR`]), and the bf16 curve against the curve's own
/// end-of-run drop ([`CURVE_MIN_DROP_NATS`] below ln V).
pub fn check_autocast_divergence(
    f32_losses: &[f64],
    bf16_losses: &[f64],
) -> Result<AutocastReport, ParityFailure> {
    const CHECK: &str = "autocast divergence";
    let gap = tiny_autocast_gap().map_err(from_ojas(CHECK))?;
    let steps = gap.f32_mean_loss.len();
    for (name, got) in [("f32", f32_losses), ("bf16", bf16_losses)] {
        if got.len() != steps {
            return Err(fail(
                CHECK,
                format!("{name} run has {} steps, the trace {steps}", got.len()),
            ));
        }
        if let Some(i) = got.iter().position(|v| !v.is_finite()) {
            return Err(fail(
                CHECK,
                format!("{name} loss at step {} is not finite", i + 1),
            ));
        }
    }
    let bound = autocast_divergence_bound()?;
    let mut worst = 0.0f64;
    for (i, (a, b)) in f32_losses.iter().zip(bf16_losses).enumerate() {
        let d = (a - b).abs();
        if d > bound {
            return Err(fail(
                CHECK,
                format!("step {}: |bf16 {b} - f32 {a}| = {d:e} > {bound:e}", i + 1),
            ));
        }
        worst = worst.max(d);
    }
    let vocab = tiny_init().map_err(from_ojas(CHECK))?.spec.vocab;
    let floor = (vocab as f64).ln() - CURVE_MIN_DROP_NATS;
    let last = bf16_losses[steps - 1];
    if last > floor {
        return Err(fail(
            CHECK,
            format!("bf16 curve ends at {last}, above ln V - {CURVE_MIN_DROP_NATS} = {floor}"),
        ));
    }
    let torch_worst = gap.gaps().into_iter().fold(0.0f64, f64::max);
    let vs_torch_bf16 = bf16_losses
        .iter()
        .zip(&gap.bf16_mean_loss)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    Ok(AutocastReport {
        worst,
        bound,
        torch_worst,
        vs_torch_bf16,
    })
}

#[cfg(test)]
mod tests {
    use super::{check_forward, Gates};

    #[test]
    fn both_zero_losses_match_and_a_non_finite_torch_loss_does_not() {
        let gates = Gates::EXACT;
        let matched = check_forward("forward", 0.0, &[0.0], 0.0, &[0.0], &gates).unwrap();
        assert_eq!(matched.loss_rel, 0.0);
        assert!(check_forward("forward", 1.0, &[1.0], 0.0, &[1.0], &gates).is_err());
        assert!(check_forward("forward", 1.0, &[1.0], f32::NAN, &[1.0], &gates).is_err());
        assert!(check_forward("forward", 1.0, &[1.0], f32::INFINITY, &[1.0], &gates).is_err());
        assert!(check_forward("forward", f32::NAN, &[1.0], 1.0, &[1.0], &gates).is_err());
        assert!(check_forward("forward", 1.0 + 2.0e-5, &[1.0], 1.0, &[1.0], &gates).is_err());
        let hit = check_forward("forward", 1.0, &[1.0], 1.0, &[1.0], &gates).unwrap();
        assert_eq!(hit.loss_rel, 0.0);
    }
}
