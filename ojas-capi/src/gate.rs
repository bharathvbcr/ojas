//! Cancellation inside a model op.
//!
//! `ojas_model::Trainer::step` and `ojas_infer::DeviceDecoder::generate` take
//! no cancel hook. Each session's backend is wrapped in [`Gated`], which polls
//! the call's cancel check before every op that can fail without changing
//! what the trainer owns, and refuses the op once the check fails.
//!
//! Which ops poll is the commit contract. `Trainer::step`
//! (`ojas-model/src/trainer.rs`, `run`) computes gradients and clips them
//! (phases 2-4) before it touches a parameter; an error there leaves
//! parameters, moments, step and cursor unchanged. Its optimizer phase
//! (`muon_ns5_step`, `adamw_step`) and its finish (`sync`, then one
//! `download`) mark the trainer poisoned on any error. So those four never
//! poll: a cancel always lands in phases 2-4, and a cancelled step commits
//! nothing. Every other op polls.
//!
//! The check is armed for one C-ABI call by [`CancelSlot::arm`] and disarmed
//! when the returned [`Armed`] drops, so a later call never sees an earlier
//! call's context. A failed poll is recorded in the slot; the call reports
//! that recorded cancel ([`Armed::tripped`]), never a reading of error text.

use std::sync::{Arc, Mutex};

use ojas_core::{
    AdamWConfig, AutocastGuard, AutocastMode, Backend, BackendId, Budget, CeChunk, GatedRmsGrad,
    GdnDecayGrad, GdnForward, GdnGrad, GdnInputs, LinearCe, MuonNs5Config, Numerics, OjasError,
    OptimizerKind, PerHeadGateGrad, Tensor, ValueResidualGrad,
};

/// The per-call cancel check: `Err` carries the cancel's own message
/// (`"cancelled: Explicit"`, which gusset maps to `context.Canceled`).
pub type Check = Box<dyn FnMut() -> Result<(), String> + Send>;

/// A test fault: the op it fires on and the error it returns.
#[cfg(test)]
type Fault = (&'static str, fn() -> OjasError);

#[derive(Default)]
struct SlotState {
    check: Option<Check>,
    tripped: Option<String>,
    #[cfg(test)]
    fault: Option<Fault>,
}

/// One per session, shared by every clone of its [`Gated`] backend.
#[derive(Default)]
pub struct CancelSlot {
    state: Mutex<SlotState>,
}

impl CancelSlot {
    fn lock(&self) -> std::sync::MutexGuard<'_, SlotState> {
        // The check is caller code; a panic in it unwinds out of the job and
        // gusset poisons the handle. The slot's own fields stay consistent.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Install `check` for one call. The call ends when the guard drops.
    pub fn arm(self: &Arc<Self>, check: Check) -> Armed {
        let mut state = self.lock();
        state.check = Some(check);
        state.tripped = None;
        Armed {
            slot: Arc::clone(self),
        }
    }

    /// Run the armed check. A failure is recorded and stays failed for the
    /// rest of the call.
    fn poll(&self) -> Result<(), ()> {
        let mut state = self.lock();
        if state.tripped.is_some() {
            return Err(());
        }
        let Some(check) = state.check.as_mut() else {
            return Ok(());
        };
        match check() {
            Ok(()) => Ok(()),
            Err(msg) => {
                state.tripped = Some(msg);
                Err(())
            }
        }
    }

    /// A test fault: the next call of `op` fails with `make()`, once.
    #[cfg(test)]
    pub fn inject(&self, op: &'static str, make: fn() -> OjasError) {
        self.lock().fault = Some((op, make));
    }

    #[cfg(test)]
    fn injected(&self, op: &'static str) -> Result<(), OjasError> {
        let mut state = self.lock();
        match state.fault {
            Some((at, make)) if at == op => {
                state.fault = None;
                Err(make())
            }
            _ => Ok(()),
        }
    }
}

/// An armed slot. Dropping it disarms the check and forgets the cancel.
pub struct Armed {
    slot: Arc<CancelSlot>,
}

impl Armed {
    /// The cancel message, when a poll in this call failed.
    pub fn tripped(&self) -> Option<String> {
        self.slot.lock().tripped.clone()
    }

    /// `err` as the call reports it: the recorded cancel when one tripped
    /// (the refused op's error is the gate's own), otherwise `err` with its
    /// kind ([`crate::ojas_error`]).
    pub fn report(&self, context: &str, err: &OjasError) -> String {
        match self.tripped() {
            Some(msg) => msg,
            None => crate::ojas_error(context, err),
        }
    }
}

impl Drop for Armed {
    fn drop(&mut self) {
        let mut state = self.slot.lock();
        state.check = None;
        state.tripped = None;
    }
}

/// `B` with a cancel poll before every op that may fail without a commit.
#[derive(Clone)]
pub struct Gated<B> {
    inner: B,
    slot: Arc<CancelSlot>,
}

impl<B: Backend> Gated<B> {
    pub fn new(inner: B, slot: Arc<CancelSlot>) -> Self {
        Self { inner, slot }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// `op` may be refused: by a test fault, then, when `cancellable`, by
    /// the armed cancel check.
    fn enter(&self, op: &'static str, cancellable: bool) -> Result<(), OjasError> {
        #[cfg(test)]
        self.slot.injected(op)?;
        if cancellable && self.slot.poll().is_err() {
            return Err(OjasError::Backend {
                id: self.inner.id(),
                detail: format!("{op}: refused after the call was cancelled"),
            });
        }
        Ok(())
    }
}

impl<B: Backend> Backend for Gated<B> {
    fn id(&self) -> BackendId {
        self.inner.id()
    }
    fn budget(&self) -> &Budget {
        self.inner.budget()
    }
    fn numerics(&self) -> Numerics {
        self.inner.numerics()
    }
    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("upload", true)?;
        self.inner.upload(tensor)
    }
    /// Never polls: the trainer's one readback follows its optimizer.
    fn download(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("download", false)?;
        self.inner.download(tensor)
    }
    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        self.enter("permute", true)?;
        self.inner.permute(input, dims)
    }
    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("embedding_forward", true)?;
        self.inner.embedding_forward(table, token_ids)
    }
    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("embedding_backward", true)?;
        self.inner.embedding_backward(table, token_ids, grad_output)
    }
    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("linear_forward", true)?;
        self.inner.linear_forward(input, weight)
    }
    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.enter("linear_backward", true)?;
        self.inner.linear_backward(input, weight, grad_output)
    }
    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        self.enter("rms_norm_forward", true)?;
        self.inner.rms_norm_forward(input, weight, eps)
    }
    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.enter("rms_norm_backward", true)?;
        self.inner
            .rms_norm_backward(input, weight, grad_output, eps)
    }
    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("rope_half_split_forward", true)?;
        self.inner.rope_half_split_forward(x, cos, sin)
    }
    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("rope_half_split_backward", true)?;
        self.inner.rope_half_split_backward(grad_output, cos, sin)
    }
    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.enter("rms_qk_norm_forward", true)?;
        self.inner
            .rms_qk_norm_forward(q, k, q_weight, k_weight, eps)
    }
    #[allow(clippy::too_many_arguments)]
    fn rms_qk_norm_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        grad_q: &Tensor,
        grad_k: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        self.enter("rms_qk_norm_backward", true)?;
        self.inner
            .rms_qk_norm_backward(q, k, q_weight, k_weight, grad_q, grad_k, eps)
    }
    fn causal_sdpa_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.enter("causal_sdpa_forward", true)?;
        self.inner.causal_sdpa_forward(q, k, v, window)
    }
    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        output: &Tensor,
        lse: &Tensor,
        grad_output: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        self.enter("causal_sdpa_backward", true)?;
        self.inner
            .causal_sdpa_backward(q, k, v, output, lse, grad_output, window)
    }
    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("per_head_sigmoid_gate_forward", true)?;
        self.inner
            .per_head_sigmoid_gate_forward(input, weight, bias, attn_out)
    }
    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        self.enter("per_head_sigmoid_gate_backward", true)?;
        self.inner
            .per_head_sigmoid_gate_backward(input, weight, bias, attn_out, grad_output)
    }
    fn per_head_sigmoid_gate_forward_saving(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>), OjasError> {
        self.enter("per_head_sigmoid_gate_forward", true)?;
        self.inner
            .per_head_sigmoid_gate_forward_saving(input, weight, bias, attn_out)
    }
    fn per_head_sigmoid_gate_backward_saved(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
        scales: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        self.enter("per_head_sigmoid_gate_backward", true)?;
        self.inner.per_head_sigmoid_gate_backward_saved(
            input,
            weight,
            bias,
            attn_out,
            grad_output,
            scales,
        )
    }
    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("value_residual_blend_forward", true)?;
        self.inner
            .value_residual_blend_forward(value, value0, lambda)
    }
    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        self.enter("value_residual_blend_backward", true)?;
        self.inner
            .value_residual_blend_backward(value, value0, lambda, grad_output)
    }
    fn chunked_gdn_forward(&self, inputs: GdnInputs<'_>) -> Result<GdnForward, OjasError> {
        self.enter("chunked_gdn_forward", true)?;
        self.inner.chunked_gdn_forward(inputs)
    }
    fn chunked_gdn_backward(
        &self,
        inputs: GdnInputs<'_>,
        checkpoints: &Tensor,
        grad_output: &Tensor,
        grad_final_state: Option<&Tensor>,
    ) -> Result<GdnGrad, OjasError> {
        self.enter("chunked_gdn_backward", true)?;
        self.inner
            .chunked_gdn_backward(inputs, checkpoints, grad_output, grad_final_state)
    }
    fn causal_conv1d_silu_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("causal_conv1d_silu_forward", true)?;
        self.inner.causal_conv1d_silu_forward(input, weight)
    }
    fn causal_conv1d_silu_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.enter("causal_conv1d_silu_backward", true)?;
        self.inner
            .causal_conv1d_silu_backward(input, weight, grad_output)
    }
    fn gated_rms_norm_forward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        self.enter("gated_rms_norm_forward", true)?;
        self.inner.gated_rms_norm_forward(input, gate, weight, eps)
    }
    fn gated_rms_norm_backward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<GatedRmsGrad, OjasError> {
        self.enter("gated_rms_norm_backward", true)?;
        self.inner
            .gated_rms_norm_backward(input, gate, weight, grad_output, eps)
    }
    fn rope_partial_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("rope_partial_forward", true)?;
        self.inner.rope_partial_forward(x, cos, sin)
    }
    fn rope_partial_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("rope_partial_backward", true)?;
        self.inner.rope_partial_backward(grad_output, cos, sin)
    }
    fn sigmoid_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("sigmoid_forward", true)?;
        self.inner.sigmoid_forward(input)
    }
    fn sigmoid_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("sigmoid_backward", true)?;
        self.inner.sigmoid_backward(input, grad_output)
    }
    fn gdn_log_decay_forward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.enter("gdn_log_decay_forward", true)?;
        self.inner.gdn_log_decay_forward(a, a_log, dt_bias)
    }
    fn gdn_log_decay_backward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        grad_output: &Tensor,
    ) -> Result<GdnDecayGrad, OjasError> {
        self.enter("gdn_log_decay_backward", true)?;
        self.inner
            .gdn_log_decay_backward(a, a_log, dt_bias, grad_output)
    }
    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("silu_forward", true)?;
        self.inner.silu_forward(input)
    }
    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("silu_backward", true)?;
        self.inner.silu_backward(input, grad_output)
    }
    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("mul_forward", true)?;
        self.inner.mul_forward(a, b)
    }
    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.enter("mul_backward", true)?;
        self.inner.mul_backward(a, b, grad_output)
    }
    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("residual_add_forward", true)?;
        self.inner.residual_add_forward(x, y)
    }
    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.enter("residual_add_backward", true)?;
        self.inner.residual_add_backward(x, y, grad_output)
    }
    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        self.enter("cross_entropy_mean_forward", true)?;
        self.inner
            .cross_entropy_mean_forward(logits, targets, ignore_index)
    }
    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        self.enter("cross_entropy_mean_backward", true)?;
        self.inner
            .cross_entropy_mean_backward(logits, targets, ignore_index)
    }
    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        self.enter("clip_grad_norm", true)?;
        self.inner.clip_grad_norm(grads, max_norm)
    }
    /// Never polls: the trainer's optimizer phase.
    fn adamw_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        moment1: &mut Tensor,
        moment2: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError> {
        self.enter("adamw_step", false)?;
        self.inner
            .adamw_step(param, grad, moment1, moment2, step, config)
    }
    /// Never polls: the trainer's optimizer phase.
    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        self.enter("muon_ns5_step", false)?;
        self.inner.muon_ns5_step(param, grad, momentum, config)
    }
    /// A size query, no op: not gated.
    fn optimizer_scratch_bytes(
        &self,
        kind: OptimizerKind,
        rows: usize,
        cols: usize,
    ) -> Result<Option<u64>, OjasError> {
        self.inner.optimizer_scratch_bytes(kind, rows, cols)
    }
    /// Never polls: the trainer's finish.
    fn sync(&self) -> Result<(), OjasError> {
        self.enter("sync", false)?;
        self.inner.sync()
    }
    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        self.enter("accumulate_grad", true)?;
        self.inner.accumulate_grad(acc, grad)
    }
    /// The tape's loss seed, applied during the backward (phase 2): polls.
    fn scale_grad(&self, grad: &mut Tensor, scale: f32) -> Result<(), OjasError> {
        self.enter("scale_grad", true)?;
        self.inner.scale_grad(grad, scale)
    }
    fn linear_cross_entropy_mean(
        &self,
        input: &Tensor,
        weight: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
        chunk: CeChunk,
        want_grad: bool,
    ) -> Result<LinearCe, OjasError> {
        self.enter("linear_cross_entropy_mean", true)?;
        self.inner
            .linear_cross_entropy_mean(input, weight, targets, ignore_index, chunk, want_grad)
    }
    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        self.enter("cached_attention_forward", true)?;
        self.inner
            .cached_attention_forward(q, k_cache, v_cache, kv_len)
    }
    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        self.enter("kv_cache_write", true)?;
        self.inner.kv_cache_write(cache, src, at)
    }
    fn argmax_rows(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("argmax_rows", true)?;
        self.inner.argmax_rows(x)
    }
    fn cast_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("cast_bf16", true)?;
        self.inner.cast_bf16(tensor)
    }
    fn bf16_operands(&self) -> bool {
        self.inner.bf16_operands()
    }
    fn to_bf16(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("to_bf16", true)?;
        self.inner.to_bf16(tensor)
    }
    fn to_f32(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        self.enter("to_f32", true)?;
        self.inner.to_f32(tensor)
    }
    fn autocast_region(&self, mode: AutocastMode) -> Result<AutocastGuard, OjasError> {
        self.enter("autocast_region", true)?;
        self.inner.autocast_region(mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_cpu::CpuBackend;

    fn gated(slot: &Arc<CancelSlot>) -> Gated<CpuBackend> {
        Gated::new(CpuBackend::new(Budget::new(1 << 20)), Arc::clone(slot))
    }

    #[test]
    fn a_cancel_refuses_polling_ops_and_never_the_commit_ops() {
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let budget = Budget::new(1 << 20);
        let x = Tensor::from_f32(&[1.0, 2.0], &[1, 2], &budget).unwrap();
        let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        let err = g.linear_forward(&x, &x).unwrap_err();
        assert!(matches!(err, OjasError::Backend { .. }), "{err}");
        assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
        assert_eq!(armed.report("op", &err), "cancelled: Explicit");
        // The commit ops still run once the call is cancelled.
        g.sync().unwrap();
        g.download(&x).unwrap();
        let mut p = Tensor::from_f32(&[0.5], &[1], &budget).unwrap();
        let grad = Tensor::from_f32(&[0.1], &[1], &budget).unwrap();
        let mut m = Tensor::from_f32(&[0.0], &[1], &budget).unwrap();
        let mut v = Tensor::from_f32(&[0.0], &[1], &budget).unwrap();
        g.adamw_step(
            &mut p,
            &grad,
            &mut m,
            &mut v,
            0,
            AdamWConfig::nanolab(1e-3, 0.0),
        )
        .unwrap();
        drop(armed);
        // Disarmed: the next call's ops run and nothing is remembered.
        g.linear_forward(&x, &x).unwrap();
        let armed = slot.arm(Box::new(|| Ok(())));
        assert_eq!(armed.tripped(), None);
        g.linear_forward(&x, &x).unwrap();
    }

    /// The gated delta rule goes through the gate to the inner backend
    /// (not the trait default's refusal), and a cancel refuses it, as it
    /// does every polling op.
    #[test]
    fn the_gated_delta_rule_reaches_the_inner_backend_and_polls() {
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let b = Budget::new(1 << 20);
        let f = |shape: &[usize], v: f32| {
            Tensor::from_f32(&vec![v; shape.iter().product()], shape, &b).unwrap()
        };
        let (q, v, gate, beta) = (
            f(&[1, 3, 1, 2], 0.5),
            f(&[1, 3, 1, 2], 0.25),
            f(&[1, 3, 1], -0.5),
            f(&[1, 3, 1], 0.5),
        );
        let x = GdnInputs {
            q: &q,
            k: &q,
            v: &v,
            g: &gate,
            beta: &beta,
            initial_state: None,
        };
        let fwd = g.chunked_gdn_forward(x).unwrap();
        let want = g.inner.chunked_gdn_forward(x).unwrap();
        assert_eq!(
            fwd.output.to_f32_vec().unwrap(),
            want.output.to_f32_vec().unwrap()
        );
        let grad = g
            .chunked_gdn_backward(x, &fwd.checkpoints, &f(&[1, 3, 1, 2], 1.0), None)
            .unwrap();
        assert!(grad.initial_state.is_none());
        let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        assert!(g.chunked_gdn_forward(x).is_err());
        assert!(g
            .chunked_gdn_backward(x, &fwd.checkpoints, &f(&[1, 3, 1, 2], 1.0), None)
            .is_err());
        assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
    }

    /// `scale_grad` reaches the inner backend: the CPU scales a uniquely
    /// owned gradient in place with no charge, where the trait default
    /// would build and charge a broadcast operand and a new product. A
    /// cancel refuses it, as it does every polling op.
    #[test]
    fn scale_grad_reaches_the_inner_backend_in_place_and_polls() {
        let slot = Arc::new(CancelSlot::default());
        let g = gated(&slot);
        let mut grad = Tensor::from_f32(&[1.0, -2.0, 4.0, 0.5], &[2, 2], g.budget()).unwrap();
        let live = g.budget().live_bytes().unwrap();
        g.budget().reset_peak();
        g.scale_grad(&mut grad, 0.25).unwrap();
        assert_eq!(grad.to_f32_vec().unwrap(), vec![0.25, -0.5, 1.0, 0.125]);
        assert_eq!(g.budget().peak_bytes(), live, "scaled in place, no charge");
        let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
        assert!(g.scale_grad(&mut grad, 0.5).is_err());
        assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
        assert_eq!(grad.to_f32_vec().unwrap(), vec![0.25, -0.5, 1.0, 0.125]);
    }

    #[test]
    fn an_error_with_no_cancel_keeps_its_kind() {
        let slot = Arc::new(CancelSlot::default());
        let armed = slot.arm(Box::new(|| Ok(())));
        let err = OjasError::NonFinite { op: "x" };
        assert!(armed.report("ctx", &err).starts_with("ojas:E_NONFINITE:"));
    }
}
