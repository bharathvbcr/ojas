use std::sync::Arc;

use ojas_core::{
    causal_conv1d_silu_backward_dims, causal_conv1d_silu_forward_dims, clip_scale,
    gated_rms_norm_backward_dims, gated_rms_norm_forward_dims, gdn_log_decay_backward_dims,
    gdn_log_decay_forward_dims, rope_partial_backward_dims, rope_partial_forward_dims,
    sigmoid_backward_dims, sigmoid_forward_dims, AdamWConfig, Backend, BackendId, Budget,
    GatedRmsGrad, GdnDecayGrad, GdnForward, GdnGrad, GdnInputs, MuonNs5Config, Numerics, OjasError,
    OptimizerKind, PerHeadGateGrad, Tensor, ValueResidualGrad,
};
// Every op below runs its `ojas_core::shapes` validator before it checks
// values, copies an input or charges the budget (docs/shape-contract.md).
use ojas_core::{
    causal_sdpa_backward_dims, causal_sdpa_forward_dims, chunked_gdn_backward_dims,
    chunked_gdn_forward_dims, linear_backward_dims, linear_forward_dims, mul_backward_dims,
    mul_forward_dims, per_head_sigmoid_gate_backward_dims, per_head_sigmoid_gate_forward_dims,
    residual_add_backward_dims, residual_add_forward_dims, rms_norm_backward_dims,
    rms_norm_forward_dims, rms_qk_norm_backward_dims, rms_qk_norm_forward_dims,
    rope_half_split_backward_dims, rope_half_split_forward_dims, silu_backward_dims,
    silu_forward_dims, value_residual_blend_backward_dims, value_residual_blend_forward_dims,
    RmsDims,
};

use crate::attn::{causal_sdpa_backward, causal_sdpa_forward, Dims as SdpaKernelDims};
use crate::gdn::{self, GradsOut, Operands as GdnOperands};
use crate::gemm::whole_call;
use crate::hybrid::{
    conv1d_silu_backward, conv1d_silu_forward, gated_rms_backward, gated_rms_forward,
    gdn_log_decay_backward, gdn_log_decay_forward, rope_partial, sigmoid_backward, sigmoid_forward,
    Turn,
};
use crate::layout::permute;
use crate::linalg::{linear_backward, linear_forward};
use crate::norm::{rms_backward, rms_forward, rope_backward, rope_forward};
use crate::optim::{
    adamw_in_place, check_muon, grad_norm, muon_ns5, muon_scratch, scale_in_place, AdamCoeffs,
};
use crate::pointwise::{
    add_backward, add_forward, ce_input, cross_entropy_grad, embedding_backward, embedding_forward,
    gate_backward, gate_forward, mul_backward, mul_forward, silu_backward, silu_forward,
    value_residual_backward, value_residual_forward,
};
use crate::pool::{Exec, Pool};
use crate::validate::{
    alloc_f32, check_f32s, f32_checked, f32_layouts, f32_operands, f32_values, fill_out, fill_outs,
    headroom, matmul_operands, nonfinite_first, payload_bytes, product, scanned_f32,
    trusted_finite_f32, u32_values, Wide,
};
use ojas_core::{
    adamw_step_dims, clip_grad_norm_dims, cross_entropy_mean_backward_dims,
    cross_entropy_mean_forward_dims, embedding_backward_dims, embedding_forward_dims,
    muon_ns5_step_dims, MuonDims, Scratch,
};

/// CPU backend. [`CpuBackend::new`] uses one thread.
/// [`CpuBackend::with_threads`] gives the backend a persistent worker pool
/// that its clones share; large ops split independent outputs across it and
/// never split a reduction.
///
/// [`Numerics::Fast`] is the default: `mul_add`, larger GEMM blocks, blocked
/// attention above 256 positions, and on macOS a GEMM of at least
/// [`crate::FAST_WHOLE_CALL_MACS`] multiply-adds (2¹³) is one Accelerate call that ignores the pool and threads on
/// its own, as PyTorch's CPU `linear` does. Those Accelerate bits are whatever
/// Apple's unspecified order gives on this machine and OS build. Every other
/// Fast op keeps bits that do not depend on the thread count.
///
/// `.with_numerics(Numerics::Exact)` selects the reference contract: every
/// reduction ascends from index 0 with separate multiply and add, and the bits
/// do not depend on the thread count, the tiling or the machine. Golden tests,
/// gradient checks and bit-identity checks select it explicitly.
#[derive(Clone, Debug)]
pub struct CpuBackend {
    budget: Budget,
    pool: Arc<Pool>,
    numerics: Numerics,
}

impl CpuBackend {
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            pool: Arc::new(Pool::serial()),
            numerics: Numerics::Fast,
        }
    }

    /// `threads == 0` and `threads > 1024` are refused. Workers are spawned
    /// on the first op large enough to use them; small ops stay on the caller.
    pub fn with_threads(budget: Budget, threads: usize) -> Result<Self, OjasError> {
        Ok(Self {
            budget,
            pool: Arc::new(Pool::new(threads)?),
            numerics: Numerics::Fast,
        })
    }

    /// Same budget and pool, different arithmetic contract.
    pub fn with_numerics(mut self, numerics: Numerics) -> Self {
        self.numerics = numerics;
        self
    }

    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Spawn the pool's workers now rather than on the first op large
    /// enough to split. A long run calls this before its first step, so a
    /// failure to spawn (a thread or memory limit) refuses the setup instead
    /// of a step part-way through; the failure is kept, and every later op
    /// that would split returns it too. A serial backend has nothing to
    /// spawn. Idempotent.
    pub fn start_workers(&self) -> Result<(), OjasError> {
        self.pool.start()
    }

    /// Called between whole output rows or attention heads.
    ///
    /// On macOS a Fast product that runs as one Accelerate call (see the
    /// type docs) calls the hook once before that call; the call itself is
    /// not interrupted. `linear_backward` makes two such calls and checks
    /// before each.
    ///
    /// The default hook returns `Ok(())`. Replacing it does not change tile
    /// or reduction order. Clones of this backend share the hook.
    pub fn set_cancel<F>(&self, hook: F)
    where
        F: Fn() -> Result<(), OjasError> + Send + Sync + 'static,
    {
        self.pool.set_cancel(std::sync::Arc::new(hook));
    }

    fn exec(&self) -> Exec<'_> {
        Exec {
            pool: &self.pool,
            numerics: self.numerics,
        }
    }

    /// Bytes `muon_ns5_step` reserves for a `[rows, cols]` matrix: the most
    /// heap scratch its phases hold at once (optim.rs `muon_scratch`). The
    /// step and [`Backend::optimizer_scratch_bytes`] both read this.
    fn muon_headroom(&self, op: &'static str, rows: usize, cols: usize) -> Result<u64, OjasError> {
        payload_bytes(op, muon_scratch(op, self.exec(), rows, cols)?)
    }

    /// Causal SDPA forward after its validator: the output and the
    /// `[B, H, T]` row log-sum-exp.
    fn sdpa_forward(
        &self,
        op: &'static str,
        dims: SdpaKernelDims,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let exec = self.exec();
        let wide = matmul_operands(op, exec, &self.budget, [q, k, v])?;
        let qkv = wide.each_ref().map(Wide::values);
        let lse_shape = &q.shape()[..q.shape().len().saturating_sub(1)];
        let [out, lse] = fill_outs(op, &self.budget, exec, [q.shape(), lse_shape], |outs| {
            causal_sdpa_forward(op, &self.budget, exec, qkv, dims, outs)
        })?;
        Ok((out, lse))
    }

    /// Causal SDPA backward after its validator, from the forward's output
    /// and log-sum-exp: `[q, k, v, output, lse, grad_output]`.
    fn sdpa_backward(
        &self,
        op: &'static str,
        dims: SdpaKernelDims,
        operands: [&Tensor; 6],
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        let exec = self.exec();
        let wide = matmul_operands(op, exec, &self.budget, operands)?;
        let ins = wide.each_ref().map(Wide::values);
        let [q, k, v, ..] = operands;
        let shapes = [q.shape(), k.shape(), v.shape()];
        let [grad_q, grad_k, grad_v] = fill_outs(op, &self.budget, exec, shapes, |grads| {
            causal_sdpa_backward(op, &self.budget, exec, ins, dims, grads)
        })?;
        Ok((grad_q, grad_k, grad_v))
    }

    /// RMSNorm forward after its validator: the op itself, and each half of
    /// `rms_qk_norm_forward` once both halves are validated.
    fn rms_norm_fwd(
        &self,
        dims: RmsDims,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rms_norm_forward";
        let exec = self.exec();
        let [x, w] = f32_operands(OP, exec, [input, weight])?;
        rms_forward(OP, &self.budget, exec, x, w, dims, eps, input.shape())
    }

    /// RMSNorm backward after its validator (see [`Self::rms_norm_fwd`]).
    fn rms_norm_bwd(
        &self,
        dims: RmsDims,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [input, weight, grad_output])?;
        let shapes = [input.shape(), weight.shape()];
        let [grad_x, grad_w] = fill_outs(OP, &self.budget, self.exec(), shapes, |grads| {
            rms_backward(OP, &self.budget, exec, ins, dims, eps, grads)
        })?;
        Ok((grad_x, grad_w))
    }

    #[allow(clippy::too_many_arguments)]
    fn gate_forward_parts(
        &self,
        exec: Exec<'_>,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        attn: &[f32],
        attn_tensor: &Tensor,
        dims: ojas_core::GateDims,
        out_shape: &[usize],
        keep_scales: bool,
    ) -> Result<(Tensor, Option<Tensor>), OjasError> {
        gate_forward(
            "per_head_sigmoid_gate_forward",
            &self.budget,
            exec,
            input,
            weight,
            bias,
            attn,
            Some(attn_tensor),
            dims,
            out_shape,
            keep_scales,
        )
    }

    /// Gate backward. On macOS Fast, `grad_y` (`v`) is refused here when it
    /// is not finite and the sigmoid (`g`) is clamped into `[0, 1]`, so
    /// `grad_attn = v * g` is not scanned. `grad_bias` and the two GEMM
    /// gradients are scanned either way. Exact scans every gradient.
    fn finish_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
        scales: Option<&[f32]>,
    ) -> Result<PerHeadGateGrad, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_backward";
        let dims = per_head_sigmoid_gate_backward_dims(input, weight, bias, attn_out, grad_output)?;
        let exec = self.exec();
        // `grad_output` is `v` in `grad_attn = v * g`. A non-finite value is
        // refused here, before any gradient is charged.
        let ins = f32_operands(OP, exec, [input, weight, bias, attn_out, grad_output])?;
        #[cfg(target_os = "macos")]
        if exec.numerics == Numerics::Fast {
            return self.publish_gate_grads(exec, ins, scales, dims, input, weight, bias, attn_out);
        }
        let shapes = [
            input.shape(),
            weight.shape(),
            bias.shape(),
            attn_out.shape(),
        ];
        let [gx, gw, gb, ga] = fill_outs(OP, &self.budget, self.exec(), shapes, |grads| {
            let proven = gate_backward(OP, &self.budget, exec, ins, dims, scales, grads)?;
            debug_assert!(!proven, "{OP}: exact path skipped the attn scan");
            let _ = proven;
            Ok(())
        })?;
        Ok(PerHeadGateGrad {
            input: gx,
            weight: gw,
            bias: gb,
            attn_out: ga,
        })
    }

    /// Charge the four gradients, run the backward, scan the GEMM outputs
    /// and `grad_bias`, and record `grad_attn` finite without a second pass
    /// when its scales were clamped into `[0, 1]`. A refusal drops every
    /// scratch.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn publish_gate_grads(
        &self,
        exec: Exec<'_>,
        ins: [&[f32]; 5],
        scales: Option<&[f32]>,
        dims: ojas_core::GateDims,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_backward";
        let mut gx = Scratch::<f32>::try_alloc(product(OP, input.shape())?, &self.budget)?;
        let mut gw = Scratch::<f32>::try_alloc(product(OP, weight.shape())?, &self.budget)?;
        let mut gb = Scratch::<f32>::try_alloc(product(OP, bias.shape())?, &self.budget)?;
        let mut ga = Scratch::<f32>::try_alloc(product(OP, attn_out.shape())?, &self.budget)?;
        let proven = gate_backward(
            OP,
            &self.budget,
            exec,
            ins,
            dims,
            scales,
            [
                gx.as_mut_slice(),
                gw.as_mut_slice(),
                gb.as_mut_slice(),
                ga.as_mut_slice(),
            ],
        )?;
        let input_g = scanned_f32(OP, exec, gx, input.shape())?;
        let weight_g = scanned_f32(OP, exec, gw, weight.shape())?;
        let bias_g = scanned_f32(OP, exec, gb, bias.shape())?;
        debug_assert!(
            proven,
            "{OP}: skipped the attn scan without a clamped scale"
        );
        let attn_g = if proven {
            trusted_finite_f32(OP, ga, attn_out.shape())?
        } else {
            scanned_f32(OP, exec, ga, attn_out.shape())?
        };
        Ok(PerHeadGateGrad {
            input: input_g,
            weight: weight_g,
            bias: bias_g,
            attn_out: attn_g,
        })
    }
}

/// The gated delta rule's operands checked and scanned in argument order
/// (`q, k, v, g, beta, initial_state`), read in place.
fn gdn_operands<'t>(
    op: &'static str,
    exec: Exec<'_>,
    inputs: GdnInputs<'t>,
) -> Result<GdnOperands<'t>, OjasError> {
    let mut ts = vec![inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta];
    ts.extend(inputs.initial_state);
    let vals = check_f32s(op, exec, &ts)?;
    Ok(GdnOperands {
        q: vals[0],
        k: vals[1],
        v: vals[2],
        g: vals[3],
        beta: vals[4],
        s0: vals.get(5).copied(),
    })
}

impl Backend for CpuBackend {
    fn id(&self) -> BackendId {
        BackendId::Cpu
    }

    fn budget(&self) -> &Budget {
        &self.budget
    }

    fn numerics(&self) -> Numerics {
        self.numerics
    }

    /// A `Bf16` operand is widened into charged `f32` scratch for the
    /// duration of the op ([`crate::validate::matmul_operands`]); the kernels
    /// and their bits are the `F32` ones.
    fn bf16_operands(&self) -> bool {
        true
    }

    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        permute("permute", &self.budget, self.exec(), input, dims)
    }

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_forward";
        let dims = embedding_forward_dims(table, token_ids)?;
        // The whole table is scanned, in place and in parallel, as Metal and
        // wgpu scan it: a NaN in any row is refused, read or not. Nothing is
        // copied; the output is the only charge.
        let table_w = f32_checked(OP, self.exec(), table)?;
        let ids = u32_values(OP, token_ids)?;
        embedding_forward(OP, &self.budget, table_w, ids, &dims)
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        let dims = embedding_backward_dims(table, token_ids, grad_output)?;
        // Only the table's shape is used, but its values are scanned as
        // every backend scans them. Operands are checked in argument order;
        // nothing is copied, and the table-shaped gradient is the only
        // charge.
        f32_checked(OP, self.exec(), table)?;
        let ids = u32_values(OP, token_ids)?;
        let grad = f32_checked(OP, self.exec(), grad_output)?;
        embedding_backward(OP, &self.budget, ids, grad, &dims)
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let dims = linear_forward_dims(input, weight)?;
        let exec = self.exec();
        let wide = matmul_operands(OP, exec, &self.budget, [input, weight])?;
        let [x, w] = wide.each_ref().map(Wide::values);
        fill_out(OP, &self.budget, self.exec(), &dims.out_shape, |y| {
            linear_forward(OP, &self.budget, exec, x, w, &dims, y)
        })
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let dims = linear_backward_dims(input, weight, grad_output)?;
        let exec = self.exec();
        let wide = matmul_operands(OP, exec, &self.budget, [input, weight, grad_output])?;
        let ins = wide.each_ref().map(Wide::values);
        // One forward row, Fast whole call: `rank1_weight_grad` checks every
        // weight-gradient element as it stores `fma(g, x, +0.0)`. That buffer
        // is not scanned again. `grad_x` is still the Accelerate product and
        // is scanned.
        if dims.rows == 1
            && whole_call(
                self.numerics,
                dims.out_features,
                dims.rows,
                dims.in_features,
            )
        {
            let gx_len = product(OP, &[dims.rows, dims.in_features])?;
            let gw_len = product(OP, &[dims.out_features, dims.in_features])?;
            let mut grad_x = Scratch::<f32>::try_alloc(gx_len, &self.budget)?;
            let mut grad_w = Scratch::<f32>::try_alloc(gw_len, &self.budget)?;
            let checked = linear_backward(
                OP,
                &self.budget,
                exec,
                ins,
                &dims,
                [grad_x.as_mut_slice(), grad_w.as_mut_slice()],
            )?;
            let grad_x = scanned_f32(OP, exec, grad_x, input.shape())?;
            let grad_w = if checked {
                trusted_finite_f32(OP, grad_w, weight.shape())?
            } else {
                scanned_f32(OP, exec, grad_w, weight.shape())?
            };
            return Ok((grad_x, grad_w));
        }
        let shapes = [input.shape(), weight.shape()];
        let [grad_x, grad_w] = fill_outs(OP, &self.budget, self.exec(), shapes, |grads| {
            let _ = linear_backward(OP, &self.budget, exec, ins, &dims, grads)?;
            Ok(())
        })?;
        Ok((grad_x, grad_w))
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        let dims = rms_norm_forward_dims(input, weight, eps)?;
        self.rms_norm_fwd(dims, input, weight, eps)
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let dims = rms_norm_backward_dims(input, weight, grad_output, eps)?;
        self.rms_norm_bwd(dims, input, weight, grad_output, eps)
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_forward";
        let dims = rope_half_split_forward_dims(x, cos, sin)?;
        let exec = self.exec();
        let [xv, cv, sv] = f32_operands(OP, exec, [x, cos, sin])?;
        rope_forward(OP, &self.budget, exec, xv, cv, sv, dims, x.shape())
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_backward";
        let dims = rope_half_split_backward_dims(grad_output, cos, sin)?;
        let exec = self.exec();
        let [gy, cv, sv] = f32_operands(OP, exec, [grad_output, cos, sin])?;
        rope_backward(
            OP,
            &self.budget,
            exec,
            gy,
            cv,
            sv,
            dims,
            grad_output.shape(),
        )
    }

    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        // Both pairs are validated before either norm charges anything.
        let (qd, kd) = rms_qk_norm_forward_dims(q, k, q_weight, k_weight, eps)?;
        let qn = self.rms_norm_fwd(qd, q, q_weight, eps)?;
        let kn = self.rms_norm_fwd(kd, k, k_weight, eps)?;
        Ok((qn, kn))
    }

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
        let (qd, kd) = rms_qk_norm_backward_dims(q, k, q_weight, k_weight, grad_q, grad_k, eps)?;
        let (gq, gqw) = self.rms_norm_bwd(qd, q, q_weight, grad_q, eps)?;
        let (gk, gkw) = self.rms_norm_bwd(kd, k, k_weight, grad_k, eps)?;
        Ok((gq, gk, gqw, gkw))
    }

    fn causal_sdpa_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        window: Option<usize>,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_forward";
        let dims = causal_sdpa_forward_dims(q, k, v, window)?;
        let dims = SdpaKernelDims::new(OP, dims, window)?;
        self.sdpa_forward(OP, dims, q, k, v)
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
        const OP: &str = "causal_sdpa_backward";
        let dims = causal_sdpa_backward_dims(q, k, v, output, lse, grad_output, window)?;
        let dims = SdpaKernelDims::new(OP, dims, window)?;
        self.sdpa_backward(OP, dims, [q, k, v, output, lse, grad_output])
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_forward";
        let dims = per_head_sigmoid_gate_forward_dims(input, weight, bias, attn_out)?;
        let exec = self.exec();
        let [x, w, b, attn] = f32_operands(OP, exec, [input, weight, bias, attn_out])?;
        self.gate_forward_parts(exec, x, w, b, attn, attn_out, dims, attn_out.shape(), false)
            .map(|(y, _)| y)
    }

    fn per_head_sigmoid_gate_forward_saving(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>), OjasError> {
        const OP: &str = "per_head_sigmoid_gate_forward";
        let dims = per_head_sigmoid_gate_forward_dims(input, weight, bias, attn_out)?;
        let exec = self.exec();
        let [x, w, b, attn] = f32_operands(OP, exec, [input, weight, bias, attn_out])?;
        // Exact does not keep the scale. Its backward recomputes the logits.
        self.gate_forward_parts(
            exec,
            x,
            w,
            b,
            attn,
            attn_out,
            dims,
            attn_out.shape(),
            exec.numerics == Numerics::Fast,
        )
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        self.finish_gate_backward(input, weight, bias, attn_out, grad_output, None)
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
        // Exact recomputes. The saved scale is not read.
        if self.numerics() == Numerics::Exact {
            return self.per_head_sigmoid_gate_backward(input, weight, bias, attn_out, grad_output);
        }
        const OP: &str = "per_head_sigmoid_gate_backward";
        let saved = f32_checked(OP, self.exec(), scales)?;
        self.finish_gate_backward(input, weight, bias, attn_out, grad_output, Some(saved))
    }

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "value_residual_blend_forward";
        value_residual_blend_forward_dims(value, value0, lambda)?;
        let [v, v0, lam] = f32_operands(OP, self.exec(), [value, value0, lambda])?;
        value_residual_forward(OP, &self.budget, self.exec(), v, v0, lam[0], value.shape())
    }

    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        const OP: &str = "value_residual_blend_backward";
        value_residual_blend_backward_dims(value, value0, lambda, grad_output)?;
        let exec = self.exec();
        let [v, v0, lam, gy] = f32_operands(OP, exec, [value, value0, lambda, grad_output])?;
        let shapes = [value.shape(), value0.shape(), lambda.shape()];
        let [gv, gv0, gl] = fill_outs(OP, &self.budget, self.exec(), shapes, |[gv, gv0, gl]| {
            gl.fill(value_residual_backward(
                OP,
                &self.budget,
                exec,
                [v, v0, gy],
                lam[0],
                [gv, gv0],
            )?);
            Ok(())
        })?;
        Ok(ValueResidualGrad {
            value: gv,
            value0: gv0,
            lambda: gl,
        })
    }

    fn chunked_gdn_forward(&self, inputs: GdnInputs<'_>) -> Result<GdnForward, OjasError> {
        const OP: &str = "chunked_gdn_forward";
        let dims = chunked_gdn_forward_dims(inputs)?;
        let exec = self.exec();
        let x = gdn_operands(OP, exec, inputs)?;
        let out_shape = inputs.v.shape();
        let [output, final_state, checkpoints] = fill_outs(
            OP,
            &self.budget,
            exec,
            [out_shape, &dims.state_shape(), &dims.checkpoint_shape()],
            |outs| gdn::forward(OP, &self.budget, exec, dims, x, outs),
        )?;
        Ok(GdnForward {
            output,
            final_state,
            checkpoints,
        })
    }

    fn chunked_gdn_backward(
        &self,
        inputs: GdnInputs<'_>,
        checkpoints: &Tensor,
        grad_output: &Tensor,
        grad_final_state: Option<&Tensor>,
    ) -> Result<GdnGrad, OjasError> {
        const OP: &str = "chunked_gdn_backward";
        let dims = chunked_gdn_backward_dims(inputs, checkpoints, grad_output, grad_final_state)?;
        let exec = self.exec();
        let x = gdn_operands(OP, exec, inputs)?;
        let mut rest = vec![checkpoints, grad_output];
        rest.extend(grad_final_state);
        let rest = check_f32s(OP, exec, &rest)?;
        let (ckpt, d_o, dfin) = (rest[0], rest[1], rest.get(2).copied());
        let (qs, vs, gs) = (inputs.q.shape(), inputs.v.shape(), inputs.g.shape());
        let run = |grads: GradsOut<'_>| {
            gdn::backward(OP, &self.budget, exec, dims, x, ckpt, d_o, dfin, grads)
        };
        match inputs.initial_state {
            Some(s0) => {
                let [q, k, v, g, beta, ds0] = fill_outs(
                    OP,
                    &self.budget,
                    exec,
                    [qs, qs, vs, gs, gs, s0.shape()],
                    |[dq, dk, dv, dg, dbeta, ds0]| {
                        run(GradsOut {
                            dq,
                            dk,
                            dv,
                            dg,
                            dbeta,
                            ds0: Some(ds0),
                        })
                    },
                )?;
                Ok(GdnGrad {
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state: Some(ds0),
                })
            }
            None => {
                let [q, k, v, g, beta] = fill_outs(
                    OP,
                    &self.budget,
                    exec,
                    [qs, qs, vs, gs, gs],
                    |[dq, dk, dv, dg, dbeta]| {
                        run(GradsOut {
                            dq,
                            dk,
                            dv,
                            dg,
                            dbeta,
                            ds0: None,
                        })
                    },
                )?;
                Ok(GdnGrad {
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state: None,
                })
            }
        }
    }

    fn causal_conv1d_silu_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "causal_conv1d_silu_forward";
        let dims = causal_conv1d_silu_forward_dims(input, weight)?;
        let exec = self.exec();
        let [x, w] = f32_operands(OP, exec, [input, weight])?;
        conv1d_silu_forward(OP, &self.budget, exec, dims, x, w, input.shape())
    }

    fn causal_conv1d_silu_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "causal_conv1d_silu_backward";
        let dims = causal_conv1d_silu_backward_dims(input, weight, grad_output)?;
        let exec = self.exec();
        let [x, w, gy] = f32_operands(OP, exec, [input, weight, grad_output])?;
        let shapes = [input.shape(), weight.shape()];
        conv1d_silu_backward(OP, &self.budget, exec, dims, x, w, gy, shapes)
    }

    fn gated_rms_norm_forward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "gated_rms_norm_forward";
        let dims = gated_rms_norm_forward_dims(input, gate, weight, eps)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [input, gate, weight])?;
        gated_rms_forward(OP, &self.budget, exec, dims, ins, eps, input.shape())
    }

    fn gated_rms_norm_backward(
        &self,
        input: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<GatedRmsGrad, OjasError> {
        const OP: &str = "gated_rms_norm_backward";
        let dims = gated_rms_norm_backward_dims(input, gate, weight, grad_output, eps)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [input, gate, weight, grad_output])?;
        let shapes = [input.shape(), gate.shape(), weight.shape()];
        let (input, gate, weight) =
            gated_rms_backward(OP, &self.budget, exec, dims, ins, eps, shapes)?;
        Ok(GatedRmsGrad {
            input,
            gate,
            weight,
        })
    }

    fn rope_partial_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_partial_forward";
        let dims = rope_partial_forward_dims(x, cos, sin)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [x, cos, sin])?;
        rope_partial(OP, &self.budget, exec, dims, ins, Turn::Forward, x.shape())
    }

    fn rope_partial_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_partial_backward";
        let dims = rope_partial_backward_dims(grad_output, cos, sin)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [grad_output, cos, sin])?;
        let shape = grad_output.shape();
        rope_partial(OP, &self.budget, exec, dims, ins, Turn::Backward, shape)
    }

    fn sigmoid_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "sigmoid_forward";
        sigmoid_forward_dims(input)?;
        let exec = self.exec();
        let [x] = f32_operands(OP, exec, [input])?;
        sigmoid_forward(OP, &self.budget, exec, x, input.shape())
    }

    fn sigmoid_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "sigmoid_backward";
        sigmoid_backward_dims(input, grad_output)?;
        let exec = self.exec();
        let [x, gy] = f32_operands(OP, exec, [input, grad_output])?;
        sigmoid_backward(OP, &self.budget, exec, x, gy, input.shape())
    }

    fn gdn_log_decay_forward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "gdn_log_decay_forward";
        let dims = gdn_log_decay_forward_dims(a, a_log, dt_bias)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [a, a_log, dt_bias])?;
        gdn_log_decay_forward(OP, &self.budget, exec, dims, ins, a.shape())
    }

    fn gdn_log_decay_backward(
        &self,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        grad_output: &Tensor,
    ) -> Result<GdnDecayGrad, OjasError> {
        const OP: &str = "gdn_log_decay_backward";
        let dims = gdn_log_decay_backward_dims(a, a_log, dt_bias, grad_output)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [a, a_log, dt_bias, grad_output])?;
        let shapes = [a.shape(), a_log.shape(), dt_bias.shape()];
        let (input, a_log, dt_bias) =
            gdn_log_decay_backward(OP, &self.budget, exec, dims, ins, shapes)?;
        Ok(GdnDecayGrad {
            input,
            a_log,
            dt_bias,
        })
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_forward";
        silu_forward_dims(input)?;
        let exec = self.exec();
        let [x] = f32_operands(OP, exec, [input])?;
        silu_forward(OP, &self.budget, exec, x, input.shape())
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        silu_backward_dims(input, grad_output)?;
        let exec = self.exec();
        let [x, gy] = f32_operands(OP, exec, [input, grad_output])?;
        silu_backward(OP, &self.budget, exec, x, gy, input.shape())
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        mul_forward_dims(a, b)?;
        let [av, bv] = f32_operands(OP, self.exec(), [a, b])?;
        mul_forward(OP, &self.budget, self.exec(), av, bv, a.shape())
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        mul_backward_dims(a, b, grad_output)?;
        let [av, bv, gy] = f32_operands(OP, self.exec(), [a, b, grad_output])?;
        mul_backward(
            OP,
            &self.budget,
            self.exec(),
            av,
            bv,
            gy,
            a.shape(),
            b.shape(),
        )
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        residual_add_forward_dims(x, y)?;
        let [xv, yv] = f32_operands(OP, self.exec(), [x, y])?;
        add_forward(OP, &self.budget, self.exec(), xv, yv, x.shape())
    }

    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        // Checked in place and copied byte for byte into two charged
        // tensors (pointwise.rs `add_backward`); nothing is decoded.
        residual_add_backward_dims(x, y, grad_output)?;
        add_backward("residual_add_backward", &self.budget, x, y, grad_output)
    }

    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_forward";
        let dims = cross_entropy_mean_forward_dims(logits, targets)?;
        // Read in place; the gradient is not formed, so the 4-byte loss is
        // the only charge. The loss pass reads every logit and is the NaN
        // scan; a refusal before it scans first (validate.rs
        // `nonfinite_first`), so a NaN still outranks it.
        let exec = self.exec();
        let input = ce_input(OP, exec, logits, targets, dims, ignore_index)?;
        let loss = input.mean(OP, exec, None)?;
        alloc_f32(OP, &self.budget, &[loss], &[])
    }

    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_backward";
        let dims = cross_entropy_mean_backward_dims(logits, targets)?;
        // Read in place; the gradient is written once into its output
        // tensor, which is the only charge. As in the forward, the pass is
        // the NaN scan and an earlier refusal scans first.
        let exec = self.exec();
        let input = ce_input(OP, exec, logits, targets, dims, ignore_index)?;
        let len = product(OP, &[dims.rows, dims.vocab])?;
        let out = Scratch::<f32>::try_alloc(len, &self.budget)
            .map_err(|err| nonfinite_first(OP, exec, &[logits], err))?;
        cross_entropy_grad(OP, exec, input, out, logits.shape())
    }

    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        const OP: &str = "clip_grad_norm";
        clip_grad_norm_dims(grads)?;
        // Layouts in argument order, then the norm, read in place. The norm
        // pass is also the NaN scan (optim.rs `grad_norm`); a layout refusal
        // of gradient `i` first scans gradients `0..i`, so an earlier NaN
        // still outranks it, as checking each gradient in turn would.
        let mut parts = Vec::with_capacity(grads.len());
        for (i, grad) in grads.iter().enumerate() {
            match f32_values(OP, grad) {
                Ok(words) => parts.push(words),
                Err(err) => {
                    let refs: Vec<&Tensor> = grads[..i].iter().collect();
                    check_f32s(OP, self.exec(), &refs)?;
                    return Err(err);
                }
            }
        }
        let norm = grad_norm(OP, self.exec(), &parts)?;
        drop(parts);
        let scale = clip_scale(max_norm, norm)?;
        if scale < 1.0 {
            scale_in_place(self.exec(), grads, scale)?;
        }
        Ok(norm)
    }

    fn adamw_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        moment1: &mut Tensor,
        moment2: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError> {
        const OP: &str = "adamw_step";
        adamw_step_dims(param, grad, moment1, moment2)?;
        // Layouts in argument order; the update pass is the NaN scan
        // (optim.rs `adamw_in_place`), and every refusal before it scans
        // first. The config comes before any charge (shape contract D14).
        let exec = self.exec();
        let inputs = [&*param, grad, &*moment1, &*moment2];
        f32_layouts(OP, exec, &inputs)?;
        let coeffs =
            AdamCoeffs::new(config, step).map_err(|err| nonfinite_first(OP, exec, &inputs, err))?;
        adamw_in_place(OP, exec, param, grad, moment1, moment2, coeffs)
    }

    /// AdamW updates in place and charges nothing beyond its operands; Muon
    /// charges [`CpuBackend::muon_headroom`], the figure its step reserves.
    fn optimizer_scratch_bytes(
        &self,
        kind: OptimizerKind,
        rows: usize,
        cols: usize,
    ) -> Result<Option<u64>, OjasError> {
        match kind {
            OptimizerKind::AdamW => Ok(Some(0)),
            OptimizerKind::MuonNs5 => self
                .muon_headroom("optimizer_scratch_bytes", rows, cols)
                .map(Some),
        }
    }

    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        const OP: &str = "muon_ns5_step";
        let MuonDims { rows, cols } = muon_ns5_step_dims(param, grad, momentum)?;
        check_f32s(OP, self.exec(), &[param, grad, momentum])?;
        // The config before any charge (shape contract D14).
        check_muon(config)?;
        let len = product(OP, &[rows, cols])?;
        // The operands are read in place; this covers what `muon_ns5`
        // builds, Newton-Schulz and its GEMMs included (optim.rs
        // `muon_scratch`).
        let _guard = headroom(OP, &self.budget, self.muon_headroom(OP, rows, cols)?)?;
        // Both targets are proven writable before anything is computed, so
        // a refusal leaves them unchanged; `muon_ns5` borrows all three
        // and its new values are written once it returns.
        param.ensure_writable_f32(len)?;
        momentum.ensure_writable_f32(len)?;
        let (new_p, new_m) = muon_ns5(
            self.exec(),
            f32_values(OP, param)?,
            f32_values(OP, grad)?,
            f32_values(OP, momentum)?,
            rows,
            cols,
            config,
        )?;
        if let Some(new_p) = new_p {
            param.write_f32(&new_p)?;
        }
        momentum.write_f32(&new_m)?;
        Ok(())
    }

    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        crate::accum::accumulate_grad("accumulate_grad", &self.budget, self.exec(), acc, grad)
    }

    fn scale_grad(&self, grad: &mut Tensor, scale: f32) -> Result<(), OjasError> {
        crate::accum::scale_grad("scale_grad", &self.budget, self.exec(), grad, scale)
    }

    fn linear_cross_entropy_mean(
        &self,
        input: &Tensor,
        weight: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
        chunk: ojas_core::CeChunk,
        want_grad: bool,
    ) -> Result<ojas_core::LinearCe, OjasError> {
        crate::fused_ce::linear_cross_entropy_mean(
            &self.budget,
            self.exec(),
            input,
            weight,
            targets,
            ignore_index,
            chunk,
            want_grad,
        )
    }

    fn cached_attention_forward(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        kv_len: usize,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cached_attention_forward";
        crate::kv::cached_attention_forward(
            OP,
            &self.budget,
            self.exec(),
            q,
            k_cache,
            v_cache,
            kv_len,
        )
    }

    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        crate::kv::kv_cache_write("kv_cache_write", &self.budget, cache, src, at)
    }
}
