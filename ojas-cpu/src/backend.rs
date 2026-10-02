use std::sync::Arc;

use ojas_core::{
    clip_scale, AdamWConfig, Backend, BackendId, Budget, MuonNs5Config, Numerics, OjasError,
    PerHeadGateGrad, Tensor, ValueResidualGrad,
};
// Every op below runs its `ojas_core::shapes` validator before it checks
// values, copies an input or charges the budget (docs/shape-contract.md).
use ojas_core::{
    causal_sdpa_backward_dims, causal_sdpa_forward_dims, linear_backward_dims, linear_forward_dims,
    mul_backward_dims, mul_forward_dims, per_head_sigmoid_gate_backward_dims,
    per_head_sigmoid_gate_forward_dims, residual_add_backward_dims, residual_add_forward_dims,
    rms_norm_backward_dims, rms_norm_forward_dims, rms_qk_norm_backward_dims,
    rms_qk_norm_forward_dims, rope_half_split_backward_dims, rope_half_split_forward_dims,
    silu_backward_dims, silu_forward_dims, value_residual_blend_backward_dims,
    value_residual_blend_forward_dims, RmsDims,
};

use crate::attn::{causal_sdpa_backward, causal_sdpa_forward, Dims as SdpaKernelDims};
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
    headroom, nonfinite_first, payload_bytes, product, u32_values,
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

    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        permute("permute", &self.budget, input, dims)
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
        let [x, w] = f32_operands(OP, exec, [input, weight])?;
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
        let ins = f32_operands(OP, exec, [input, weight, grad_output])?;
        let shapes = [input.shape(), weight.shape()];
        let [grad_x, grad_w] = fill_outs(OP, &self.budget, self.exec(), shapes, |grads| {
            linear_backward(OP, &self.budget, exec, ins, &dims, grads)
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

    fn causal_sdpa_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "causal_sdpa_forward";
        let dims = SdpaKernelDims::new(OP, causal_sdpa_forward_dims(q, k, v)?)?;
        let exec = self.exec();
        let qkv = f32_operands(OP, exec, [q, k, v])?;
        fill_out(OP, &self.budget, self.exec(), q.shape(), |out| {
            causal_sdpa_forward(OP, &self.budget, exec, qkv, dims, out)
        })
    }

    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_backward";
        let dims = SdpaKernelDims::new(OP, causal_sdpa_backward_dims(q, k, v, grad_output)?)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [q, k, v, grad_output])?;
        let shapes = [q.shape(), k.shape(), v.shape()];
        let [grad_q, grad_k, grad_v] = fill_outs(OP, &self.budget, self.exec(), shapes, |grads| {
            causal_sdpa_backward(OP, &self.budget, exec, ins, dims, grads)
        })?;
        Ok((grad_q, grad_k, grad_v))
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
        gate_forward(
            OP,
            &self.budget,
            exec,
            x,
            w,
            b,
            attn,
            dims,
            attn_out.shape(),
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
        const OP: &str = "per_head_sigmoid_gate_backward";
        let dims = per_head_sigmoid_gate_backward_dims(input, weight, bias, attn_out, grad_output)?;
        let exec = self.exec();
        let ins = f32_operands(OP, exec, [input, weight, bias, attn_out, grad_output])?;
        let shapes = [
            input.shape(),
            weight.shape(),
            bias.shape(),
            attn_out.shape(),
        ];
        let [gx, gw, gb, ga] = fill_outs(OP, &self.budget, self.exec(), shapes, |grads| {
            gate_backward(OP, &self.budget, exec, ins, dims, grads)
        })?;
        Ok(PerHeadGateGrad {
            input: gx,
            weight: gw,
            bias: gb,
            attn_out: ga,
        })
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
        let work = muon_scratch(OP, self.exec(), rows, cols)?;
        let _guard = headroom(OP, &self.budget, payload_bytes(OP, work)?)?;
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
