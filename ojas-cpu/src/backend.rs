use std::sync::Arc;

use ojas_core::{
    clip_scale, AdamWConfig, Backend, BackendId, Budget, MuonNs5Config, Numerics, OjasError,
    PerHeadGateGrad, Reservation, Tensor, ValueResidualGrad,
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
    alloc_f32, alloc_out, check_f32s, claim_writable, copy_checked, f32_checked, f32_in,
    f32_inputs, f32_layouts, f32_words, headroom, nonfinite_first, payload_bytes, product,
    room_for, u32_words, F32Out,
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
/// attention above 256 positions, and on macOS a GEMM of at least 2²¹
/// multiply-adds is one Accelerate call that ignores the pool and threads on
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

    /// The charge for an output of `elements` f32 values, taken before the
    /// kernel that computes it runs and handed to [`alloc_out`] with the
    /// result.
    ///
    /// A kernel's own hold covers the buffer it fills and ends when it
    /// returns. While the kernel assembles per-task parts into that buffer,
    /// and afterwards while [`alloc_out`] copies it into the returned tensor,
    /// a second output-sized buffer is live; this charge covers it until
    /// the copy exists and the buffer is freed.
    fn out_charge(&self, op: &'static str, elements: usize) -> Result<Reservation, OjasError> {
        room_for(op, &self.budget, elements)
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
        let [x, w] = f32_inputs(OP, &self.budget, [input, weight])?;
        let charge = self.out_charge(OP, x.data.len())?;
        let y = rms_forward(OP, &self.budget, self.exec(), x.data, w.data, dims, eps)?;
        finish(OP, &self.budget, y, charge, &x.shape)
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
        let [x, w, gy] = f32_inputs(OP, &self.budget, [input, weight, grad_output])?;
        let gx_charge = self.out_charge(OP, x.data.len())?;
        let gw_charge = self.out_charge(OP, w.data.len())?;
        // norm.rs `rms_backward` keeps one rstd per row in its task parts and
        // again in the joined vector, and its hold counts them once. With
        // many short rows that second copy outgrows the two weight-gradient
        // charges, so it is charged here until the kernel returns.
        let rstd = room_for(OP, &self.budget, dims.rows)?;
        let (gx, gw) = rms_backward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            w.data,
            gy.data,
            dims,
            eps,
        )?;
        drop(rstd);
        let grad_x = finish(OP, &self.budget, gx, gx_charge, &x.shape)?;
        let grad_w = finish(OP, &self.budget, gw, gw_charge, &w.shape)?;
        Ok((grad_x, grad_w))
    }
}

/// `out` plus the charge [`CpuBackend::out_charge`] took for it, as a tensor.
fn finish(
    op: &'static str,
    budget: &Budget,
    data: Vec<f32>,
    charge: Reservation,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    alloc_out(op, budget, F32Out { data, charge }, shape)
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
        let ids = u32_words(OP, token_ids)?;
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
        let ids = u32_words(OP, token_ids)?;
        let grad = f32_checked(OP, self.exec(), grad_output)?;
        embedding_backward(OP, &self.budget, ids, grad, &dims)
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let dims = linear_forward_dims(input, weight)?;
        let [x, w] = f32_inputs(OP, &self.budget, [input, weight])?;
        let y = linear_forward(OP, &self.budget, self.exec(), x.data, w.data, &dims)?;
        alloc_out(OP, &self.budget, y, &dims.out_shape)
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let dims = linear_backward_dims(input, weight, grad_output)?;
        let [x, w, gy] = f32_inputs(OP, &self.budget, [input, weight, grad_output])?;
        let (gx, gw) = linear_backward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            w.data,
            gy.data,
            &dims,
        )?;
        let grad_x = alloc_out(OP, &self.budget, gx, &x.shape)?;
        let grad_w = alloc_out(OP, &self.budget, gw, &w.shape)?;
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
        let [xv, cv, sv] = f32_inputs(OP, &self.budget, [x, cos, sin])?;
        let charge = self.out_charge(OP, xv.data.len())?;
        let y = rope_forward(
            OP,
            &self.budget,
            self.exec(),
            xv.data,
            cv.data,
            sv.data,
            dims,
        )?;
        finish(OP, &self.budget, y, charge, &xv.shape)
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_backward";
        let dims = rope_half_split_backward_dims(grad_output, cos, sin)?;
        let [gy, cv, sv] = f32_inputs(OP, &self.budget, [grad_output, cos, sin])?;
        let charge = self.out_charge(OP, gy.data.len())?;
        let gx = rope_backward(
            OP,
            &self.budget,
            self.exec(),
            gy.data,
            cv.data,
            sv.data,
            dims,
        )?;
        finish(OP, &self.budget, gx, charge, &gy.shape)
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
        let [qv, kv, vv] = f32_inputs(OP, &self.budget, [q, k, v])?;
        let charge = self.out_charge(OP, qv.data.len())?;
        let y = causal_sdpa_forward(
            OP,
            &self.budget,
            self.exec(),
            qv.data,
            kv.data,
            vv.data,
            dims,
        )?;
        finish(OP, &self.budget, y, charge, &qv.shape)
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
        let [qv, kv, vv, gy] = f32_inputs(OP, &self.budget, [q, k, v, grad_output])?;
        let gq_charge = self.out_charge(OP, qv.data.len())?;
        let gk_charge = self.out_charge(OP, kv.data.len())?;
        let gv_charge = self.out_charge(OP, vv.data.len())?;
        let (gq, gk, gv) = causal_sdpa_backward(
            OP,
            &self.budget,
            self.exec(),
            qv.data,
            kv.data,
            vv.data,
            gy.data,
            dims,
        )?;
        let grad_q = finish(OP, &self.budget, gq, gq_charge, &qv.shape)?;
        let grad_k = finish(OP, &self.budget, gk, gk_charge, &kv.shape)?;
        let grad_v = finish(OP, &self.budget, gv, gv_charge, &vv.shape)?;
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
        let [x, w, b, attn] = f32_inputs(OP, &self.budget, [input, weight, bias, attn_out])?;
        let charge = self.out_charge(OP, attn.data.len())?;
        let y = gate_forward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            w.data,
            &b.data,
            attn.data,
            dims,
        )?;
        finish(OP, &self.budget, y, charge, &attn.shape)
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
        let [x, w, b, attn, gy] = f32_inputs(
            OP,
            &self.budget,
            [input, weight, bias, attn_out, grad_output],
        )?;
        let gx_charge = self.out_charge(OP, x.data.len())?;
        let gw_charge = self.out_charge(OP, w.data.len())?;
        let gb_charge = self.out_charge(OP, b.data.len())?;
        let ga_charge = self.out_charge(OP, attn.data.len())?;
        let (gx, gw, gb, ga) = gate_backward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            w.data,
            &b.data,
            attn.data,
            gy.data,
            dims,
        )?;
        Ok(PerHeadGateGrad {
            input: finish(OP, &self.budget, gx, gx_charge, &x.shape)?,
            weight: finish(OP, &self.budget, gw, gw_charge, &w.shape)?,
            bias: finish(OP, &self.budget, gb, gb_charge, &b.shape)?,
            attn_out: finish(OP, &self.budget, ga, ga_charge, &attn.shape)?,
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
        let [v, v0, lam] = f32_inputs(OP, &self.budget, [value, value0, lambda])?;
        let charge = self.out_charge(OP, v.data.len())?;
        let y = value_residual_forward(OP, v.data, v0.data, lam.data[0])?;
        finish(OP, &self.budget, y, charge, &v.shape)
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
        let [v, v0, lam_t, gy] =
            f32_inputs(OP, &self.budget, [value, value0, lambda, grad_output])?;
        let gv_charge = self.out_charge(OP, v.data.len())?;
        let gv0_charge = self.out_charge(OP, v0.data.len())?;
        let gl_charge = self.out_charge(OP, 1)?;
        let (gv, gv0, gl) = value_residual_backward(
            OP,
            &self.budget,
            self.exec(),
            v.data,
            v0.data,
            lam_t.data[0],
            gy.data,
        )?;
        Ok(ValueResidualGrad {
            value: finish(OP, &self.budget, gv, gv_charge, &v.shape)?,
            value0: finish(OP, &self.budget, gv0, gv0_charge, &v0.shape)?,
            lambda: finish(OP, &self.budget, vec![gl], gl_charge, &lam_t.shape)?,
        })
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_forward";
        silu_forward_dims(input)?;
        let x = f32_in(OP, &self.budget, input)?;
        let charge = self.out_charge(OP, x.data.len())?;
        let y = silu_forward(OP, &self.budget, self.exec(), x.data)?;
        finish(OP, &self.budget, y, charge, &x.shape)
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        silu_backward_dims(input, grad_output)?;
        let [x, gy] = f32_inputs(OP, &self.budget, [input, grad_output])?;
        let charge = self.out_charge(OP, x.data.len())?;
        let gx = silu_backward(OP, &self.budget, self.exec(), x.data, gy.data)?;
        finish(OP, &self.budget, gx, charge, &x.shape)
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        mul_forward_dims(a, b)?;
        let [av, bv] = f32_inputs(OP, &self.budget, [a, b])?;
        let charge = self.out_charge(OP, av.data.len())?;
        let y = mul_forward(OP, av.data, bv.data)?;
        finish(OP, &self.budget, y, charge, &av.shape)
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        mul_backward_dims(a, b, grad_output)?;
        let [av, bv, gy] = f32_inputs(OP, &self.budget, [a, b, grad_output])?;
        let ga_charge = self.out_charge(OP, av.data.len())?;
        let gb_charge = self.out_charge(OP, bv.data.len())?;
        let (ga, gb) = mul_backward(OP, av.data, bv.data, gy.data)?;
        let grad_a = finish(OP, &self.budget, ga, ga_charge, &av.shape)?;
        let grad_b = finish(OP, &self.budget, gb, gb_charge, &bv.shape)?;
        Ok((grad_a, grad_b))
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        residual_add_forward_dims(x, y)?;
        let [xv, yv] = f32_inputs(OP, &self.budget, [x, y])?;
        let charge = self.out_charge(OP, xv.data.len())?;
        let z = add_forward(OP, xv.data, yv.data)?;
        finish(OP, &self.budget, z, charge, &xv.shape)
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
        let bytes = product(OP, &[dims.rows, dims.vocab, 4])?;
        let out = Scratch::<u8>::try_alloc(bytes, &self.budget)
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
            match f32_words(OP, grad) {
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
            scale_in_place(OP, &self.budget, self.exec(), grads, scale)?;
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
        adamw_in_place(
            OP,
            &self.budget,
            exec,
            param,
            grad,
            moment1,
            moment2,
            coeffs,
        )
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
        let p = copy_checked(OP, &self.budget, param, len)?;
        let g = copy_checked(OP, &self.budget, grad, len)?;
        let m = copy_checked(OP, &self.budget, momentum, len)?;
        // The three input copies are charged above; this covers what
        // `muon_ns5` builds, Newton-Schulz and its GEMMs included (optim.rs
        // `muon_scratch`).
        let work = muon_scratch(OP, self.exec(), rows, cols)?;
        let _guard = headroom(OP, &self.budget, payload_bytes(OP, work)?)?;
        claim_writable(&mut [(param, &p.data), (momentum, &m.data)])?;
        let (new_p, new_m) = muon_ns5(self.exec(), p.data, g.data, m.data, rows, cols, config)?;
        param.write_f32(&new_p)?;
        momentum.write_f32(&new_m)?;
        Ok(())
    }

    fn accumulate_grad(&self, acc: &mut Tensor, grad: &Tensor) -> Result<(), OjasError> {
        crate::accum::accumulate_grad("accumulate_grad", &self.budget, acc, grad)
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
        let y = crate::kv::cached_attention_forward(
            OP,
            &self.budget,
            self.exec(),
            q,
            k_cache,
            v_cache,
            kv_len,
        )?;
        alloc_out(OP, &self.budget, y, q.shape())
    }

    fn kv_cache_write(&self, cache: &mut Tensor, src: &Tensor, at: usize) -> Result<(), OjasError> {
        crate::kv::kv_cache_write("kv_cache_write", &self.budget, cache, src, at)
    }
}
