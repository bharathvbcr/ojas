use std::sync::Arc;

use ojas_core::{
    clip_scale, AdamWConfig, Backend, BackendId, Budget, MuonNs5Config, Numerics, OjasError,
    PerHeadGateGrad, Reservation, Tensor, ValueResidualGrad,
};

use crate::attn::{causal_sdpa_backward, causal_sdpa_forward};
use crate::layout::permute;
use crate::linalg::{linear_backward, linear_forward};
use crate::norm::{rms_backward, rms_forward, rope_backward, rope_forward};
use crate::optim::{adamw, muon_ns5, total_norm};
use crate::pointwise::{
    add_forward, cross_entropy, embedding_backward, embedding_forward, gate_backward, gate_forward,
    mul_backward, mul_forward, silu_backward, silu_forward, value_residual_backward,
    value_residual_forward,
};
use crate::pool::{Exec, Pool};
use crate::optim::{adamw_scratch, muon_scratch};
use crate::validate::{
    alloc_f32, alloc_out, check_f32, check_u32, claim_writable, f32_in, f32_input_list, f32_inputs,
    headroom, payload_bytes, product, room_for, same_shape, shape, u32_in, F32Out,
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
        check_u32(OP, token_ids)?;
        let table_v = f32_in(OP, &self.budget, table)?;
        let ids = u32_in(OP, &self.budget, token_ids)?;
        // A table that is not `[vocab, dim]` is refused by the kernel before
        // it allocates; it takes no output charge, so it is a Shape error
        // under any cap.
        let dim = match table_v.shape.as_slice() {
            [_, dim] => *dim,
            _ => 0,
        };
        let charge = self.out_charge(OP, product(OP, &[ids.data.len(), dim])?)?;
        let (out, out_shape) = embedding_forward(
            OP,
            &self.budget,
            &table_v.data,
            &table_v.shape,
            &ids.data,
            &ids.shape,
        )?;
        finish(OP, &self.budget, out, charge, &out_shape)
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        // Only the table's shape is read, so it is validated in place and
        // never copied.
        check_u32(OP, token_ids)?;
        check_f32(OP, table)?;
        let grad = f32_in(OP, &self.budget, grad_output)?;
        let ids = u32_in(OP, &self.budget, token_ids)?;
        let charge = self.out_charge(OP, table.num_elements()?)?;
        let out = embedding_backward(
            OP,
            &self.budget,
            table.shape(),
            &ids.data,
            &ids.shape,
            &grad.data,
            &grad.shape,
        )?;
        finish(OP, &self.budget, out, charge, table.shape())
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let [x, w] = f32_inputs(OP, &self.budget, [input, weight])?;
        let (y, y_shape) = linear_forward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            &x.shape,
            w.data,
            &w.shape,
        )?;
        alloc_out(OP, &self.budget, y, &y_shape)
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let [x, w, gy] = f32_inputs(OP, &self.budget, [input, weight, grad_output])?;
        let (gx, gw) = linear_backward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            &x.shape,
            w.data,
            &w.shape,
            gy.data,
            &gy.shape,
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
        const OP: &str = "rms_norm_forward";
        let [x, w] = f32_inputs(OP, &self.budget, [input, weight])?;
        let charge = self.out_charge(OP, x.data.len())?;
        let y = rms_forward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            &x.shape,
            w.data,
            &w.shape,
            eps,
        )?;
        finish(OP, &self.budget, y, charge, &x.shape)
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let [x, w, gy] = f32_inputs(OP, &self.budget, [input, weight, grad_output])?;
        let gx_charge = self.out_charge(OP, x.data.len())?;
        let gw_charge = self.out_charge(OP, w.data.len())?;
        let (gx, gw) = rms_backward(
            OP,
            &self.budget,
            self.exec(),
            x.data,
            &x.shape,
            w.data,
            &w.shape,
            gy.data,
            &gy.shape,
            eps,
        )?;
        let grad_x = finish(OP, &self.budget, gx, gx_charge, &x.shape)?;
        let grad_w = finish(OP, &self.budget, gw, gw_charge, &w.shape)?;
        Ok((grad_x, grad_w))
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_forward";
        let [xv, cv, sv] = f32_inputs(OP, &self.budget, [x, cos, sin])?;
        let charge = self.out_charge(OP, xv.data.len())?;
        let y = rope_forward(
            OP,
            &self.budget,
            self.exec(),
            xv.data,
            &xv.shape,
            cv.data,
            &cv.shape,
            sv.data,
            &sv.shape,
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
        let [gy, cv, sv] = f32_inputs(OP, &self.budget, [grad_output, cos, sin])?;
        let charge = self.out_charge(OP, gy.data.len())?;
        let gx = rope_backward(
            OP,
            &self.budget,
            self.exec(),
            gy.data,
            &gy.shape,
            cv.data,
            &cv.shape,
            sv.data,
            &sv.shape,
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
        let qn = self.rms_norm_forward(q, q_weight, eps)?;
        let kn = self.rms_norm_forward(k, k_weight, eps)?;
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
        let (gq, gqw) = self.rms_norm_backward(q, q_weight, grad_q, eps)?;
        let (gk, gkw) = self.rms_norm_backward(k, k_weight, grad_k, eps)?;
        Ok((gq, gk, gqw, gkw))
    }

    fn causal_sdpa_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "causal_sdpa_forward";
        let [qv, kv, vv] = f32_inputs(OP, &self.budget, [q, k, v])?;
        let charge = self.out_charge(OP, qv.data.len())?;
        let y = causal_sdpa_forward(
            OP,
            &self.budget,
            self.exec(),
            qv.data,
            &qv.shape,
            kv.data,
            &kv.shape,
            vv.data,
            &vv.shape,
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
        let [qv, kv, vv, gy] = f32_inputs(OP, &self.budget, [q, k, v, grad_output])?;
        let gq_charge = self.out_charge(OP, qv.data.len())?;
        let gk_charge = self.out_charge(OP, kv.data.len())?;
        let gv_charge = self.out_charge(OP, vv.data.len())?;
        let (gq, gk, gv) = causal_sdpa_backward(
            OP,
            &self.budget,
            self.exec(),
            qv.data,
            &qv.shape,
            kv.data,
            &kv.shape,
            vv.data,
            &vv.shape,
            gy.data,
            &gy.shape,
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
        let [x, w, b, attn] = f32_inputs(OP, &self.budget, [input, weight, bias, attn_out])?;
        let charge = self.out_charge(OP, attn.data.len())?;
        let y = gate_forward(
            OP,
            &self.budget,
            &x.data,
            &x.shape,
            &w.data,
            &w.shape,
            &b.data,
            &b.shape,
            &attn.data,
            &attn.shape,
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
            &x.data,
            &x.shape,
            &w.data,
            &w.shape,
            &b.data,
            &b.shape,
            &attn.data,
            &attn.shape,
            &gy.data,
            &gy.shape,
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
        let [v, v0, lam] = f32_inputs(OP, &self.budget, [value, value0, lambda])?;
        if lam.data.len() != 1 {
            return Err(shape(OP, "expected a scalar tensor"));
        }
        same_shape(OP, &v.shape, &v0.shape)?;
        let charge = self.out_charge(OP, v.data.len())?;
        let (y, _) = value_residual_forward(OP, &v.data, &v0.data, lam.data[0])?;
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
        let [v, v0, lam_t, gy] =
            f32_inputs(OP, &self.budget, [value, value0, lambda, grad_output])?;
        if lam_t.data.len() != 1 {
            return Err(shape(OP, "value residual lambda must be a scalar"));
        }
        same_shape(OP, &v.shape, &v0.shape)?;
        same_shape(OP, &v.shape, &gy.shape)?;
        let gv_charge = self.out_charge(OP, v.data.len())?;
        let gv0_charge = self.out_charge(OP, v0.data.len())?;
        let gl_charge = self.out_charge(OP, 1)?;
        let (gv, gv0, gl) =
            value_residual_backward(OP, &v.data, &v0.data, lam_t.data[0], &gy.data)?;
        Ok(ValueResidualGrad {
            value: finish(OP, &self.budget, gv, gv_charge, &v.shape)?,
            value0: finish(OP, &self.budget, gv0, gv0_charge, &v0.shape)?,
            lambda: finish(OP, &self.budget, vec![gl], gl_charge, &lam_t.shape)?,
        })
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_forward";
        let x = f32_in(OP, &self.budget, input)?;
        let charge = self.out_charge(OP, x.data.len())?;
        finish(OP, &self.budget, silu_forward(&x.data), charge, &x.shape)
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        let [x, gy] = f32_inputs(OP, &self.budget, [input, grad_output])?;
        same_shape(OP, &x.shape, &gy.shape)?;
        let charge = self.out_charge(OP, x.data.len())?;
        let gx = silu_backward(OP, &x.data, &gy.data)?;
        finish(OP, &self.budget, gx, charge, &x.shape)
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        let [av, bv] = f32_inputs(OP, &self.budget, [a, b])?;
        same_shape(OP, &av.shape, &bv.shape)?;
        let charge = self.out_charge(OP, av.data.len())?;
        let y = mul_forward(OP, &av.data, &bv.data)?;
        finish(OP, &self.budget, y, charge, &av.shape)
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        let [av, bv, gy] = f32_inputs(OP, &self.budget, [a, b, grad_output])?;
        same_shape(OP, &av.shape, &bv.shape)?;
        same_shape(OP, &av.shape, &gy.shape)?;
        let ga_charge = self.out_charge(OP, av.data.len())?;
        let gb_charge = self.out_charge(OP, bv.data.len())?;
        let (ga, gb) = mul_backward(OP, &av.data, &bv.data, &gy.data)?;
        let grad_a = finish(OP, &self.budget, ga, ga_charge, &av.shape)?;
        let grad_b = finish(OP, &self.budget, gb, gb_charge, &bv.shape)?;
        Ok((grad_a, grad_b))
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        let [xv, yv] = f32_inputs(OP, &self.budget, [x, y])?;
        same_shape(OP, &xv.shape, &yv.shape)?;
        let charge = self.out_charge(OP, xv.data.len())?;
        let z = add_forward(OP, &xv.data, &yv.data)?;
        finish(OP, &self.budget, z, charge, &xv.shape)
    }

    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "residual_add_backward";
        let [xv, yv, gy] = f32_inputs(OP, &self.budget, [x, y, grad_output])?;
        same_shape(OP, &xv.shape, &yv.shape)?;
        same_shape(OP, &xv.shape, &gy.shape)?;
        // Both tensors are copies of `gy`'s charged host copy; no buffer is
        // live that the input charges and the tensors' own do not cover.
        Ok((
            alloc_f32(OP, &self.budget, &gy.data, &xv.shape)?,
            alloc_f32(OP, &self.budget, &gy.data, &yv.shape)?,
        ))
    }

    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_forward";
        check_u32(OP, targets)?;
        let logits_v = f32_in(OP, &self.budget, logits)?;
        let targets_v = u32_in(OP, &self.budget, targets)?;
        let (loss, _) = cross_entropy(
            OP,
            &self.budget,
            self.exec(),
            logits_v.data,
            &logits_v.shape,
            targets_v.data,
            &targets_v.shape,
            ignore_index,
        )?;
        alloc_f32(OP, &self.budget, &[loss], &[])
    }

    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_backward";
        check_u32(OP, targets)?;
        let logits_v = f32_in(OP, &self.budget, logits)?;
        let targets_v = u32_in(OP, &self.budget, targets)?;
        let (_, grad) = cross_entropy(
            OP,
            &self.budget,
            self.exec(),
            logits_v.data,
            &logits_v.shape,
            targets_v.data,
            &targets_v.shape,
            ignore_index,
        )?;
        // No separate output charge: the logits copy is moved into the kernel
        // and freed when its row tasks are dropped, before the gradient is
        // assembled, and its charge (`logits_v.charge`, as large as the
        // gradient) is held until this function returns, through the copy
        // below. `redteam_ops_heap.rs` measures this. On a pooled backend a
        // worker drops its handle to the finished batch, and with it the
        // copy, only after `unpublish` (pool.rs `worker`), which can be after
        // assembly starts; that window is not charged here.
        alloc_f32(OP, &self.budget, &grad, &logits_v.shape)
    }

    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        const OP: &str = "clip_grad_norm";
        if grads.is_empty() {
            return Err(shape(OP, "empty tensor"));
        }
        let mut parts = Vec::with_capacity(grads.len());
        // Each copy's charge is kept here for as long as `parts` holds it.
        let mut charges = Vec::with_capacity(grads.len());
        let mut bytes = 0u64;
        let refs: Vec<&Tensor> = grads.iter().collect();
        for view in f32_input_list(OP, &self.budget, &refs)? {
            bytes = bytes
                .checked_add(payload_bytes(OP, view.data.len())?)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: OP,
                    detail: "clip payload overflows".to_string(),
                })?;
            parts.push(view.data);
            charges.push(view.charge);
        }
        // The input copies are charged above; this covers `scaled`.
        let _guard = headroom(OP, &self.budget, bytes)?;
        let norm = total_norm(OP, &parts)?;
        let scale = clip_scale(max_norm, norm)?;
        if scale < 1.0 {
            let mut scaled = Vec::with_capacity(parts.len());
            for part in &parts {
                let mut next = Vec::with_capacity(part.len());
                for value in part {
                    let y = value * scale;
                    if !y.is_finite() {
                        return Err(OjasError::NonFinite { op: OP });
                    }
                    next.push(y);
                }
                scaled.push(next);
            }
            let mut targets: Vec<(&mut Tensor, &[f32])> = grads
                .iter_mut()
                .zip(parts.iter())
                .map(|(grad, part)| (grad, part.as_slice()))
                .collect();
            claim_writable(&mut targets)?;
            drop(targets);
            for (grad, next) in grads.iter_mut().zip(scaled.iter()) {
                grad.write_f32(next)?;
            }
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
        let [p, g, m1, m2] = f32_inputs(OP, &self.budget, [param, grad, moment1, moment2])?;
        same_shape(OP, &p.shape, &g.shape)?;
        same_shape(OP, &p.shape, &m1.shape)?;
        same_shape(OP, &p.shape, &m2.shape)?;
        let len = p.data.len();
        let _guard = headroom(OP, &self.budget, payload_bytes(OP, len)?)?;
        claim_writable(&mut [(param, &p.data), (moment1, &m1.data), (moment2, &m2.data)])?;
        let (new_p, new_m, new_v) =
            adamw(self.exec(), p.data, g.data, m1.data, m2.data, step, config)?;
        param.write_f32(&new_p)?;
        moment1.write_f32(&new_m)?;
        moment2.write_f32(&new_v)?;
        Ok(())
    }

    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        const OP: &str = "muon_ns5_step";
        let [p, g, m] = f32_inputs(OP, &self.budget, [param, grad, momentum])?;
        if p.shape.len() != 2 {
            return Err(shape(OP, "muon parameter must be a matrix"));
        }
        same_shape(OP, &p.shape, &g.shape)?;
        same_shape(OP, &p.shape, &m.shape)?;
        let len = p.data.len();
        let _guard = headroom(OP, &self.budget, payload_bytes(OP, len)?)?;
        claim_writable(&mut [(param, &p.data), (momentum, &m.data)])?;
        let (new_p, new_m) = muon_ns5(
            self.exec(),
            p.data,
            g.data,
            m.data,
            p.shape[0],
            p.shape[1],
            config,
        )?;
        param.write_f32(&new_p)?;
        momentum.write_f32(&new_m)?;
        Ok(())
    }
}
