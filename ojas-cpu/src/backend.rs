use ojas_core::{
    AdamWConfig, Backend, BackendId, Budget, MuonNs5Config, OjasError, PerHeadGateGrad, Tensor,
    ValueResidualGrad,
};

use crate::attn::{causal_sdpa_backward, causal_sdpa_forward};
use crate::linalg::{linear_backward, linear_forward};
use crate::norm::{rms_backward, rms_forward, rope_backward, rope_forward};
use crate::optim::{adamw, clip_scale, muon_ns5, total_norm};
use crate::pointwise::{
    add_forward, cross_entropy, embedding_backward, embedding_forward, gate_backward, gate_forward,
    mul_backward, mul_forward, silu_backward, silu_forward, value_residual_backward,
    value_residual_forward,
};
use crate::validate::{alloc_f32, f32_in, headroom, payload_bytes, same_shape, shape, u32_in};

/// Deterministic CPU reference. [`CpuBackend::new`] uses one thread.
/// [`CpuBackend::with_threads`] runs large linears on a Rayon pool, in fixed
/// row tiles. Reductions stay in increasing index order.
#[derive(Clone, Debug)]
pub struct CpuBackend {
    budget: Budget,
    threads: usize,
}

impl CpuBackend {
    pub fn new(budget: Budget) -> Self {
        Self { budget, threads: 1 }
    }

    /// `threads == 0` is refused. Small ops stay on the caller even when
    /// `threads` is larger.
    pub fn with_threads(budget: Budget, threads: usize) -> Result<Self, OjasError> {
        if threads == 0 {
            return Err(OjasError::OutOfRange {
                op: "CpuBackend::with_threads",
                detail: "thread count is 0".to_string(),
            });
        }
        Ok(Self { budget, threads })
    }

    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    pub fn threads(&self) -> usize {
        self.threads
    }
}

impl Backend for CpuBackend {
    fn id(&self) -> BackendId {
        BackendId::Cpu
    }

    fn budget(&self) -> &Budget {
        &self.budget
    }

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_forward";
        let table_v = f32_in(OP, table)?;
        let ids = u32_in(OP, token_ids)?;
        let (out, out_shape) = embedding_forward(
            OP,
            &self.budget,
            &table_v.data,
            &table_v.shape,
            &ids.data,
            &ids.shape,
        )?;
        alloc_f32(OP, &self.budget, &out, &out_shape)
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        let table_v = f32_in(OP, table)?;
        let ids = u32_in(OP, token_ids)?;
        let grad = f32_in(OP, grad_output)?;
        let out = embedding_backward(
            OP,
            &table_v.shape,
            &ids.data,
            &ids.shape,
            &grad.data,
            &grad.shape,
        )?;
        alloc_f32(OP, &self.budget, &out, &table_v.shape)
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let x = f32_in(OP, input)?;
        let w = f32_in(OP, weight)?;
        let (y, y_shape) = linear_forward(
            OP,
            &self.budget,
            self.threads,
            &x.data,
            &x.shape,
            &w.data,
            &w.shape,
        )?;
        alloc_f32(OP, &self.budget, &y, &y_shape)
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let x = f32_in(OP, input)?;
        let w = f32_in(OP, weight)?;
        let gy = f32_in(OP, grad_output)?;
        let (gx, gw) = linear_backward(
            OP,
            &self.budget,
            self.threads,
            &x.data,
            &x.shape,
            &w.data,
            &w.shape,
            &gy.data,
            &gy.shape,
        )?;
        let grad_x = alloc_f32(OP, &self.budget, &gx, &x.shape)?;
        let grad_w = alloc_f32(OP, &self.budget, &gw, &w.shape)?;
        Ok((grad_x, grad_w))
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rms_norm_forward";
        let x = f32_in(OP, input)?;
        let w = f32_in(OP, weight)?;
        let y = rms_forward(OP, &x.data, &x.shape, &w.data, &w.shape, eps)?;
        alloc_f32(OP, &self.budget, &y, &x.shape)
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let x = f32_in(OP, input)?;
        let w = f32_in(OP, weight)?;
        let gy = f32_in(OP, grad_output)?;
        let (gx, gw) = rms_backward(
            OP, &x.data, &x.shape, &w.data, &w.shape, &gy.data, &gy.shape, eps,
        )?;
        Ok((
            alloc_f32(OP, &self.budget, &gx, &x.shape)?,
            alloc_f32(OP, &self.budget, &gw, &w.shape)?,
        ))
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_forward";
        let xv = f32_in(OP, x)?;
        let cv = f32_in(OP, cos)?;
        let sv = f32_in(OP, sin)?;
        let y = rope_forward(
            OP, &xv.data, &xv.shape, &cv.data, &cv.shape, &sv.data, &sv.shape,
        )?;
        alloc_f32(OP, &self.budget, &y, &xv.shape)
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_backward";
        let gy = f32_in(OP, grad_output)?;
        let cv = f32_in(OP, cos)?;
        let sv = f32_in(OP, sin)?;
        let gx = rope_backward(
            OP, &gy.data, &gy.shape, &cv.data, &cv.shape, &sv.data, &sv.shape,
        )?;
        alloc_f32(OP, &self.budget, &gx, &gy.shape)
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
        let qv = f32_in(OP, q)?;
        let kv = f32_in(OP, k)?;
        let vv = f32_in(OP, v)?;
        let y = causal_sdpa_forward(
            OP, &qv.data, &qv.shape, &kv.data, &kv.shape, &vv.data, &vv.shape,
        )?;
        alloc_f32(OP, &self.budget, &y, &qv.shape)
    }

    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_backward";
        let qv = f32_in(OP, q)?;
        let kv = f32_in(OP, k)?;
        let vv = f32_in(OP, v)?;
        let gy = f32_in(OP, grad_output)?;
        let (gq, gk, gv) = causal_sdpa_backward(
            OP, &qv.data, &qv.shape, &kv.data, &kv.shape, &vv.data, &vv.shape, &gy.data, &gy.shape,
        )?;
        Ok((
            alloc_f32(OP, &self.budget, &gq, &qv.shape)?,
            alloc_f32(OP, &self.budget, &gk, &kv.shape)?,
            alloc_f32(OP, &self.budget, &gv, &vv.shape)?,
        ))
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_forward";
        let x = f32_in(OP, input)?;
        let w = f32_in(OP, weight)?;
        let b = f32_in(OP, bias)?;
        let attn = f32_in(OP, attn_out)?;
        let y = gate_forward(
            OP,
            &x.data,
            &x.shape,
            &w.data,
            &w.shape,
            &b.data,
            &b.shape,
            &attn.data,
            &attn.shape,
        )?;
        alloc_f32(OP, &self.budget, &y, &attn.shape)
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
        let x = f32_in(OP, input)?;
        let w = f32_in(OP, weight)?;
        let b = f32_in(OP, bias)?;
        let attn = f32_in(OP, attn_out)?;
        let gy = f32_in(OP, grad_output)?;
        let (gx, gw, gb, ga) = gate_backward(
            OP,
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
            input: alloc_f32(OP, &self.budget, &gx, &x.shape)?,
            weight: alloc_f32(OP, &self.budget, &gw, &w.shape)?,
            bias: alloc_f32(OP, &self.budget, &gb, &b.shape)?,
            attn_out: alloc_f32(OP, &self.budget, &ga, &attn.shape)?,
        })
    }

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "value_residual_blend_forward";
        let v = f32_in(OP, value)?;
        let v0 = f32_in(OP, value0)?;
        let lam = scalar(OP, lambda)?;
        same_shape(OP, &v.shape, &v0.shape)?;
        let (y, _) = value_residual_forward(OP, &v.data, &v0.data, lam)?;
        alloc_f32(OP, &self.budget, &y, &v.shape)
    }

    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        const OP: &str = "value_residual_blend_backward";
        let v = f32_in(OP, value)?;
        let v0 = f32_in(OP, value0)?;
        let lam_t = f32_in(OP, lambda)?;
        let gy = f32_in(OP, grad_output)?;
        if lam_t.data.len() != 1 {
            return Err(shape(OP, "value residual lambda must be a scalar"));
        }
        same_shape(OP, &v.shape, &v0.shape)?;
        same_shape(OP, &v.shape, &gy.shape)?;
        let (gv, gv0, gl) =
            value_residual_backward(OP, &v.data, &v0.data, lam_t.data[0], &gy.data)?;
        Ok(ValueResidualGrad {
            value: alloc_f32(OP, &self.budget, &gv, &v.shape)?,
            value0: alloc_f32(OP, &self.budget, &gv0, &v0.shape)?,
            lambda: alloc_f32(OP, &self.budget, &[gl], &lam_t.shape)?,
        })
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_forward";
        let x = f32_in(OP, input)?;
        alloc_f32(OP, &self.budget, &silu_forward(&x.data), &x.shape)
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        let x = f32_in(OP, input)?;
        let gy = f32_in(OP, grad_output)?;
        same_shape(OP, &x.shape, &gy.shape)?;
        let gx = silu_backward(OP, &x.data, &gy.data)?;
        alloc_f32(OP, &self.budget, &gx, &x.shape)
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        let av = f32_in(OP, a)?;
        let bv = f32_in(OP, b)?;
        same_shape(OP, &av.shape, &bv.shape)?;
        let y = mul_forward(OP, &av.data, &bv.data)?;
        alloc_f32(OP, &self.budget, &y, &av.shape)
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        let av = f32_in(OP, a)?;
        let bv = f32_in(OP, b)?;
        let gy = f32_in(OP, grad_output)?;
        same_shape(OP, &av.shape, &bv.shape)?;
        same_shape(OP, &av.shape, &gy.shape)?;
        let (ga, gb) = mul_backward(OP, &av.data, &bv.data, &gy.data)?;
        Ok((
            alloc_f32(OP, &self.budget, &ga, &av.shape)?,
            alloc_f32(OP, &self.budget, &gb, &bv.shape)?,
        ))
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        let xv = f32_in(OP, x)?;
        let yv = f32_in(OP, y)?;
        same_shape(OP, &xv.shape, &yv.shape)?;
        let z = add_forward(OP, &xv.data, &yv.data)?;
        alloc_f32(OP, &self.budget, &z, &xv.shape)
    }

    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "residual_add_backward";
        let xv = f32_in(OP, x)?;
        let yv = f32_in(OP, y)?;
        let gy = f32_in(OP, grad_output)?;
        same_shape(OP, &xv.shape, &yv.shape)?;
        same_shape(OP, &xv.shape, &gy.shape)?;
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
        let logits_v = f32_in(OP, logits)?;
        let targets_v = u32_in(OP, targets)?;
        let (loss, _) = cross_entropy(
            OP,
            &logits_v.data,
            &logits_v.shape,
            &targets_v.data,
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
        let logits_v = f32_in(OP, logits)?;
        let targets_v = u32_in(OP, targets)?;
        let (_, grad) = cross_entropy(
            OP,
            &logits_v.data,
            &logits_v.shape,
            &targets_v.data,
            &targets_v.shape,
            ignore_index,
        )?;
        alloc_f32(OP, &self.budget, &grad, &logits_v.shape)
    }

    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        const OP: &str = "clip_grad_norm";
        if grads.is_empty() {
            return Err(shape(OP, "empty tensor"));
        }
        let mut parts = Vec::with_capacity(grads.len());
        let mut bytes = 0u64;
        for grad in grads.iter() {
            let view = f32_in(OP, grad)?;
            bytes = bytes
                .checked_add(payload_bytes(OP, view.data.len())?)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: OP,
                    detail: "clip payload overflows".to_string(),
                })?;
            parts.push(view.data);
        }
        let norm = total_norm(OP, &parts)?;
        let scale = clip_scale(max_norm, norm)?;
        let _guard = headroom(OP, &self.budget, bytes)?;
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
        let p = f32_in(OP, param)?;
        let g = f32_in(OP, grad)?;
        let m1 = f32_in(OP, moment1)?;
        let m2 = f32_in(OP, moment2)?;
        same_shape(OP, &p.shape, &g.shape)?;
        same_shape(OP, &p.shape, &m1.shape)?;
        same_shape(OP, &p.shape, &m2.shape)?;
        let (new_p, new_m, new_v) = adamw(&p.data, &g.data, &m1.data, &m2.data, step, config)?;
        let _guard = headroom(OP, &self.budget, payload_bytes(OP, p.data.len())?)?;
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
        let p = f32_in(OP, param)?;
        let g = f32_in(OP, grad)?;
        let m = f32_in(OP, momentum)?;
        if p.shape.len() != 2 {
            return Err(shape(OP, "muon parameter must be a matrix"));
        }
        same_shape(OP, &p.shape, &g.shape)?;
        same_shape(OP, &p.shape, &m.shape)?;
        let (new_p, new_m) = muon_ns5(&p.data, &g.data, &m.data, p.shape[0], p.shape[1], config)?;
        let _guard = headroom(OP, &self.budget, payload_bytes(OP, p.data.len())?)?;
        param.write_f32(&new_p)?;
        momentum.write_f32(&new_m)?;
        Ok(())
    }
}

fn scalar(op: &'static str, tensor: &Tensor) -> Result<f32, OjasError> {
    let view = f32_in(op, tensor)?;
    if view.data.len() != 1 {
        return Err(shape(op, "expected a scalar tensor"));
    }
    Ok(view.data[0])
}
