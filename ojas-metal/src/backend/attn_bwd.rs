//! [`ojas_core::Backend::attention_backward`] on Metal: the general tiled
//! backward (`attn_gen_bwd.metal`, run by `Worker::attn_gen_bwd`).
//!
//! Bidirectional or causal attention with a window, a query length that
//! differs from the key length, an additive score bias with its gradient,
//! per-batch key lengths and grouped-query heads, at head dims through
//! `METAL_MAX_HEAD_DIM`. The only scratch is each query row's
//! `Dr = dO . O`, `B * H * Tq` values; grouped-query gradients are summed in
//! place onto their KV head, in increasing query-head order, and a bias that
//! broadcasts is summed in a fixed order, so a result repeats bit for bit.
//! The operand rules, the dtype rules and the refusals are the validator's
//! ([`attention_backward_dims`]); the device limits are this backend's.

use ojas_core::{attention_backward_dims, sdpa_scale};

use super::*;
use crate::link::AttnBwdGeom;

impl MetalBackend {
    /// The gradients of `attention_forward_saving` from its `output` and
    /// `lse`, with the bias gradient when `bias_grad` asks for it.
    ///
    /// A device without TensorOps refuses: the forward has scalar fallbacks,
    /// this backward is matrix-unit code only, and a refusal is the only
    /// honest answer there (never a host computation).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        output: &Tensor,
        lse: &Tensor,
        grad_output: &Tensor,
        opts: AttentionOpts<'_>,
        bias_grad: bool,
    ) -> Res<AttentionGrad> {
        const OP: &str = "attention_backward";
        let dims = attention_backward_dims(q, k, v, output, lse, grad_output, &opts, bias_grad)?;
        // The head-dim cap is the same on every device, so it is checked
        // before the device-dependent TensorOps refusal.
        let geom = attn_launch(OP, &dims)?;
        if !self.link.has_tensorops() {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: "Metal's attention backward runs on the TensorOps matrix units, which \
                         this device lacks"
                    .to_string(),
            });
        }
        // The log-sum-exp stays F32 (the validator refuses a Bf16 one).
        let [q, k, v, output, grad_output] =
            [q, k, v, output, grad_output].map(|t| self.widened(t));
        let (q, k, v, output, grad_output) = (q?, k?, v?, output?, grad_output?);
        let bias = opts.bias.map(|b| self.widened(b)).transpose()?;
        let args = [
            self.f32(OP, &q)?,
            self.f32(OP, &k)?,
            self.f32(OP, &v)?,
            self.f32(OP, &output)?,
            self.f32(OP, lse)?,
            self.f32(OP, &grad_output)?,
        ];
        let bias_arg = bias.as_ref().map(|b| self.f32(OP, b)).transpose()?;
        let lens = opts
            .key_lens
            .map(|l| self.arg(OP, l, DType::U32))
            .transpose()?;
        let geom = AttnBwdGeom {
            attn: geom,
            scale: sdpa_scale(geom.d)?,
            radius: 0,
            dead_rows: false,
            bias_grad,
        };
        let scratch = product(OP, lse_shape(q.shape()))?;
        let mut shapes: Vec<&[usize]> = vec![q.shape(), k.shape(), v.shape()];
        if let (true, Some(b)) = (bias_grad, bias.as_ref()) {
            shapes.push(b.shape());
        }
        let cmd = Cmd::AttnBwd {
            op: OP,
            args,
            bias: bias_arg,
            lens,
            geom,
        };
        let mut out = self.outputs(OP, &shapes, scratch, cmd)?.into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(q), Some(k), Some(v)) => Ok(AttentionGrad {
                q,
                k,
                v,
                bias: out.next(),
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }
}
