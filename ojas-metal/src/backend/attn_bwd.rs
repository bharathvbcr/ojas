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

use ojas_core::{attention_backward_dims, sdpa_scale, AttentionGrad};

use super::*;
use crate::link::AttnBwdGeom;

/// The launch of a validated call: the forward's geometry `attn` plus the
/// mask features the kernels carry beyond what `AttentionOpts` names today.
///
/// Today the score scale is `sdpa_scale(D)`, there is no symmetric radius and
/// no dead query rows, because `AttentionOpts` has no field for them. When
/// `AttentionOpts::scale` / `radius` land (the core lane), this function is
/// the one place that changes, and it takes `op`, `&dims` and `&opts` too:
///
/// ```text
/// scale:     attention_scale(attn.d, opts.scale)?
/// radius:    dims.radius.map_or(Ok(0), |r| u32_dim(op, r))?
/// dead_rows: dims.key_lens && dims.q_len == dims.kv_len
/// ```
///
/// The kernels already run all three (`device/attn_bwd_tests.rs` drives them
/// on the device command), so the merge is these three lines, not a kernel
/// change. Until then a call cannot ask for a scale or a radius, so none is
/// silently dropped.
fn bwd_geom(attn: AttnGeom, bias_grad: bool) -> Res<AttnBwdGeom> {
    Ok(AttnBwdGeom {
        attn,
        scale: sdpa_scale(attn.d)?,
        radius: 0,
        dead_rows: false,
        bias_grad,
    })
}

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
        let geom = bwd_geom(geom, bias_grad)?;
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
