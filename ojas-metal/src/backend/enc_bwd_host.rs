//! Host side of the encoder backward ops: validate with the pure `*_dims`
//! validators, charge the budget for the outputs and the scratch, and send one
//! command. The kernels, their reduction order and their bound are described
//! in `kernels/encoder_bwd.metal`; the device side is `device/enc_bwd_dev.rs`.
//!
//! Numerics against the CPU reference (which sums in `f64` and rounds once):
//!
//! - Every sum is a double-single sum of exact products in a fixed order, then
//!   rounded once, so a sum differs from the CPU's by well under one `f32`
//!   ulp of the sum itself. Row statistics (`mean`, `rstd`) carry the same
//!   bound, except that `rstd` is formed in `f32` (`1 / sqrt(var + eps)`, two
//!   roundings, `<= 1.5e-7` relative).
//! - A LayerNorm `dx` is `rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat *
//!   xhat))` formed in `f32` from those: each element is within about
//!   `6 * 2^-24 * rstd * (|dxhat| + |mean(dxhat)| + |xhat * mean(dxhat *
//!   xhat)|)` of the CPU's, which is what cancellation leaves of a few `f32`
//!   roundings.
//! - `dw`, `db`, the softmax dot and the broadcast gradients are the sums
//!   above, so they differ from the CPU's by at most the rounding of the sum
//!   (broadcast gradients and `db` of adds: one `f32` ulp).
//! - GELU's derivative uses the forward's erfc fit (relative error `1.2e-7`),
//!   so it is within `2e-7 * |dy|` of the exact derivative for `|x| <= 9`.
//!
//! None of it is bit-identical to the CPU; all of it repeats bit for bit.

use ojas_core::{
    broadcast_add_backward_dims, broadcast_mul_backward_dims, broadcast_shapes,
    channel_layer_norm_backward_dims, gelu_erf_backward_dims, layer_norm_backward_dims,
    linear_bias_backward_dims, softmax_backward_dims, AffineGrad, Backend, BroadcastDims,
    OjasError, Tensor,
};

use super::{gated_rms_geom, metal_err, overflow, product, u32_dim, MetalBackend, Res};
use crate::encoder_bwd::{BcastGradGeom, ClnBwdGeom, LnBwdGeom};
use crate::link::Cmd;

/// `(float)dim` must be exact in the kernels' mean (`dim < 2^24`).
const EXACT_FLOAT: usize = 1 << 24;

fn exact_extent(op: &'static str, what: &str, n: usize) -> Res<u32> {
    if n >= EXACT_FLOAT {
        return Err(OjasError::Unsupported {
            op,
            detail: format!("{what} {n} is past the 2^24 a float mean divides exactly"),
        });
    }
    u32_dim(op, n)
}

fn three(op: &'static str, out: Vec<Tensor>) -> Res<AffineGrad> {
    let mut it = out.into_iter();
    match (it.next(), it.next(), it.next()) {
        (Some(input), Some(weight), Some(bias)) => Ok(AffineGrad {
            input,
            weight,
            bias,
        }),
        _ => Err(metal_err(format!("{op}: missing outputs"))),
    }
}

impl MetalBackend {
    pub(super) fn enc_layer_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Res<AffineGrad> {
        const OP: &str = "layer_norm_backward";
        let dims = layer_norm_backward_dims(input, weight, bias, grad_output, eps)?;
        let (x, w, b, gy) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, grad_output)?,
        );
        let (rows, dim) = gated_rms_geom(OP, &dims)?;
        exact_extent(OP, "dim", dims.dim)?;
        let geom = LnBwdGeom::new(rows, dim);
        let scratch = geom.scratch().ok_or_else(|| overflow(OP))?;
        let cmd = Cmd::LnBwd {
            x,
            w,
            b,
            gy,
            geom,
            eps,
        };
        let shapes = [input.shape(), weight.shape(), bias.shape()];
        three(OP, self.outputs(OP, &shapes, scratch, cmd)?)
    }

    /// `[N, C]` is already channels-last, so it takes the row path; an image
    /// shaped input normalizes each position over its strided channels.
    pub(super) fn enc_channel_layer_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Res<AffineGrad> {
        const OP: &str = "channel_layer_norm_backward";
        let dims = channel_layer_norm_backward_dims(input, weight, bias, grad_output, eps)?;
        if dims.spatial == 1 {
            let shape = [dims.batch, dims.channels];
            let flat = input.reshape(&shape)?;
            let gy = grad_output.reshape(&shape)?;
            let g = Backend::layer_norm_backward(self, &flat, weight, bias, &gy, eps)?;
            return Ok(AffineGrad {
                input: g.input.reshape(input.shape())?,
                ..g
            });
        }
        let (x, w, b, gy) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, grad_output)?,
        );
        let positions = u32_dim(OP, product(OP, &[dims.batch, dims.spatial])?)?;
        let channels = exact_extent(OP, "channels", dims.channels)?;
        let geom = ClnBwdGeom::new(positions, channels, u32_dim(OP, dims.spatial)?);
        let scratch = geom.scratch().ok_or_else(|| overflow(OP))?;
        let cmd = Cmd::ClnBwd {
            x,
            w,
            b,
            gy,
            geom,
            eps,
        };
        let shapes = [input.shape(), weight.shape(), bias.shape()];
        three(OP, self.outputs(OP, &shapes, scratch, cmd)?)
    }

    pub(super) fn enc_gelu_erf_backward(
        &self,
        input: &Tensor,
        grad_output: &Tensor,
    ) -> Res<Tensor> {
        const OP: &str = "gelu_erf_backward";
        gelu_erf_backward_dims(input, grad_output)?;
        let (x, gy) = (self.f32(OP, input)?, self.f32(OP, grad_output)?);
        self.one(OP, input.shape(), 0, Cmd::GeluErfBwd { x, gy })
    }

    pub(super) fn enc_softmax_backward(
        &self,
        output: &Tensor,
        grad_output: &Tensor,
    ) -> Res<Tensor> {
        const OP: &str = "softmax_backward";
        let dims = softmax_backward_dims(output, grad_output)?;
        let (y, gy) = (self.f32(OP, output)?, self.f32(OP, grad_output)?);
        let (rows, dim) = gated_rms_geom(OP, &dims)?;
        self.one(OP, output.shape(), 0, Cmd::SoftmaxBwd { y, gy, rows, dim })
    }

    pub(super) fn enc_broadcast_add_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Res<(Tensor, Tensor)> {
        const OP: &str = "broadcast_add_backward";
        let dims = broadcast_add_backward_dims(a, b, grad_output)?;
        self.broadcast_grads(OP, false, [a, b], grad_output, &dims)
    }

    pub(super) fn enc_broadcast_mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Res<(Tensor, Tensor)> {
        const OP: &str = "broadcast_mul_backward";
        let dims = broadcast_mul_backward_dims(a, b, grad_output)?;
        self.broadcast_grads(OP, true, [a, b], grad_output, &dims)
    }

    /// Each operand's gradient is its own command: the output gradient summed
    /// over the axes the operand was stretched along. An add scans both
    /// operands (nothing reads them); a multiply reads each as the other's
    /// factor, every element of it.
    fn broadcast_grads(
        &self,
        op: &'static str,
        mul: bool,
        [a, b]: [&Tensor; 2],
        grad_output: &Tensor,
        dims: &BroadcastDims,
    ) -> Res<(Tensor, Tensor)> {
        let (aa, ba) = (self.f32(op, a)?, self.f32(op, b)?);
        let g = self.f32(op, grad_output)?;
        let geom_a = BcastGradGeom::new(op, dims, true, mul)?;
        let geom_b = BcastGradGeom::new(op, dims, false, mul)?;
        let ga = self.one(
            op,
            a.shape(),
            geom_a.scratch().ok_or_else(|| overflow(op))?,
            Cmd::BcastGrad {
                op,
                g,
                other: mul.then_some(ba),
                scan: if mul { vec![] } else { vec![aa, ba] },
                geom: geom_a,
            },
        )?;
        let gb = self.one(
            op,
            b.shape(),
            geom_b.scratch().ok_or_else(|| overflow(op))?,
            Cmd::BcastGrad {
                op,
                g,
                other: mul.then_some(aa),
                scan: vec![],
                geom: geom_b,
            },
        )?;
        Ok((ga, gb))
    }

    /// `linear_backward`'s `dx` and `dw` (same bits, tessl's GEMMs on the
    /// lane its operands pick), and the bias gradient as the sum of the
    /// output gradient's rows.
    pub(super) fn enc_linear_bias_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        grad_output: &Tensor,
    ) -> Res<AffineGrad> {
        const OP: &str = "linear_bias_backward";
        let dims = linear_bias_backward_dims(input, weight, bias, grad_output)?;
        let (gx, gw) = Backend::linear_backward(self, input, weight, grad_output)?;
        let gy = self.widened(grad_output)?;
        let rows = gy.reshape(&[dims.rows, dims.out_features])?;
        let (g, bias_arg) = (self.f32(OP, &rows)?, self.f32(OP, bias)?);
        let bdims = broadcast_shapes(OP, rows.shape(), bias.shape())?;
        let geom = BcastGradGeom::new(OP, &bdims, false, false)?;
        let scratch = geom.scratch().ok_or_else(|| overflow(OP))?;
        let gb = self.one(
            OP,
            bias.shape(),
            scratch,
            Cmd::BcastGrad {
                op: OP,
                g,
                other: None,
                scan: vec![bias_arg],
                geom,
            },
        )?;
        Ok(AffineGrad {
            input: gx,
            weight: gw,
            bias: gb,
        })
    }
}
