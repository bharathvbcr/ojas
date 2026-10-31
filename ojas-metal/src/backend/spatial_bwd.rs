//! The host half of the convolution backward passes
//! ([`ojas_core::Backend::conv2d_backward`], [`ojas_core::Backend::conv_transpose2d_backward`]):
//! validate, refuse what the kernels do not run, charge the outputs and the
//! scratch to the budget, and send one command. The device half is
//! `src/device/spatial_bwd.rs`.
//!
//! The configurations are the forward's (`refuse_unsupported_conv2d`,
//! `refuse_unsupported_conv_transpose2d`): dilation 1 with one group or one
//! filter per channel, any stride and padding, which covers SAM's patch embed
//! (kernel 14, stride 14), 1 x 1, 3 x 3 with padding 1, the strided 3 x 3 and
//! the depthwise 7 x 7 with padding 3; and a transposed convolution whose
//! kernel equals its stride. Anything else is [`OjasError::Unsupported`]
//! naming the backward and the configuration, never a host fallback.

use ojas_core::{
    conv2d_backward_dims, conv_transpose2d_backward_dims, Conv2dParams, ConvGrad,
    ConvTranspose2dDims, ConvTranspose2dParams, OjasError, Tensor,
};

use super::{
    conv_geom, copied_elems, metal_err, plus, product, refuse_unsupported_conv2d,
    refuse_unsupported_conv_transpose2d, u32_dim, Cmd, ConvGeom, MetalBackend, ReadAs, Res,
};

/// The forward's geometry of a transposed convolution (`conv_transpose2d_forward`
/// builds the same words): the input extents, the output channels and the
/// kernel, stride and padding, and the output extents.
fn convt_geom(op: &'static str, d: &ConvTranspose2dDims) -> Res<ConvGeom> {
    let pair = |v: [usize; 2]| -> Res<[u32; 2]> { Ok([u32_dim(op, v[0])?, u32_dim(op, v[1])?]) };
    Ok(ConvGeom {
        batch: u32_dim(op, d.batch)?,
        c: u32_dim(op, d.in_channels)?,
        h: u32_dim(op, d.height)?,
        w: u32_dim(op, d.width)?,
        o: u32_dim(op, d.out_channels)?,
        kernel: pair(d.kernel)?,
        stride: pair(d.params.stride)?,
        padding: pair(d.params.padding)?,
        oh: u32_dim(op, d.out_height)?,
        ow: u32_dim(op, d.out_width)?,
    })
}

impl MetalBackend {
    /// [`ojas_core::Backend::conv2d_backward`]: im2col, two exact-f32 GEMMs and a
    /// col2im gather for one group; three direct kernels for a depthwise
    /// convolution. Scratch (one group): the im2col matrix, the permuted
    /// `grad_output`, the column gradient, and a copy of the weight if it
    /// does not start on 16 bytes.
    pub(super) fn conv2d_backward_routed(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        grad_output: &Tensor,
        params: Conv2dParams,
    ) -> Result<ConvGrad, OjasError> {
        const OP: &str = "conv2d_backward";
        let dims = conv2d_backward_dims(input, weight, bias, grad_output, params)?;
        refuse_unsupported_conv2d(OP, &dims)?;
        let (input, weight) = (self.widened(input)?, self.widened(weight)?);
        let grad_output = self.widened(grad_output)?;
        let bias = bias.map(|b| self.widened(b)).transpose()?;
        let (x, w, gy) = (
            self.f32(OP, &input)?,
            self.f32(OP, &weight)?,
            self.f32(OP, &grad_output)?,
        );
        let bias_arg = bias.as_ref().map(|b| self.f32(OP, b)).transpose()?;
        let geom = conv_geom(OP, &dims)?;
        let bias_shape = [dims.out_channels];
        let mut shapes: Vec<&[usize]> = vec![input.shape(), weight.shape()];
        if bias_arg.is_some() {
            shapes.push(&bias_shape);
        }
        let (cmd, scratch) = if dims.params.groups > 1 {
            let cmd = Cmd::DepthwiseConv2dBwd {
                op: OP,
                x,
                w,
                bias: bias_arg,
                gy,
                geom,
            };
            (cmd, 0)
        } else {
            let rows = product(OP, &[dims.batch, dims.out_height, dims.out_width])?;
            let width = product(OP, &[dims.in_channels, dims.kernel[0], dims.kernel[1]])?;
            let cols = product(OP, &[rows, width])?;
            let cmd = Cmd::Conv2dBwd {
                op: OP,
                x,
                w,
                bias: bias_arg,
                gy,
                geom,
            };
            let scratch = plus(
                OP,
                plus(OP, plus(OP, cols, cols)?, gy.n)?,
                copied_elems(OP, ReadAs::Matrix, &[w])?,
            )?;
            (cmd, scratch)
        };
        u32_dim(OP, product(OP, &dims.out_shape())?)?;
        grads(OP, self.outputs(OP, &shapes, scratch, cmd)?)
    }

    /// [`ojas_core::Backend::conv_transpose2d_backward`] for kernel == stride: gy read
    /// as im2col columns on the input's grid, two exact-f32 GEMMs and a
    /// layout permute. Scratch: the NHWC input, the column gradient, the
    /// NHWC input gradient, and a copy of the weight if it does not start on
    /// 16 bytes.
    pub(super) fn conv_transpose2d_backward_routed(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        grad_output: &Tensor,
        params: ConvTranspose2dParams,
    ) -> Result<ConvGrad, OjasError> {
        const OP: &str = "conv_transpose2d_backward";
        let dims = conv_transpose2d_backward_dims(input, weight, bias, grad_output, params)?;
        refuse_unsupported_conv_transpose2d(OP, &dims)?;
        let (input, weight) = (self.widened(input)?, self.widened(weight)?);
        let grad_output = self.widened(grad_output)?;
        let bias = bias.map(|b| self.widened(b)).transpose()?;
        let (x, w, gy) = (
            self.f32(OP, &input)?,
            self.f32(OP, &weight)?,
            self.f32(OP, &grad_output)?,
        );
        let bias_arg = bias.as_ref().map(|b| self.f32(OP, b)).transpose()?;
        let geom = convt_geom(OP, &dims)?;
        let rows = product(OP, &[dims.batch, dims.height, dims.width])?;
        let okk = product(OP, &[dims.out_channels, dims.kernel[0], dims.kernel[1]])?;
        let cols = product(OP, &[rows, okk])?;
        let scratch = plus(
            OP,
            plus(OP, plus(OP, x.n, cols)?, x.n)?,
            copied_elems(OP, ReadAs::Matrix, &[w])?,
        )?;
        u32_dim(OP, cols)?;
        u32_dim(OP, product(OP, &dims.out_shape())?)?;
        let bias_shape = [dims.out_channels];
        let mut shapes: Vec<&[usize]> = vec![input.shape(), weight.shape()];
        if bias_arg.is_some() {
            shapes.push(&bias_shape);
        }
        let cmd = Cmd::ConvTranspose2dBwd {
            op: OP,
            x,
            w,
            bias: bias_arg,
            gy,
            geom,
        };
        grads(OP, self.outputs(OP, &shapes, scratch, cmd)?)
    }
}

/// The device's `dx`, `dw` and optional `db` as a [`ConvGrad`].
fn grads(op: &'static str, outs: Vec<Tensor>) -> Result<ConvGrad, OjasError> {
    let mut outs = outs.into_iter();
    match (outs.next(), outs.next()) {
        (Some(input), Some(weight)) => Ok(ConvGrad {
            input,
            weight,
            bias: outs.next(),
        }),
        _ => Err(metal_err(format!("{op}: missing outputs"))),
    }
}
