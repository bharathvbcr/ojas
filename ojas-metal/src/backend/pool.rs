//! `max_pool2d_forward` and `max_pool2d_backward` on `MetalBackend`: the
//! host side. `ojas_core`'s validators run first, so a malformed call is
//! refused with the CPU's variant and op name; the operands must be
//! device-resident; the 32-bit index caps come after. The kernels
//! (`kernels/max_pool.metal`) need no scratch, so only the output is charged.
//! Parity with the CPU is bitwise: the same first-maximum tie rule, the same
//! ascending gather order, and no arithmetic but one add per covered window.

use ojas_core::{
    max_pool2d_backward_dims, max_pool2d_forward_dims, MaxPool2dDims, OjasError, Tensor,
};

use super::{product, u32_dim, MetalBackend};
use crate::link::{Cmd, PoolGeom, Res};

/// The kernel geometry of validated `dims`, with the 32-bit caps applied.
fn geometry(op: &'static str, dims: &MaxPool2dDims) -> Res<PoolGeom> {
    let pair = |v: [usize; 2]| -> Res<[u32; 2]> { Ok([u32_dim(op, v[0])?, u32_dim(op, v[1])?]) };
    // Both the input and the output are indexed by one 32-bit thread id.
    u32_dim(
        op,
        product(op, &[dims.batch, dims.channels, dims.height, dims.width])?,
    )?;
    u32_dim(op, product(op, &dims.out_shape())?)?;
    Ok(PoolGeom {
        planes: u32_dim(op, product(op, &[dims.batch, dims.channels])?)?,
        h: u32_dim(op, dims.height)?,
        w: u32_dim(op, dims.width)?,
        oh: u32_dim(op, dims.out_height)?,
        ow: u32_dim(op, dims.out_width)?,
        kernel: pair(dims.kernel)?,
        stride: pair(dims.stride)?,
    })
}

impl MetalBackend {
    /// [`ojas_core::Backend::max_pool2d_forward`] on `ojas_max_pool2d_fwd`.
    pub(super) fn pool_forward(
        &self,
        input: &Tensor,
        kernel: [usize; 2],
        stride: [usize; 2],
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "max_pool2d_forward";
        let dims = max_pool2d_forward_dims(input, kernel, stride)?;
        let x = self.f32(OP, input)?;
        let geom = geometry(OP, &dims)?;
        self.one(OP, &dims.out_shape(), 0, Cmd::MaxPool2d { x, geom })
    }

    /// [`ojas_core::Backend::max_pool2d_backward`] on `ojas_max_pool2d_bwd`.
    pub(super) fn pool_backward(
        &self,
        input: &Tensor,
        grad_output: &Tensor,
        kernel: [usize; 2],
        stride: [usize; 2],
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "max_pool2d_backward";
        let dims = max_pool2d_backward_dims(input, grad_output, kernel, stride)?;
        let (x, dy) = (self.f32(OP, input)?, self.f32(OP, grad_output)?);
        let geom = geometry(OP, &dims)?;
        self.one(
            OP,
            &[dims.batch, dims.channels, dims.height, dims.width],
            0,
            Cmd::MaxPool2dBwd { x, dy, geom },
        )
    }
}
