//! Host side of the EmbeddingGemma 2 encoder ops that are not attention:
//! validate with the pure `*_dims` validators, charge the budget for the
//! outputs and the scratch, and send one command. The kernels are described
//! in `src/gemma_ops.rs`; the device side is `device/gemma_dev.rs`.
//!
//! Numerics against the CPU reference, which sums in `f64` and rounds once:
//!
//! - `gelu_tanh_forward` is tessl's `tessl_gelu_pytorch_tanh` in `f32`, one
//!   thread an element. The CPU evaluates the same formula in `f64`, so the
//!   two agree to a few `f32` ulps of the result (the `f32` `tanh` and the
//!   cubic), not bit for bit; the bound the tests hold is 2e-6 absolute plus
//!   1e-5 relative. The backward is the same evaluation of the derivative.
//! - `seq_mean_forward` adds a column in row order in `f32` (tessl's
//!   `segment_mean_rows`) and divides by the count, so it is within the `f32`
//!   rounding of a sequential sum of the CPU's `f64` result. The device scans
//!   the whole operand for NaN and infinity, padding rows included, and so
//!   does the CPU (`f32_operands`), so a non-finite value in padding is a
//!   fault on both: the mean never reads those rows, but neither backend
//!   accepts an operand that holds one.
//! - `seq_mean_backward` scans the whole gradient the same way and divides
//!   by `min(len, T)` in `f32`, where the CPU divides in `f64` and rounds
//!   once, so the two agree to the `f32` rounding of one division. `x` is
//!   read for its shape only on both, so a non-finite value in it refuses
//!   neither.
//! - `l2_normalize_forward` sums squares in `f32` in tessl's lane order and
//!   multiplies by `1 / max(sqrt(sum), 1e-12)`. A row whose squares leave the
//!   `f32` range (components past about 1.8e19 overflow the sum and give a
//!   zero row; past about 1e-22 they underflow to 0 and the row is scaled by
//!   1e12 rather than normalized) does not match the CPU's `f64` result;
//!   embedding activations are many orders inside that range. Only eps `1e-12`
//!   runs: it is the constant of tessl's kernel and the one `F.normalize`
//!   uses, and any other value is `OjasError::Unsupported`, never ignored.

use ojas_core::{
    gelu_tanh_backward_dims, gelu_tanh_forward_dims, l2_normalize_backward_dims,
    l2_normalize_forward_dims, seq_mean_backward_dims, seq_mean_forward_dims, DType, OjasError,
    Tensor,
};

use super::{copied_elems, plus, product, u32_dim, MetalBackend, Res};
use crate::gemma_ops::{SeqMeanGeom, L2_EPS};
use crate::link::{Cmd, ReadAs};

impl MetalBackend {
    /// One element a thread on `ojas_gelu_tanh_fwd`; a view is read in place.
    pub(super) fn gemma_gelu_tanh_forward(&self, input: &Tensor) -> Res<Tensor> {
        const OP: &str = "gelu_tanh_forward";
        gelu_tanh_forward_dims(input)?;
        let x = self.f32(OP, input)?;
        self.one(OP, input.shape(), 0, Cmd::GeluTanh { x })
    }

    /// `grad_output * gelu_tanh'(input)`, one element a thread on
    /// `ojas_gelu_tanh_bwd`.
    pub(super) fn gemma_gelu_tanh_backward(
        &self,
        input: &Tensor,
        grad_output: &Tensor,
    ) -> Res<Tensor> {
        const OP: &str = "gelu_tanh_backward";
        gelu_tanh_backward_dims(input, grad_output)?;
        let (x, gy) = (self.f32(OP, input)?, self.f32(OP, grad_output)?);
        self.one(OP, input.shape(), 0, Cmd::GeluTanhBwd { x, gy })
    }

    /// The masked mean over time. Scratch: the `[batch, 2]` row ranges, and
    /// a copy of each operand that is a view (tessl's kernel takes a buffer,
    /// not a byte offset).
    pub(super) fn gemma_seq_mean_forward(&self, x: &Tensor, key_lens: &Tensor) -> Res<Tensor> {
        const OP: &str = "seq_mean_forward";
        let dims = seq_mean_forward_dims(x, key_lens)?;
        let xa = self.f32(OP, x)?;
        let lens = self.arg(OP, key_lens, DType::U32)?;
        let geom = SeqMeanGeom {
            batch: u32_dim(OP, dims.batch)?,
            seq: u32_dim(OP, dims.seq)?,
            dim: u32_dim(OP, dims.dim)?,
        };
        geom.rows().ok_or_else(|| OjasError::Unsupported {
            op: OP,
            detail: format!(
                "{} x {} rows exceed the kernels' 32-bit row indices",
                dims.batch, dims.seq
            ),
        })?;
        let ranges = product(OP, &[2, dims.batch])?;
        let scratch = plus(OP, ranges, copied_elems(OP, ReadAs::Whole, &[xa, lens])?)?;
        self.one(
            OP,
            &[dims.batch, dims.dim],
            scratch,
            Cmd::SeqMean { x: xa, lens, geom },
        )
    }

    /// The gradient of the masked mean, `[B, T, D]`, on ojas's own
    /// `ojas_seq_mean_bwd`: one output element a thread, no scratch (the
    /// kernel takes the operands' byte offsets). `x` is read for its shape
    /// only, so it is not sent.
    pub(super) fn gemma_seq_mean_backward(
        &self,
        x: &Tensor,
        key_lens: &Tensor,
        grad_output: &Tensor,
    ) -> Res<Tensor> {
        const OP: &str = "seq_mean_backward";
        let dims = seq_mean_backward_dims(x, key_lens, grad_output)?;
        let grad = self.f32(OP, grad_output)?;
        let lens = self.arg(OP, key_lens, DType::U32)?;
        let geom = SeqMeanGeom {
            batch: u32_dim(OP, dims.batch)?,
            seq: u32_dim(OP, dims.seq)?,
            dim: u32_dim(OP, dims.dim)?,
        };
        geom.in_elems().ok_or_else(|| OjasError::Unsupported {
            op: OP,
            detail: format!(
                "a [{}, {}, {}] gradient exceeds the kernel's 32-bit element indices",
                dims.batch, dims.seq, dims.dim
            ),
        })?;
        self.one(OP, x.shape(), 0, Cmd::SeqMeanBwd { grad, lens, geom })
    }

    /// `x / max(||x||, 1e-12)` per row, for eps `1e-12` only.
    pub(super) fn gemma_l2_normalize_forward(&self, x: &Tensor, eps: f32) -> Res<Tensor> {
        const OP: &str = "l2_normalize_forward";
        let dims = l2_normalize_forward_dims(x, eps)?;
        if eps.to_bits() != L2_EPS.to_bits() {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: format!(
                    "Metal normalizes on tessl's l2_normalize_rows, whose epsilon is fixed \
                     at 1e-12; eps {eps} is not routed"
                ),
            });
        }
        let xa = self.f32(OP, x)?;
        let cmd = Cmd::L2Normalize {
            x: xa,
            rows: u32_dim(OP, dims.rows)?,
            dim: u32_dim(OP, dims.dim)?,
        };
        self.one(OP, x.shape(), 0, cmd)
    }

    /// The gradient of the row-wise normalize on ojas's own
    /// `ojas_l2_normalize_bwd`, one row a thread, for eps `1e-12` only: the
    /// forward of this backend refuses every other eps, so a gradient of a
    /// normalize it could not have computed is refused here too.
    pub(super) fn gemma_l2_normalize_backward(
        &self,
        x: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Res<Tensor> {
        const OP: &str = "l2_normalize_backward";
        let dims = l2_normalize_backward_dims(x, grad_output, eps)?;
        if eps.to_bits() != L2_EPS.to_bits() {
            return Err(OjasError::Unsupported {
                op: OP,
                detail: format!(
                    "Metal normalizes on tessl's l2_normalize_rows, whose epsilon is fixed \
                     at 1e-12, so its gradient is routed for that epsilon only; eps {eps} \
                     is not routed"
                ),
            });
        }
        let cmd = Cmd::L2NormalizeBwd {
            x: self.f32(OP, x)?,
            grad: self.f32(OP, grad_output)?,
            rows: u32_dim(OP, dims.rows)?,
            dim: u32_dim(OP, dims.dim)?,
            eps,
        };
        self.one(OP, x.shape(), 0, cmd)
    }
}
