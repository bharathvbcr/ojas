//! The encoder ops EmbeddingGemma 2 adds to [`crate::encoder_ops`]: the
//! tanh-approximate GELU (forward and backward), the masked mean over time
//! of a batch of sequences, and the L2 normalization of rows.
//!
//! Every sum ascends from index 0 in `f64` with a separate multiply and add,
//! is rounded once to `f32`, and belongs to exactly one output row, so no
//! task splits a reduction and the bits do not depend on the thread count;
//! `Fast` and `Exact` agree bit for bit. The GELU is
//! [`ojas_core::gelu_tanh_exact`] and its derivative
//! [`ojas_core::gelu_tanh_backward_exact`], the same bits on every platform.
//! Each output is scanned for NaN and infinity as it is stored
//! ([`fill_rows`]), so a value that overflows `f32` is
//! [`ojas_core::OjasError::NonFinite`], not `inf`.

use ojas_core::{
    gelu_tanh_backward_exact, gelu_tanh_exact, Budget, OjasError, RmsDims, SeqMeanDims, Tensor,
};

use crate::pool::Exec;
use crate::validate::{fill_rows, product, shape};

/// The tanh-approximate GELU of every element.
pub(crate) fn gelu_tanh_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    fill_rows(op, budget, exec, out_shape, x.len(), 1, |range, y| {
        for (out, &v) in y.iter_mut().zip(&x[range]) {
            *out = gelu_tanh_exact(v);
        }
        Ok(())
    })
}

/// `dy * gelu_tanh'(x)` of every element.
pub(crate) fn gelu_tanh_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    grad_y: &[f32],
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    if grad_y.len() != x.len() {
        return Err(shape(
            op,
            "gelu_tanh backward grad length does not match input",
        ));
    }
    fill_rows(op, budget, exec, out_shape, x.len(), 1, |range, dx| {
        let src = x[range.clone()].iter().zip(&grad_y[range]);
        for (out, (&v, &g)) in dx.iter_mut().zip(src) {
            *out = gelu_tanh_backward_exact(v, g);
        }
        Ok(())
    })
}

/// Columns of `x` summed together per pass over a sequence: one cache line
/// of `f32`. Each column's sum still ascends over `t` on its own.
const MEAN_COLUMNS: usize = 16;

/// The mean of the first `min(lens[b], T)` rows of every sequence `b` of
/// `x` (`[B, T, D]`), `[B, D]`. A sequence with no rows is a row of zeros.
/// `dims` comes from [`ojas_core::seq_mean_forward_dims`].
pub(crate) fn seq_mean_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    lens: &[u32],
    dims: SeqMeanDims,
) -> Result<Tensor, OjasError> {
    let SeqMeanDims { batch, seq, dim } = dims;
    if x.len() != product(op, &[batch, seq, dim])? || lens.len() != batch {
        return Err(shape(op, "seq_mean data length does not match its shape"));
    }
    fill_rows(op, budget, exec, &[batch, dim], batch, dim, |range, out| {
        for (b, dst) in range.zip(out.chunks_exact_mut(dim)) {
            let rows = (lens[b] as usize).min(seq);
            if rows == 0 {
                continue;
            }
            let sequence = &x[b * seq * dim..b * seq * dim + rows * dim];
            let count = rows as f64;
            for (block, cols) in dst.chunks_mut(MEAN_COLUMNS).enumerate() {
                let first = block * MEAN_COLUMNS;
                let mut sums = [0.0f64; MEAN_COLUMNS];
                for row in sequence.chunks_exact(dim) {
                    let src = &row[first..first + cols.len()];
                    for (sum, &v) in sums.iter_mut().zip(src) {
                        *sum += f64::from(v);
                    }
                }
                for (slot, sum) in cols.iter_mut().zip(sums) {
                    *slot = (sum / count) as f32;
                }
            }
        }
        Ok(())
    })
}

/// `x / max(||x||_2, eps)` of every row of `x` (`[rows, dim]`): the squared
/// norm an ascending `f64` sum, its square root (correctly rounded) clamped
/// from below by `eps`, and each quotient rounded once. `dims` comes from
/// [`ojas_core::l2_normalize_forward_dims`].
pub(crate) fn l2_normalize_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    dims: RmsDims,
    eps: f32,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RmsDims { rows, dim } = dims;
    if x.len() != product(op, &[rows, dim])? {
        return Err(shape(
            op,
            "l2_normalize data length does not match its shape",
        ));
    }
    let eps = f64::from(eps);
    fill_rows(op, budget, exec, out_shape, rows, dim, |range, out| {
        let src = x[range.start * dim..range.end * dim].chunks_exact(dim);
        for (dst, row) in out.chunks_exact_mut(dim).zip(src) {
            let mut sum_sq = 0.0f64;
            for &v in row {
                let v = f64::from(v);
                sum_sq += v * v;
            }
            let denom = sum_sq.sqrt().max(eps);
            for (slot, &v) in dst.iter_mut().zip(row) {
                *slot = (f64::from(v) / denom) as f32;
            }
        }
        Ok(())
    })
}
