//! `y = x @ W^T` and its gradients, as strided views over one GEMM core.
//!
//! `W` is `[out, in]`, the `nn.Linear(..., bias=False)` layout. The forward
//! is `x · Wᵀ`, `grad_x` is `g · W`, and `grad_w` is `gᵀ · x`; the transposes
//! are strides that [`crate::gemm`] packs through, not copies. Under
//! [`ojas_core::Numerics::Exact`] each output sums its reduction axis from
//! index 0 upward in `f32` without `mul_add`, and the bits do not depend on
//! the thread count.

use ojas_core::{Budget, LinearDims, OjasError};

use crate::gemm::{gemm_out, scratch, single_task, whole_call, Mat, TASK_MACS};
use crate::pool::{scoped, Exec};
use crate::validate::{product, room_for, shape};

pub(crate) fn transpose(
    op: &'static str,
    a: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, OjasError> {
    let len = product(op, &[rows, cols])?;
    if a.len() != len {
        return Err(shape(
            op,
            format!("transpose data len {} != {rows}*{cols}", a.len()),
        ));
    }
    let mut out = vec![0.0f32; len];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = a[row * cols + col];
        }
    }
    Ok(out)
}

/// `x` is `[rows, in]`, `weight` is `[out, in]`, and the output `y`
/// (`[rows, out]`, zeroed, charged by the caller) is written in place;
/// `dims` comes from [`ojas_core::linear_forward_dims`]. The operands are
/// read where they are (a tensor's own storage), so beside the output only
/// the GEMM scratch is charged.
pub(crate) fn linear_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    weight: &[f32],
    dims: &LinearDims,
    y: &mut [f32],
) -> Result<(), OjasError> {
    let (rows, kin, nout) = (dims.rows, dims.in_features, dims.out_features);
    if x.len() != product(op, &[rows, kin])?
        || weight.len() != product(op, &[nout, kin])?
        || y.len() != product(op, &[rows, nout])?
    {
        return Err(shape(op, "linear data length does not match shape"));
    }
    let _scratch = room_for(op, budget, scratch(op, exec, rows, kin, nout)?)?;
    let x = Mat::row_major(x, rows, kin);
    let w = Mat::row_major(weight, nout, kin);
    gemm_out(op, exec, &x, &w.t(), y)
}

/// `grad_x` (`[rows, in]`) and `grad_w` (`[out, in]`) for `grad_y`
/// `[rows, out]`, written into `grads` (zeroed, charged by the caller);
/// `dims` comes from [`ojas_core::linear_backward_dims`]. The operands are
/// read where they are, as in [`linear_forward`].
pub(crate) fn linear_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [x, weight, grad_y]: [&[f32]; 3],
    dims: &LinearDims,
    [grad_x, grad_w]: [&mut [f32]; 2],
) -> Result<(), OjasError> {
    let (rows, kin, nout) = (dims.rows, dims.in_features, dims.out_features);
    if x.len() != product(op, &[rows, kin])?
        || weight.len() != product(op, &[nout, kin])?
        || grad_y.len() != product(op, &[rows, nout])?
        || grad_x.len() != product(op, &[rows, kin])?
        || grad_w.len() != product(op, &[nout, kin])?
    {
        return Err(shape(
            op,
            "linear backward data length does not match shape",
        ));
    }
    // Two products that each fit in one task run side by side, so their
    // scratch is live at once; larger ones split themselves in turn. A whole
    // `ojas-simd` call must not run inside a pool task. On macOS a Fast
    // product of at least `TASK_MACS / 2` is always a whole call, so only
    // Exact pairs there; off macOS Fast pairs too.
    let pair = exec.pool.threads() > 1
        && single_task(exec, rows, kin, nout)
        && single_task(exec, nout, kin, rows)
        && !whole_call(exec.numerics, rows, nout, kin)
        && rows.saturating_mul(nout).saturating_mul(kin) >= TASK_MACS / 2;
    let (sx, sw) = (
        scratch(op, exec, rows, nout, kin)?,
        scratch(op, exec, nout, rows, kin)?,
    );
    let work = if pair {
        sx.checked_add(sw).ok_or_else(|| overflow(op))?
    } else {
        sx.max(sw)
    };
    let _work = room_for(op, budget, work)?;
    let x = Mat::row_major(x, rows, kin);
    let w = Mat::row_major(weight, nout, kin);
    let g = Mat::row_major(grad_y, rows, nout);
    // grad_x[row, inner] = sum_col g[row, col] * W[col, inner], col from 0.
    // grad_w[col, inner] = sum_row g[row, col] * x[row, inner], row from 0.
    // A paired product fits in one task, so it runs whole on its scoped
    // thread, which reads the borrowed operands and writes its own output.
    if pair {
        scoped::fill_parts(exec, vec![grad_x, grad_w], |i, out| {
            if i == 0 {
                gemm_out(op, exec, &g, &w, out)
            } else {
                gemm_out(op, exec, &g.t(), &x, out)
            }
        })?;
    } else {
        gemm_out(op, exec, &g, &w, grad_x)?;
        gemm_out(op, exec, &g.t(), &x, grad_w)?;
    }
    Ok(())
}

fn overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "linear scratch length overflows".to_string(),
    }
}
