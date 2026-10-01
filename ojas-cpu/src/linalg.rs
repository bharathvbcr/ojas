//! `y = x @ W^T` and its gradients, as strided views over one GEMM core.
//!
//! `W` is `[out, in]`, the `nn.Linear(..., bias=False)` layout. The forward
//! is `x · Wᵀ`, `grad_x` is `g · W`, and `grad_w` is `gᵀ · x`; the transposes
//! are strides that [`crate::gemm`] packs through, not copies. Under
//! [`ojas_core::Numerics::Exact`] each output sums its reduction axis from
//! index 0 upward in `f32` without `mul_add`, and the bits do not depend on
//! the thread count.

use std::sync::Arc;

use ojas_core::{BackendId, Budget, OjasError};

use crate::gemm::{gemm, scratch, single_task, whole_call, Mat, TASK_MACS};
use crate::pool::Exec;
use crate::validate::{product, room_for, shape, F32Out};

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

/// `x` is `[..., in]`, `weight` is `[out, in]`, output is `[..., out]`.
pub(crate) fn linear_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Vec<f32>,
    x_shape: &[usize],
    weight: Vec<f32>,
    weight_shape: &[usize],
) -> Result<(F32Out, Vec<usize>), OjasError> {
    let (rows, kin, nout, y_shape) = linear_dims(op, x_shape, weight_shape)?;
    if x.len() != product(op, &[rows, kin])? || weight.len() != product(op, &[nout, kin])? {
        return Err(shape(op, "linear data length does not match shape"));
    }
    let y_len = product(op, &[rows, nout])?;
    let work = scratch(op, exec, rows, kin, nout)?;
    // The output's charge leaves with it and is held until its tensor copy
    // exists; the scratch charge ends here.
    let charge = room_for(op, budget, y_len)?;
    let _scratch = room_for(op, budget, work)?;
    let x = Mat::row_major(Arc::new(x), rows, kin);
    let w = Mat::row_major(Arc::new(weight), nout, kin);
    let data = gemm(op, exec, &x, &w.t())?;
    Ok((F32Out { data, charge }, y_shape))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Vec<f32>,
    x_shape: &[usize],
    weight: Vec<f32>,
    weight_shape: &[usize],
    grad_y: Vec<f32>,
    grad_shape: &[usize],
) -> Result<(F32Out, F32Out), OjasError> {
    let (rows, kin, nout, y_shape) = linear_dims(op, x_shape, weight_shape)?;
    if grad_shape != y_shape.as_slice() {
        return Err(shape(
            op,
            format!("grad shape {grad_shape:?} != forward shape {y_shape:?}"),
        ));
    }
    if x.len() != product(op, &[rows, kin])?
        || weight.len() != product(op, &[nout, kin])?
        || grad_y.len() != product(op, &[rows, nout])?
    {
        return Err(shape(
            op,
            "linear backward data length does not match shape",
        ));
    }
    let gx_len = product(op, &[rows, kin])?;
    let gw_len = product(op, &[nout, kin])?;
    // Two products that each fit in one task run side by side, so their
    // scratch is live at once; larger ones split themselves in turn. A whole
    // `ojas-simd` call must not run inside a pool task.
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
    // Each gradient's charge leaves with it (see `linear_forward`).
    let gx_charge = room_for(op, budget, gx_len)?;
    let gw_charge = room_for(op, budget, gw_len)?;
    let _work = room_for(op, budget, work)?;
    let x = Mat::row_major(Arc::new(x), rows, kin);
    let w = Mat::row_major(Arc::new(weight), nout, kin);
    let g = Mat::row_major(Arc::new(grad_y), rows, nout);
    // grad_x[row, inner] = sum_col g[row, col] * W[col, inner], col from 0.
    // grad_w[col, inner] = sum_row g[row, col] * x[row, inner], row from 0.
    let (grad_x, grad_w) = if pair {
        let pool = Arc::clone(exec.pool);
        let numerics = exec.numerics;
        let both = exec.pool.run(2, move |i| {
            let exec = Exec {
                pool: &pool,
                numerics,
            };
            if i == 0 {
                gemm(op, exec, &g, &w)
            } else {
                gemm(op, exec, &g.t(), &x)
            }
        })?;
        let [gx, gw] = <[_; 2]>::try_from(both).map_err(|_| OjasError::Backend {
            id: BackendId::Cpu,
            detail: format!("{op}: pool returned the wrong number of results"),
        })?;
        (gx?, gw?)
    } else {
        (gemm(op, exec, &g, &w)?, gemm(op, exec, &g.t(), &x)?)
    };
    Ok((
        F32Out {
            data: grad_x,
            charge: gx_charge,
        },
        F32Out {
            data: grad_w,
            charge: gw_charge,
        },
    ))
}

fn overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "linear scratch length overflows".to_string(),
    }
}

fn linear_dims(
    op: &'static str,
    x_shape: &[usize],
    weight_shape: &[usize],
) -> Result<(usize, usize, usize, Vec<usize>), OjasError> {
    if weight_shape.len() != 2 {
        return Err(shape(
            op,
            format!("weight rank {} != 2 ([out, in])", weight_shape.len()),
        ));
    }
    let kin = match x_shape.last().copied() {
        Some(dim) => dim,
        None => return Err(shape(op, "input rank 0 has no in-features")),
    };
    if kin == 0 {
        return Err(shape(op, "in_dim is 0"));
    }
    let nout = weight_shape[0];
    let w_in = weight_shape[1];
    if kin != w_in {
        return Err(shape(
            op,
            format!("input in-features {kin} != weight in-features {w_in}"),
        ));
    }
    let rows = product(op, &x_shape[..x_shape.len() - 1])?;
    if rows == 0 || nout == 0 {
        return Err(shape(op, "empty tensor"));
    }
    let mut y_shape = Vec::with_capacity(x_shape.len());
    y_shape.extend_from_slice(&x_shape[..x_shape.len() - 1]);
    y_shape.push(nout);
    Ok((rows, kin, nout, y_shape))
}
