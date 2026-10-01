//! Dense GEMM for `y = x @ W^T`.
//!
//! `W` is `[out, in]`, the `nn.Linear(..., bias=False)` layout. The reduction
//! over `in` runs from index 0 upward in `f32`, so the result does not depend
//! on a thread count.

use ojas_core::{Budget, OjasError};

use crate::validate::{flat, get, product, room_for, shape};

pub(crate) fn matmul(
    op: &'static str,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, OjasError> {
    let len = m.checked_mul(n).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "matmul output length overflows".to_string(),
    })?;
    let mut out = vec![0.0f32; len];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                let av = get(op, a, flat(op, row, inner, k)?)?;
                let bv = get(op, b, flat(op, inner, col, n)?)?;
                acc += av * bv;
            }
            let slot = flat(op, row, col, n)?;
            let dest = out.get_mut(slot).ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: "matmul store index exceeds output".to_string(),
            })?;
            *dest = acc;
        }
    }
    Ok(out)
}

pub(crate) fn transpose(
    op: &'static str,
    a: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, OjasError> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "transpose length overflows".to_string(),
        })?;
    if a.len() != len {
        return Err(shape(
            op,
            format!("transpose data len {} != {rows}*{cols}", a.len()),
        ));
    }
    let mut out = vec![0.0f32; len];
    for row in 0..rows {
        for col in 0..cols {
            let src = get(op, a, flat(op, row, col, cols)?)?;
            let dest = flat(op, col, row, rows)?;
            let slot = out.get_mut(dest).ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: "transpose store index exceeds output".to_string(),
            })?;
            *slot = src;
        }
    }
    Ok(out)
}

/// `x` is `[..., in]`, `weight` is `[out, in]`, output is `[..., out]`.
pub(crate) fn linear_forward(
    op: &'static str,
    budget: &Budget,
    x: &[f32],
    x_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), OjasError> {
    let (rows, kin, nout, y_shape) = linear_dims(op, x_shape, weight_shape)?;
    if x.len() != product(op, &[rows, kin])? || weight.len() != product(op, &[nout, kin])? {
        return Err(shape(op, "linear data length does not match shape"));
    }
    let y_len = product(op, &[rows, nout])?;
    room_for(op, budget, y_len)?;
    // y[r, c] = sum_i x[r, i] * W[c, i]
    let mut y = vec![0.0f32; y_len];
    for row in 0..rows {
        for col in 0..nout {
            let mut acc = 0.0f32;
            for inner in 0..kin {
                acc += get(op, x, flat(op, row, inner, kin)?)?
                    * get(op, weight, flat(op, col, inner, kin)?)?;
            }
            y[flat(op, row, col, nout)?] = acc;
        }
    }
    Ok((y, y_shape))
}

pub(crate) fn linear_backward(
    op: &'static str,
    x: &[f32],
    x_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    grad_y: &[f32],
    grad_shape: &[usize],
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    let (rows, kin, nout, y_shape) = linear_dims(op, x_shape, weight_shape)?;
    if grad_shape != y_shape.as_slice() {
        return Err(shape(
            op,
            format!("grad shape {grad_shape:?} != forward shape {y_shape:?}"),
        ));
    }
    let mut grad_x = vec![0.0f32; rows * kin];
    let mut grad_w = vec![0.0f32; nout * kin];
    for row in 0..rows {
        for inner in 0..kin {
            let mut acc = 0.0f32;
            for col in 0..nout {
                acc += get(op, grad_y, flat(op, row, col, nout)?)?
                    * get(op, weight, flat(op, col, inner, kin)?)?;
            }
            grad_x[flat(op, row, inner, kin)?] = acc;
        }
    }
    for col in 0..nout {
        for inner in 0..kin {
            let mut acc = 0.0f32;
            for row in 0..rows {
                acc += get(op, grad_y, flat(op, row, col, nout)?)?
                    * get(op, x, flat(op, row, inner, kin)?)?;
            }
            grad_w[flat(op, col, inner, kin)?] = acc;
        }
    }
    Ok((grad_x, grad_w))
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
    let nout = weight_shape[0];
    let w_in = weight_shape[1];
    if kin != w_in {
        return Err(shape(
            op,
            format!("input in-features {kin} != weight in-features {w_in}"),
        ));
    }
    let rows = product(op, &x_shape[..x_shape.len() - 1])?;
    if rows == 0 {
        return Err(shape(op, "empty tensor"));
    }
    let mut y_shape = x_shape[..x_shape.len() - 1].to_vec();
    y_shape.push(nout);
    Ok((rows, kin, nout, y_shape))
}
