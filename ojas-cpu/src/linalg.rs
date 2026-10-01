//! Dense GEMM for `y = x @ W^T`.
//!
//! `W` is `[out, in]`, the `nn.Linear(..., bias=False)` layout. The reduction
//! over `in` runs from index 0 upward in `f32`. Threads partition output
//! rows with `std::thread::scope` and never split that reduction, so the bits
//! do not depend on the thread count. The inner loop does not call `mul_add`.

use ojas_core::{Budget, OjasError};

use crate::validate::{product, room_for, shape};

/// Multiplies below this stay on the calling thread.
/// The tiny step's largest linear is `4*16*32` = 2048.
const GRAIN: usize = 4_096;

pub(crate) fn matmul(
    op: &'static str,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, OjasError> {
    let a_len = product(op, &[m, k])?;
    let b_len = product(op, &[k, n])?;
    let len = product(op, &[m, n])?;
    if a.len() != a_len || b.len() != b_len {
        return Err(shape(
            op,
            format!(
                "matmul lengths a {} b {} != {m}*{k} and {k}*{n}",
                a.len(),
                b.len()
            ),
        ));
    }
    let mut out = vec![0.0f32; len];
    gemm(1, a, b, &mut out, m, k, n);
    Ok(out)
}

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
    threads: usize,
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
    let packed_len = product(op, &[kin, nout])?;
    let peak = y_len
        .checked_add(packed_len)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "linear scratch length overflows".to_string(),
        })?;
    room_for(op, budget, peak)?;
    let packed = pack_weight(weight, nout, kin);
    let mut y = vec![0.0f32; y_len];
    gemm(threads, x, &packed, &mut y, rows, kin, nout);
    Ok((y, y_shape))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_backward(
    op: &'static str,
    budget: &Budget,
    threads: usize,
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
    let peak = gx_len
        .checked_add(gw_len)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "linear backward scratch length overflows".to_string(),
        })?;
    room_for(op, budget, peak)?;
    let mut grad_x = vec![0.0f32; gx_len];
    let mut grad_w = vec![0.0f32; gw_len];
    // grad_x[row, inner] = sum_col grad_y[row, col] * W[col, inner], col from 0.
    // The saxpy walks `inner` contiguously so this stays the same sum.
    map_rows(threads, rows, kin * nout, &mut grad_x, kin, |row, dest| {
        let g_row = &grad_y[row * nout..(row + 1) * nout];
        for col in 0..nout {
            saxpy(dest, &weight[col * kin..(col + 1) * kin], g_row[col]);
        }
    });
    // grad_w[col, inner] = sum_row grad_y[row, col] * x[row, inner], row from 0.
    map_rows(threads, nout, rows * kin, &mut grad_w, kin, |col, dest| {
        for row in 0..rows {
            saxpy(
                dest,
                &x[row * kin..(row + 1) * kin],
                grad_y[row * nout + col],
            );
        }
    });
    Ok((grad_x, grad_w))
}

fn saxpy(dst: &mut [f32], src: &[f32], scale: f32) {
    for (slot, value) in dst.iter_mut().zip(src) {
        *slot += scale * *value;
    }
}

/// `W` is `[out, in]`. The packed matrix is `[in, out]`, so one input index
/// walks output columns contiguously.
fn pack_weight(weight: &[f32], nout: usize, kin: usize) -> Vec<f32> {
    let mut packed = vec![0.0f32; kin * nout];
    for col in 0..nout {
        let row = &weight[col * kin..(col + 1) * kin];
        for (inner, &value) in row.iter().enumerate() {
            packed[inner * nout + col] = value;
        }
    }
    packed
}

/// `b` is row-major `[k, n]`. Each output lane sums `k` in ascending order.
fn gemm(threads: usize, a: &[f32], b: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    map_rows(threads, m, k.saturating_mul(n), y, n, |row, dest| {
        gemm_row(&a[row * k..(row + 1) * k], b, dest, k, n);
    });
}

fn gemm_row(arow: &[f32], b: &[f32], yrow: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(arow.len(), k);
    debug_assert_eq!(yrow.len(), n);
    let mut col = 0;
    while col + 8 <= n {
        let mut acc = [0.0f32; 8];
        for (inner, &av) in arow.iter().enumerate() {
            let base = inner * n + col;
            acc[0] += av * b[base];
            acc[1] += av * b[base + 1];
            acc[2] += av * b[base + 2];
            acc[3] += av * b[base + 3];
            acc[4] += av * b[base + 4];
            acc[5] += av * b[base + 5];
            acc[6] += av * b[base + 6];
            acc[7] += av * b[base + 7];
        }
        yrow[col..col + 8].copy_from_slice(&acc);
        col += 8;
    }
    while col < n {
        let mut acc = 0.0f32;
        for (inner, &av) in arow.iter().enumerate() {
            acc += av * b[inner * n + col];
        }
        yrow[col] = acc;
        col += 1;
    }
}

fn map_rows(
    threads: usize,
    rows: usize,
    work_per_row: usize,
    dest: &mut [f32],
    width: usize,
    body: impl Fn(usize, &mut [f32]) + Send + Sync,
) {
    let work = rows.saturating_mul(work_per_row);
    let workers = if threads <= 1 || rows <= 1 || work < GRAIN || width == 0 {
        1
    } else {
        threads.min(rows)
    };
    if workers == 1 {
        for row in 0..rows {
            body(row, &mut dest[row * width..(row + 1) * width]);
        }
        return;
    }
    let chunk = rows.div_ceil(workers);
    let body = &body;
    std::thread::scope(|scope| {
        let mut rest = dest;
        let mut left = rows;
        let mut row0 = 0usize;
        for _ in 0..workers {
            if left == 0 {
                break;
            }
            let take = chunk.min(left);
            let (mine, tail) = rest.split_at_mut(take * width);
            rest = tail;
            left -= take;
            let start = row0;
            row0 += take;
            scope.spawn(move || {
                for local in 0..take {
                    body(start + local, &mut mine[local * width..(local + 1) * width]);
                }
            });
        }
    });
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
    let mut y_shape = x_shape[..x_shape.len() - 1].to_vec();
    y_shape.push(nout);
    Ok((rows, kin, nout, y_shape))
}
