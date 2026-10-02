//! RMSNorm (`eps` default `1e-6`) and half-split RoPE.
//!
//! RoPE matches nanolab `apply_rope`: split the last axis in half,
//! `rot = cat(-x2, x1)`, then `x * cos + rot * sin`. That is the opposite
//! sign from metal-native's partial RoPE.
//!
//! RMSNorm's row sums (the sum of squares and the backward dot) ascend from
//! index 0 in `f32` without `mul_add` under both contracts, so Fast and
//! Exact give the same bits. A single such sum is one dependent chain of
//! adds; [`GROUP`] rows are summed side by side so that many independent
//! chains run at once, which leaves every row's order unchanged.

use ojas_core::{Budget, OjasError, RmsDims, RopeDims, RopeLayout, Tensor};

use crate::pool::{scoped, Exec, ROW_MIN_ELEMS};
use crate::validate::{fill_rows, nonfinite, product, room_for, shape};

/// Rows whose sums run side by side.
const GROUP: usize = 8;

/// `dims` comes from [`ojas_core::rms_norm_forward_dims`], which also
/// refused a non-finite `eps`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    weight: &[f32],
    dims: RmsDims,
    eps: f32,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RmsDims { rows, dim } = dims;
    let n = product(op, &[rows, dim])?;
    if x.len() != n || weight.len() != dim {
        return Err(shape(op, "rms data length does not match shape"));
    }
    // One rstd per row, across the tasks, held while the output is charged
    // and written.
    let _hold = room_for(op, budget, rows)?;
    fill_rows(op, budget, exec, out_shape, rows, dim, |range, y| {
        let src = &x[range.start * dim..range.end * dim];
        let rstd = rstds(op, src, dim, eps)?;
        let rows = y
            .chunks_exact_mut(dim)
            .zip(src.chunks_exact(dim))
            .zip(&rstd);
        for ((dst, src), &rstd) in rows {
            for ((out, &value), &w) in dst.iter_mut().zip(src).zip(weight.iter()) {
                *out = value * rstd * w;
            }
        }
        Ok(())
    })
}

/// `grad_x` runs in row chunks. `grad_w` is a sum over rows, so it runs in
/// column chunks that each add their rows in increasing order. `dims` comes
/// from [`ojas_core::rms_norm_backward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [x, weight, grad_y]: [&[f32]; 3],
    dims: RmsDims,
    eps: f32,
    [grad_x, grad_w]: [&mut [f32]; 2],
) -> Result<(), OjasError> {
    let RmsDims { rows, dim } = dims;
    let n = product(op, &[rows, dim])?;
    if x.len() != n
        || grad_y.len() != n
        || weight.len() != dim
        || grad_x.len() != n
        || grad_w.len() != dim
    {
        return Err(shape(op, "rms backward data length does not match shape"));
    }
    // One rstd and one dot per row in the tasks, then the rstds joined for
    // the column pass; both gradients are the caller's, written in place.
    let scratch = rows.checked_mul(3).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "rms scratch length overflows".to_string(),
    })?;
    let _hold = room_for(op, budget, scratch)?;
    let inv_dim = 1.0f32 / dim as f32;
    let parts = {
        let min_rows = scoped::min_rows(dim);
        scoped::chunks_into(exec, grad_x, rows, dim, min_rows, |range, grad_x| {
            let src = &x[range.start * dim..range.end * dim];
            let gy = &grad_y[range.start * dim..range.end * dim];
            let rstd = rstds(op, src, dim, eps)?;
            let dots = backward_dots(src, gy, weight, &rstd, dim);
            let rows = grad_x
                .chunks_exact_mut(dim)
                .zip(src.chunks_exact(dim))
                .zip(gy.chunks_exact(dim))
                .zip(rstd.iter().zip(&dots));
            for (((dst, src), gy), (&rstd, &dot)) in rows {
                let mean = dot * inv_dim;
                for (((out, &g), &w), &value) in dst.iter_mut().zip(gy).zip(weight.iter()).zip(src)
                {
                    let dxhat = g * w;
                    let xhat = value * rstd;
                    *out = (dxhat - xhat * mean) * rstd;
                }
            }
            Ok::<_, OjasError>(rstd)
        })?
    };
    let rstd = parts.concat();
    let min_cols = (ROW_MIN_ELEMS / rows).max(1);
    scoped::chunks_into(exec, grad_w, dim, 1, min_cols, |cols, acc| {
        for (row, &r) in rstd.iter().enumerate() {
            let src = &x[row * dim + cols.start..row * dim + cols.end];
            let gy = &grad_y[row * dim + cols.start..row * dim + cols.end];
            for ((slot, &g), &value) in acc.iter_mut().zip(gy).zip(src) {
                let xhat = value * r;
                *slot += g * xhat;
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// `1 / sqrt(mean(row²) + eps)` for each `dim`-wide row of `rows`. Each
/// sum of squares ascends from index 0; [`GROUP`] rows run side by side.
// One column index addresses all [`GROUP`] rows of a block.
#[allow(clippy::needless_range_loop)]
fn rstds(op: &'static str, rows: &[f32], dim: usize, eps: f32) -> Result<Vec<f32>, OjasError> {
    let mut out = Vec::with_capacity(rows.len() / dim.max(1));
    let mut blocks = rows.chunks_exact(GROUP * dim);
    for block in &mut blocks {
        let row: [&[f32]; GROUP] = std::array::from_fn(|r| &block[r * dim..(r + 1) * dim]);
        let mut acc = [0.0f32; GROUP];
        for col in 0..dim {
            for r in 0..GROUP {
                let value = row[r][col];
                acc[r] += value * value;
            }
        }
        for sum_sq in acc {
            out.push(rstd_of_sum(op, sum_sq, dim, eps)?);
        }
    }
    for row in blocks.remainder().chunks_exact(dim) {
        let mut sum_sq = 0.0f32;
        for &value in row {
            sum_sq += value * value;
        }
        out.push(rstd_of_sum(op, sum_sq, dim, eps)?);
    }
    Ok(out)
}

/// `sum_col (gy * w) * (x * rstd)` for each row, ascending from index 0;
/// [`GROUP`] rows run side by side.
fn backward_dots(x: &[f32], gy: &[f32], weight: &[f32], rstd: &[f32], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rstd.len());
    let mut xs = x.chunks_exact(GROUP * dim);
    let mut gs = gy.chunks_exact(GROUP * dim);
    let (rs, rs_rest) = rstd.as_chunks::<GROUP>();
    for ((xb, gb), rb) in (&mut xs).zip(&mut gs).zip(rs) {
        let xr: [&[f32]; GROUP] = std::array::from_fn(|r| &xb[r * dim..(r + 1) * dim]);
        let gr: [&[f32]; GROUP] = std::array::from_fn(|r| &gb[r * dim..(r + 1) * dim]);
        let mut acc = [0.0f32; GROUP];
        for (col, &w) in weight.iter().enumerate().take(dim) {
            for r in 0..GROUP {
                acc[r] += (gr[r][col] * w) * (xr[r][col] * rb[r]);
            }
        }
        out.extend_from_slice(&acc);
    }
    let rows = xs
        .remainder()
        .chunks_exact(dim)
        .zip(gs.remainder().chunks_exact(dim))
        .zip(rs_rest);
    for ((xr, gr), &r) in rows {
        let mut dot = 0.0f32;
        for ((&g, &w), &value) in gr.iter().zip(weight).zip(xr) {
            dot += (g * w) * (value * r);
        }
        out.push(dot);
    }
    out
}

/// `1 / sqrt(sum_sq / dim + eps)`, refusing a result that is not finite.
fn rstd_of_sum(op: &'static str, sum_sq: f32, dim: usize, eps: f32) -> Result<f32, OjasError> {
    let mean_sq = sum_sq / dim as f32;
    let denom = mean_sq + eps;
    if !(denom.is_finite() && denom > 0.0) {
        return Err(nonfinite(op));
    }
    let rstd = 1.0 / denom.sqrt();
    if !rstd.is_finite() {
        return Err(nonfinite(op));
    }
    Ok(rstd)
}

/// `dims` comes from [`ojas_core::rope_half_split_forward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    dims: RopeDims,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RopeDims { rows, dim, .. } = dims;
    let n = product(op, &[rows, dim])?;
    if x.len() != n {
        return Err(shape(op, "rope data length does not match shape"));
    }
    rope_rows(
        op,
        budget,
        exec,
        dims,
        x,
        [cos, sin],
        Direction::Forward,
        out_shape,
    )
}

/// `dims` comes from [`ojas_core::rope_half_split_backward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    grad_y: &[f32],
    cos: &[f32],
    sin: &[f32],
    dims: RopeDims,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RopeDims { rows, dim, .. } = dims;
    let n = product(op, &[rows, dim])?;
    if grad_y.len() != n {
        return Err(shape(op, "rope grad length does not match shape"));
    }
    rope_rows(
        op,
        budget,
        exec,
        dims,
        grad_y,
        [cos, sin],
        Direction::Backward,
        out_shape,
    )
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Backward,
}

#[allow(clippy::too_many_arguments)]
fn rope_rows(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    dims: RopeDims,
    x: &[f32],
    [cos, sin]: [&[f32]; 2],
    direction: Direction,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RopeDims { rows, dim, layout } = dims;
    check_rope_tables(op, layout, rows, dim, cos, sin)?;
    let half = dim / 2;
    fill_rows(op, budget, exec, out_shape, rows, dim, |range, y| {
        for (local, row) in range.enumerate() {
            let (cos_row, sin_row) = coeff_row(layout, cos, sin, row, dim);
            let src = &x[row * dim..(row + 1) * dim];
            let dst = &mut y[local * dim..(local + 1) * dim];
            for col in 0..half {
                let a = src[col];
                let b = src[col + half];
                let c1 = cos_row[col];
                let s1 = sin_row[col];
                let c2 = cos_row[col + half];
                let s2 = sin_row[col + half];
                match direction {
                    // y = x * cos + cat(-x2, x1) * sin
                    Direction::Forward => {
                        dst[col] = a * c1 + (-b) * s1;
                        dst[col + half] = b * c2 + a * s2;
                    }
                    // d/dx1 = c1 from the first half and s2 from the second half.
                    Direction::Backward => {
                        dst[col] = a * c1 + b * s2;
                        dst[col + half] = -a * s1 + b * c2;
                    }
                }
            }
        }
        Ok(())
    })
}

fn check_rope_tables(
    op: &'static str,
    layout: RopeLayout,
    rows: usize,
    dim: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), OjasError> {
    let need = match layout {
        RopeLayout::Same => product(op, &[rows, dim])?,
        RopeLayout::TimeDim { time, .. } => product(op, &[time, dim])?,
    };
    if cos.len() != need || sin.len() != need {
        return Err(shape(op, "rope cos/sin length does not match shape"));
    }
    Ok(())
}

fn coeff_row<'a>(
    layout: RopeLayout,
    cos: &'a [f32],
    sin: &'a [f32],
    row: usize,
    dim: usize,
) -> (&'a [f32], &'a [f32]) {
    let base = match layout {
        RopeLayout::Same => row * dim,
        RopeLayout::TimeDim { time, heads } => {
            let time_index = (row / heads) % time;
            time_index * dim
        }
    };
    (&cos[base..base + dim], &sin[base..base + dim])
}
