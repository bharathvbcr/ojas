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

use std::sync::Arc;

use ojas_core::{Budget, OjasError, RmsDims, RopeDims, RopeLayout};

use crate::pool::{Exec, ROW_MIN_ELEMS};
use crate::validate::{nonfinite, product, room_for, shape, Shared};

/// Rows whose sums run side by side.
const GROUP: usize = 8;

/// `dims` comes from [`ojas_core::rms_norm_forward_dims`], which also
/// refused a non-finite `eps`.
pub(crate) fn rms_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Shared,
    weight: Shared,
    dims: RmsDims,
    eps: f32,
) -> Result<Vec<f32>, OjasError> {
    let RmsDims { rows, dim } = dims;
    let n = product(op, &[rows, dim])?;
    if x.values()?.len() != n || weight.values()?.len() != dim {
        return Err(shape(op, "rms data length does not match shape"));
    }
    // The output's task parts and one rstd per row; the caller charges the
    // joined output.
    let scratch = n.checked_add(rows).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "rms scratch length overflows".to_string(),
    })?;
    let _hold = room_for(op, budget, scratch)?;
    exec.rows(rows, dim, move |range| {
        let (x, weight) = (x.values()?, weight.values()?);
        let mut y = vec![0.0f32; range.len() * dim];
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
        Ok(y)
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
    x: Shared,
    weight: Shared,
    grad_y: Shared,
    dims: RmsDims,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    let RmsDims { rows, dim } = dims;
    let n = product(op, &[rows, dim])?;
    if x.values()?.len() != n || grad_y.values()?.len() != n || weight.values()?.len() != dim {
        return Err(shape(op, "rms backward data length does not match shape"));
    }
    // grad_x, the weight gradient, and one rstd and one dot per row are
    // live together.
    let scratch = n
        .checked_add(dim)
        .and_then(|v| v.checked_add(rows))
        .and_then(|v| v.checked_add(rows))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "rms scratch length overflows".to_string(),
        })?;
    let _hold = room_for(op, budget, scratch)?;
    let inv_dim = 1.0f32 / dim as f32;
    let parts = {
        let (x, grad_y) = (x.clone(), grad_y.clone());
        let min_rows = (ROW_MIN_ELEMS / dim).max(1);
        exec.chunks(rows, min_rows, move |range| {
            let (x, grad_y, weight) = (x.values()?, grad_y.values()?, weight.values()?);
            let mut grad_x = vec![0.0f32; range.len() * dim];
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
            Ok::<_, OjasError>((grad_x, rstd))
        })?
    };
    let mut grad_x = Vec::with_capacity(n);
    let mut rstd = Vec::with_capacity(rows);
    for part in parts {
        let (gx, rs) = part?;
        grad_x.extend_from_slice(&gx);
        rstd.extend_from_slice(&rs);
    }
    let rstd = Arc::new(rstd);
    let min_cols = (ROW_MIN_ELEMS / rows).max(1);
    let cols = exec.chunks(dim, min_cols, move |cols| {
        let (x, grad_y) = (x.values()?, grad_y.values()?);
        let mut acc = vec![0.0f32; cols.len()];
        for (row, &r) in rstd.iter().enumerate() {
            let src = &x[row * dim + cols.start..row * dim + cols.end];
            let gy = &grad_y[row * dim + cols.start..row * dim + cols.end];
            for ((slot, &g), &value) in acc.iter_mut().zip(gy).zip(src) {
                let xhat = value * r;
                *slot += g * xhat;
            }
        }
        Ok::<_, OjasError>(acc)
    })?;
    let cols = cols.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok((grad_x, cols.concat()))
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
pub(crate) fn rope_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Shared,
    cos: Shared,
    sin: Shared,
    dims: RopeDims,
) -> Result<Vec<f32>, OjasError> {
    let RopeDims { rows, dim, layout } = dims;
    let n = product(op, &[rows, dim])?;
    if x.values()?.len() != n {
        return Err(shape(op, "rope data length does not match shape"));
    }
    rope_rows(
        op,
        budget,
        exec,
        layout,
        rows,
        dim,
        x,
        cos,
        sin,
        Direction::Forward,
    )
}

/// `dims` comes from [`ojas_core::rope_half_split_backward_dims`].
pub(crate) fn rope_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    grad_y: Shared,
    cos: Shared,
    sin: Shared,
    dims: RopeDims,
) -> Result<Vec<f32>, OjasError> {
    let RopeDims { rows, dim, layout } = dims;
    let n = product(op, &[rows, dim])?;
    if grad_y.values()?.len() != n {
        return Err(shape(op, "rope grad length does not match shape"));
    }
    rope_rows(
        op,
        budget,
        exec,
        layout,
        rows,
        dim,
        grad_y,
        cos,
        sin,
        Direction::Backward,
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
    layout: RopeLayout,
    rows: usize,
    dim: usize,
    x: Shared,
    cos: Shared,
    sin: Shared,
    direction: Direction,
) -> Result<Vec<f32>, OjasError> {
    check_rope_tables(op, layout, rows, dim, cos.values()?, sin.values()?)?;
    let n = product(op, &[rows, dim])?;
    let _hold = room_for(op, budget, n)?;
    let half = dim / 2;
    exec.rows(rows, dim, move |range| {
        let (x, cos, sin) = (x.values()?, cos.values()?, sin.values()?);
        let mut y = vec![0.0f32; range.len() * dim];
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
        Ok(y)
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
