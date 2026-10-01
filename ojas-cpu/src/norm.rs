//! RMSNorm (`eps` default `1e-6`) and half-split RoPE.
//!
//! RoPE matches nanolab `apply_rope`: split the last axis in half,
//! `rot = cat(-x2, x1)`, then `x * cos + rot * sin`. That is the opposite
//! sign from metal-native's partial RoPE.

use std::sync::Arc;

use ojas_core::{Budget, OjasError};

use crate::pool::{Exec, ROW_MIN_ELEMS};
use crate::validate::{nonfinite, product, room_for, same_shape, shape};

#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Vec<f32>,
    x_shape: &[usize],
    weight: Vec<f32>,
    weight_shape: &[usize],
    eps: f32,
) -> Result<Vec<f32>, OjasError> {
    let (rows, dim) = rms_layout(op, x_shape, weight_shape, eps)?;
    let n = product(op, &[rows, dim])?;
    if x.len() != n || weight.len() != dim {
        return Err(shape(op, "rms data length does not match shape"));
    }
    let _hold = room_for(op, budget, n)?;
    let (x, weight) = (Arc::new(x), Arc::new(weight));
    exec.rows(rows, dim, move |range| {
        let mut y = vec![0.0f32; range.len() * dim];
        for (local, row) in range.enumerate() {
            let src = &x[row * dim..(row + 1) * dim];
            let dst = &mut y[local * dim..(local + 1) * dim];
            let rstd = rstd_of_slice(op, src, eps)?;
            for col in 0..dim {
                dst[col] = src[col] * rstd * weight[col];
            }
        }
        Ok(y)
    })
}

/// `grad_x` runs in row chunks. `grad_w` is a sum over rows, so it runs in
/// column chunks that each add their rows in increasing order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Vec<f32>,
    x_shape: &[usize],
    weight: Vec<f32>,
    weight_shape: &[usize],
    grad_y: Vec<f32>,
    grad_shape: &[usize],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    same_shape(op, x_shape, grad_shape)?;
    let (rows, dim) = rms_layout(op, x_shape, weight_shape, eps)?;
    let n = product(op, &[rows, dim])?;
    if x.len() != n || grad_y.len() != n || weight.len() != dim {
        return Err(shape(op, "rms backward data length does not match shape"));
    }
    // grad_x, the weight gradient, and one rstd per row are live together.
    let scratch = n
        .checked_add(dim)
        .and_then(|v| v.checked_add(rows))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "rms scratch length overflows".to_string(),
        })?;
    let _hold = room_for(op, budget, scratch)?;
    let (x, weight, grad_y) = (Arc::new(x), Arc::new(weight), Arc::new(grad_y));
    let inv_dim = 1.0f32 / dim as f32;
    let parts = {
        let (x, grad_y) = (Arc::clone(&x), Arc::clone(&grad_y));
        let min_rows = (ROW_MIN_ELEMS / dim).max(1);
        exec.chunks(rows, min_rows, move |range| {
            let mut grad_x = vec![0.0f32; range.len() * dim];
            let mut rstds = Vec::with_capacity(range.len());
            for (local, row) in range.enumerate() {
                let src = &x[row * dim..(row + 1) * dim];
                let gy = &grad_y[row * dim..(row + 1) * dim];
                let dst = &mut grad_x[local * dim..(local + 1) * dim];
                let rstd = rstd_of_slice(op, src, eps)?;
                let mut dot = 0.0f32;
                for col in 0..dim {
                    let dxhat = gy[col] * weight[col];
                    let xhat = src[col] * rstd;
                    dot += dxhat * xhat;
                }
                let mean = dot * inv_dim;
                for col in 0..dim {
                    let dxhat = gy[col] * weight[col];
                    let xhat = src[col] * rstd;
                    dst[col] = (dxhat - xhat * mean) * rstd;
                }
                rstds.push(rstd);
            }
            Ok::<_, OjasError>((grad_x, rstds))
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
        let mut acc = vec![0.0f32; cols.len()];
        for (row, &r) in rstd.iter().enumerate() {
            let src = &x[row * dim + cols.start..row * dim + cols.end];
            let gy = &grad_y[row * dim + cols.start..row * dim + cols.end];
            for ((slot, &g), &value) in acc.iter_mut().zip(gy).zip(src) {
                let xhat = value * r;
                *slot += g * xhat;
            }
        }
        acc
    })?;
    Ok((grad_x, cols.concat()))
}

fn rms_layout(
    op: &'static str,
    x_shape: &[usize],
    weight_shape: &[usize],
    eps: f32,
) -> Result<(usize, usize), OjasError> {
    if !eps.is_finite() {
        return Err(nonfinite(op));
    }
    let dim = match x_shape.last().copied() {
        Some(dim) => dim,
        None => return Err(shape(op, "rms_norm input rank 0")),
    };
    if weight_shape.len() != 1 || weight_shape[0] != dim {
        return Err(shape(
            op,
            format!("rms weight {weight_shape:?} != last dim {dim}"),
        ));
    }
    let rows = product(op, &x_shape[..x_shape.len() - 1])?;
    if rows == 0 {
        return Err(shape(op, "empty tensor"));
    }
    Ok((rows, dim))
}

fn rstd_of_slice(op: &'static str, row: &[f32], eps: f32) -> Result<f32, OjasError> {
    let mut sum_sq = 0.0f32;
    for &value in row {
        sum_sq += value * value;
    }
    let mean_sq = sum_sq / row.len() as f32;
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

#[derive(Clone, Copy)]
enum RopeLayout {
    Same,
    /// `x` is `[batch, time, heads, dim]`, cos/sin are `[time, dim]`.
    TimeDim {
        time: usize,
        heads: usize,
    },
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: Vec<f32>,
    x_shape: &[usize],
    cos: Vec<f32>,
    cos_shape: &[usize],
    sin: Vec<f32>,
    sin_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let (layout, rows, dim) = rope_layout(op, x_shape, cos_shape, sin_shape)?;
    let n = product(op, &[rows, dim])?;
    if x.len() != n {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    grad_y: Vec<f32>,
    grad_shape: &[usize],
    cos: Vec<f32>,
    cos_shape: &[usize],
    sin: Vec<f32>,
    sin_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let (layout, rows, dim) = rope_layout(op, grad_shape, cos_shape, sin_shape)?;
    let n = product(op, &[rows, dim])?;
    if grad_y.len() != n {
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

fn rope_layout(
    op: &'static str,
    x_shape: &[usize],
    cos_shape: &[usize],
    sin_shape: &[usize],
) -> Result<(RopeLayout, usize, usize), OjasError> {
    if cos_shape != sin_shape {
        return Err(shape(
            op,
            format!("cos shape {cos_shape:?} != sin shape {sin_shape:?}"),
        ));
    }
    let dim = match x_shape.last().copied() {
        Some(dim) => dim,
        None => return Err(shape(op, "rope input rank 0")),
    };
    if dim % 2 != 0 {
        return Err(shape(op, format!("rope last dim {dim} is odd")));
    }
    let rows = product(op, &x_shape[..x_shape.len() - 1])?;
    if rows == 0 {
        return Err(shape(op, "empty tensor"));
    }
    if cos_shape == x_shape {
        return Ok((RopeLayout::Same, rows, dim));
    }
    if x_shape.len() == 4
        && cos_shape.len() == 2
        && cos_shape[0] == x_shape[1]
        && cos_shape[1] == dim
    {
        return Ok((
            RopeLayout::TimeDim {
                time: x_shape[1],
                heads: x_shape[2],
            },
            rows,
            dim,
        ));
    }
    Err(shape(
        op,
        format!("cos/sin shape {cos_shape:?} does not broadcast onto {x_shape:?}"),
    ))
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
    x: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    direction: Direction,
) -> Result<Vec<f32>, OjasError> {
    check_rope_tables(op, layout, rows, dim, &cos, &sin)?;
    let n = product(op, &[rows, dim])?;
    let _hold = room_for(op, budget, n)?;
    let half = dim / 2;
    let (x, cos, sin) = (Arc::new(x), Arc::new(cos), Arc::new(sin));
    exec.rows(rows, dim, move |range| {
        let mut y = vec![0.0f32; range.len() * dim];
        for (local, row) in range.enumerate() {
            let (cos_row, sin_row) = coeff_row(layout, &cos, &sin, row, dim);
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
        RopeLayout::TimeDim { time, heads } => {
            if heads == 0 || time == 0 {
                return Err(shape(op, "rope time or heads is 0"));
            }
            product(op, &[time, dim])?
        }
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
