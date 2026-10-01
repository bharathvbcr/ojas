//! RMSNorm (`eps` default `1e-6`) and half-split RoPE.
//!
//! RoPE matches nanolab `apply_rope`: split the last axis in half,
//! `rot = cat(-x2, x1)`, then `x * cos + rot * sin`. That is the opposite
//! sign from metal-native's partial RoPE.

use ojas_core::OjasError;

use crate::validate::{flat, get, nonfinite, product, same_shape, shape};

pub(crate) fn rms_forward(
    op: &'static str,
    x: &[f32],
    x_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    eps: f32,
) -> Result<Vec<f32>, OjasError> {
    let (rows, dim) = rms_layout(op, x_shape, weight_shape, eps)?;
    let mut y = vec![0.0f32; x.len()];
    for row in 0..rows {
        let rstd = rstd_of_row(op, x, row, dim, eps)?;
        for col in 0..dim {
            let index = flat(op, row, col, dim)?;
            y[index] = get(op, x, index)? * rstd * get(op, weight, col)?;
        }
    }
    Ok(y)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rms_backward(
    op: &'static str,
    x: &[f32],
    x_shape: &[usize],
    weight: &[f32],
    weight_shape: &[usize],
    grad_y: &[f32],
    grad_shape: &[usize],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    same_shape(op, x_shape, grad_shape)?;
    let (rows, dim) = rms_layout(op, x_shape, weight_shape, eps)?;
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_w = vec![0.0f32; dim];
    let inv_dim = 1.0f32 / dim as f32;
    for row in 0..rows {
        let rstd = rstd_of_row(op, x, row, dim, eps)?;
        let mut dot = 0.0f32;
        for col in 0..dim {
            let index = flat(op, row, col, dim)?;
            let dxhat = get(op, grad_y, index)? * get(op, weight, col)?;
            let xhat = get(op, x, index)? * rstd;
            dot += dxhat * xhat;
        }
        let mean = dot * inv_dim;
        for (col, grad_weight) in grad_w.iter_mut().enumerate() {
            let index = flat(op, row, col, dim)?;
            let dxhat = get(op, grad_y, index)? * get(op, weight, col)?;
            let xhat = get(op, x, index)? * rstd;
            grad_x[index] = (dxhat - xhat * mean) * rstd;
            *grad_weight += get(op, grad_y, index)? * xhat;
        }
    }
    Ok((grad_x, grad_w))
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

fn rstd_of_row(
    op: &'static str,
    x: &[f32],
    row: usize,
    dim: usize,
    eps: f32,
) -> Result<f32, OjasError> {
    let mut sum_sq = 0.0f32;
    for col in 0..dim {
        let value = get(op, x, flat(op, row, col, dim)?)?;
        sum_sq += value * value;
    }
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

#[derive(Clone, Copy)]
enum RopeLayout {
    Same,
    /// `x` is `[batch, time, heads, dim]`, cos/sin are `[time, dim]`.
    TimeDim {
        time: usize,
        heads: usize,
    },
}

pub(crate) fn rope_forward(
    op: &'static str,
    x: &[f32],
    x_shape: &[usize],
    cos: &[f32],
    cos_shape: &[usize],
    sin: &[f32],
    sin_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let (layout, rows, dim) = rope_layout(op, x_shape, cos_shape, sin_shape)?;
    let half = dim / 2;
    let mut y = vec![0.0f32; x.len()];
    for row in 0..rows {
        for col in 0..half {
            let i1 = flat(op, row, col, dim)?;
            let i2 = flat(op, row, col + half, dim)?;
            let x1 = get(op, x, i1)?;
            let x2 = get(op, x, i2)?;
            let (c1, s1) = coeff(op, layout, cos, sin, row, col, dim)?;
            let (c2, s2) = coeff(op, layout, cos, sin, row, col + half, dim)?;
            // y = x * cos + cat(-x2, x1) * sin
            y[i1] = x1 * c1 + (-x2) * s1;
            y[i2] = x2 * c2 + x1 * s2;
        }
    }
    Ok(y)
}

pub(crate) fn rope_backward(
    op: &'static str,
    grad_y: &[f32],
    grad_shape: &[usize],
    cos: &[f32],
    cos_shape: &[usize],
    sin: &[f32],
    sin_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let (layout, rows, dim) = rope_layout(op, grad_shape, cos_shape, sin_shape)?;
    let half = dim / 2;
    let mut grad_x = vec![0.0f32; grad_y.len()];
    for row in 0..rows {
        for col in 0..half {
            let i1 = flat(op, row, col, dim)?;
            let i2 = flat(op, row, col + half, dim)?;
            let g1 = get(op, grad_y, i1)?;
            let g2 = get(op, grad_y, i2)?;
            let (c1, s1) = coeff(op, layout, cos, sin, row, col, dim)?;
            let (c2, s2) = coeff(op, layout, cos, sin, row, col + half, dim)?;
            // d/dx1 = c1 from the first half and s2 from the second half.
            grad_x[i1] = g1 * c1 + g2 * s2;
            grad_x[i2] = -g1 * s1 + g2 * c2;
        }
    }
    Ok(grad_x)
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

fn coeff(
    op: &'static str,
    layout: RopeLayout,
    cos: &[f32],
    sin: &[f32],
    row: usize,
    col: usize,
    dim: usize,
) -> Result<(f32, f32), OjasError> {
    let index = match layout {
        RopeLayout::Same => flat(op, row, col, dim)?,
        RopeLayout::TimeDim { time, heads } => {
            if heads == 0 || time == 0 {
                return Err(shape(op, "rope time or heads is 0"));
            }
            let time_index = (row / heads) % time;
            flat(op, time_index, col, dim)?
        }
    };
    Ok((get(op, cos, index)?, get(op, sin, index)?))
}
