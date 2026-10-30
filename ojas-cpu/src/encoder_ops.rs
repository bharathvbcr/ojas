//! Forward ops of a transformer encoder that is not nanolab's GPT: LayerNorm
//! with a bias (plain, across the channel axis of an image, and per channel
//! group), the exact GELU, a linear layer's bias, broadcast add and multiply, a scalar scale,
//! and a standalone softmax; and the backward of all of them but GroupNorm.
//!
//! Every row, channel and softmax reduction ascends from index 0 without
//! `mul_add` (LayerNorm's in `f64`, the softmax sum in `f32`), and no output
//! value depends on which chunk computed it, so the bits do not depend on the
//! thread count, and `Fast` and `Exact` agree bit for bit, except for the
//! softmax exponential (below). Each output is scanned for NaN and infinity
//! as it is stored ([`fill_rows`]).
//!
//! - LayerNorm is torch's: `(x - mean) * rstd * weight + bias` with the
//!   biased variance, `mean = sum / dim` and `var = sum((x - mean)^2) / dim`,
//!   `rstd = 1 / sqrt(var + eps)`. The reference is exact rather than fast: both
//!   sums are `f64`, ascending, and the deviation is taken from the `f64`
//!   mean, so a row whose mean is far from zero (SAM's residual stream) is not
//!   limited by an `f32` sum's rounding. The normalized value `(x - mean) *
//!   rstd` is rounded once to `f32`, then scaled and shifted in `f32`.
//! - GroupNorm is torch's `nn.GroupNorm`: each `(batch, group)` row of
//!   `channels / groups` channels and every position is normalized with the
//!   same two-pass `f64` statistics and [`normalized`] arithmetic as LayerNorm,
//!   `weight` and `bias` indexed by the channel.
//! - GELU is [`ojas_core::gelu_erf_exact`], the same bits on every platform.
//! - Softmax subtracts the row maximum, exponentiates with
//!   [`ojas_core::exp_exact`] under [`ojas_core::Numerics::Exact`] and with
//!   the vector exponential of [`crate::exp`] under
//!   [`ojas_core::Numerics::Fast`], sums ascending
//!   (Fast: in the fixed lane order of [`crate::exp::exp_sub_store`]) and
//!   divides. The row kernel is [`crate::attn::softmax_row`], the one the
//!   attention softmax uses.
//!
//! The backwards are gathers: no scatter, no atomics, and no sum is split
//! across threads. A value that is a sum over rows (`dw` and `db` of a norm,
//! a bias gradient, a broadcast operand's gradient) is one `f64` sum in
//! ascending order, chunked over the values it produces, and rounded once;
//! so no backward depends on the thread count, and none uses the
//! exponential, so `Fast` and `Exact` agree bit for bit. A norm's backward
//! takes its row statistics from the forward's [`row_stats`] and
//! [`channel_stats`], so it sees the forward's own `rstd`. Each gradient
//! is scanned for NaN and infinity as it is stored, so an overflowing
//! gradient is [`OjasError::NonFinite`], not `inf`.

use ojas_core::{
    gelu_erf_backward_exact, gelu_erf_exact, BroadcastDims, Budget, ChannelNormDims, GroupNormDims,
    OjasError, RmsDims, Tensor,
};

use crate::attn::softmax_row;
use crate::pool::{scoped, Exec, ROW_MIN_ELEMS};
use crate::validate::{fill_rows, nonfinite, product, room_for, shape};

/// `1 / sqrt(sum_sq / dim + eps)`, refusing a result that is not finite.
fn layer_rstd(op: &'static str, sum_sq_dev: f64, dim: usize, eps: f32) -> Result<f64, OjasError> {
    let denom = sum_sq_dev / dim as f64 + f64::from(eps);
    if !(denom.is_finite() && denom > 0.0) {
        return Err(nonfinite(op));
    }
    let rstd = 1.0 / denom.sqrt();
    if !rstd.is_finite() {
        return Err(nonfinite(op));
    }
    Ok(rstd)
}

/// The `f64` mean and reciprocal deviation `1 / sqrt(var + eps)` of one row:
/// two ascending passes, the deviation taken from the `f64` mean. The forward
/// and the backward of LayerNorm and GroupNorm all take their statistics
/// here, so the backward sees exactly the `rstd` the forward used.
fn row_stats(op: &'static str, row: &[f32], eps: f32) -> Result<(f64, f64), OjasError> {
    let mut sum = 0.0f64;
    for &v in row {
        sum += f64::from(v);
    }
    let mean = sum / row.len() as f64;
    let mut dev = 0.0f64;
    for &v in row {
        let d = f64::from(v) - mean;
        dev += d * d;
    }
    Ok((mean, layer_rstd(op, dev, row.len(), eps)?))
}

/// The `f64` mean and reciprocal deviation of each of an image's `spatial`
/// positions over its `channels` planes (`img` is `[channels, spatial]`),
/// into `mean` and `rstd`. The walk visits the channels in order, each
/// contiguous over `spatial`, so a position's sums ascend over its channels
/// exactly as [`row_stats`]'s do.
fn channel_stats(
    op: &'static str,
    img: &[f32],
    channels: usize,
    spatial: usize,
    eps: f32,
    mean: &mut [f64],
    rstd: &mut [f64],
) -> Result<(), OjasError> {
    mean.fill(0.0);
    for plane in img.chunks_exact(spatial) {
        for (m, &v) in mean.iter_mut().zip(plane) {
            *m += f64::from(v);
        }
    }
    for m in mean.iter_mut() {
        *m /= channels as f64;
    }
    rstd.fill(0.0);
    for plane in img.chunks_exact(spatial) {
        for ((r, &v), &m) in rstd.iter_mut().zip(plane).zip(mean.iter()) {
            let d = f64::from(v) - m;
            *r += d * d;
        }
    }
    for r in rstd.iter_mut() {
        *r = layer_rstd(op, *r, channels, eps)?;
    }
    Ok(())
}

/// One normalized, scaled and shifted element: the shared arithmetic of
/// [`layer_norm_forward`] and [`channel_layer_norm_forward`], so the two agree
/// bit for bit.
fn normalized(v: f32, mean: f64, rstd: f64, weight: f32, bias: f32) -> f32 {
    (((f64::from(v) - mean) * rstd) as f32) * weight + bias
}

/// `dims` comes from [`ojas_core::layer_norm_forward_dims`], which also
/// refused a non-finite `eps`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    dims: RmsDims,
    eps: f32,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RmsDims { rows, dim } = dims;
    if x.len() != product(op, &[rows, dim])? || weight.len() != dim || bias.len() != dim {
        return Err(shape(op, "layer_norm data length does not match shape"));
    }
    fill_rows(op, budget, exec, out_shape, rows, dim, |range, y| {
        let src = &x[range.start * dim..range.end * dim];
        for (dst, row) in y.chunks_exact_mut(dim).zip(src.chunks_exact(dim)) {
            let (mean, rstd) = row_stats(op, row, eps)?;
            for (((out, &v), &w), &b) in dst.iter_mut().zip(row).zip(weight).zip(bias) {
                *out = normalized(v, mean, rstd, w, b);
            }
        }
        Ok(())
    })
}

/// LayerNorm over the channel axis of `[batch, channels, spatial]`. A batch
/// element is one chunk: its `spatial` means, variances and reciprocal
/// deviations are built by walking the channels in order, each walk
/// contiguous over `spatial`, so a position's sums ascend over its channels
/// exactly as [`layer_norm_forward`]'s row sums do.
/// `dims` comes from [`ojas_core::channel_layer_norm_forward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn channel_layer_norm_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    dims: ChannelNormDims,
    eps: f32,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let ChannelNormDims {
        batch,
        channels,
        spatial,
    } = dims;
    let width = product(op, &[channels, spatial])?;
    if x.len() != product(op, &[batch, width])?
        || weight.len() != channels
        || bias.len() != channels
    {
        return Err(shape(
            op,
            "channel_layer_norm data length does not match shape",
        ));
    }
    // Per task: the mean and reciprocal deviation of each position, in `f64`,
    // held beside the output (at most two tasks a thread, as [`fill_rows`]
    // cuts). Charged in `f32` elements, two apiece.
    let tasks = exec.pool.threads().saturating_mul(2).max(1);
    let _hold = room_for(op, budget, spatial.saturating_mul(4).saturating_mul(tasks))?;
    fill_rows(op, budget, exec, out_shape, batch, width, |range, y| {
        let mut mean = vec![0.0f64; spatial];
        let mut rstd = vec![0.0f64; spatial];
        for (dst, img) in y
            .chunks_exact_mut(width)
            .zip(x[range.start * width..range.end * width].chunks_exact(width))
        {
            channel_stats(op, img, channels, spatial, eps, &mut mean, &mut rstd)?;
            for ((out_plane, plane), (&w, &b)) in dst
                .chunks_exact_mut(spatial)
                .zip(img.chunks_exact(spatial))
                .zip(weight.iter().zip(bias))
            {
                for (((out, &v), &m), &r) in out_plane.iter_mut().zip(plane).zip(&mean).zip(&rstd) {
                    *out = normalized(v, m, r, w, b);
                }
            }
        }
        Ok(())
    })
}

/// GroupNorm over `[batch, channels, spatial]`. A `(batch, group)` pair is a
/// row: its `channels / groups` channels are contiguous, so the row's two
/// `f64` sums ascend over them and `spatial` exactly as
/// [`layer_norm_forward`]'s row sums do, and a row needs no scratch beyond
/// the output [`fill_rows`] charges. Channel `g * (channels / groups) + j`
/// supplies `weight` and `bias` for the `j`th run of `spatial` values.
/// `dims` comes from [`ojas_core::group_norm_forward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_norm_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    dims: GroupNormDims,
    eps: f32,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let GroupNormDims {
        batch,
        channels,
        groups,
        spatial,
    } = dims;
    if groups == 0 || !channels.is_multiple_of(groups) {
        return Err(shape(op, "group_norm groups do not divide the channels"));
    }
    let per_group = channels / groups;
    let width = product(op, &[per_group, spatial])?;
    let rows = product(op, &[batch, groups])?;
    if x.len() != product(op, &[rows, width])? || weight.len() != channels || bias.len() != channels
    {
        return Err(shape(op, "group_norm data length does not match shape"));
    }
    fill_rows(op, budget, exec, out_shape, rows, width, |range, y| {
        let src = &x[range.start * width..range.end * width];
        for (row, (dst, grp)) in range.zip(y.chunks_exact_mut(width).zip(src.chunks_exact(width))) {
            let (mean, rstd) = row_stats(op, grp, eps)?;
            let first = (row % groups) * per_group;
            for ((out_plane, plane), (&w, &b)) in dst
                .chunks_exact_mut(spatial)
                .zip(grp.chunks_exact(spatial))
                .zip(
                    weight[first..first + per_group]
                        .iter()
                        .zip(&bias[first..first + per_group]),
                )
            {
                for (out, &v) in out_plane.iter_mut().zip(plane) {
                    *out = normalized(v, mean, rstd, w, b);
                }
            }
        }
        Ok(())
    })
}

/// The exact (erf) GELU of every element.
pub(crate) fn gelu_erf_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    fill_rows(op, budget, exec, out_shape, x.len(), 1, |range, y| {
        for (out, &v) in y.iter_mut().zip(&x[range]) {
            *out = gelu_erf_exact(v);
        }
        Ok(())
    })
}

/// `max(x, 0)` of every element: `x` unless it is negative, then `+0`. A
/// negative zero passes through as it is (it is not negative); torch's own
/// sign for that one value was not checked.
pub(crate) fn relu_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    fill_rows(op, budget, exec, out_shape, x.len(), 1, |range, y| {
        for (out, &v) in y.iter_mut().zip(&x[range]) {
            *out = if v < 0.0 { 0.0 } else { v };
        }
        Ok(())
    })
}

/// `y[r, :] += bias` over `[rows, out]` rows in place: the one `f32` add
/// [`ojas_core::Backend::linear_bias_forward`] makes after the product.
pub(crate) fn add_row_bias(op: &'static str, y: &mut [f32], bias: &[f32]) -> Result<(), OjasError> {
    if bias.is_empty() || !y.len().is_multiple_of(bias.len()) {
        return Err(shape(op, "bias length does not divide the output"));
    }
    for row in y.chunks_exact_mut(bias.len()) {
        for (v, &b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
    Ok(())
}

/// `f(a, b)` over the broadcast of `a` and `b`: each output row (the last
/// output axis) reads `a` and `b` at strides 0 or 1, and starts at the
/// offsets its leading indices give. `dims` comes from
/// [`ojas_core::broadcast_add_forward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn broadcast_zip(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    a: &[f32],
    b: &[f32],
    dims: &BroadcastDims,
    f: impl Fn(f32, f32) -> f32 + Sync,
) -> Result<Tensor, OjasError> {
    let out_shape = dims.out_shape.as_slice();
    let n = product(op, out_shape)?;
    let width = out_shape.last().copied().unwrap_or(1);
    let rows = n / width.max(1);
    let (sa, sb) = (
        dims.a_strides.last().copied().unwrap_or(0),
        dims.b_strides.last().copied().unwrap_or(0),
    );
    let lead = out_shape.len().saturating_sub(1);
    fill_rows(op, budget, exec, out_shape, rows, width, |range, y| {
        let mut idx = vec![0usize; lead];
        // Row `range.start`'s leading indices and operand offsets.
        let (mut ao, mut bo) = (0usize, 0usize);
        let mut rest = range.start;
        for axis in (0..lead).rev() {
            idx[axis] = rest % out_shape[axis];
            rest /= out_shape[axis];
            ao += idx[axis] * dims.a_strides[axis];
            bo += idx[axis] * dims.b_strides[axis];
        }
        for dst in y.chunks_exact_mut(width) {
            let last_a = ao + sa * (width - 1);
            let last_b = bo + sb * (width - 1);
            if last_a >= a.len() || last_b >= b.len() {
                return Err(shape(op, "broadcast operand shorter than its shape"));
            }
            match (sa, sb) {
                (1, 1) => {
                    for ((out, &x), &y) in dst.iter_mut().zip(&a[ao..]).zip(&b[bo..]) {
                        *out = f(x, y);
                    }
                }
                (1, 0) => {
                    let y = b[bo];
                    for (out, &x) in dst.iter_mut().zip(&a[ao..]) {
                        *out = f(x, y);
                    }
                }
                (0, 1) => {
                    let x = a[ao];
                    for (out, &y) in dst.iter_mut().zip(&b[bo..]) {
                        *out = f(x, y);
                    }
                }
                _ => dst.fill(f(a[ao], b[bo])),
            }
            // Next row: carry through the leading indices.
            for axis in (0..lead).rev() {
                idx[axis] += 1;
                ao += dims.a_strides[axis];
                bo += dims.b_strides[axis];
                if idx[axis] < out_shape[axis] {
                    break;
                }
                ao -= idx[axis] * dims.a_strides[axis];
                bo -= idx[axis] * dims.b_strides[axis];
                idx[axis] = 0;
            }
        }
        Ok(())
    })
}

/// `x * scale` for every element.
pub(crate) fn scale_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    scale: f32,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    fill_rows(op, budget, exec, out_shape, x.len(), 1, |range, y| {
        for (out, &v) in y.iter_mut().zip(&x[range]) {
            *out = v * scale;
        }
        Ok(())
    })
}

/// Softmax over the last axis. `dims` comes from
/// [`ojas_core::softmax_forward_dims`].
pub(crate) fn softmax_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    dims: RmsDims,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RmsDims { rows, dim } = dims;
    if x.len() != product(op, &[rows, dim])? {
        return Err(shape(op, "softmax data length does not match shape"));
    }
    let numerics = exec.numerics;
    fill_rows(op, budget, exec, out_shape, rows, dim, |range, y| {
        let src = &x[range.start * dim..range.end * dim];
        for (dst, row) in y.chunks_exact_mut(dim).zip(src.chunks_exact(dim)) {
            let _ = softmax_row(numerics, row, dst);
        }
        Ok(())
    })
}

/// LayerNorm's three gradients. With `xhat = (x - mean) * rstd` and
/// `dxhat = g * w`, a row's input gradient is `rstd * (dxhat - mean(dxhat) -
/// xhat * mean(dxhat * xhat))`, all in `f64` and rounded once. It runs in row
/// chunks. `grad_w` (`sum g * xhat`) and `grad_b` (`sum g`) are sums over the
/// rows, so they run in column chunks that each add their rows in increasing
/// order, from the `(mean, rstd)` the row pass kept. `dims` comes from
/// [`ojas_core::layer_norm_backward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [x, weight, grad_y]: [&[f32]; 3],
    dims: RmsDims,
    eps: f32,
    [grad_x, grad_w, grad_b]: [&mut [f32]; 3],
) -> Result<(), OjasError> {
    let RmsDims { rows, dim } = dims;
    let n = product(op, &[rows, dim])?;
    if x.len() != n
        || grad_y.len() != n
        || weight.len() != dim
        || grad_x.len() != n
        || grad_w.len() != dim
        || grad_b.len() != dim
    {
        return Err(shape(
            op,
            "layer_norm backward data length does not match shape",
        ));
    }
    // A `(mean, rstd)` pair of `f64` per row, held in the tasks' parts and
    // again joined for the column pass (four `f32` elements each time), and
    // the column pass's two `f64` accumulators per column.
    let held = product(op, &[rows, 8])?.saturating_add(dim.saturating_mul(4));
    let _hold = room_for(op, budget, held)?;
    let parts = scoped::chunks_into(
        exec,
        grad_x,
        rows,
        dim,
        scoped::min_rows(dim),
        |range, grad_x| {
            let src = &x[range.start * dim..range.end * dim];
            let gy = &grad_y[range.start * dim..range.end * dim];
            let mut stats = Vec::with_capacity(range.len());
            let rows = grad_x
                .chunks_exact_mut(dim)
                .zip(src.chunks_exact(dim))
                .zip(gy.chunks_exact(dim));
            for ((dst, row), g_row) in rows {
                let (mean, rstd) = row_stats(op, row, eps)?;
                let (mut sum_g, mut sum_gx) = (0.0f64, 0.0f64);
                for ((&v, &g), &w) in row.iter().zip(g_row).zip(weight) {
                    let dxhat = f64::from(g) * f64::from(w);
                    sum_g += dxhat;
                    sum_gx += dxhat * ((f64::from(v) - mean) * rstd);
                }
                let (mean_g, mean_gx) = (sum_g / dim as f64, sum_gx / dim as f64);
                for (((out, &v), &g), &w) in dst.iter_mut().zip(row).zip(g_row).zip(weight) {
                    let xhat = (f64::from(v) - mean) * rstd;
                    let dxhat = f64::from(g) * f64::from(w);
                    *out = (rstd * (dxhat - mean_g - xhat * mean_gx)) as f32;
                }
                stats.push((mean, rstd));
            }
            Ok::<_, OjasError>(stats)
        },
    )?;
    let stats = parts.concat();
    let min_cols = (ROW_MIN_ELEMS / rows).max(1);
    scoped::chunks_into_n(
        exec,
        [grad_w, grad_b],
        dim,
        [1, 1],
        min_cols,
        |cols, [grad_w, grad_b]| {
            let mut sum_gx = vec![0.0f64; cols.len()];
            let mut sum_g = vec![0.0f64; cols.len()];
            for (row, &(mean, rstd)) in stats.iter().enumerate() {
                let at = row * dim;
                let src = &x[at + cols.start..at + cols.end];
                let gy = &grad_y[at + cols.start..at + cols.end];
                let acc = sum_gx.iter_mut().zip(sum_g.iter_mut());
                for (((a_gx, a_g), &v), &g) in acc.zip(src).zip(gy) {
                    let g = f64::from(g);
                    *a_gx += g * ((f64::from(v) - mean) * rstd);
                    *a_g += g;
                }
            }
            for (out, sum) in grad_w.iter_mut().zip(&sum_gx) {
                *out = *sum as f32;
            }
            for (out, sum) in grad_b.iter_mut().zip(&sum_g) {
                *out = *sum as f32;
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// [`layer_norm_backward`] across the channel axis of `[batch, channels,
/// spatial]`. A batch element is one chunk: its positions' statistics are
/// [`channel_stats`], and each position's two sums over its channels walk
/// the channels in order, each walk contiguous over `spatial`, so a sum
/// ascends exactly as [`layer_norm_backward`]'s row sums do. `grad_w[c]` and
/// `grad_b[c]` sum over the batch and then the positions, both ascending.
/// `dims` comes from [`ojas_core::channel_layer_norm_backward_dims`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn channel_layer_norm_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [x, weight, grad_y]: [&[f32]; 3],
    dims: ChannelNormDims,
    eps: f32,
    [grad_x, grad_w, grad_b]: [&mut [f32]; 3],
) -> Result<(), OjasError> {
    let ChannelNormDims {
        batch,
        channels,
        spatial,
    } = dims;
    let width = product(op, &[channels, spatial])?;
    let n = product(op, &[batch, width])?;
    if x.len() != n
        || grad_y.len() != n
        || weight.len() != channels
        || grad_x.len() != n
        || grad_w.len() != channels
        || grad_b.len() != channels
    {
        return Err(shape(
            op,
            "channel_layer_norm backward data length does not match shape",
        ));
    }
    let positions = product(op, &[batch, spatial])?;
    // The `(mean, rstd)` of every position, in the tasks' parts and joined
    // (four `f32` elements a position each time), and per task the four
    // `f64` rows `mean`, `rstd` and the two sums, `spatial` long.
    let tasks = exec.pool.threads().saturating_mul(2).max(1);
    let held = positions
        .saturating_mul(8)
        .saturating_add(spatial.saturating_mul(8).saturating_mul(tasks));
    let _hold = room_for(op, budget, held)?;
    let parts = scoped::chunks_into(
        exec,
        grad_x,
        batch,
        width,
        scoped::min_rows(width),
        |range, grad_x| {
            let mut mean = vec![0.0f64; spatial];
            let mut rstd = vec![0.0f64; spatial];
            let mut sum_g = vec![0.0f64; spatial];
            let mut sum_gx = vec![0.0f64; spatial];
            let mut stats = Vec::with_capacity(range.len().saturating_mul(spatial));
            let images = grad_x.chunks_exact_mut(width).zip(range);
            for (dst, image) in images {
                let img = &x[image * width..(image + 1) * width];
                let g_img = &grad_y[image * width..(image + 1) * width];
                channel_stats(op, img, channels, spatial, eps, &mut mean, &mut rstd)?;
                sum_g.fill(0.0);
                sum_gx.fill(0.0);
                let planes = img.chunks_exact(spatial).zip(g_img.chunks_exact(spatial));
                for ((plane, g_plane), &w) in planes.zip(weight) {
                    let w = f64::from(w);
                    let acc = sum_g.iter_mut().zip(sum_gx.iter_mut());
                    let stat = mean.iter().zip(&rstd);
                    for ((((a_g, a_gx), &v), &g), (&m, &r)) in acc.zip(plane).zip(g_plane).zip(stat)
                    {
                        let dxhat = f64::from(g) * w;
                        *a_g += dxhat;
                        *a_gx += dxhat * ((f64::from(v) - m) * r);
                    }
                }
                let channels_f = channels as f64;
                let planes = dst
                    .chunks_exact_mut(spatial)
                    .zip(img.chunks_exact(spatial))
                    .zip(g_img.chunks_exact(spatial));
                for (((out_plane, plane), g_plane), &w) in planes.zip(weight) {
                    let w = f64::from(w);
                    for (s, ((out, &v), &g)) in
                        out_plane.iter_mut().zip(plane).zip(g_plane).enumerate()
                    {
                        let xhat = (f64::from(v) - mean[s]) * rstd[s];
                        let dxhat = f64::from(g) * w;
                        let (mean_g, mean_gx) = (sum_g[s] / channels_f, sum_gx[s] / channels_f);
                        *out = (rstd[s] * (dxhat - mean_g - xhat * mean_gx)) as f32;
                    }
                }
                stats.extend(mean.iter().copied().zip(rstd.iter().copied()));
            }
            Ok::<_, OjasError>(stats)
        },
    )?;
    let stats = parts.concat();
    let min_cols = (ROW_MIN_ELEMS / positions).max(1);
    scoped::chunks_into_n(
        exec,
        [grad_w, grad_b],
        channels,
        [1, 1],
        min_cols,
        |cols, [grad_w, grad_b]| {
            for (local, c) in cols.enumerate() {
                let (mut sum_gx, mut sum_g) = (0.0f64, 0.0f64);
                for b in 0..batch {
                    let at = (b * channels + c) * spatial;
                    let src = &x[at..at + spatial];
                    let gy = &grad_y[at..at + spatial];
                    let stat = &stats[b * spatial..(b + 1) * spatial];
                    for ((&v, &g), &(m, r)) in src.iter().zip(gy).zip(stat) {
                        let g = f64::from(g);
                        sum_gx += g * ((f64::from(v) - m) * r);
                        sum_g += g;
                    }
                }
                grad_w[local] = sum_gx as f32;
                grad_b[local] = sum_g as f32;
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// `dy * gelu'(x)` of every element ([`gelu_erf_backward_exact`]).
pub(crate) fn gelu_erf_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    grad_y: &[f32],
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    if grad_y.len() != x.len() {
        return Err(shape(op, "gelu backward grad length does not match input"));
    }
    fill_rows(op, budget, exec, out_shape, x.len(), 1, |range, dx| {
        let src = x[range.clone()].iter().zip(&grad_y[range]);
        for (out, (&v, &g)) in dx.iter_mut().zip(src) {
            *out = gelu_erf_backward_exact(v, g);
        }
        Ok(())
    })
}

/// Softmax's input gradient `y * (g - sum(g * y))` per row, the dot an
/// ascending `f64` sum. `dims` comes from [`ojas_core::softmax_backward_dims`].
pub(crate) fn softmax_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    y: &[f32],
    grad_y: &[f32],
    dims: RmsDims,
    out_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let RmsDims { rows, dim } = dims;
    let n = product(op, &[rows, dim])?;
    if y.len() != n || grad_y.len() != n {
        return Err(shape(
            op,
            "softmax backward data length does not match shape",
        ));
    }
    fill_rows(op, budget, exec, out_shape, rows, dim, |range, dx| {
        let ys = &y[range.start * dim..range.end * dim];
        let gs = &grad_y[range.start * dim..range.end * dim];
        let src = ys.chunks_exact(dim).zip(gs.chunks_exact(dim));
        for (dst, (y_row, g_row)) in dx.chunks_exact_mut(dim).zip(src) {
            let mut dot = 0.0f64;
            for (&p, &g) in y_row.iter().zip(g_row) {
                dot += f64::from(p) * f64::from(g);
            }
            for ((out, &p), &g) in dst.iter_mut().zip(y_row).zip(g_row) {
                *out = (f64::from(p) * (f64::from(g) - dot)) as f32;
            }
        }
        Ok(())
    })
}

/// Which operand of a broadcast op a gradient is for.
#[derive(Clone, Copy)]
pub(crate) enum Operand {
    A,
    B,
}

/// One axis of a broadcast output: its extent, the row-major stride it has
/// in the output, and the stride of the other operand along it.
#[derive(Clone, Copy)]
struct Axis {
    extent: usize,
    out: usize,
    mate: usize,
}

/// The gradient of one operand of a broadcast op, shaped `grad_shape`: the
/// output gradient `g` summed over the axes the operand was stretched along
/// (the axes where its stride is 0 and the output is longer than 1). Each
/// element is one `f64` sum, ascending over those axes in output order and
/// rounded once: of `g[o]` for an add, of `g[o] * other[p]` for a multiply
/// (`other` is the other operand, and its product with an `f32` is exact in
/// `f64`). It is a gather over the operand's own elements, so no two threads
/// write one value and no sum is split. `dims` comes from
/// [`ojas_core::broadcast_add_backward_dims`] or its multiply twin.
#[allow(clippy::too_many_arguments)]
pub(crate) fn broadcast_grad(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    dims: &BroadcastDims,
    which: Operand,
    g: &[f32],
    other: Option<&[f32]>,
    grad_shape: &[usize],
) -> Result<Tensor, OjasError> {
    let out_shape = dims.out_shape.as_slice();
    let (own, mate) = match which {
        Operand::A => (&dims.a_strides, &dims.b_strides),
        Operand::B => (&dims.b_strides, &dims.a_strides),
    };
    if g.len() != product(op, out_shape)? {
        return Err(shape(op, "broadcast grad length does not match its shape"));
    }
    let rank = out_shape.len();
    let mut kept = Vec::with_capacity(rank);
    let mut reduced = Vec::with_capacity(rank);
    let mut run = 1usize;
    let mut farthest = 0usize;
    for axis in (0..rank).rev() {
        let extent = out_shape[axis];
        let a = Axis {
            extent,
            out: run,
            mate: mate[axis],
        };
        run = run.saturating_mul(extent);
        farthest = farthest.saturating_add((extent - 1).saturating_mul(mate[axis]));
        if own[axis] != 0 {
            kept.push(a);
        } else if extent > 1 {
            reduced.push(a);
        }
    }
    // Collected from the last axis; the row-major walks run from the first.
    kept.reverse();
    reduced.reverse();
    if let Some(other) = other {
        if farthest >= other.len() {
            return Err(shape(op, "broadcast operand shorter than its shape"));
        }
    }
    let kept_extents: Vec<usize> = kept.iter().map(|a| a.extent).collect();
    let count = product(op, &kept_extents)?;
    if count != product(op, grad_shape)? {
        return Err(shape(op, "broadcast grad shape does not match its operand"));
    }
    // The innermost reduced axis is a tight loop; the rest is an odometer.
    let (inner, outer) = match reduced.split_last() {
        Some((inner, outer)) => (Some(*inner), outer),
        None => (None, &[][..]),
    };
    let term = |o: usize, p: usize| match other {
        Some(other) => f64::from(g[o]) * f64::from(other[p]),
        None => f64::from(g[o]),
    };
    fill_rows(op, budget, exec, grad_shape, count, 1, |range, y| {
        let mut idx = vec![0usize; outer.len()];
        for (slot, t) in y.iter_mut().zip(range) {
            // The operand element `t`'s coordinates give the offsets of its
            // first output element and of the other operand's.
            let (mut base_o, mut base_p) = (0usize, 0usize);
            let mut rest = t;
            for a in kept.iter().rev() {
                let coord = rest % a.extent;
                rest /= a.extent;
                base_o += coord * a.out;
                base_p += coord * a.mate;
            }
            let (mut o, mut p) = (base_o, base_p);
            idx.fill(0);
            let mut acc = 0.0f64;
            loop {
                match inner {
                    Some(a) => {
                        let (mut oi, mut pi) = (o, p);
                        for _ in 0..a.extent {
                            acc += term(oi, pi);
                            oi += a.out;
                            pi += a.mate;
                        }
                    }
                    None => acc += term(o, p),
                }
                let mut done = true;
                for (k, a) in outer.iter().enumerate().rev() {
                    idx[k] += 1;
                    o += a.out;
                    p += a.mate;
                    if idx[k] < a.extent {
                        done = false;
                        break;
                    }
                    o -= idx[k] * a.out;
                    p -= idx[k] * a.mate;
                    idx[k] = 0;
                }
                if done {
                    break;
                }
            }
            *slot = acc as f32;
        }
        Ok(())
    })
}
