//! Forward ops of a transformer encoder that is not nanolab's GPT: LayerNorm
//! with a bias (plain, across the channel axis of an image, and per channel
//! group), the exact GELU, a linear layer's bias, broadcast add and multiply, a scalar scale,
//! and a standalone softmax.
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
//!   [`ojas_core::exp_exact`] under [`Numerics::Exact`] and with the vector
//!   exponential of [`crate::exp`] under [`Numerics::Fast`], sums ascending
//!   (Fast: in the fixed lane order of [`exp_sub_store`]) and divides.

use ojas_core::{
    exp_exact, gelu_erf_exact, BroadcastDims, Budget, ChannelNormDims, GroupNormDims, Numerics,
    OjasError, RmsDims, Tensor,
};

use crate::exp::exp_sub_store;
use crate::pool::Exec;
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
            let mut sum = 0.0f64;
            for &v in row {
                sum += f64::from(v);
            }
            let mean = sum / dim as f64;
            let mut dev = 0.0f64;
            for &v in row {
                let d = f64::from(v) - mean;
                dev += d * d;
            }
            let rstd = layer_rstd(op, dev, dim, eps)?;
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
            mean.fill(0.0);
            for plane in img.chunks_exact(spatial) {
                for (m, &v) in mean.iter_mut().zip(plane) {
                    *m += f64::from(v);
                }
            }
            for m in &mut mean {
                *m /= channels as f64;
            }
            rstd.fill(0.0);
            for plane in img.chunks_exact(spatial) {
                for ((r, &v), &m) in rstd.iter_mut().zip(plane).zip(&mean) {
                    let d = f64::from(v) - m;
                    *r += d * d;
                }
            }
            for r in &mut rstd {
                *r = layer_rstd(op, *r, channels, eps)?;
            }
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
            let mut sum = 0.0f64;
            for &v in grp {
                sum += f64::from(v);
            }
            let mean = sum / width as f64;
            let mut dev = 0.0f64;
            for &v in grp {
                let d = f64::from(v) - mean;
                dev += d * d;
            }
            let rstd = layer_rstd(op, dev, width, eps)?;
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

/// One softmax row into `dst`.
fn softmax_row(numerics: Numerics, row: &[f32], dst: &mut [f32]) {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let sum = match numerics {
        Numerics::Exact => {
            let mut sum = 0.0f32;
            for (out, &v) in dst.iter_mut().zip(row) {
                let e = exp_exact(v - max);
                *out = e;
                sum += e;
            }
            sum
        }
        Numerics::Fast => exp_sub_store(row, max, dst, None),
    };
    for out in dst.iter_mut() {
        *out /= sum;
    }
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
            softmax_row(numerics, row, dst);
        }
        Ok(())
    })
}
