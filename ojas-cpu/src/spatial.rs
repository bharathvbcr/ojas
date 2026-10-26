//! Spatial ops on the CPU: 2D convolution as im2col and one GEMM per image
//! and group (or a direct loop for the depthwise case), transposed
//! convolution as a GEMM and a gather, max pooling, bilinear resampling and
//! interleaved RoPE, for SAM-style vision models.
//!
//! `NCHW` images, PyTorch weight layouts. Every sum ascends from index 0 in
//! f32 with no `mul_add` outside the GEMM, whose own contract
//! ([`crate::linalg`]) covers the convolution's reduction over `(channel,
//! kh, kw)`. Row-parallel passes cut whole rows ([`fill_rows`],
//! [`scoped::rows_into`]) and never split a sum, so the bits do not depend on
//! the thread count. The shapes are checked by the `ojas_core::spatial`
//! validators before these run.

use ojas_core::{
    BilinearDims, Budget, Conv2dDims, ConvTranspose2dDims, InterleavedRopeDims, MaxPool2dDims,
    OjasError, Scratch, Tensor,
};

use crate::gemm::{gemm_out, scratch, Mat};
use crate::pool::{scoped, Exec};
use crate::validate::{check_f32s, fill_out, fill_rows, product, room_for};

/// A convolution's checked `input`, `weight` and optional `bias` values.
pub(crate) type ConvOperands<'t> = (&'t [f32], &'t [f32], Option<&'t [f32]>);

/// The checked values of a convolution's `input` and `weight`, and of its
/// `bias` when there is one.
pub(crate) fn operands_with_bias<'t>(
    op: &'static str,
    exec: Exec<'_>,
    input: &'t Tensor,
    weight: &'t Tensor,
    bias: Option<&'t Tensor>,
) -> Result<ConvOperands<'t>, OjasError> {
    let mut operands = vec![input, weight];
    operands.extend(bias);
    let values = check_f32s(op, exec, &operands)?;
    Ok((values[0], values[1], values.get(2).copied()))
}

/// `bias[row % channels]` added to every value of the `[rows, width]`
/// matrix `out`, whose rows are `(image, channel)` pairs.
fn add_channel_bias(
    exec: Exec<'_>,
    out: &mut [f32],
    rows: usize,
    width: usize,
    channels: usize,
    bias: &[f32],
) -> Result<(), OjasError> {
    scoped::rows_into(exec, out, rows, width, |range, part| {
        for (local, row) in range.enumerate() {
            let b = bias[row % channels];
            for v in &mut part[local * width..(local + 1) * width] {
                *v += b;
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// The input coordinate that output index `o` and kernel tap `k` read on an
/// axis, or `None` where it is zero padding: `o * stride + k * dilation -
/// pad` inside `0..size`.
fn tap(
    o: usize,
    k: usize,
    stride: usize,
    dilation: usize,
    pad: usize,
    size: usize,
) -> Option<usize> {
    (o * stride + k * dilation)
        .checked_sub(pad)
        .filter(|&i| i < size)
}

/// The im2col matrix of one group's input planes `x` (`[cg, H, W]`):
/// `[cg * kh * kw, Ho * Wo]`, row `(c * kh + i) * kw + j`, zero where a tap
/// falls on padding.
fn im2col(exec: Exec<'_>, d: &Conv2dDims, x: &[f32], cols: &mut [f32]) -> Result<(), OjasError> {
    let Conv2dDims {
        height,
        width,
        kernel: [kh, kw],
        params,
        out_height,
        out_width,
        ..
    } = *d;
    let p = out_height * out_width;
    let rows = d.in_per_group() * kh * kw;
    scoped::rows_into(exec, cols, rows, p, |range, part| {
        for (local, row) in range.enumerate() {
            let (c, i, j) = (row / (kh * kw), (row / kw) % kh, row % kw);
            let plane = &x[c * height * width..(c + 1) * height * width];
            let dst = &mut part[local * p..(local + 1) * p];
            for oh in 0..out_height {
                let line = &mut dst[oh * out_width..(oh + 1) * out_width];
                match tap(
                    oh,
                    i,
                    params.stride[0],
                    params.dilation[0],
                    params.padding[0],
                    height,
                ) {
                    None => line.fill(0.0),
                    Some(ih) => {
                        for (ow, v) in line.iter_mut().enumerate() {
                            *v = tap(
                                ow,
                                j,
                                params.stride[1],
                                params.dilation[1],
                                params.padding[1],
                                width,
                            )
                            .map_or(0.0, |iw| plane[ih * width + iw]);
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}

/// [`ojas_core::Backend::conv2d_forward`].
///
/// A group with one input channel (depthwise, and its channel multipliers)
/// is a direct loop over the taps in `(kh, kw)` order. Any other group is
/// `out[Og, Ho * Wo] = weight[Og, K] . cols[K, Ho * Wo]` with `K = cg * kh *
/// kw`; a 1x1 stride-1 unpadded convolution reads the image in place as
/// `cols`. The bias is added last.
pub(crate) fn conv2d_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: Conv2dDims,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Result<Tensor, OjasError> {
    let Conv2dDims {
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel: [kh, kw],
        params,
        out_height,
        out_width,
    } = d;
    let (cg, og) = (d.in_per_group(), d.out_per_group());
    let p = out_height * out_width;
    let shape = d.out_shape();
    if cg == 1 {
        return fill_rows(
            op,
            budget,
            exec,
            &shape,
            batch * out_channels,
            p,
            |range, out| {
                for (local, row) in range.enumerate() {
                    let (b, oc) = (row / out_channels, row % out_channels);
                    let plane_at = (b * in_channels + oc / og) * height * width;
                    let plane = &x[plane_at..plane_at + height * width];
                    let taps = &w[oc * kh * kw..(oc + 1) * kh * kw];
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let mut acc = 0.0f32;
                            for i in 0..kh {
                                let Some(ih) = tap(
                                    oh,
                                    i,
                                    params.stride[0],
                                    params.dilation[0],
                                    params.padding[0],
                                    height,
                                ) else {
                                    continue;
                                };
                                for j in 0..kw {
                                    if let Some(iw) = tap(
                                        ow,
                                        j,
                                        params.stride[1],
                                        params.dilation[1],
                                        params.padding[1],
                                        width,
                                    ) {
                                        acc += taps[i * kw + j] * plane[ih * width + iw];
                                    }
                                }
                            }
                            out[local * p + oh * out_width + ow] =
                                bias.map_or(acc, |b| acc + b[oc]);
                        }
                    }
                }
                Ok(())
            },
        );
    }
    let k = product(op, &[cg, kh, kw])?;
    let in_place = [kh, kw] == [1, 1] && params.stride == [1, 1] && params.padding == [0, 0];
    fill_out(op, budget, exec, &shape, |out| {
        let mut cols = if in_place {
            None
        } else {
            Some(Scratch::<f32>::try_alloc(product(op, &[k, p])?, budget)?)
        };
        let _gemm = room_for(op, budget, scratch(op, exec, og, k, p)?)?;
        let plane = height * width;
        for b in 0..batch {
            for g in 0..params.groups {
                let at = (b * in_channels + g * cg) * plane;
                let xs = &x[at..at + cg * plane];
                if let Some(cols) = cols.as_mut() {
                    im2col(exec, &d, xs, cols.as_mut_slice())?;
                }
                let cols = cols.as_ref().map_or(xs, Scratch::as_slice);
                let weights = Mat::row_major(&w[g * og * k..(g + 1) * og * k], og, k);
                let patches = Mat::row_major(cols, k, p);
                let to = (b * out_channels + g * og) * p;
                gemm_out(op, exec, &weights, &patches, &mut out[to..to + og * p])?;
            }
        }
        if let Some(bias) = bias {
            add_channel_bias(exec, out, batch * out_channels, p, out_channels, bias)?;
        }
        Ok(())
    })
}

/// [`ojas_core::Backend::conv_transpose2d_forward`]: per image, the GEMM
/// `cols[O * kh * kw, H * W] = weight^T[O * kh * kw, C] . x[C, H * W]`
/// (`weight` is `[C, O * kh * kw]`, read transposed in place), then a gather:
/// output pixel `(oh, ow)` of channel `oc` sums `cols[(oc, i, j), (ih, iw)]`
/// over the taps with `ih * stride + i = oh + pad` and
/// `iw * stride + j = ow + pad`, in ascending `(i, j)`, then adds the bias. A
/// gather rather than a scatter, so no two threads write one value and the
/// order is fixed.
pub(crate) fn conv_transpose2d_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: ConvTranspose2dDims,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Result<Tensor, OjasError> {
    let ConvTranspose2dDims {
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel: [kh, kw],
        params,
        out_height,
        out_width,
    } = d;
    let hw = height * width;
    let p = out_height * out_width;
    let okk = product(op, &[out_channels, kh, kw])?;
    let shape = d.out_shape();
    fill_out(op, budget, exec, &shape, |out| {
        let mut cols = Scratch::<f32>::try_alloc(product(op, &[okk, hw])?, budget)?;
        let _gemm = room_for(op, budget, scratch(op, exec, okk, in_channels, hw)?)?;
        let weights = Mat::row_major(w, in_channels, okk).t();
        for b in 0..batch {
            let image = Mat::row_major(
                &x[b * in_channels * hw..(b + 1) * in_channels * hw],
                in_channels,
                hw,
            );
            gemm_out(op, exec, &weights, &image, cols.as_mut_slice())?;
            let cols = cols.as_slice();
            let rows_at = b * out_channels * p;
            let target = &mut out[rows_at..rows_at + out_channels * p];
            scoped::rows_into(exec, target, out_channels, p, |range, part| {
                for (local, oc) in range.enumerate() {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let mut acc = 0.0f32;
                            for i in 0..kh {
                                let Some(ih) =
                                    source(oh, i, params.stride[0], params.padding[0], height)
                                else {
                                    continue;
                                };
                                for j in 0..kw {
                                    if let Some(iw) =
                                        source(ow, j, params.stride[1], params.padding[1], width)
                                    {
                                        acc +=
                                            cols[((oc * kh + i) * kw + j) * hw + ih * width + iw];
                                    }
                                }
                            }
                            part[local * p + oh * out_width + ow] =
                                bias.map_or(acc, |b| acc + b[oc]);
                        }
                    }
                }
                Ok(())
            })?;
        }
        Ok(())
    })
}

/// The input index whose tap `k` lands on output `o` of a transposed
/// convolution: `i * stride + k = o + pad` with `i` inside `0..size`.
fn source(o: usize, k: usize, stride: usize, pad: usize, size: usize) -> Option<usize> {
    let at = (o + pad).checked_sub(k)?;
    (at % stride == 0)
        .then_some(at / stride)
        .filter(|&i| i < size)
}

/// [`ojas_core::Backend::max_pool2d_forward`].
pub(crate) fn max_pool2d_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: MaxPool2dDims,
    x: &[f32],
) -> Result<Tensor, OjasError> {
    let MaxPool2dDims {
        batch,
        channels,
        height,
        width,
        kernel: [kh, kw],
        stride: [sh, sw],
        out_height,
        out_width,
    } = d;
    let p = out_height * out_width;
    fill_rows(
        op,
        budget,
        exec,
        &d.out_shape(),
        batch * channels,
        p,
        |range, out| {
            for (local, plane_index) in range.enumerate() {
                let plane = &x[plane_index * height * width..(plane_index + 1) * height * width];
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut best = f32::NEG_INFINITY;
                        for i in 0..kh {
                            let row = (oh * sh + i) * width + ow * sw;
                            for &v in &plane[row..row + kw] {
                                if v > best {
                                    best = v;
                                }
                            }
                        }
                        out[local * p + oh * out_width + ow] = best;
                    }
                }
            }
            Ok(())
        },
    )
}

/// One output index of a bilinear axis: the two source indices and the weight
/// of the second.
type BilinearTap = (usize, usize, f32);

/// One axis of the bilinear resample: for each output index, the two source
/// indices and the weight of the second, as `torch` computes them
/// (`area_pixel_compute_source_index`, `guard_index_and_lambda`).
fn bilinear_axis(size: usize, out: usize, align_corners: bool) -> Vec<BilinearTap> {
    let scale = if align_corners {
        if out > 1 {
            (size - 1) as f32 / (out - 1) as f32
        } else {
            0.0
        }
    } else {
        size as f32 / out as f32
    };
    (0..out)
        .map(|o| {
            let src = if align_corners {
                scale * o as f32
            } else {
                (scale * (o as f32 + 0.5) - 0.5).max(0.0)
            };
            let i0 = (src.floor() as usize).min(size - 1);
            let lambda = (src - i0 as f32).clamp(0.0, 1.0);
            (i0, i0 + usize::from(i0 + 1 < size), lambda)
        })
        .collect()
}

/// [`ojas_core::Backend::interpolate_bilinear_forward`]: per output value
/// `(1 - ly) * ((1 - lx) * v00 + lx * v01) + ly * ((1 - lx) * v10 + lx *
/// v11)`.
pub(crate) fn interpolate_bilinear_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: BilinearDims,
    x: &[f32],
    align_corners: bool,
) -> Result<Tensor, OjasError> {
    let BilinearDims {
        batch,
        channels,
        height,
        width,
        out_height,
        out_width,
    } = d;
    // The two axis tables are heap scratch, charged before they exist so a
    // cap refuses a skinny `1 x n` output instead of growing the process.
    let table_f32s = std::mem::size_of::<BilinearTap>()
        .div_ceil(std::mem::size_of::<f32>())
        .saturating_mul(out_height.saturating_add(out_width));
    let _tables = room_for(op, budget, table_f32s)?;
    let ys = bilinear_axis(height, out_height, align_corners);
    let xs = bilinear_axis(width, out_width, align_corners);
    let p = out_height * out_width;
    fill_rows(
        op,
        budget,
        exec,
        &d.out_shape(),
        batch * channels,
        p,
        |range, out| {
            for (local, plane_index) in range.enumerate() {
                let plane = &x[plane_index * height * width..(plane_index + 1) * height * width];
                for (oh, &(y0, y1, ly)) in ys.iter().enumerate() {
                    for (ow, &(x0, x1, lx)) in xs.iter().enumerate() {
                        let (w0, w1) = (1.0 - lx, lx);
                        let top = w0 * plane[y0 * width + x0] + w1 * plane[y0 * width + x1];
                        let bottom = w0 * plane[y1 * width + x0] + w1 * plane[y1 * width + x1];
                        out[local * p + oh * out_width + ow] = (1.0 - ly) * top + ly * bottom;
                    }
                }
            }
            Ok(())
        },
    )
}

/// [`ojas_core::Backend::rope_interleaved_forward`] (`inverse == false`) and
/// [`ojas_core::Backend::rope_interleaved_backward`] (`inverse == true`, the
/// rotation by `-angle`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_interleaved(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: InterleavedRopeDims,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    inverse: bool,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let InterleavedRopeDims {
        rows,
        time,
        heads,
        dim,
    } = d;
    let half = dim / 2;
    let sign = if inverse { -1.0f32 } else { 1.0 };
    fill_rows(op, budget, exec, shape, rows, dim, |range, out| {
        for (local, row) in range.enumerate() {
            let t = (row / heads) % time;
            let (c, s) = (
                &cos[t * half..(t + 1) * half],
                &sin[t * half..(t + 1) * half],
            );
            let src = &x[row * dim..(row + 1) * dim];
            let dst = &mut out[local * dim..(local + 1) * dim];
            for i in 0..half {
                let (a, b) = (src[2 * i], src[2 * i + 1]);
                let s = sign * s[i];
                dst[2 * i] = a * c[i] - b * s;
                dst[2 * i + 1] = a * s + b * c[i];
            }
        }
        Ok(())
    })
}
