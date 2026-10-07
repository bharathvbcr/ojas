//! Qwen3.5's hybrid-layer ops on the CPU: the depthwise causal conv with
//! SiLU in front of the gated delta rule, gated RMSNorm, and partial RoPE.
//!
//! Every value is the scalar formula below, evaluated as written in f32
//! with [`sigmoid`] ([`ojas_core::exp_exact`], no `mul_add`), and every sum
//! ascends from index 0, under either [`ojas_core::Numerics`]. Row-parallel
//! passes cut whole rows ([`fill_rows`]), and the weight gradients, which
//! sum over rows, run on the calling thread, so the bits do not depend on
//! the thread count. The shapes are checked by the `ojas_core::shapes`
//! validators before these run.

use ojas_core::{Budget, Conv1dDims, OjasError, PartialRopeDims, RmsDims, Scratch, Tensor};

use crate::pointwise::sigmoid;
use crate::pool::Exec;
use crate::validate::{fill_outs, fill_rows, nonfinite};

/// `silu(a)` and `d silu / d a` at `a`, in the CPU SiLU's expressions.
fn silu_and_slope(a: f32) -> (f32, f32) {
    let s = sigmoid(a);
    (a * s, s * (1.0 + a * (1.0 - s)))
}

/// The conv's pre-activation at `(b, t, c)`: taps `j` ascending, input
/// before time 0 zero.
fn conv_pre(d: Conv1dDims, x: &[f32], w: &[f32], b: usize, t: usize, c: usize) -> f32 {
    let Conv1dDims {
        time,
        channels,
        width,
        ..
    } = d;
    let mut acc = 0.0f32;
    for j in 0..width {
        // Source time t + j - (width - 1), skipped below 0.
        if let Some(src) = (t + j).checked_sub(width - 1) {
            acc += w[c * width + j] * x[(b * time + src) * channels + c];
        }
    }
    acc
}

/// [`ojas_core::Backend::causal_conv1d_silu_forward`].
pub(crate) fn conv1d_silu_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: Conv1dDims,
    x: &[f32],
    w: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let rows = d.batch * d.time;
    fill_rows(op, budget, exec, shape, rows, d.channels, |range, out| {
        for (local, row) in range.enumerate() {
            let (b, t) = (row / d.time, row % d.time);
            for c in 0..d.channels {
                out[local * d.channels + c] = silu_and_slope(conv_pre(d, x, w, b, t, c)).0;
            }
        }
        Ok(())
    })
}

/// [`ojas_core::Backend::causal_conv1d_silu_backward`]: `(grad_input,
/// grad_weight)`. `da = gy * silu'(a)` is formed once; then
/// `gx[b, s, c] = sum_j w[c, j] * da[b, s + (K - 1) - j, c]` and
/// `gw[c, j] = sum_(b, t) da[b, t, c] * x[b, t + j - (K - 1), c]`, each in
/// ascending order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_silu_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: Conv1dDims,
    x: &[f32],
    w: &[f32],
    gy: &[f32],
    shapes: [&[usize]; 2],
) -> Result<(Tensor, Tensor), OjasError> {
    let Conv1dDims {
        batch,
        time,
        channels,
        width,
    } = d;
    let [gx, gw] = fill_outs(op, budget, exec, shapes, |[gx, gw]| {
        // `da`, charged beside the two outputs.
        let mut scratch = Scratch::try_alloc(x.len(), budget)?;
        let da = scratch.as_mut_slice();
        for b in 0..batch {
            for t in 0..time {
                for c in 0..channels {
                    let at = (b * time + t) * channels + c;
                    da[at] = gy[at] * silu_and_slope(conv_pre(d, x, w, b, t, c)).1;
                }
            }
        }
        for b in 0..batch {
            for s in 0..time {
                for c in 0..channels {
                    let mut acc = 0.0f32;
                    for j in 0..width {
                        let t = s + (width - 1) - j;
                        if t < time {
                            acc += w[c * width + j] * da[(b * time + t) * channels + c];
                        }
                    }
                    gx[(b * time + s) * channels + c] = acc;
                }
            }
        }
        for c in 0..channels {
            for j in 0..width {
                let mut acc = 0.0f32;
                for b in 0..batch {
                    for t in 0..time {
                        if let Some(src) = (t + j).checked_sub(width - 1) {
                            acc += da[(b * time + t) * channels + c]
                                * x[(b * time + src) * channels + c];
                        }
                    }
                }
                gw[c * width + j] = acc;
            }
        }
        Ok(())
    })?;
    Ok((gx, gw))
}

/// `1 / sqrt(sum_sq / dim + eps)` of one row, refusing a non-finite one
/// (the RMSNorm rule).
fn rstd(op: &'static str, row: &[f32], eps: f32) -> Result<f32, OjasError> {
    let mut sum_sq = 0.0f32;
    for &v in row {
        sum_sq += v * v;
    }
    let denom = sum_sq / row.len() as f32 + eps;
    if !(denom.is_finite() && denom > 0.0) {
        return Err(nonfinite(op));
    }
    let r = 1.0 / denom.sqrt();
    if !r.is_finite() {
        return Err(nonfinite(op));
    }
    Ok(r)
}

/// [`ojas_core::Backend::gated_rms_norm_forward`]:
/// `out = (w * (x * rstd)) * silu(z)`, transformers' order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_rms_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: RmsDims,
    [x, z, w]: [&[f32]; 3],
    eps: f32,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let dim = d.dim;
    fill_rows(op, budget, exec, shape, d.rows, dim, |range, out| {
        for (local, row) in range.enumerate() {
            let xs = &x[row * dim..(row + 1) * dim];
            let zs = &z[row * dim..(row + 1) * dim];
            let r = rstd(op, xs, eps)?;
            let dst = &mut out[local * dim..(local + 1) * dim];
            for (((o, &xv), &zv), &wv) in dst.iter_mut().zip(xs).zip(zs).zip(w) {
                *o = (wv * (xv * r)) * silu_and_slope(zv).0;
            }
        }
        Ok(())
    })
}

/// [`ojas_core::Backend::gated_rms_norm_backward`]: with `n = x * rstd`,
/// `g = silu(z)` and `dn = gy * w * g`,
/// `dx = rstd * (dn - n * sum(dn * n) / dim)`, `dz = gy * w * n * silu'(z)`
/// and `dw = sum_rows gy * n * g`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_rms_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: RmsDims,
    [x, z, w, gy]: [&[f32]; 4],
    eps: f32,
    shapes: [&[usize]; 3],
) -> Result<(Tensor, Tensor, Tensor), OjasError> {
    let dim = d.dim;
    let [gx, gz, gw] = fill_outs(op, budget, exec, shapes, |[gx, gz, gw]| {
        gw.fill(0.0);
        let rows = x
            .chunks_exact(dim)
            .zip(z.chunks_exact(dim))
            .zip(gy.chunks_exact(dim))
            .zip(gx.chunks_exact_mut(dim).zip(gz.chunks_exact_mut(dim)));
        for (((xs, zs), gs), (gxs, gzs)) in rows {
            let r = rstd(op, xs, eps)?;
            let mut dot = 0.0f32;
            for (((&xv, &zv), &gv), &wv) in xs.iter().zip(zs).zip(gs).zip(w) {
                let dn = (gv * wv) * silu_and_slope(zv).0;
                dot += dn * (xv * r);
            }
            let mean = dot / dim as f32;
            let each = xs.iter().zip(zs).zip(gs).zip(w);
            let outs = gxs.iter_mut().zip(gzs.iter_mut()).zip(gw.iter_mut());
            for ((((&xv, &zv), &gv), &wv), ((dx, dz), dw)) in each.zip(outs) {
                let n = xv * r;
                let (g, slope) = silu_and_slope(zv);
                let dn = (gv * wv) * g;
                *dx = r * (dn - n * mean);
                *dz = ((gv * wv) * n) * slope;
                *dw += (gv * n) * g;
            }
        }
        Ok(())
    })?;
    Ok((gx, gz, gw))
}

/// Which way [`rope_partial`] turns.
#[derive(Clone, Copy)]
pub(crate) enum Turn {
    Forward,
    Backward,
}

/// [`ojas_core::Backend::rope_partial_forward`] and its backward: the
/// half-split rotation of [`crate::norm`]'s RoPE, in its expressions, on the
/// leading `rotary` values of each head, the rest copied.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_partial(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: PartialRopeDims,
    [x, cos, sin]: [&[f32]; 3],
    turn: Turn,
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let PartialRopeDims {
        rows,
        time,
        heads,
        dim,
        rotary,
    } = d;
    let half = rotary / 2;
    fill_rows(op, budget, exec, shape, rows, dim, |range, out| {
        for (local, row) in range.enumerate() {
            let t = (row / heads) % time;
            let (c, s) = (
                &cos[t * rotary..(t + 1) * rotary],
                &sin[t * rotary..(t + 1) * rotary],
            );
            let src = &x[row * dim..(row + 1) * dim];
            let dst = &mut out[local * dim..(local + 1) * dim];
            for col in 0..half {
                let (a, b) = (src[col], src[col + half]);
                let (c1, s1, c2, s2) = (c[col], s[col], c[col + half], s[col + half]);
                match turn {
                    Turn::Forward => {
                        dst[col] = a * c1 + (-b) * s1;
                        dst[col + half] = b * c2 + a * s2;
                    }
                    Turn::Backward => {
                        dst[col] = a * c1 + b * s2;
                        dst[col + half] = -a * s1 + b * c2;
                    }
                }
            }
            dst[rotary..].copy_from_slice(&src[rotary..]);
        }
        Ok(())
    })
}
