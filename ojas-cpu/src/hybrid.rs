//! Qwen3.5's hybrid-layer ops on the CPU: the depthwise causal conv with
//! SiLU in front of the gated delta rule, gated RMSNorm, and partial RoPE.
//!
//! Every value is the scalar formula below, evaluated as written in f32
//! with [`sigmoid`] ([`ojas_core::exp_exact`], no `mul_add`), and every sum
//! ascends from index 0, under either [`ojas_core::Numerics`]. Row-parallel
//! passes cut whole rows ([`fill_rows`], [`scoped::rows_into`]), and the
//! weight gradients, which sum over rows, are cut by weight column instead,
//! each piece walking every row in ascending order (plain RMSNorm's split,
//! [`crate::norm`]), so the bits do not depend on the thread count. The
//! shapes are checked by the `ojas_core::shapes` validators before these
//! run.

use ojas_core::{Budget, Conv1dDims, OjasError, PartialRopeDims, RmsDims, Scratch, Tensor};

use crate::pointwise::sigmoid;
use crate::pool::{scoped, Exec, ROW_MIN_ELEMS};
use crate::validate::{fill_outs, fill_rows, nonfinite, room_for};

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
///
/// `da` and `gx` are row passes over the `B * T` rows ([`scoped::rows_into`]).
/// `gw` is cut by channel ([`scoped::chunks_into`], at least
/// [`ROW_MIN_ELEMS`] `/ (B * T)` channels a piece): each piece walks every
/// `(b, t)` row in ascending order, reading that row's channels of the piece
/// contiguously, and adds into its own `[c, j]` sums, so each sum takes its
/// terms in the serial order whatever the cut (before 2026-10-07 each
/// `(c, j)` walked the whole input at channel stride on the calling thread).
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
    let rows = batch * time;
    let [gx, gw] = fill_outs(op, budget, exec, shapes, |[gx, gw]| {
        // `da`, charged beside the two outputs.
        let mut scratch = Scratch::try_alloc(x.len(), budget)?;
        let da = scratch.as_mut_slice();
        scoped::rows_into(exec, da, rows, channels, |range, part| {
            for (local, row) in range.enumerate() {
                let (b, t) = (row / time, row % time);
                let base = row * channels;
                for c in 0..channels {
                    let slope = silu_and_slope(conv_pre(d, x, w, b, t, c)).1;
                    part[local * channels + c] = gy[base + c] * slope;
                }
            }
            Ok(())
        })?;
        let da = &*da;
        scoped::rows_into(exec, gx, rows, channels, |range, part| {
            for (local, row) in range.enumerate() {
                let (b, s) = (row / time, row % time);
                for c in 0..channels {
                    let mut acc = 0.0f32;
                    for j in 0..width {
                        let t = s + (width - 1) - j;
                        if t < time {
                            acc += w[c * width + j] * da[(b * time + t) * channels + c];
                        }
                    }
                    part[local * channels + c] = acc;
                }
            }
            Ok(())
        })?;
        let min_channels = (ROW_MIN_ELEMS / rows.max(1)).max(1);
        scoped::chunks_into(exec, gw, channels, width, min_channels, |cs, sums| {
            sums.fill(0.0);
            for b in 0..batch {
                for t in 0..time {
                    let row = (b * time + t) * channels;
                    for j in 0..width {
                        // Source time t + j - (width - 1), skipped below 0.
                        let Some(src) = (t + j).checked_sub(width - 1) else {
                            continue;
                        };
                        let from = (b * time + src) * channels;
                        for (k, c) in cs.clone().enumerate() {
                            sums[k * width + j] += da[row + c] * x[from + c];
                        }
                    }
                }
            }
            Ok(())
        })?;
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
///
/// Plain RMSNorm's split ([`crate::norm`]): `dx` and `dz` are one row pass
/// ([`scoped::chunks_into_n`] over whole rows), which also returns each row's
/// `rstd`; then `dw` is cut by column, each piece summing its columns over
/// every row in ascending order, so each sum takes its terms in the serial
/// order whatever the cut. The `rstd`s are charged while they live.
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
    let RmsDims { rows, dim } = d;
    let _hold = room_for(op, budget, rows)?;
    let [gx, gz, gw] = fill_outs(op, budget, exec, shapes, |[gx, gz, gw]| {
        let parts = scoped::chunks_into_n(
            exec,
            [gx, gz],
            rows,
            [dim, dim],
            scoped::min_rows(dim),
            |range, [gxs, gzs]| {
                let mut rstds = Vec::with_capacity(range.len());
                for (local, row) in range.enumerate() {
                    let xs = &x[row * dim..(row + 1) * dim];
                    let zs = &z[row * dim..(row + 1) * dim];
                    let gs = &gy[row * dim..(row + 1) * dim];
                    let r = rstd(op, xs, eps)?;
                    let mut dot = 0.0f32;
                    for (((&xv, &zv), &gv), &wv) in xs.iter().zip(zs).zip(gs).zip(w) {
                        let dn = (gv * wv) * silu_and_slope(zv).0;
                        dot += dn * (xv * r);
                    }
                    let mean = dot / dim as f32;
                    let each = xs.iter().zip(zs).zip(gs).zip(w);
                    let outs = gxs[local * dim..(local + 1) * dim]
                        .iter_mut()
                        .zip(&mut gzs[local * dim..(local + 1) * dim]);
                    for ((((&xv, &zv), &gv), &wv), (dx, dz)) in each.zip(outs) {
                        let n = xv * r;
                        let (g, slope) = silu_and_slope(zv);
                        let dn = (gv * wv) * g;
                        *dx = r * (dn - n * mean);
                        *dz = ((gv * wv) * n) * slope;
                    }
                    rstds.push(r);
                }
                Ok(rstds)
            },
        )?;
        let rstds = parts.concat();
        let min_cols = (ROW_MIN_ELEMS / rows.max(1)).max(1);
        scoped::chunks_into(exec, gw, dim, 1, min_cols, |cols, sums| {
            sums.fill(0.0);
            for (row, &r) in rstds.iter().enumerate() {
                let at = row * dim;
                let span = at + cols.start..at + cols.end;
                let each = x[span.clone()].iter().zip(&z[span.clone()]).zip(&gy[span]);
                for (dw, ((&xv, &zv), &gv)) in sums.iter_mut().zip(each) {
                    let n = xv * r;
                    *dw += (gv * n) * silu_and_slope(zv).0;
                }
            }
            Ok(())
        })?;
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
