//! Qwen3.5's hybrid-layer ops on the CPU: the depthwise causal conv with
//! SiLU in front of the gated delta rule, gated RMSNorm, partial RoPE, the
//! sigmoid of the attention output gate and the delta rule's `beta`, and the
//! delta rule's log decay.
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

use ojas_core::{
    exp_exact, Budget, Conv1dDims, GdnDecayDims, OjasError, PartialRopeDims, RmsDims, Scratch,
    Tensor,
};

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
/// contiguously, and adds into its own tap-major sums (charged while the
/// piece runs, then transposed into `gw`), so each sum takes its terms in the
/// serial order whatever the cut (before 2026-10-07 each `(c, j)` walked the
/// whole input at channel stride on the calling thread; until 2026-10-08 the
/// sums were written at stride `K` in place).
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
            // Tap-major sums (`[j][k]`), so the inner loop reads and writes
            // contiguously; transposed into `gw`'s `[c, j]` at the end.
            let n = cs.len();
            let mut tap_major = Scratch::try_alloc(width * n, budget)?;
            let acc = tap_major.as_mut_slice();
            for b in 0..batch {
                for t in 0..time {
                    let row = (b * time + t) * channels + cs.start;
                    for j in 0..width {
                        // Source time t + j - (width - 1), skipped below 0.
                        let Some(src) = (t + j).checked_sub(width - 1) else {
                            continue;
                        };
                        let from = (b * time + src) * channels + cs.start;
                        let terms = da[row..row + n].iter().zip(&x[from..from + n]);
                        for (sum, (&dv, &xv)) in acc[j * n..(j + 1) * n].iter_mut().zip(terms) {
                            *sum += dv * xv;
                        }
                    }
                }
            }
            for (k, out) in sums.chunks_exact_mut(width).enumerate() {
                for (j, value) in out.iter_mut().enumerate() {
                    *value = acc[j * n + k];
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

/// [`ojas_core::Backend::sigmoid_forward`]: [`sigmoid`] of each value.
pub(crate) fn sigmoid_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    fill_rows(op, budget, exec, shape, x.len(), 1, |range, out| {
        for (o, &v) in out.iter_mut().zip(&x[range]) {
            *o = sigmoid(v);
        }
        Ok(())
    })
}

/// [`ojas_core::Backend::sigmoid_backward`]: `gy * s(x) * s(-x)`, which is
/// `s * (1 - s)` without the cancellation: above `x` ~ 17 `s` rounds to 1
/// in f32 and `1 - s` to 0, while `s(-x)` keeps the slope.
pub(crate) fn sigmoid_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    x: &[f32],
    gy: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    fill_rows(op, budget, exec, shape, x.len(), 1, |range, out| {
        let each = x[range.clone()].iter().zip(&gy[range]);
        for (o, (&v, &g)) in out.iter_mut().zip(each) {
            *o = g * sigmoid(v) * sigmoid(-v);
        }
        Ok(())
    })
}

/// torch's `F.softplus` at its defaults (beta 1, threshold 20): `x` above
/// 20, `log1p(e^x)` otherwise. `e^x` is [`exp_exact`]; `log1p` is taken in
/// `f64` and rounded once, since `ln(1 + e)` in `f32` loses most of the
/// result where `a + dt_bias` sits (around -2 to -15).
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        f64::from(exp_exact(x)).ln_1p() as f32
    }
}

/// [`ojas_core::Backend::gdn_log_decay_forward`]:
/// `g = -exp(a_log[h]) * softplus(a + dt_bias[h])`, `exp(a_log)` once per
/// head.
pub(crate) fn gdn_log_decay_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: GdnDecayDims,
    [a, a_log, dt_bias]: [&[f32]; 3],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let GdnDecayDims { rows, heads } = d;
    let rate: Vec<f32> = a_log.iter().map(|&l| -exp_exact(l)).collect();
    fill_rows(op, budget, exec, shape, rows, heads, |range, out| {
        let src = &a[range.start * heads..range.end * heads];
        for (row_out, row_a) in out.chunks_exact_mut(heads).zip(src.chunks_exact(heads)) {
            for h in 0..heads {
                row_out[h] = rate[h] * softplus(row_a[h] + dt_bias[h]);
            }
        }
        Ok(())
    })
}

/// [`ojas_core::Backend::gdn_log_decay_backward`]: `(da, da_log, ddt_bias)`.
/// `da = gy * -exp(a_log) * softplus'(a + dt_bias)`, with `softplus' = 1`
/// above 20 and [`sigmoid`] below; `ddt_bias[h]` sums `da` and `da_log[h]`
/// sums `gy * g` over rows in ascending order from `+0.0`, each head on one
/// [`scoped`] thread, so the bits do not depend on the thread count. `da`
/// is a row pass.
pub(crate) fn gdn_log_decay_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    d: GdnDecayDims,
    [a, a_log, dt_bias, gy]: [&[f32]; 4],
    shapes: [&[usize]; 3],
) -> Result<(Tensor, Tensor, Tensor), OjasError> {
    let GdnDecayDims { rows, heads } = d;
    let rate: Vec<f32> = a_log.iter().map(|&l| -exp_exact(l)).collect();
    let [da, dlog, ddt] = fill_outs(op, budget, exec, shapes, |[da, dlog, ddt]| {
        scoped::rows_into(exec, da, rows, heads, |range, out| {
            for (local, row) in range.enumerate() {
                for h in 0..heads {
                    let i = row * heads + h;
                    let x = a[i] + dt_bias[h];
                    let slope = if x > 20.0 { 1.0 } else { sigmoid(x) };
                    out[local * heads + h] = gy[i] * rate[h] * slope;
                }
            }
            Ok(())
        })?;
        let da = &*da;
        // Each value costs a softplus, so a piece is worth a thread at an
        // eighth of the usual row-pass size.
        let min_heads = (ROW_MIN_ELEMS / 8 / rows.max(1)).max(1);
        scoped::chunks_into_n(
            exec,
            [dlog, ddt],
            heads,
            [1, 1],
            min_heads,
            |cols, [dlog, ddt]| {
                for ((h, dl), dd) in cols.zip(dlog.iter_mut()).zip(ddt.iter_mut()) {
                    let (mut sum_log, mut sum_dt) = (0.0f32, 0.0f32);
                    for row in 0..rows {
                        let i = row * heads + h;
                        let g = rate[h] * softplus(a[i] + dt_bias[h]);
                        sum_log += gy[i] * g;
                        sum_dt += da[i];
                    }
                    *dl = sum_log;
                    *dd = sum_dt;
                }
                Ok(())
            },
        )?;
        Ok(())
    })?;
    Ok((da, dlog, ddt))
}
