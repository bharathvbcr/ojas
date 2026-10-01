//! Blocked causal attention for [`ojas_core::Numerics::Fast`].
//!
//! Keys are visited in blocks of `BK` with an online softmax (running max
//! and sum per query row), so no `T x T` score matrix exists. Scratch is the
//! transposed `K` (and `V` in the backward pass), `O(T·D)` per task. Dots and
//! value mixes use `mul_add`. Each query row `t` visits the key blocks
//! `0, BK, 2·BK, ..` up to `t` in that order whatever task or query block it
//! lands in, so the bits do not depend on the thread count.
//!
//! The backward pass recomputes each row's log-sum-exp and
//! `delta_t = dO_t · O_t`, then walks key blocks once, accumulating `dK` and
//! `dV` for the block and `dQ` for every row that attends to it.

use std::ops::Range;

use ojas_core::OjasError;

use crate::gemm::fma;

/// Keys per block; a `[dim, BK]` slice of `Kᵀ` and `BK` rows of `V` stay in L1.
const BK: usize = 64;
/// Query rows that reuse one key block before the next block is loaded.
const BQ: usize = 32;

/// `[dim, padded]` transpose with `padded = round_up(time, BK)`, zero padded.
fn transpose_padded(x: &[f32], time: usize, dim: usize) -> (Vec<f32>, usize) {
    let padded = time.div_ceil(BK) * BK;
    let mut out = vec![0.0f32; dim * padded];
    for t in 0..time {
        for (d, &value) in x[t * dim..(t + 1) * dim].iter().enumerate() {
            out[d * padded + t] = value;
        }
    }
    (out, padded)
}

/// `out[j] = scale * sum_d a[d] * xt[d][kb + j]` for `j < BK`.
#[inline]
fn block_dots(a: &[f32], xt: &[f32], padded: usize, kb: usize, scale: f32) -> [f32; BK] {
    let mut acc = [0.0f32; BK];
    for (d, &ad) in a.iter().enumerate() {
        let base = d * padded + kb;
        if let Ok(lane) = <&[f32; BK]>::try_from(&xt[base..base + BK]) {
            for j in 0..BK {
                acc[j] = fma(ad, lane[j], acc[j]);
            }
        }
    }
    for value in acc.iter_mut() {
        *value *= scale;
    }
    acc
}

#[inline]
fn axpy(dst: &mut [f32], scale: f32, src: &[f32]) {
    for (slot, &value) in dst.iter_mut().zip(src) {
        *slot = fma(scale, value, *slot);
    }
}

fn row(data: &[f32], t: usize, dim: usize) -> &[f32] {
    &data[t * dim..(t + 1) * dim]
}

/// Output rows `rows` of one head, plus each row's log-sum-exp.
#[allow(clippy::too_many_arguments)]
fn forward_core(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: Range<usize>,
    time: usize,
    dim: usize,
    scale: f32,
    cancel: &dyn Fn() -> Result<(), OjasError>,
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    let (kt, padded) = transpose_padded(k, time, dim);
    let mut out = vec![0.0f32; rows.len() * dim];
    let mut lse = vec![0.0f32; rows.len()];
    let mut qb = rows.start;
    while qb < rows.end {
        cancel()?;
        let nq = BQ.min(rows.end - qb);
        let base = qb - rows.start;
        let mut max = [f32::NEG_INFINITY; BQ];
        let mut sum = [0.0f32; BQ];
        let mut kb = 0;
        while kb < qb + nq {
            for i in 0..nq {
                let t = qb + i;
                if kb > t {
                    continue;
                }
                let valid = BK.min(t + 1 - kb);
                let s = block_dots(row(q, t, dim), &kt, padded, kb, scale);
                let mut block_max = f32::NEG_INFINITY;
                for &value in &s[..valid] {
                    block_max = block_max.max(value);
                }
                let new_max = max[i].max(block_max);
                let o = &mut out[(base + i) * dim..(base + i + 1) * dim];
                if max[i] != new_max {
                    let corr = (max[i] - new_max).exp();
                    sum[i] *= corr;
                    for value in o.iter_mut() {
                        *value *= corr;
                    }
                }
                for (j, &score) in s[..valid].iter().enumerate() {
                    let p = (score - new_max).exp();
                    sum[i] += p;
                    axpy(o, p, row(v, kb + j, dim));
                }
                max[i] = new_max;
            }
            kb += BK;
        }
        for i in 0..nq {
            let inv = 1.0 / sum[i];
            for value in out[(base + i) * dim..(base + i + 1) * dim].iter_mut() {
                *value *= inv;
            }
            lse[base + i] = max[i] + sum[i].ln();
        }
        qb += nq;
    }
    Ok((out, lse))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_rows(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: Range<usize>,
    time: usize,
    dim: usize,
    scale: f32,
    cancel: &dyn Fn() -> Result<(), OjasError>,
) -> Result<Vec<f32>, OjasError> {
    Ok(forward_core(q, k, v, rows, time, dim, scale, cancel)?.0)
}

type HeadGrads = (Vec<f32>, Vec<f32>, Vec<f32>);

/// `(grad_q, grad_k, grad_v)` of one `[time, dim]` head.
#[allow(clippy::too_many_arguments)]
pub(super) fn backward_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_y: &[f32],
    time: usize,
    dim: usize,
    scale: f32,
    cancel: &dyn Fn() -> Result<(), OjasError>,
) -> Result<HeadGrads, OjasError> {
    cancel()?;
    let (out, lse) = forward_core(q, k, v, 0..time, time, dim, scale, cancel)?;
    let delta: Vec<f32> = (0..time)
        .map(|t| {
            row(grad_y, t, dim)
                .iter()
                .zip(row(&out, t, dim))
                .fold(0.0f32, |acc, (&g, &o)| fma(g, o, acc))
        })
        .collect();
    drop(out);
    let (kt, padded) = transpose_padded(k, time, dim);
    let (vt, _) = transpose_padded(v, time, dim);
    let width = time * dim;
    let mut grad_q = vec![0.0f32; width];
    let mut grad_k = vec![0.0f32; width];
    let mut grad_v = vec![0.0f32; width];
    let mut kb = 0;
    while kb < time {
        let nk = BK.min(time - kb);
        for t in kb..time {
            let valid = nk.min(t + 1 - kb);
            let q_row = row(q, t, dim);
            let g_row = row(grad_y, t, dim);
            let s = block_dots(q_row, &kt, padded, kb, scale);
            let dp = block_dots(g_row, &vt, padded, kb, 1.0);
            let gq = &mut grad_q[t * dim..(t + 1) * dim];
            for j in 0..valid {
                let key = kb + j;
                let p = (s[j] - lse[t]).exp();
                let ds = scale * (p * (dp[j] - delta[t]));
                axpy(&mut grad_v[key * dim..(key + 1) * dim], p, g_row);
                axpy(&mut grad_k[key * dim..(key + 1) * dim], ds, q_row);
                axpy(gq, ds, row(k, key, dim));
            }
        }
        kb += BK;
    }
    Ok((grad_q, grad_k, grad_v))
}
