//! Blocked causal attention for [`ojas_core::Numerics::Fast`] above
//! [`super::FLASH_MIN_TIME`] positions.
//!
//! A head's queries are cut into blocks of [`BQ`] rows. Query block `b`
//! (rows `qb..qb + nq`) sees keys `0..e` with `e = qb + nq`, so one block's
//! scores are a `[nq, e]` product and each row's softmax is taken over its
//! whole causal prefix at once: no online rescaling, no log-sum-exp pass and
//! no `T x T` matrix (scratch is `O(BQ·T)` per task). Every product is one
//! [`gemm`] call on a pool-free [`Exec`]; on macOS a product of at least
//! [`crate::FAST_WHOLE_CALL_MACS`] is one Accelerate call, and several pool workers may make
//! such calls at once. The exponential is [`crate::exp::exp2_affine`].
//!
//! Forward, per (head, query block): `S = Q_b·K[0..e]ᵀ`, softmax over each
//! row's prefix (the masked tail is zero), `O_b = P·V[0..e]`.
//!
//! Backward, per (head, chunk of consecutive query blocks), for each block
//! in ascending order: `P` as in the forward, `dP = dO_b·V[0..e]ᵀ`,
//! `delta_t = sum_j P_tj·dP_tj` (Exact's `expected`; no forward output is
//! recomputed), `dS = scale·P∘(dP - delta)`, `dQ_b = dS·K[0..e]`, and the
//! chunk's `dK[0..e] += dSᵀ·Q_b`, `dV[0..e] += Pᵀ·dO_b`. A head's chunk
//! partials are summed in ascending chunk order after the pool returns.
//!
//! Bits: the block size, the chunk cuts and so every product's shape and
//! operands depend only on `(T, D)`. The pool size decides which worker runs
//! a task, never what it computes, so the bits do not depend on the thread
//! count. The one exception is the gemm.rs contract itself: an Accelerate
//! call is only as repeatable as Accelerate is on this machine and OS build.
//!
//! `gemm` views have no offset, so `K` and `V` are first split into one
//! buffer per head (`Mat::row_major(head, e, dim)` is then the key prefix);
//! query and output-gradient blocks are copied per task.

use std::sync::Arc;

use ojas_core::{Budget, Numerics, OjasError};

use super::{scratch_overflow, Dims, SdpaGrads, TASK_WORK};
use crate::exp::exp2_affine;
use crate::gemm::{fma, gemm, scratch, Mat};
use crate::pool::{Exec, Pool};
use crate::validate::{nonfinite, room_for};

/// Query rows per block. At `[1, 12, 1024, 64]` on an M5 Pro, 256 beat 128
/// by about 5% at 6 threads and tied at 18 (min of 5 interleaved runs);
/// 64 and 512 were slower.
const BQ: usize = 256;
/// Most chunks one head's backward is cut into. Each chunk holds its own
/// `dK` and `dV` partial over the keys its last block sees.
const CHUNKS: usize = 4;

/// First row and row count of query block `b`.
fn block(b: usize, time: usize) -> (usize, usize) {
    let qb = b * BQ;
    (qb, BQ.min(time - qb))
}

/// A one-thread pool for the products inside a task, polling the backend's
/// cancel hook. The backend's pool must not be re-entered from a worker.
fn inner_pool(exec: Exec<'_>) -> Arc<Pool> {
    let pool = Arc::new(Pool::serial());
    pool.set_cancel(exec.pool.cancel_hook());
    pool
}

fn fast(pool: &Arc<Pool>) -> Exec<'_> {
    Exec {
        pool,
        numerics: Numerics::Fast,
    }
}

fn inflight(exec: Exec<'_>, tasks: usize) -> usize {
    if exec.pool.threads() <= 1 {
        1
    } else {
        tasks.min(exec.pool.threads()).max(1)
    }
}

/// One buffer per `[time, dim]` head.
fn split_heads(data: Vec<f32>, stride: usize) -> Arc<Vec<Arc<Vec<f32>>>> {
    Arc::new(
        data.chunks_exact(stride)
            .map(|head| Arc::new(head.to_vec()))
            .collect(),
    )
}

/// Rows `start..start + rows` of the head at `base`, as a `[rows, dim]` view.
fn rows_of(data: &[f32], base: usize, start: usize, rows: usize, dim: usize) -> Mat {
    let at = base + start * dim;
    Mat::row_major(Arc::new(data[at..at + rows * dim].to_vec()), rows, dim)
}

fn sum(op: &'static str, terms: &[usize]) -> Result<usize, OjasError> {
    terms
        .iter()
        .try_fold(0usize, |acc, &t| acc.checked_add(t))
        .ok_or_else(|| scratch_overflow(op))
}

fn mul(op: &'static str, a: usize, b: usize) -> Result<usize, OjasError> {
    a.checked_mul(b).ok_or_else(|| scratch_overflow(op))
}

pub(super) fn forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    d: Dims,
) -> Result<Vec<f32>, OjasError> {
    let heads = d.batch * d.heads;
    let (time, dim, stride) = (d.time, d.dim, d.stride_h());
    let len = q.len();
    let blocks = time.div_ceil(BQ);
    let tasks = heads * blocks;
    let inner = inner_pool(exec);
    // One task: its query block, scores (reused as probabilities) and output
    // block, plus each product's packing. Every task's output block is held
    // until the pool returns, and K and V are copied once per head.
    let (nq, sx) = (BQ.min(time), fast(&inner));
    let per_task = sum(
        op,
        &[
            mul(op, 2 * nq, dim)?,
            nq,
            mul(op, nq, time)?,
            scratch(op, sx, nq, dim, time)?,
            scratch(op, sx, nq, time, dim)?,
        ],
    )?;
    let hold = sum(
        op,
        &[mul(op, len, 3)?, mul(op, inflight(exec, tasks), per_task)?],
    )?;
    let _hold = room_for(op, budget, hold)?;
    let (kh, vh) = (split_heads(k, stride), split_heads(v, stride));
    let q = Arc::new(q);
    let cancel = exec.pool.cancel_hook();
    // Late blocks see the most keys, so they are handed out first.
    let order = move |task: usize| (task % heads, blocks - 1 - task / heads);
    let parts = exec.map(tasks, d.work(), 2 * TASK_WORK, move |task| {
        cancel()?;
        let (head, b) = order(task);
        let (qb, nq) = block(b, time);
        let e = qb + nq;
        let sx = fast(&inner);
        let qm = rows_of(&q, head * stride, qb, nq, dim);
        let km = Mat::row_major(Arc::clone(&kh[head]), e, dim);
        let mut s = gemm(op, sx, &qm, &km.t())?;
        let inv = softmax_rows(op, &mut s, qb, nq, e, d.scale)?;
        let pm = Mat::row_major(Arc::new(s), nq, e);
        let mut o = gemm(op, sx, &pm, &Mat::row_major(Arc::clone(&vh[head]), e, dim))?;
        scale_rows(&mut o, &inv, dim);
        Ok(o)
    })?;
    let parts = parts.into_iter().collect::<Result<Vec<_>, _>>()?;
    // Output order is head-major, blocks ascending: task `(blocks - 1 - b)
    // * heads + head` holds block `b` of `head`.
    let mut out = Vec::with_capacity(len);
    for head in 0..heads {
        for b in 0..blocks {
            out.extend_from_slice(&parts[(blocks - 1 - b) * heads + head]);
        }
    }
    Ok(out)
}

/// Block boundaries of at most [`CHUNKS`] chunks of about equal causal work
/// (block `b` costs about `b + 1`), from the block count alone.
fn chunk_cuts(blocks: usize) -> Vec<usize> {
    let chunks = CHUNKS.min(blocks).max(1);
    let mut cuts = vec![0usize];
    for c in 1..chunks {
        let cut = ((blocks as f64) * (c as f64 / chunks as f64).sqrt()).round() as usize;
        if cut > *cuts.last().unwrap_or(&0) && cut < blocks {
            cuts.push(cut);
        }
    }
    cuts.push(blocks);
    cuts
}

/// `(dQ rows of the chunk, dK partial, dV partial)`.
type ChunkGrads = (Vec<f32>, Vec<f32>, Vec<f32>);

#[allow(clippy::too_many_arguments)]
pub(super) fn backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    grad_y: Vec<f32>,
    d: Dims,
) -> Result<SdpaGrads, OjasError> {
    let heads = d.batch * d.heads;
    let (time, dim, stride) = (d.time, d.dim, d.stride_h());
    let len = q.len();
    let blocks = time.div_ceil(BQ);
    let cuts = Arc::new(chunk_cuts(blocks));
    let chunks = cuts.len() - 1;
    let tasks = heads * chunks;
    let inner = inner_pool(exec);
    // One block: its query and output-gradient rows, P and dP/dS, the dQ
    // block, and the dK and dV products before they are added, plus each
    // product's packing.
    let (nq, sx) = (BQ.min(time), fast(&inner));
    let per_task = sum(
        op,
        &[
            mul(op, 3 * nq, dim)?,
            nq,
            mul(op, 2 * nq, time)?,
            mul(op, 2 * time, dim)?,
            mul(op, scratch(op, sx, nq, dim, time)?, 2)?,
            scratch(op, sx, nq, time, dim)?,
            mul(op, scratch(op, sx, time, nq, dim)?, 2)?,
        ],
    )?;
    // Held until the pool returns: every chunk's dQ rows (one output) and
    // its dK and dV partials. K and V are copied once per head.
    let partial_rows = cuts[1..]
        .iter()
        .try_fold(0usize, |acc, &c| acc.checked_add((c * BQ).min(time)))
        .ok_or_else(|| scratch_overflow(op))?;
    let partials = mul(op, mul(op, heads, partial_rows)?, 2 * dim)?;
    let hold = sum(
        op,
        &[
            mul(op, len, 3)?,
            partials,
            mul(op, inflight(exec, tasks), per_task)?,
        ],
    )?;
    let _hold = room_for(op, budget, hold)?;
    let (kh, vh) = (split_heads(k, stride), split_heads(v, stride));
    let (q, g) = (Arc::new(q), Arc::new(grad_y));
    let cancel = exec.pool.cancel_hook();
    let task_cuts = Arc::clone(&cuts);
    let parts = exec.map(tasks, d.work(), 2 * TASK_WORK, move |task| {
        let (head, c) = (task / chunks, task % chunks);
        let (first, last) = (task_cuts[c], task_cuts[c + 1]);
        let row0 = first * BQ;
        let keys = (last * BQ).min(time);
        let base = head * stride;
        let sx = fast(&inner);
        let mut dq = Vec::with_capacity((keys - row0) * dim);
        let mut dk = vec![0.0f32; keys * dim];
        let mut dv = vec![0.0f32; keys * dim];
        for b in first..last {
            cancel()?;
            let (qb, nq) = block(b, time);
            let e = qb + nq;
            let qm = rows_of(&q, base, qb, nq, dim);
            let gm = rows_of(&g, base, qb, nq, dim);
            let km = Mat::row_major(Arc::clone(&kh[head]), e, dim);
            let vm = Mat::row_major(Arc::clone(&vh[head]), e, dim);
            let mut p = gemm(op, sx, &qm, &km.t())?;
            let inv = softmax_rows(op, &mut p, qb, nq, e, d.scale)?;
            let mut ds = gemm(op, sx, &gm, &vm.t())?;
            score_grads(&mut p, &mut ds, &inv, qb, nq, e, d.scale);
            let pm = Mat::row_major(Arc::new(p), nq, e);
            let dsm = Mat::row_major(Arc::new(ds), nq, e);
            dq.extend_from_slice(&gemm(op, sx, &dsm, &km)?);
            add_into(&mut dk[..e * dim], &gemm(op, sx, &dsm.t(), &qm)?);
            add_into(&mut dv[..e * dim], &gemm(op, sx, &pm.t(), &gm)?);
        }
        Ok::<ChunkGrads, OjasError>((dq, dk, dv))
    })?;
    let mut grad_q = Vec::with_capacity(len);
    let mut grad_k = vec![0.0f32; len];
    let mut grad_v = vec![0.0f32; len];
    // Task order is head-major, chunks ascending: the dQ rows concatenate
    // in output order, and the partial sums run in ascending chunk order
    // for every head.
    for (task, part) in parts.into_iter().enumerate() {
        let (dq, dk, dv) = part?;
        let base = (task / chunks) * stride;
        grad_q.extend_from_slice(&dq);
        add_into(&mut grad_k[base..base + dk.len()], &dk);
        add_into(&mut grad_v[base..base + dv.len()], &dv);
    }
    Ok((grad_q, grad_k, grad_v))
}

/// Row `i` of the `[nq, e]` raw scores `q·k` is query `qb + i` against
/// keys `0..e`; its causal prefix is the first `qb + i + 1` keys. A
/// non-finite score in a prefix is refused (with `scale <= 1` a finite raw
/// score is a finite scaled one). Each prefix becomes the unnormalized
/// weights `e^(scale·(s - max)) = 2^(s·c - max·c)`, `c = scale·log2(e)`,
/// and the masked tail zero. Returns each row's `1 / sum` of weights; the
/// maximum contributes about 1, so the sum is at least about 1.
fn softmax_rows(
    op: &'static str,
    s: &mut [f32],
    qb: usize,
    nq: usize,
    e: usize,
    scale: f32,
) -> Result<Vec<f32>, OjasError> {
    let c = scale * std::f32::consts::LOG2_E;
    let mut inv = Vec::with_capacity(nq);
    for (i, row) in s.chunks_exact_mut(e).take(nq).enumerate() {
        let (live, masked) = row.split_at_mut(qb + i + 1);
        let max = max_finite(live).ok_or_else(|| nonfinite(op))?;
        inv.push(1.0 / exp2_affine(live, c, -(max * c)));
        masked.fill(0.0);
    }
    Ok(inv)
}

/// With `p` the unnormalized weights of [`softmax_rows`] and `inv` their
/// row scales: `p` becomes `P = inv·p` and `dp` becomes
/// `dS = scale·P∘(dP - delta)`, zero past each row's causal prefix, with
/// `delta = sum_j P_j·dP_j` over the prefix.
#[allow(clippy::too_many_arguments)]
fn score_grads(
    p: &mut [f32],
    dp: &mut [f32],
    inv: &[f32],
    qb: usize,
    nq: usize,
    e: usize,
    scale: f32,
) {
    for (i, ((p_row, d_row), &inv)) in p
        .chunks_exact_mut(e)
        .zip(dp.chunks_exact_mut(e))
        .zip(inv)
        .take(nq)
        .enumerate()
    {
        let valid = qb + i + 1;
        let delta = inv * dot8(&p_row[..valid], &d_row[..valid]);
        let (live, masked) = d_row.split_at_mut(valid);
        for (slot, pj) in live.iter_mut().zip(p_row.iter_mut()) {
            let prob = *pj * inv;
            *pj = prob;
            *slot = scale * (prob * (*slot - delta));
        }
        masked.fill(0.0);
    }
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (slot, &value) in dst.iter_mut().zip(src) {
        *slot += value;
    }
}

/// Multiply row `i` of a `[rows, dim]` matrix by `inv[i]`.
fn scale_rows(m: &mut [f32], inv: &[f32], dim: usize) {
    for (row, &inv) in m.chunks_exact_mut(dim).zip(inv) {
        for value in row.iter_mut() {
            *value *= inv;
        }
    }
}

/// Lane count of the fixed-order reductions below; the lanes are combined
/// in index order, then the tail is added.
const LANES: usize = 8;
/// `f32` bits without the sign; at or above this the value is not finite.
const MAGNITUDE: u32 = 0x7fff_ffff;
const NON_FINITE: u32 = 0x7f80_0000;

/// The largest value, or `None` if any value is NaN or infinite.
fn max_finite(x: &[f32]) -> Option<f32> {
    let (chunks, rest) = x.as_chunks::<LANES>();
    let mut top = [f32::NEG_INFINITY; LANES];
    let mut mag = [0u32; LANES];
    for chunk in chunks {
        for lane in 0..LANES {
            top[lane] = top[lane].max(chunk[lane]);
            mag[lane] = mag[lane].max(chunk[lane].to_bits() & MAGNITUDE);
        }
    }
    let mut max = f32::NEG_INFINITY;
    let mut bits = 0u32;
    for (&t, &m) in top.iter().zip(&mag) {
        max = max.max(t);
        bits = bits.max(m);
    }
    for &v in rest {
        max = max.max(v);
        bits = bits.max(v.to_bits() & MAGNITUDE);
    }
    (bits < NON_FINITE).then_some(max)
}

fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let (ca, ra) = a.as_chunks::<LANES>();
    let (cb, rb) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.iter().zip(cb) {
        for lane in 0..LANES {
            acc[lane] = fma(x[lane], y[lane], acc[lane]);
        }
    }
    let mut total = acc.iter().fold(0.0f32, |s, &v| s + v);
    for (&x, &y) in ra.iter().zip(rb) {
        total = fma(x, y, total);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_cuts_cover_every_block_once_in_order() {
        for blocks in 1..200 {
            let cuts = chunk_cuts(blocks);
            assert_eq!(cuts[0], 0);
            assert_eq!(*cuts.last().unwrap(), blocks);
            assert!(cuts.windows(2).all(|w| w[0] < w[1]), "{blocks}: {cuts:?}");
            assert!(cuts.len() - 1 <= CHUNKS.min(blocks));
        }
        assert_eq!(chunk_cuts(8), vec![0, 4, 6, 7, 8]);
    }

    #[test]
    fn lane_reductions_match_their_definitions() {
        for len in [0usize, 1, 7, 8, 9, 37, 64] {
            let x: Vec<f32> = (0..len).map(|i| ((i * 7919) % 101) as f32 - 50.0).collect();
            let y: Vec<f32> = (0..len).map(|i| ((i * 31) % 17) as f32 * 0.25).collect();
            let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            assert_eq!(max_finite(&x), Some(max), "len {len}");
            // Small integers times quarters: every partial sum is exact.
            let dot: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            assert_eq!(dot8(&x, &y), dot, "len {len}");
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for at in 0..len {
                    let mut z = x.clone();
                    z[at] = bad;
                    assert_eq!(max_finite(&z), None, "len {len} at {at} {bad}");
                }
            }
        }
    }
}
