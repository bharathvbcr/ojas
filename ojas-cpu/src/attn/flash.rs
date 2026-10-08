//! Blocked causal attention for [`ojas_core::Numerics::Fast`] above
//! [`super::FLASH_MIN_TIME`] positions.
//!
//! A head's queries are cut into blocks of [`BQ`] rows. Query block `b`
//! (rows `qb..qb + nq`) sees keys `lo..e` with `e = qb + nq` and `lo` the
//! first key its first row sees (0 without a window), so one block's scores
//! are a `[nq, e - lo]` product and each row's softmax is taken over its
//! whole window at once: no online rescaling and no `T x T` matrix (scratch
//! is `O(BQ·T)` per task, `O(BQ·(BQ + W))` with a window `W`). Every
//! product is one [`gemm`] call on a pool-free [`Exec`]; on macOS a product
//! of at least [`crate::FAST_WHOLE_CALL_MACS`] is one Accelerate call, and
//! several pool workers may make such calls at once. The exponential is
//! [`crate::exp::exp2_affine`].
//!
//! Forward, per (head, query block): `S = Q_b·K[lo..e]ᵀ`, softmax over each
//! row's window (the rest of the row is zero), `O_b = P·V[lo..e]`, and
//! each row's log-sum-exp `lse = scale·max + ln(sum)`.
//!
//! Backward, per (head, chunk of consecutive query blocks), for each block
//! in ascending order: `P = e^(scale·S - lse)` from the forward's
//! log-sum-exp (no row sum is formed), `dP = dO_b·V[lo..e]ᵀ`, `delta_t =
//! sum_j P_tj·dP_tj` over this `P` (not `dO_t · O_t`, see
//! [`score_grads`]), `dS = scale·P∘(dP - delta)`, `dQ_b = dS·K[lo..e]`, and the chunk's
//! `dK[lo..e] += dSᵀ·Q_b`, `dV[lo..e] += Pᵀ·dO_b`. A head's chunk partials
//! are summed in ascending chunk order after the pool returns.
//!
//! Bits: the block size, the chunk cuts and so every product's shape and
//! operands depend only on `(T, D)`. The pool size decides which worker runs
//! a task, never what it computes, so the bits do not depend on the thread
//! count. The one exception is the gemm.rs contract itself: an Accelerate
//! call is only as repeatable as Accelerate is on this machine and OS build.
//!
//! Every operand is read where it is: a head of `K` or `V` is a slice of
//! the tensor (`Mat::row_major(head, e, dim)` is then the key prefix), and
//! query and output-gradient blocks are views of the same slices.
//! Grouped-query heads share one KV plane. A query head's `dK` and `dV`
//! partials are added onto that plane after the pool returns, in increasing
//! query-head order.

use std::sync::Arc;

use ojas_core::{Budget, Numerics, OjasError};

use super::{fill_parts, scratch_overflow, Dims, SdpaGrads};
use crate::exp::exp2_affine;
use crate::gemm::{fma, gemm, scratch, Mat};
use crate::pool::{scoped, Exec, Pool};
use crate::validate::{nonfinite, room_for, shape};

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

/// Rows `start..start + rows` of the head at `base`, as a `[rows, dim]` view.
fn rows_of(data: &[f32], base: usize, start: usize, rows: usize, dim: usize) -> Mat<'_> {
    let at = base + start * dim;
    Mat::row_major(&data[at..at + rows * dim], rows, dim)
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

/// Most keys one query block's scores span: the whole prefix, or the
/// block's rows plus the window behind its first row.
fn block_keys(d: &Dims) -> usize {
    match d.window {
        Some(w) => (BQ + w - 1).min(d.time),
        None => d.time,
    }
}

pub(super) fn forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k, v]: [&[f32]; 3],
    d: Dims,
    [out, lse]: [&mut [f32]; 2],
) -> Result<(), OjasError> {
    let heads = d.batch * d.heads;
    let (time, dim, stride) = (d.time, d.dim, d.stride_h());
    let blocks = time.div_ceil(BQ);
    let tasks = heads * blocks;
    let inner = inner_pool(exec);
    // One task: its scores (reused as probabilities) and output block, plus
    // each product's packing; its query block is a view. The output block
    // and its log-sum-exp are copied into `out` and `lse` before the task
    // returns.
    let (nq, width, sx) = (BQ.min(time), block_keys(&d), fast(&inner));
    let per_task = sum(
        op,
        &[
            mul(op, nq, dim)?,
            2 * nq,
            mul(op, nq, width)?,
            scratch(op, sx, nq, dim, width)?,
            scratch(op, sx, nq, width, dim)?,
        ],
    )?;
    let _hold = room_for(op, budget, mul(op, inflight(exec, tasks), per_task)?)?;
    let cancel = exec.pool.cancel_hook();
    // `out` and `lse` are head-major, blocks ascending. Late blocks see the
    // most keys, so they are handed out first: task `t` writes block
    // `blocks - 1 - t / heads` of head `t % heads`.
    let order = |task: usize| (task % heads, blocks - 1 - task / heads);
    let rows: Vec<usize> = (0..tasks).map(|i| block(i % blocks, time).1).collect();
    let lens: Vec<usize> = rows.iter().map(|r| r * dim).collect();
    let mut by_place: Vec<Option<(&mut [f32], &mut [f32])>> = scoped::cut(out, &lens)?
        .into_iter()
        .zip(scoped::cut(lse, &rows)?)
        .map(Some)
        .collect();
    let mut parts = Vec::with_capacity(tasks);
    for task in 0..tasks {
        let (head, b) = order(task);
        let part = by_place
            .get_mut(head * blocks + b)
            .and_then(Option::take)
            .ok_or_else(|| shape(op, "flash output parts do not tile the output"))?;
        parts.push(part);
    }
    fill_parts(exec, d.work(), parts, |task, (part, lse_part)| {
        cancel()?;
        let (head, b) = order(task);
        let (qb, nq) = block(b, time);
        let (lo, e) = (d.first_key(qb), qb + nq);
        let sx = fast(&inner);
        let kv = d.kv_plane(head) * stride;
        let span = kv + lo * dim..kv + e * dim;
        let qm = rows_of(q, head * stride, qb, nq, dim);
        let km = Mat::row_major(&k[span.clone()], e - lo, dim);
        let mut s = gemm(op, sx, &qm, &km.t())?;
        let (inv, row_lse) = softmax_rows(op, &mut s, &d, qb, nq, lo)?;
        let pm = Mat::row_major(&s, nq, e - lo);
        let mut o = gemm(op, sx, &pm, &Mat::row_major(&v[span], e - lo, dim))?;
        scale_rows(&mut o, &inv, dim);
        if o.len() != part.len() || row_lse.len() != lse_part.len() {
            return Err(shape(op, "flash output block does not match its rows"));
        }
        part.copy_from_slice(&o);
        lse_part.copy_from_slice(&row_lse);
        Ok(())
    })?;
    Ok(())
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

/// One chunk's dK and dV partials, `[keys, dim]` each, over the keys from
/// the first one its first row sees.
type ChunkPartials = (Vec<f32>, Vec<f32>);

/// `(first key, end key)` of the rows of chunk `c`: the keys its partials
/// cover.
fn chunk_keys(d: &Dims, cuts: &[usize], c: usize) -> (usize, usize) {
    let row0 = cuts[c] * BQ;
    (d.first_key(row0), (cuts[c + 1] * BQ).min(d.time))
}

pub(super) fn backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k, v, lse, g]: [&[f32]; 5],
    d: Dims,
    [grad_q, grad_k, grad_v]: SdpaGrads<'_>,
) -> Result<(), OjasError> {
    let heads = d.batch * d.heads;
    let (time, dim, stride) = (d.time, d.dim, d.stride_h());
    let blocks = time.div_ceil(BQ);
    let cuts = chunk_cuts(blocks);
    let chunks = cuts.len() - 1;
    let tasks = heads * chunks;
    let inner = inner_pool(exec);
    // One block: P and dP/dS, the dQ block, and the dK and dV products
    // before they are added, plus each product's packing; its query,
    // output and output-gradient rows are views.
    let (nq, width, sx) = (BQ.min(time), block_keys(&d), fast(&inner));
    let per_task = sum(
        op,
        &[
            mul(op, nq, dim)?,
            nq,
            mul(op, 2 * nq, width)?,
            mul(op, 2 * width, dim)?,
            mul(op, scratch(op, sx, nq, dim, width)?, 2)?,
            scratch(op, sx, nq, width, dim)?,
            mul(op, scratch(op, sx, width, nq, dim)?, 2)?,
        ],
    )?;
    // Held until every task returns: each chunk's dK and dV partials. The
    // dQ rows go straight into `grad_q`.
    let partial_rows = (0..chunks)
        .map(|c| chunk_keys(&d, &cuts, c))
        .try_fold(0usize, |acc, (lo, end)| acc.checked_add(end - lo))
        .ok_or_else(|| scratch_overflow(op))?;
    let partials = mul(op, mul(op, heads, partial_rows)?, 2 * dim)?;
    let hold = sum(op, &[partials, mul(op, inflight(exec, tasks), per_task)?])?;
    let _hold = room_for(op, budget, hold)?;
    let cancel = exec.pool.cancel_hook();
    // Task `head * chunks + c` writes the dQ rows of chunk `c` of `head`: in
    // task order the parts tile `grad_q` from the start.
    let dq_lens: Vec<usize> = (0..tasks)
        .map(|task| {
            let c = task % chunks;
            ((cuts[c + 1] * BQ).min(time) - cuts[c] * BQ) * dim
        })
        .collect();
    let partials = fill_parts(
        exec,
        d.work(),
        scoped::cut(grad_q, &dq_lens)?,
        |task, dq| {
            let (head, c) = (task / chunks, task % chunks);
            let (first, last) = (cuts[c], cuts[c + 1]);
            let row0 = first * BQ;
            let (klo, keys) = chunk_keys(&d, &cuts, c);
            let base = head * stride;
            let kv = d.kv_plane(head) * stride;
            let sx = fast(&inner);
            let mut dk = vec![0.0f32; (keys - klo) * dim];
            let mut dv = vec![0.0f32; (keys - klo) * dim];
            for b in first..last {
                cancel()?;
                let (qb, nq) = block(b, time);
                let (lo, e) = (d.first_key(qb), qb + nq);
                let span = kv + lo * dim..kv + e * dim;
                let qm = rows_of(q, base, qb, nq, dim);
                let gm = rows_of(g, base, qb, nq, dim);
                let km = Mat::row_major(&k[span.clone()], e - lo, dim);
                let vm = Mat::row_major(&v[span], e - lo, dim);
                let rows = head * time + qb..head * time + e;
                let mut p = gemm(op, sx, &qm, &km.t())?;
                probs_from_lse(op, &mut p, &d, qb, lo, &lse[rows])?;
                let mut ds = gemm(op, sx, &gm, &vm.t())?;
                score_grads(op, &p, &mut ds, nq, &d, qb, lo)?;
                let pm = Mat::row_major(&p, nq, e - lo);
                let dsm = Mat::row_major(&ds, nq, e - lo);
                let rows = (qb - row0) * dim..(e - row0) * dim;
                let block_dq = gemm(op, sx, &dsm, &km)?;
                dq.get_mut(rows)
                    .filter(|rows| rows.len() == block_dq.len())
                    .ok_or_else(|| shape(op, "flash dQ block does not match its rows"))?
                    .copy_from_slice(&block_dq);
                let at = (lo - klo) * dim..(e - klo) * dim;
                add_into(&mut dk[at.clone()], &gemm(op, sx, &dsm.t(), &qm)?);
                add_into(&mut dv[at], &gemm(op, sx, &pm.t(), &gm)?);
            }
            Ok::<ChunkPartials, OjasError>((dk, dv))
        },
    )?;
    // Task order is head-major, chunks ascending: the partial sums run in
    // ascending chunk order for every head, and the query heads of one KV
    // head in increasing order.
    for (task, (dk, dv)) in partials.into_iter().enumerate() {
        let (klo, _) = chunk_keys(&d, &cuts, task % chunks);
        let base = d.kv_plane(task / chunks) * stride + klo * dim;
        add_into(&mut grad_k[base..base + dk.len()], &dk);
        add_into(&mut grad_v[base..base + dv.len()], &dv);
    }
    Ok(())
}

/// The keys of query `qb + i` within a block's score row whose first
/// column is key `lo`.
fn live(d: &Dims, qb: usize, i: usize, lo: usize) -> std::ops::Range<usize> {
    let keys = d.keys(qb + i);
    keys.start - lo..keys.end - lo
}

/// Row `i` of the `[nq, e - lo]` raw scores `q·k` is query `qb + i`
/// against keys `lo..e`; [`live`] is its window. A non-finite score there
/// is refused (with `scale <= 1` a finite raw score is a finite scaled
/// one). Each window becomes the unnormalized weights
/// `e^(scale·(s - max)) = 2^(s·c - max·c)`, `c = scale·log2(e)`, and the
/// rest of the row zero. Returns each row's `1 / sum` of weights (the
/// maximum contributes about 1, so the sum is at least about 1) and its
/// log-sum-exp `scale·max + ln(sum)`.
fn softmax_rows(
    op: &'static str,
    s: &mut [f32],
    d: &Dims,
    qb: usize,
    nq: usize,
    lo: usize,
) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
    let c = d.scale * std::f32::consts::LOG2_E;
    let width = s.len() / nq.max(1);
    let mut inv = Vec::with_capacity(nq);
    let mut lse = Vec::with_capacity(nq);
    for (i, row) in s.chunks_exact_mut(width).take(nq).enumerate() {
        let span = live(d, qb, i, lo);
        let max = max_finite(&row[span.clone()]).ok_or_else(|| nonfinite(op))?;
        let off = -(max * c);
        let total = exp2_affine(&mut row[span.clone()], c, off);
        inv.push(1.0 / total);
        // `P = w / total = 2^(s·c + off - log2(total))`: the log-sum-exp
        // that reproduces it, in `f64` from the `f32` offset the weights
        // used, rounded once.
        let row_lse = -f64::from(off) * std::f64::consts::LN_2 + f64::from(total).ln();
        lse.push(row_lse as f32);
        row[..span.start].fill(0.0);
        row[span.end..].fill(0.0);
    }
    Ok((inv, lse))
}

/// The probabilities `P = e^(scale·s - lse)` of the raw scores `s` laid out
/// as in [`softmax_rows`], from each row's forward log-sum-exp, zero outside
/// the row's window. No row sum is formed.
///
/// `2^(s·c - lse·log2(e))` with the offset rounded to `f32` would put that
/// rounding (about one ulp of `lse·log2(e)`) on every weight of the row
/// uncancelled, unlike the forward, whose `1 / sum` absorbs any common
/// factor. So the weights are taken as in the forward, `w = 2^(s·c - m·c)`
/// with `m` the row maximum, and scaled by `2^(-lse·log2(e) - off)`, where
/// `off = -(m·c)` is the offset `w` used, formed in `f64` and rounded once.
/// A non-finite score in a window, or a non-finite `lse`, is refused.
fn probs_from_lse(
    op: &'static str,
    s: &mut [f32],
    d: &Dims,
    qb: usize,
    lo: usize,
    lse: &[f32],
) -> Result<(), OjasError> {
    let c = d.scale * std::f32::consts::LOG2_E;
    let width = s.len() / lse.len().max(1);
    for (i, (row, &l)) in s.chunks_exact_mut(width).zip(lse).enumerate() {
        let span = live(d, qb, i, lo);
        let max = max_finite(&row[span.clone()]).ok_or_else(|| nonfinite(op))?;
        if !l.is_finite() {
            return Err(nonfinite(op));
        }
        let off = -(max * c);
        exp2_affine(&mut row[span.clone()], c, off);
        let corr = (-f64::from(l) * std::f64::consts::LOG2_E - f64::from(off)).exp2() as f32;
        for p in &mut row[span.clone()] {
            *p *= corr;
        }
        row[..span.start].fill(0.0);
        row[span.end..].fill(0.0);
    }
    Ok(())
}

/// With `p` the probabilities of [`probs_from_lse`], `dp` becomes
/// `dS = scale·P∘(dP - delta)`, zero outside each row's window, with
/// `delta = sum_j P_j·dP_j` over the window, from this same `P`.
///
/// `delta` is also `dO · O`, but only for the forward's own `P`. Fast's
/// `exp2_affine` weights differ from the forward's by a few ulps, and `dP -
/// delta` nearly cancels on rows with few keys, so a `delta` from the
/// forward's output puts that mismatch on every `dS` of the row (at `[1, 1,
/// 257, 64]` grad_q went from 2e-7 to 7e-7 of f64). Summing over this `P`
/// keeps `sum_j dS_j` the rounding of zero, as Exact's per-row kernel does.
/// A non-finite `delta` is refused.
fn score_grads(
    op: &'static str,
    p: &[f32],
    dp: &mut [f32],
    nq: usize,
    d: &Dims,
    qb: usize,
    lo: usize,
) -> Result<(), OjasError> {
    let width = (p.len() / nq.max(1)).max(1);
    for (i, (p_row, d_row)) in p
        .chunks_exact(width)
        .zip(dp.chunks_exact_mut(width))
        .enumerate()
    {
        let span = live(d, qb, i, lo);
        let delta = dot8(&p_row[span.clone()], &d_row[span.clone()]);
        if !delta.is_finite() {
            return Err(nonfinite(op));
        }
        for (slot, &prob) in d_row[span.clone()].iter_mut().zip(&p_row[span.clone()]) {
            *slot = d.scale * (prob * (*slot - delta));
        }
        d_row[..span.start].fill(0.0);
        d_row[span.end..].fill(0.0);
    }
    Ok(())
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
