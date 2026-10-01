//! Causal scaled dot-product attention.
//!
//! Layout is `[batch, heads, time, head_dim]`, the same layout
//! `F.scaled_dot_product_attention` uses after nanolab transposes. Scale is
//! `1/sqrt(head_dim)`. Query `t` attends to keys `0..=t` only. This function
//! does not apply the Metal head-dimension cap.
//!
//! The forward pass splits heads and query-row blocks across the pool; the
//! backward pass splits heads, since `grad_k` and `grad_v` sum over every
//! query of a head. Under `Numerics::Fast` with more than 256 positions the
//! blocked kernel in [`flash`] runs instead of the per-row kernel.

use std::ops::Range;
use std::sync::Arc;

use ojas_core::{sdpa_scale, Budget, Numerics, OjasError};

use crate::pool::Exec;
use crate::validate::{nonfinite, product, room_for, shape};

mod flash;

/// Work units ([`Dims::work`]) per pool task. A pass with fewer than two
/// tasks' worth stays on the calling thread.
const TASK_WORK: usize = 1 << 20;
/// The blocked fast path runs above this sequence length.
const FLASH_MIN_TIME: usize = 256;

#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_sdpa_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    q: Vec<f32>,
    q_shape: &[usize],
    k: Vec<f32>,
    k_shape: &[usize],
    v: Vec<f32>,
    v_shape: &[usize],
) -> Result<Vec<f32>, OjasError> {
    let dims = sdpa_dims(op, q_shape, k_shape, v_shape)?;
    let len = dims.checked_len(op)?;
    if q.len() != len || k.len() != len || v.len() != len {
        return Err(shape(
            op,
            format!(
                "sdpa data lengths q {} k {} v {} != shape product {len}",
                q.len(),
                k.len(),
                v.len()
            ),
        ));
    }
    if dims.time == 0 || dims.dim == 0 || len == 0 {
        return Ok(vec![0.0f32; len]);
    }
    let heads = dims.batch * dims.heads;
    let work = dims.work();
    // A head's rows are independent in the forward pass, so a few heads can
    // still fill the pool. Cuts follow the causal cost, which grows with `t`.
    let tasks = (work / TASK_WORK).min(exec.pool.threads().saturating_mul(2));
    let blocks = if exec.pool.threads() > 1 && tasks >= 2 {
        tasks.div_ceil(heads).clamp(1, dims.time)
    } else {
        1
    };
    let cuts = Arc::new(causal_cuts(dims.time, blocks));
    let flash = exec.numerics == Numerics::Fast && dims.time > FLASH_MIN_TIME;
    let launched = heads.saturating_mul(blocks).max(1);
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        launched.min(exec.pool.threads()).max(1)
    };
    let _hold = room_for(op, budget, forward_scratch(op, &dims, inflight, flash)?)?;
    let cancel = exec.pool.cancel_hook();
    let (q, k, v) = (Arc::new(q), Arc::new(k), Arc::new(v));
    let d = dims;
    let parts = exec.map(heads * blocks, work, 2 * TASK_WORK, {
        let cancel = Arc::clone(&cancel);
        move |task| {
            cancel()?;
            let (head, block) = (task / blocks, task % blocks);
            let span = head * d.stride_h()..(head + 1) * d.stride_h();
            let rows = cuts[block]..cuts[block + 1];
            let (qh, kh, vh) = (&q[span.clone()], &k[span.clone()], &v[span]);
            if flash {
                flash::forward_rows(qh, kh, vh, rows, d.time, d.dim, d.scale, cancel.as_ref())
            } else {
                forward_rows(
                    op,
                    qh,
                    kh,
                    vh,
                    rows,
                    d.time,
                    d.dim,
                    d.scale,
                    cancel.as_ref(),
                )
            }
        }
    })?;
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend_from_slice(&part?);
    }
    Ok(out)
}

/// `blocks + 1` row boundaries with about equal causal work `sum (t + 1)`
/// between them. The cut does not change any row's arithmetic.
fn causal_cuts(time: usize, blocks: usize) -> Vec<usize> {
    let mut cuts = Vec::with_capacity(blocks + 1);
    cuts.push(0);
    for b in 1..blocks {
        let frac = b as f64 / blocks as f64;
        let cut = ((time as f64) * frac.sqrt()).round() as usize;
        let prev = cuts.last().copied().unwrap_or(0);
        cuts.push(cut.clamp(prev, time));
    }
    cuts.push(time);
    cuts
}

/// Causal forward for rows `rows` of one `[time, dim]` head.
///
/// Query `t` scores keys `0..=t` only. Each score sums the head dimension
/// from 0, then the value mix adds those keys from 0, one output lane at a
/// time. Eight key columns share that reduction when the prefix is long
/// enough; the tail uses the same product.
#[allow(clippy::too_many_arguments)]
fn forward_rows(
    op: &'static str,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: Range<usize>,
    time: usize,
    dim: usize,
    scale: f32,
    cancel: &dyn Fn() -> Result<(), OjasError>,
) -> Result<Vec<f32>, OjasError> {
    let width = time.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "sdpa head length overflows".to_string(),
    })?;
    if q.len() != width || k.len() != width || v.len() != width || rows.end > time {
        return Err(shape(op, "sdpa head length does not match time*dim"));
    }
    let mut out = vec![0.0f32; rows.len() * dim];
    if rows.is_empty() {
        return Ok(out);
    }
    // Scratch is length `time` or `time * dim`, never a `T x T` score matrix.
    let mut scores = vec![0.0f32; time];
    let mut probs = vec![0.0f32; time];
    let mut packed = vec![0.0f32; width];
    pack_keys(k, &mut packed, time, dim);
    for t in rows.clone() {
        cancel()?;
        let keys = t + 1;
        score_prefix(op, &packed, row(q, t, dim), &mut scores, keys, time, scale)?;
        softmax_prefix(op, &scores, &mut probs, keys)?;
        let local = t - rows.start;
        mix_values(
            &mut out[local * dim..(local + 1) * dim],
            v,
            &probs,
            keys,
            dim,
        );
    }
    Ok(out)
}

/// `[time, dim]` keys become `[dim, time]` so one dimension step has the
/// keys for that lane in a contiguous span.
fn pack_keys(k: &[f32], packed: &mut [f32], time: usize, dim: usize) {
    for d in 0..dim {
        let dest = &mut packed[d * time..(d + 1) * time];
        for j in 0..time {
            dest[j] = k[j * dim + d];
        }
    }
}

fn score_prefix(
    op: &'static str,
    packed: &[f32],
    q_row: &[f32],
    scores: &mut [f32],
    keys: usize,
    time: usize,
    scale: f32,
) -> Result<(), OjasError> {
    let mut j = 0usize;
    while j + 8 <= keys {
        let mut acc = [0.0f32; 8];
        for (lane, &qv) in packed.chunks_exact(time).zip(q_row) {
            let tile = &lane[j..j + 8];
            acc[0] += qv * tile[0];
            acc[1] += qv * tile[1];
            acc[2] += qv * tile[2];
            acc[3] += qv * tile[3];
            acc[4] += qv * tile[4];
            acc[5] += qv * tile[5];
            acc[6] += qv * tile[6];
            acc[7] += qv * tile[7];
        }
        for lane in 0..8 {
            let score = acc[lane] * scale;
            if !score.is_finite() {
                return Err(nonfinite(op));
            }
            scores[j + lane] = score;
        }
        j += 8;
    }
    while j < keys {
        let mut dot = 0.0f32;
        for (lane, &qv) in packed.chunks_exact(time).zip(q_row) {
            dot += qv * lane[j];
        }
        let score = dot * scale;
        if !score.is_finite() {
            return Err(nonfinite(op));
        }
        scores[j] = score;
        j += 1;
    }
    Ok(())
}

fn softmax_prefix(
    op: &'static str,
    scores: &[f32],
    probs: &mut [f32],
    keys: usize,
) -> Result<(), OjasError> {
    let mut max_score = f32::NEG_INFINITY;
    for &score in scores.iter().take(keys) {
        if score > max_score {
            max_score = score;
        }
    }
    let mut sum = 0.0f32;
    for j in 0..keys {
        let e = (scores[j] - max_score).exp();
        if !e.is_finite() {
            return Err(nonfinite(op));
        }
        probs[j] = e;
        sum += e;
    }
    if !(sum.is_finite() && sum > 0.0) {
        return Err(nonfinite(op));
    }
    for p in probs.iter_mut().take(keys) {
        *p /= sum;
    }
    Ok(())
}

/// `out[d] += probs[j] * v[j, d]` for `j` from 0 and `d` from 0.
fn mix_values(out_row: &mut [f32], v: &[f32], probs: &[f32], keys: usize, dim: usize) {
    let mut j = 0usize;
    while j + 4 <= keys {
        let p0 = probs[j];
        let p1 = probs[j + 1];
        let p2 = probs[j + 2];
        let p3 = probs[j + 3];
        let v0 = row(v, j, dim);
        let v1 = row(v, j + 1, dim);
        let v2 = row(v, j + 2, dim);
        let v3 = row(v, j + 3, dim);
        for d in 0..dim {
            let mut acc = out_row[d];
            acc += p0 * v0[d];
            acc += p1 * v1[d];
            acc += p2 * v2[d];
            acc += p3 * v3[d];
            out_row[d] = acc;
        }
        j += 4;
    }
    while j < keys {
        saxpy_up(out_row, row(v, j, dim), probs[j]);
        j += 1;
    }
}

/// `(grad_q, grad_k, grad_v)`.
type SdpaGrads = (Vec<f32>, Vec<f32>, Vec<f32>);

#[allow(clippy::too_many_arguments)]
pub(crate) fn causal_sdpa_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    q: Vec<f32>,
    q_shape: &[usize],
    k: Vec<f32>,
    k_shape: &[usize],
    v: Vec<f32>,
    v_shape: &[usize],
    grad_y: Vec<f32>,
    grad_shape: &[usize],
) -> Result<SdpaGrads, OjasError> {
    if grad_shape != q_shape {
        return Err(shape(
            op,
            format!("sdpa grad shape {grad_shape:?} != query {q_shape:?}"),
        ));
    }
    let dims = sdpa_dims(op, q_shape, k_shape, v_shape)?;
    let len = dims.checked_len(op)?;
    if q.len() != len || k.len() != len || v.len() != len || grad_y.len() != len {
        return Err(shape(
            op,
            format!(
                "sdpa data lengths q {} k {} v {} grad {} != shape product {len}",
                q.len(),
                k.len(),
                v.len(),
                grad_y.len()
            ),
        ));
    }
    if dims.time == 0 || dims.dim == 0 || len == 0 {
        return Ok((vec![0.0f32; len], vec![0.0f32; len], vec![0.0f32; len]));
    }
    // One head is an independent reduction; grad_k and grad_v accumulate in
    // increasing query index inside it, so a head is never split.
    let heads = dims.batch * dims.heads;
    let flash = exec.numerics == Numerics::Fast && dims.time > FLASH_MIN_TIME;
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        heads.min(exec.pool.threads()).max(1)
    };
    let _hold = room_for(op, budget, backward_scratch(op, &dims, inflight, flash)?)?;
    let cancel = exec.pool.cancel_hook();
    let (q, k, v, g) = (Arc::new(q), Arc::new(k), Arc::new(v), Arc::new(grad_y));
    let d = dims;
    let parts = exec.map(heads, dims.work(), 2 * TASK_WORK, {
        let cancel = Arc::clone(&cancel);
        move |head| {
            cancel()?;
            let span = head * d.stride_h()..(head + 1) * d.stride_h();
            let (qh, kh, vh, gh) = (
                &q[span.clone()],
                &k[span.clone()],
                &v[span.clone()],
                &g[span],
            );
            if flash {
                flash::backward_head(qh, kh, vh, gh, d.time, d.dim, d.scale, cancel.as_ref())
            } else {
                backward_head(op, qh, kh, vh, gh, d.time, d.dim, d.scale, cancel.as_ref())
            }
        }
    })?;
    let mut grad_q = Vec::with_capacity(len);
    let mut grad_k = Vec::with_capacity(len);
    let mut grad_v = Vec::with_capacity(len);
    for part in parts {
        let (gq, gk, gv) = part?;
        grad_q.extend_from_slice(&gq);
        grad_k.extend_from_slice(&gk);
        grad_v.extend_from_slice(&gv);
    }
    Ok((grad_q, grad_k, grad_v))
}

/// Causal backward for one `[time, dim]` head.
///
/// Each row is contiguous. Dots and softmax sums walk the contracted index
/// upward from 0, the same order as the per-element index loop.
#[allow(clippy::too_many_arguments)]
fn backward_head(
    op: &'static str,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_y: &[f32],
    time: usize,
    dim: usize,
    scale: f32,
    cancel: &dyn Fn() -> Result<(), OjasError>,
) -> Result<SdpaGrads, OjasError> {
    let width = time.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "sdpa head length overflows".to_string(),
    })?;
    if q.len() != width || k.len() != width || v.len() != width || grad_y.len() != width {
        return Err(shape(op, "sdpa head length does not match time*dim"));
    }
    let mut grad_q = vec![0.0f32; width];
    let mut grad_k = vec![0.0f32; width];
    let mut grad_v = vec![0.0f32; width];
    if time == 0 || dim == 0 {
        return Ok((grad_q, grad_k, grad_v));
    }
    let mut scores = vec![0.0f32; time];
    let mut probs = vec![0.0f32; time];
    let mut dprobs = vec![0.0f32; time];
    for t in 0..time {
        cancel()?;
        let q_row = row(q, t, dim);
        let gy_row = row(grad_y, t, dim);
        let mut max_score = f32::NEG_INFINITY;
        for (j, slot) in scores.iter_mut().enumerate().take(t + 1) {
            let dot = dot_up(q_row, row(k, j, dim));
            let score = dot * scale;
            if !score.is_finite() {
                return Err(nonfinite(op));
            }
            *slot = score;
            if score > max_score {
                max_score = score;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..=t {
            let e = (scores[j] - max_score).exp();
            if !e.is_finite() {
                return Err(nonfinite(op));
            }
            probs[j] = e;
            sum += e;
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(nonfinite(op));
        }
        for p in probs.iter_mut().take(t + 1) {
            *p /= sum;
        }
        for (j, slot) in dprobs.iter_mut().enumerate().take(t + 1) {
            *slot = dot_up(gy_row, row(v, j, dim));
        }
        let mut expected = 0.0f32;
        for j in 0..=t {
            expected += probs[j] * dprobs[j];
        }
        for j in 0..=t {
            let p = probs[j];
            let coef = scale * (p * (dprobs[j] - expected));
            saxpy_up(&mut grad_q[t * dim..(t + 1) * dim], row(k, j, dim), coef);
            saxpy_up(&mut grad_k[j * dim..(j + 1) * dim], q_row, coef);
            saxpy_up(&mut grad_v[j * dim..(j + 1) * dim], gy_row, p);
        }
    }
    Ok((grad_q, grad_k, grad_v))
}

fn row(data: &[f32], t: usize, dim: usize) -> &[f32] {
    let start = t * dim;
    &data[start..start + dim]
}

/// Sum `a[d] * b[d]` for `d` from 0 upward.
fn dot_up(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        acc += *x * *y;
    }
    acc
}

/// `dst[d] += scale * src[d]` for `d` from 0 upward.
fn saxpy_up(dst: &mut [f32], src: &[f32], scale: f32) {
    for (slot, value) in dst.iter_mut().zip(src) {
        *slot += scale * *value;
    }
}

#[derive(Clone, Copy)]
struct Dims {
    batch: usize,
    heads: usize,
    time: usize,
    dim: usize,
    scale: f32,
}

impl Dims {
    /// Elements of one `[time, dim]` head. Only called after
    /// [`Dims::checked_len`] proved the whole product fits.
    fn stride_h(&self) -> usize {
        self.time * self.dim
    }

    fn checked_len(&self, op: &'static str) -> Result<usize, OjasError> {
        product(op, &[self.batch, self.heads, self.time, self.dim])
    }

    /// `B*H*T*T*D`: the causal score and value-mix multiply-adds of one
    /// forward pass, `T*T*D/2` each per head.
    fn work(&self) -> usize {
        self.batch
            .saturating_mul(self.heads)
            .saturating_mul(self.time)
            .saturating_mul(self.time)
            .saturating_mul(self.dim)
    }
}

/// Scores, probabilities, and packed keys, once per in-flight head.
/// The fast path's padded transpose replaces the packed-key buffer.
fn forward_scratch(
    op: &'static str,
    dims: &Dims,
    inflight: usize,
    flash: bool,
) -> Result<usize, OjasError> {
    let width = dims
        .time
        .checked_mul(dims.dim)
        .ok_or_else(|| scratch_overflow(op))?;
    let per_task = if flash {
        dims.dim
            .checked_mul(dims.time.div_ceil(64) * 64)
            .ok_or_else(|| scratch_overflow(op))?
    } else {
        width
            .checked_add(dims.time)
            .and_then(|n| n.checked_add(dims.time))
            .ok_or_else(|| scratch_overflow(op))?
    };
    inflight
        .checked_mul(per_task)
        .ok_or_else(|| scratch_overflow(op))
}

/// Exact backward keeps three length-`time` buffers per head. The fast path
/// also transposes K and V and the forward output used to form delta.
fn backward_scratch(
    op: &'static str,
    dims: &Dims,
    inflight: usize,
    flash: bool,
) -> Result<usize, OjasError> {
    let per_task = if flash {
        let padded = dims.time.div_ceil(64) * 64;
        dims.dim
            .checked_mul(padded)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(dims.time))
            .ok_or_else(|| scratch_overflow(op))?
    } else {
        dims.time
            .checked_mul(3)
            .ok_or_else(|| scratch_overflow(op))?
    };
    inflight
        .checked_mul(per_task)
        .ok_or_else(|| scratch_overflow(op))
}

fn scratch_overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "sdpa scratch length overflows".to_string(),
    }
}

fn sdpa_dims(
    op: &'static str,
    q_shape: &[usize],
    k_shape: &[usize],
    v_shape: &[usize],
) -> Result<Dims, OjasError> {
    if q_shape.len() != 4 {
        return Err(shape(
            op,
            format!("sdpa query rank {} != 4 [B, H, T, D]", q_shape.len()),
        ));
    }
    if k_shape != q_shape || v_shape != q_shape {
        return Err(shape(
            op,
            format!("sdpa shapes q {q_shape:?} k {k_shape:?} v {v_shape:?} differ"),
        ));
    }
    let dim = q_shape[3];
    let dim_u32 = u32::try_from(dim).map_err(|_| OjasError::OutOfRange {
        op,
        detail: format!("head dim {dim} does not fit in u32"),
    })?;
    let scale = sdpa_scale(dim_u32)?;
    Ok(Dims {
        batch: q_shape[0],
        heads: q_shape[1],
        time: q_shape[2],
        dim,
        scale,
    })
}
