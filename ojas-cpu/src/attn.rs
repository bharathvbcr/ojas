//! Causal scaled dot-product attention.
//!
//! Layout is `[batch, heads, time, head_dim]`, the same layout
//! `F.scaled_dot_product_attention` uses after nanolab transposes. Scale is
//! `1/sqrt(head_dim)`. Query `t` attends to keys `0..=t` only. This function
//! does not apply the Metal head-dimension cap.
//!
//! Under `Numerics::Exact`, and under `Numerics::Fast` up to 256 positions,
//! the per-row kernels below run: the forward pass splits query heads and
//! query-row blocks, the backward pass splits KV heads. Query head `h` reads
//! KV head `h / (H / Hkv)`. The `rep` query heads that share one KV head are
//! contiguous, and `grad_k` / `grad_v` accumulate them in increasing
//! query-head order, so a KV group is never split across tasks. Under
//! `Numerics::Fast` with more than 256 positions the blocked GEMM kernel in
//! [`flash`] runs instead; it splits query heads and query blocks in both
//! passes and its bits do not depend on the thread count (see that module).
//!
//! Both passes write into the output tensors the caller charged: each task
//! fills its own slice on [`scoped`] threads (on the calling thread when the
//! pass has under two tasks' worth of work), so nothing is joined or copied.

use std::ops::Range;

use ojas_core::{exp_exact, sdpa_scale, Budget, Numerics, OjasError, SdpaDims};

use crate::pool::{scoped, Exec};
use crate::validate::{nonfinite, product, room_for, shape};

mod flash;

/// Work units ([`Dims::work`]) per pool task. A pass with fewer than two
/// tasks' worth stays on the calling thread.
pub(crate) const TASK_WORK: usize = 1 << 20;
/// The blocked fast path runs above this sequence length.
const FLASH_MIN_TIME: usize = 256;

/// `dims` is [`Dims::new`] of the validator's result. Writes the output
/// into `out` (`len` zeroed values, charged by the caller), each task its
/// own rows, so there are no per-task results to join.
pub(crate) fn causal_sdpa_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k, v]: [&[f32]; 3],
    dims: Dims,
    out: &mut [f32],
) -> Result<(), OjasError> {
    let q_len = dims.checked_len(op)?;
    let kv_len = dims.checked_kv_len(op)?;
    if q.len() != q_len || out.len() != q_len || k.len() != kv_len || v.len() != kv_len {
        return Err(shape(
            op,
            format!(
                "sdpa data lengths q {} k {} v {} out {} != query {q_len} kv {kv_len}",
                q.len(),
                k.len(),
                v.len(),
                out.len()
            ),
        ));
    }
    if dims.time == 0 || dims.dim == 0 || q_len == 0 {
        return Ok(());
    }
    if exec.numerics == Numerics::Fast && dims.time > FLASH_MIN_TIME {
        return flash::forward(op, budget, exec, [q, k, v], dims, out);
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
    let cuts = causal_cuts(dims.time, blocks);
    let launched = heads.saturating_mul(blocks).max(1);
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        launched.min(exec.pool.threads()).max(1)
    };
    let _hold = room_for(op, budget, forward_scratch(op, &dims, inflight)?)?;
    let cancel = exec.pool.cancel_hook();
    let d = dims;
    // Task `head * blocks + block` writes rows `cuts[block]..cuts[block + 1]`
    // of `head`: in task order the parts tile `out` from the start.
    let part_lens: Vec<usize> = (0..launched)
        .map(|task| (cuts[task % blocks + 1] - cuts[task % blocks]) * d.dim)
        .collect();
    fill_parts(exec, work, scoped::cut(out, &part_lens)?, |task, part| {
        cancel()?;
        let (head, block) = (task / blocks, task % blocks);
        let span = head * d.stride_h()..(head + 1) * d.stride_h();
        let kv = d.kv_plane(head) * d.stride_h();
        let kv_span = kv..kv + d.stride_h();
        let rows = cuts[block]..cuts[block + 1];
        let (qh, kh, vh) = (&q[span], &k[kv_span.clone()], &v[kv_span]);
        forward_rows(
            op,
            [qh, kh, vh],
            rows,
            d.time,
            d.dim,
            d.scale,
            cancel.as_ref(),
            part,
        )
    })?;
    Ok(())
}

/// `task(i, parts[i])` for every part: on [`scoped`] threads when the pass
/// has at least two tasks' worth of `work`, as the pool's `map` decides, and
/// on the calling thread otherwise, so a small pass spawns nothing.
fn fill_parts<P, R, F>(
    exec: Exec<'_>,
    work: usize,
    parts: Vec<P>,
    task: F,
) -> Result<Vec<R>, OjasError>
where
    P: Send,
    R: Send,
    F: Fn(usize, P) -> Result<R, OjasError> + Sync,
{
    if work >= 2 * TASK_WORK {
        scoped::fill_parts(exec, parts, task)
    } else {
        parts
            .into_iter()
            .enumerate()
            .map(|(i, part)| task(i, part))
            .collect()
    }
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

/// Causal forward for rows `rows` of one `[time, dim]` head, written into
/// `out` (`rows.len() * dim` values).
///
/// Query `t` scores keys `0..=t` only. Each score sums the head dimension
/// from 0, then the value mix adds those keys from 0, one output lane at a
/// time. Eight key columns share that reduction when the prefix is long
/// enough; the tail uses the same product.
#[allow(clippy::too_many_arguments)]
fn forward_rows(
    op: &'static str,
    [q, k, v]: [&[f32]; 3],
    rows: Range<usize>,
    time: usize,
    dim: usize,
    scale: f32,
    cancel: &dyn Fn() -> Result<(), OjasError>,
    out: &mut [f32],
) -> Result<(), OjasError> {
    let width = time.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "sdpa head length overflows".to_string(),
    })?;
    if q.len() != width || k.len() != width || v.len() != width || rows.end > time {
        return Err(shape(op, "sdpa head length does not match time*dim"));
    }
    if out.len() != rows.len() * dim {
        return Err(shape(op, "sdpa output part does not match its rows"));
    }
    if rows.is_empty() {
        return Ok(());
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
    Ok(())
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

pub(crate) fn score_prefix(
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

pub(crate) fn softmax_prefix(
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
        let e = exp_exact(scores[j] - max_score);
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
pub(crate) fn mix_values(out_row: &mut [f32], v: &[f32], probs: &[f32], keys: usize, dim: usize) {
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

/// `[grad_q, grad_k, grad_v]`, each `len` zeroed values the caller charged.
type SdpaGrads<'a> = [&'a mut [f32]; 3];

/// `dims` is [`Dims::new`] of the validator's result. Writes the three
/// gradients into `grads`, each head's rows by the task that computes them.
pub(crate) fn causal_sdpa_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k, v, grad_y]: [&[f32]; 4],
    dims: Dims,
    grads: SdpaGrads<'_>,
) -> Result<(), OjasError> {
    let q_len = dims.checked_len(op)?;
    let kv_len = dims.checked_kv_len(op)?;
    let [gq, gk, gv] = grads;
    if q.len() != q_len
        || grad_y.len() != q_len
        || gq.len() != q_len
        || k.len() != kv_len
        || v.len() != kv_len
        || gk.len() != kv_len
        || gv.len() != kv_len
    {
        return Err(shape(
            op,
            format!(
                "sdpa data lengths q {} k {} v {} grad {} != query {q_len} kv {kv_len}",
                q.len(),
                k.len(),
                v.len(),
                grad_y.len()
            ),
        ));
    }
    if dims.time == 0 || dims.dim == 0 || q_len == 0 {
        return Ok(());
    }
    if exec.numerics == Numerics::Fast && dims.time > FLASH_MIN_TIME {
        return flash::backward(op, budget, exec, [q, k, v, grad_y], dims, [gq, gk, gv]);
    }
    // One KV group is an independent reduction. The `rep` query heads that
    // share it are contiguous and accumulate into the same grad_k / grad_v
    // in increasing query-head order, so a group is never split. `rep == 1`
    // is one head and one call, the multi-head cut.
    let groups = dims.batch * dims.kv_heads;
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        groups.min(exec.pool.threads()).max(1)
    };
    let _hold = room_for(op, budget, backward_scratch(op, &dims, inflight)?)?;
    let cancel = exec.pool.cancel_hook();
    let d = dims;
    let rep = d.rep();
    let stride = d.stride_h();
    let q_lens = vec![stride * rep; groups];
    let kv_lens = vec![stride; groups];
    let parts: Vec<SdpaGrads<'_>> = scoped::cut(gq, &q_lens)?
        .into_iter()
        .zip(scoped::cut(gk, &kv_lens)?)
        .zip(scoped::cut(gv, &kv_lens)?)
        .map(|((q, k), v)| [q, k, v])
        .collect();
    fill_parts(exec, dims.work(), parts, |group, part| {
        cancel()?;
        let [mut gq_rest, gk, gv] = part;
        let kv_span = group * stride..(group + 1) * stride;
        let q0 = group * rep;
        for r in 0..rep {
            let (gq_r, rest) = gq_rest.split_at_mut(stride);
            gq_rest = rest;
            let span = (q0 + r) * stride..(q0 + r + 1) * stride;
            let heads_in = [
                &q[span.clone()],
                &k[kv_span.clone()],
                &v[kv_span.clone()],
                &grad_y[span],
            ];
            // Reborrow the KV grads: every query head in the group
            // accumulates into the same slices, in increasing `r`.
            backward_head(
                op,
                heads_in,
                d.time,
                d.dim,
                d.scale,
                cancel.as_ref(),
                [gq_r, &mut gk[..], &mut gv[..]],
            )?;
        }
        Ok(())
    })?;
    Ok(())
}

/// Causal backward for one `[time, dim]` head, accumulated into `grads`
/// (three zeroed `[time, dim]` slices).
///
/// Each row is contiguous. Dots and softmax sums walk the contracted index
/// upward from 0, the same order as the per-element index loop.
fn backward_head(
    op: &'static str,
    [q, k, v, grad_y]: [&[f32]; 4],
    time: usize,
    dim: usize,
    scale: f32,
    cancel: &dyn Fn() -> Result<(), OjasError>,
    [grad_q, grad_k, grad_v]: SdpaGrads<'_>,
) -> Result<(), OjasError> {
    let width = time.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
        op,
        detail: "sdpa head length overflows".to_string(),
    })?;
    if q.len() != width || k.len() != width || v.len() != width || grad_y.len() != width {
        return Err(shape(op, "sdpa head length does not match time*dim"));
    }
    if [grad_q.len(), grad_k.len(), grad_v.len()] != [width; 3] {
        return Err(shape(op, "sdpa gradient part does not match time*dim"));
    }
    if time == 0 || dim == 0 {
        return Ok(());
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
            let e = exp_exact(scores[j] - max_score);
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
    Ok(())
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
pub(crate) struct Dims {
    batch: usize,
    heads: usize,
    kv_heads: usize,
    time: usize,
    dim: usize,
    scale: f32,
}

impl Dims {
    /// Query heads that share one KV head. `1` for multi-head, including
    /// the empty `0 == 0` case, so this never divides by zero.
    fn rep(&self) -> usize {
        if self.kv_heads == 0 || self.kv_heads == self.heads {
            1
        } else {
            self.heads / self.kv_heads
        }
    }

    /// KV plane read by query plane `q_plane`. Equal to `q_plane` when
    /// [`Self::rep`] is 1. `heads` is non-zero wherever this is called: a
    /// zero head count makes the tensor length zero and both passes return
    /// before indexing.
    fn kv_plane(&self, q_plane: usize) -> usize {
        let rep = self.rep();
        (q_plane / self.heads) * self.kv_heads + (q_plane % self.heads) / rep
    }

    /// Elements of one `[time, dim]` head. Only called after
    /// [`Dims::checked_len`] proved the whole product fits.
    fn stride_h(&self) -> usize {
        self.time * self.dim
    }

    fn checked_len(&self, op: &'static str) -> Result<usize, OjasError> {
        product(op, &[self.batch, self.heads, self.time, self.dim])
    }

    fn checked_kv_len(&self, op: &'static str) -> Result<usize, OjasError> {
        product(op, &[self.batch, self.kv_heads, self.time, self.dim])
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
fn forward_scratch(op: &'static str, dims: &Dims, inflight: usize) -> Result<usize, OjasError> {
    let per_task = dims
        .time
        .checked_mul(dims.dim)
        .and_then(|n| n.checked_add(dims.time))
        .and_then(|n| n.checked_add(dims.time))
        .ok_or_else(|| scratch_overflow(op))?;
    inflight
        .checked_mul(per_task)
        .ok_or_else(|| scratch_overflow(op))
}

/// The per-row backward keeps three length-`time` buffers per in-flight
/// head; each head writes its gradients straight into the outputs, which
/// the caller charges.
fn backward_scratch(op: &'static str, dims: &Dims, inflight: usize) -> Result<usize, OjasError> {
    let per_task = dims
        .time
        .checked_mul(3)
        .ok_or_else(|| scratch_overflow(op))?;
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

impl Dims {
    /// The kernel's dimensions from [`ojas_core::causal_sdpa_forward_dims`]
    /// or its backward. A head dimension past `u32` is this backend's limit
    /// (shape-contract D15), checked here, after the validator and before
    /// anything is copied or charged.
    pub(crate) fn new(op: &'static str, dims: SdpaDims) -> Result<Self, OjasError> {
        let dim = dims.head_dim;
        let dim_u32 = u32::try_from(dim).map_err(|_| OjasError::OutOfRange {
            op,
            detail: format!("head dim {dim} does not fit in u32"),
        })?;
        if dims.heads != dims.kv_heads {
            let bad = dims.kv_heads == 0
                || dims.heads == 0
                || dims.kv_heads > dims.heads
                || !dims.heads.is_multiple_of(dims.kv_heads);
            if bad {
                return Err(shape(
                    op,
                    format!(
                        "sdpa heads {} kv heads {} differ",
                        dims.heads, dims.kv_heads
                    ),
                ));
            }
        }
        Ok(Dims {
            batch: dims.batch,
            heads: dims.heads,
            kv_heads: dims.kv_heads,
            time: dims.seq,
            dim,
            scale: sdpa_scale(dim_u32)?,
        })
    }
}
