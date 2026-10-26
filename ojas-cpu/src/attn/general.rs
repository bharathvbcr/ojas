//! General scaled dot-product attention: bidirectional or causal, a query
//! length that differs from the key length, an additive score bias and
//! per-batch key lengths ([`ojas_core::AttentionOpts`]).
//!
//! Layout is `[batch, heads, time, head_dim]`; K and V have `kv_heads`
//! heads and `Tk` positions. Scale is `1/sqrt(head_dim)`. Every mask these
//! options can express leaves each query a contiguous run of keys
//! ([`Geom::keys`]), so the row kernels of [`super`] (`score_range`,
//! `softmax_range`, `mix_values`) apply unchanged and give the same bits as
//! the causal kernels where the two overlap: with `causal`, `Tq == Tk`, no
//! bias and no key lengths, the output, log-sum-exp and gradients equal
//! [`super::causal_sdpa_forward`]'s and [`super::causal_sdpa_backward`]'s.
//!
//! The row arithmetic is the same under `Numerics::Exact` and
//! `Numerics::Fast`: no pass here splits a reduction, so results do not
//! depend on the thread count. A query that sees no key outputs zeros and
//! stores a log-sum-exp of 0, and has no gradient.
//!
//! The bias gradient is its own pass. A bias that broadcasts over the batch
//! or heads sums the gradient of every `(b, h)` it serves; to keep that sum
//! in one order and one task, each task owns one `[Hb, Bb]` plane of the
//! bias gradient and walks the batches and heads that read it in increasing
//! order.

use std::ops::Range;

use ojas_core::{exp_exact, sdpa_scale, AttentionBias, AttentionDims, Budget, OjasError};

use super::{
    dot_up, fill_parts, mix_values, pack_keys, row, saxpy_up, score_range, softmax_range, TASK_WORK,
};
use crate::pool::{scoped, Exec};
use crate::validate::{nonfinite, product, room_for, shape};

/// `[output, lse]`: `[B, H, Tq, D]` and `[B, H, Tq]` zeroed values the
/// caller charged.
type Out<'a> = [&'a mut [f32]; 2];

/// `[grad_q, grad_k, grad_v]`, zeroed values the caller charged.
type Grads<'a> = [&'a mut [f32]; 3];

/// The operands of one pass, besides Q, K and V: the bias values and the
/// key lengths, each when the options carry one.
#[derive(Clone, Copy)]
pub(crate) struct Extras<'a> {
    pub(crate) bias: Option<&'a [f32]>,
    pub(crate) key_lens: Option<&'a [u32]>,
}

/// Kernel geometry from [`AttentionDims`].
#[derive(Clone, Copy)]
pub(crate) struct Geom {
    batch: usize,
    heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    dim: usize,
    scale: f32,
    causal: bool,
    window: Option<usize>,
    bias: Option<AttentionBias>,
}

impl Geom {
    /// A head dimension past `u32` is this backend's limit (shape-contract
    /// D15), checked after the validator and before anything is charged.
    pub(crate) fn new(op: &'static str, dims: AttentionDims) -> Result<Self, OjasError> {
        let dim = dims.head_dim;
        let dim_u32 = u32::try_from(dim).map_err(|_| OjasError::OutOfRange {
            op,
            detail: format!("head dim {dim} does not fit in u32"),
        })?;
        Ok(Geom {
            batch: dims.batch,
            heads: dims.heads,
            kv_heads: dims.kv_heads,
            q_len: dims.q_len,
            kv_len: dims.kv_len,
            dim,
            scale: sdpa_scale(dim_u32)?,
            causal: dims.causal,
            window: dims.window,
            bias: dims.bias,
        })
    }

    /// Query heads that share one KV head.
    fn rep(&self) -> usize {
        self.heads / self.kv_heads
    }

    fn q_plane_len(&self) -> usize {
        self.q_len * self.dim
    }

    fn kv_plane_len(&self) -> usize {
        self.kv_len * self.dim
    }

    fn checked_q_len(&self, op: &'static str) -> Result<usize, OjasError> {
        product(op, &[self.batch, self.heads, self.q_len, self.dim])
    }

    fn checked_kv_len(&self, op: &'static str) -> Result<usize, OjasError> {
        product(op, &[self.batch, self.kv_heads, self.kv_len, self.dim])
    }

    fn checked_rows(&self, op: &'static str) -> Result<usize, OjasError> {
        product(op, &[self.batch, self.heads, self.q_len])
    }

    fn checked_bias_len(&self, op: &'static str) -> Result<Option<usize>, OjasError> {
        self.bias
            .map(|b| product(op, &[b.batch, b.heads, b.queries, self.kv_len]))
            .transpose()
    }

    /// Keys query `i` of a batch row whose key length is `len` sees. A
    /// causal query `i` sits at position `i + Tk - Tq`.
    fn keys(&self, len: usize, i: usize) -> Range<usize> {
        let len = len.min(self.kv_len);
        let (lo, hi) = if self.causal {
            // `pos + 1` keys precede and include the query; none when the
            // query sits before the first key.
            let Some(seen) = (i + self.kv_len).checked_sub(self.q_len).map(|p| p + 1) else {
                return 0..0;
            };
            let lo = self.window.map_or(0, |w| seen.saturating_sub(w));
            (lo, seen.min(len))
        } else {
            (0, len)
        };
        if lo < hi {
            lo..hi
        } else {
            0..0
        }
    }

    /// The key length of batch row `b`.
    fn len_of(&self, lens: Option<&[u32]>, b: usize) -> usize {
        lens.map_or(self.kv_len, |l| (l[b] as usize).min(self.kv_len))
    }

    /// The `Tk` bias values added to query `i` of head `h` of batch `b`.
    fn bias_row<'a>(
        &self,
        bias: Option<&'a [f32]>,
        b: usize,
        h: usize,
        i: usize,
    ) -> Option<&'a [f32]> {
        let (data, shape) = (bias?, self.bias?);
        let pick = |extent: usize, at: usize| if extent == 1 { 0 } else { at };
        let plane = pick(shape.batch, b) * shape.heads + pick(shape.heads, h);
        let at = (plane * shape.queries + pick(shape.queries, i)) * self.kv_len;
        Some(&data[at..at + self.kv_len])
    }

    /// Multiply-adds of one pass over every key: `B*H*Tq*Tk*D`.
    fn work(&self) -> usize {
        self.batch
            .saturating_mul(self.heads)
            .saturating_mul(self.q_len)
            .saturating_mul(self.kv_len)
            .saturating_mul(self.dim)
    }
}

/// `scores[j] += bias[j]` for `j` in `keys`; a non-finite sum is refused.
fn add_bias(
    op: &'static str,
    scores: &mut [f32],
    bias: Option<&[f32]>,
    keys: Range<usize>,
) -> Result<(), OjasError> {
    let Some(bias) = bias else {
        return Ok(());
    };
    for j in keys {
        let s = scores[j] + bias[j];
        if !s.is_finite() {
            return Err(nonfinite(op));
        }
        scores[j] = s;
    }
    Ok(())
}

fn scratch_overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "attention scratch length overflows".to_string(),
    }
}

/// `inflight` tasks' scores, probabilities and packed keys.
fn forward_scratch(op: &'static str, g: &Geom, inflight: usize) -> Result<usize, OjasError> {
    let per_task = g
        .kv_len
        .checked_mul(g.dim)
        .and_then(|n| n.checked_add(g.kv_len))
        .and_then(|n| n.checked_add(g.kv_len))
        .ok_or_else(|| scratch_overflow(op))?;
    inflight
        .checked_mul(per_task)
        .ok_or_else(|| scratch_overflow(op))
}

/// Two length-`Tk` buffers per in-flight task.
fn backward_scratch(op: &'static str, g: &Geom, inflight: usize) -> Result<usize, OjasError> {
    g.kv_len
        .checked_mul(2)
        .and_then(|n| n.checked_mul(inflight))
        .ok_or_else(|| scratch_overflow(op))
}

/// `blocks + 1` row boundaries cutting `0..rows` into near-equal runs.
fn even_cuts(rows: usize, blocks: usize) -> Vec<usize> {
    (0..=blocks).map(|b| b * rows / blocks).collect()
}

fn check_extras(
    op: &'static str,
    g: &Geom,
    extras: &Extras<'_>,
    batch_lens: bool,
) -> Result<(), OjasError> {
    let bias_len = g.checked_bias_len(op)?;
    if extras.bias.map(<[f32]>::len) != bias_len {
        return Err(shape(op, "attention bias data does not match its shape"));
    }
    if batch_lens && extras.key_lens.is_some_and(|l| l.len() != g.batch) {
        return Err(shape(
            op,
            "attention key_lens data does not match the batch",
        ));
    }
    Ok(())
}

/// Writes the output into `out` and each query row's log-sum-exp into `lse`
/// (zeroed values the caller charged), each task its own rows.
pub(crate) fn attention_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k, v]: [&[f32]; 3],
    extras: Extras<'_>,
    g: Geom,
    [out, lse]: Out<'_>,
) -> Result<(), OjasError> {
    let (q_len, kv_len, rows_len) = (
        g.checked_q_len(op)?,
        g.checked_kv_len(op)?,
        g.checked_rows(op)?,
    );
    if q.len() != q_len
        || out.len() != q_len
        || lse.len() != rows_len
        || k.len() != kv_len
        || v.len() != kv_len
    {
        return Err(shape(
            op,
            format!(
                "attention data lengths q {} k {} v {} out {} lse {} != query {q_len} kv {kv_len} rows {rows_len}",
                q.len(),
                k.len(),
                v.len(),
                out.len(),
                lse.len()
            ),
        ));
    }
    check_extras(op, &g, &extras, true)?;
    let planes = g.batch * g.heads;
    let work = g.work();
    let tasks = (work / TASK_WORK).min(exec.pool.threads().saturating_mul(2));
    let blocks = if exec.pool.threads() > 1 && tasks >= 2 {
        tasks.div_ceil(planes).clamp(1, g.q_len)
    } else {
        1
    };
    let cuts = even_cuts(g.q_len, blocks);
    let launched = planes.saturating_mul(blocks).max(1);
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        launched.min(exec.pool.threads()).max(1)
    };
    let _hold = room_for(op, budget, forward_scratch(op, &g, inflight)?)?;
    let cancel = exec.pool.cancel_hook();
    // Task `plane * blocks + block` writes rows `cuts[block]..cuts[block + 1]`
    // of `plane`: in task order the parts tile `out` and `lse` from the start.
    let row_lens: Vec<usize> = (0..launched)
        .map(|task| cuts[task % blocks + 1] - cuts[task % blocks])
        .collect();
    let part_lens: Vec<usize> = row_lens.iter().map(|rows| rows * g.dim).collect();
    let parts: Vec<Out<'_>> = scoped::cut(out, &part_lens)?
        .into_iter()
        .zip(scoped::cut(lse, &row_lens)?)
        .map(|(o, l)| [o, l])
        .collect();
    fill_parts(exec, work, parts, |task, part| {
        cancel()?;
        let (plane, block) = (task / blocks, task % blocks);
        let (b, h) = (plane / g.heads, plane % g.heads);
        let kv_plane = b * g.kv_heads + h / g.rep();
        let span = plane * g.q_plane_len()..(plane + 1) * g.q_plane_len();
        let kv_span = kv_plane * g.kv_plane_len()..(kv_plane + 1) * g.kv_plane_len();
        let rows = cuts[block]..cuts[block + 1];
        forward_rows(
            op,
            [&q[span], &k[kv_span.clone()], &v[kv_span]],
            extras,
            (b, h),
            rows,
            &g,
            cancel.as_ref(),
            part,
        )
    })?;
    Ok(())
}

/// Forward for rows `rows` of head `h` of batch `b`, written into `out`
/// (`rows.len() * dim` values) and `lse` (`rows.len()` values).
///
/// Each score sums the head dimension from 0 and adds the bias; the softmax
/// and the value mix walk the visible keys in increasing order.
#[allow(clippy::too_many_arguments)]
fn forward_rows(
    op: &'static str,
    [q, k, v]: [&[f32]; 3],
    extras: Extras<'_>,
    (b, h): (usize, usize),
    rows: Range<usize>,
    g: &Geom,
    cancel: &dyn Fn() -> Result<(), OjasError>,
    [out, lse]: Out<'_>,
) -> Result<(), OjasError> {
    let (kv_len, dim) = (g.kv_len, g.dim);
    if q.len() != g.q_plane_len()
        || k.len() != g.kv_plane_len()
        || v.len() != g.kv_plane_len()
        || rows.end > g.q_len
    {
        return Err(shape(op, "attention head length does not match its shape"));
    }
    if out.len() != rows.len() * dim || lse.len() != rows.len() {
        return Err(shape(op, "attention output part does not match its rows"));
    }
    if rows.is_empty() {
        return Ok(());
    }
    let len = g.len_of(extras.key_lens, b);
    let mut scores = vec![0.0f32; kv_len];
    let mut probs = vec![0.0f32; kv_len];
    let mut packed = vec![0.0f32; k.len()];
    pack_keys(k, &mut packed, kv_len, dim);
    for i in rows.clone() {
        cancel()?;
        let local = i - rows.start;
        let keys = g.keys(len, i);
        if keys.is_empty() {
            // Zeros out, and 0 for a log-sum-exp that is not finite.
            continue;
        }
        score_range(
            op,
            &packed,
            row(q, i, dim),
            &mut scores,
            keys.clone(),
            kv_len,
            g.scale,
        )?;
        add_bias(
            op,
            &mut scores,
            g.bias_row(extras.bias, b, h, i),
            keys.clone(),
        )?;
        lse[local] = softmax_range(op, &scores, &mut probs, keys.clone())?;
        mix_values(
            &mut out[local * dim..(local + 1) * dim],
            v,
            &probs,
            keys,
            dim,
            dim,
        );
    }
    Ok(())
}

/// The three gradients, each KV group's by the task that computes it. The
/// forward's `out` and `lse` are for the same operands.
pub(crate) fn attention_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k, v, out, lse, grad_y]: [&[f32]; 6],
    extras: Extras<'_>,
    g: Geom,
    grads: Grads<'_>,
) -> Result<(), OjasError> {
    let (q_len, kv_len, rows_len) = (
        g.checked_q_len(op)?,
        g.checked_kv_len(op)?,
        g.checked_rows(op)?,
    );
    let [gq, gk, gv] = grads;
    if q.len() != q_len
        || out.len() != q_len
        || lse.len() != rows_len
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
                "attention data lengths q {} k {} v {} out {} lse {} grad {} != query {q_len} kv {kv_len} rows {rows_len}",
                q.len(),
                k.len(),
                v.len(),
                out.len(),
                lse.len(),
                grad_y.len()
            ),
        ));
    }
    check_extras(op, &g, &extras, true)?;
    // One KV group is an independent reduction: its `rep` query heads are
    // contiguous and accumulate into the same grad_k / grad_v in increasing
    // query-head order, so a group is never split.
    let groups = g.batch * g.kv_heads;
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        groups.min(exec.pool.threads()).max(1)
    };
    let _hold = room_for(op, budget, backward_scratch(op, &g, inflight)?)?;
    let cancel = exec.pool.cancel_hook();
    let rep = g.rep();
    let (q_stride, kv_stride) = (g.q_plane_len(), g.kv_plane_len());
    let q_lens = vec![q_stride * rep; groups];
    let kv_lens = vec![kv_stride; groups];
    let parts: Vec<Grads<'_>> = scoped::cut(gq, &q_lens)?
        .into_iter()
        .zip(scoped::cut(gk, &kv_lens)?)
        .zip(scoped::cut(gv, &kv_lens)?)
        .map(|((q, k), v)| [q, k, v])
        .collect();
    fill_parts(exec, g.work(), parts, |group, part| {
        cancel()?;
        let [mut gq_rest, gk, gv] = part;
        let (b, kv_head) = (group / g.kv_heads, group % g.kv_heads);
        let kv_span = group * kv_stride..(group + 1) * kv_stride;
        for r in 0..rep {
            let (gq_r, rest) = gq_rest.split_at_mut(q_stride);
            gq_rest = rest;
            let h = kv_head * rep + r;
            let plane = b * g.heads + h;
            let span = plane * q_stride..(plane + 1) * q_stride;
            let heads_in = [
                &q[span.clone()],
                &k[kv_span.clone()],
                &v[kv_span.clone()],
                &out[span.clone()],
                &lse[plane * g.q_len..(plane + 1) * g.q_len],
                &grad_y[span],
            ];
            backward_head(
                op,
                heads_in,
                extras,
                (b, h),
                &g,
                cancel.as_ref(),
                [gq_r, &mut gk[..], &mut gv[..]],
            )?;
        }
        Ok(())
    })?;
    Ok(())
}

/// Backward for head `h` of batch `b`, accumulated into `grads` (zeroed
/// `[Tq, D]`, `[Tk, D]` and `[Tk, D]` slices), from the forward's output rows
/// and log-sum-exp: `P = e^(S - lse)` and `Dr = dO . O`.
fn backward_head(
    op: &'static str,
    [q, k, v, out, lse, grad_y]: [&[f32]; 6],
    extras: Extras<'_>,
    (b, h): (usize, usize),
    g: &Geom,
    cancel: &dyn Fn() -> Result<(), OjasError>,
    [grad_q, grad_k, grad_v]: Grads<'_>,
) -> Result<(), OjasError> {
    let (dim, scale) = (g.dim, g.scale);
    let (q_w, kv_w) = (g.q_plane_len(), g.kv_plane_len());
    if [q.len(), out.len(), grad_y.len(), grad_q.len()] != [q_w; 4]
        || [k.len(), v.len(), grad_k.len(), grad_v.len()] != [kv_w; 4]
        || lse.len() != g.q_len
    {
        return Err(shape(op, "attention head length does not match its shape"));
    }
    let len = g.len_of(extras.key_lens, b);
    let mut probs = vec![0.0f32; g.kv_len];
    let mut dprobs = vec![0.0f32; g.kv_len];
    for i in 0..g.q_len {
        cancel()?;
        let keys = g.keys(len, i);
        if keys.is_empty() {
            continue;
        }
        let q_row = row(q, i, dim);
        let gy_row = row(grad_y, i, dim);
        let row_lse = lse[i];
        let expected = dot_up(gy_row, row(out, i, dim));
        if !(row_lse.is_finite() && expected.is_finite()) {
            return Err(nonfinite(op));
        }
        let bias = g.bias_row(extras.bias, b, h, i);
        for j in keys.clone() {
            let mut score = dot_up(q_row, row(k, j, dim)) * scale;
            if let Some(bias) = bias {
                score += bias[j];
            }
            if !score.is_finite() {
                return Err(nonfinite(op));
            }
            let p = exp_exact(score - row_lse);
            if !p.is_finite() {
                return Err(nonfinite(op));
            }
            probs[j] = p;
            dprobs[j] = dot_up(gy_row, row(v, j, dim));
        }
        for j in keys {
            let p = probs[j];
            let coef = scale * (p * (dprobs[j] - expected));
            saxpy_up(&mut grad_q[i * dim..(i + 1) * dim], row(k, j, dim), coef);
            saxpy_up(&mut grad_k[j * dim..(j + 1) * dim], q_row, coef);
            saxpy_up(&mut grad_v[j * dim..(j + 1) * dim], gy_row, p);
        }
    }
    Ok(())
}

/// The bias gradient `dS = P (dP - Dr)`, summed over the extents the bias
/// broadcast across, into `grad_bias` (`[Bb, Hb, Tqb, Tk]`, zeroed values
/// the caller charged). One task per `(Bb, Hb)` plane.
pub(crate) fn bias_backward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k, v, out, lse, grad_y]: [&[f32]; 6],
    extras: Extras<'_>,
    g: Geom,
    grad_bias: &mut [f32],
) -> Result<(), OjasError> {
    let Some(bias) = g.bias else {
        return Err(shape(op, "attention bias gradient without a bias"));
    };
    let (q_len, kv_len, rows_len) = (
        g.checked_q_len(op)?,
        g.checked_kv_len(op)?,
        g.checked_rows(op)?,
    );
    if q.len() != q_len
        || out.len() != q_len
        || lse.len() != rows_len
        || grad_y.len() != q_len
        || k.len() != kv_len
        || v.len() != kv_len
        || Some(grad_bias.len()) != g.checked_bias_len(op)?
    {
        return Err(shape(op, "attention bias gradient lengths do not match"));
    }
    check_extras(op, &g, &extras, true)?;
    let planes = bias.batch * bias.heads;
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        planes.min(exec.pool.threads()).max(1)
    };
    let _hold = room_for(op, budget, backward_scratch(op, &g, inflight)?)?;
    let cancel = exec.pool.cancel_hook();
    let plane_len = bias.queries * g.kv_len;
    let parts: Vec<&mut [f32]> = grad_bias.chunks_mut(plane_len).collect();
    let rep = g.rep();
    fill_parts(exec, g.work(), parts, |plane, part| {
        cancel()?;
        let (bb, bh) = (plane / bias.heads, plane % bias.heads);
        let batches = if bias.batch == 1 {
            0..g.batch
        } else {
            bb..bb + 1
        };
        let heads = if bias.heads == 1 {
            0..g.heads
        } else {
            bh..bh + 1
        };
        let mut probs = vec![0.0f32; g.kv_len];
        let mut dprobs = vec![0.0f32; g.kv_len];
        for b in batches {
            let len = g.len_of(extras.key_lens, b);
            for h in heads.clone() {
                let head = b * g.heads + h;
                let kv_head = b * g.kv_heads + h / rep;
                let (qh, kh, vh) = (
                    &q[head * g.q_plane_len()..(head + 1) * g.q_plane_len()],
                    &k[kv_head * g.kv_plane_len()..(kv_head + 1) * g.kv_plane_len()],
                    &v[kv_head * g.kv_plane_len()..(kv_head + 1) * g.kv_plane_len()],
                );
                for i in 0..g.q_len {
                    cancel()?;
                    let keys = g.keys(len, i);
                    if keys.is_empty() {
                        continue;
                    }
                    let q_row = row(qh, i, g.dim);
                    let gy_row = row(grad_y, head * g.q_len + i, g.dim);
                    let expected = dot_up(gy_row, row(out, head * g.q_len + i, g.dim));
                    let row_lse = lse[head * g.q_len + i];
                    if !(row_lse.is_finite() && expected.is_finite()) {
                        return Err(nonfinite(op));
                    }
                    let brow = g.bias_row(extras.bias, b, h, i);
                    for j in keys.clone() {
                        let mut score = dot_up(q_row, row(kh, j, g.dim)) * g.scale;
                        if let Some(brow) = brow {
                            score += brow[j];
                        }
                        let p = exp_exact(score - row_lse);
                        if !(score.is_finite() && p.is_finite()) {
                            return Err(nonfinite(op));
                        }
                        probs[j] = p;
                        dprobs[j] = dot_up(gy_row, row(vh, j, g.dim));
                    }
                    let target = if bias.queries == 1 { 0 } else { i };
                    for j in keys {
                        part[target * g.kv_len + j] += probs[j] * (dprobs[j] - expected);
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(())
}
