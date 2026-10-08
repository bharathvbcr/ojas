//! KV-cache attention and cache writes, for decode and for prefill onto a
//! non-empty cache.
//!
//! Caches are time-major `[B, Tcap, Hkv, D]` rings: position `j` is slot
//! `j % Tcap`. Only the slots of positions some query reads are touched:
//! they are checked for NaN and infinity and read in place, and every other
//! slot is never read, so a cache's unused capacity may hold anything.
//! Query `i` of `Tq` sits at position `p = kv_len - Tq + i` and attends to
//! positions [`KvDims::visible`] (`0..=p`, or the last `window` of them),
//! oldest first; head `h` reads KV head `h / (H / Hkv)`.
//!
//! Each query row runs the per-row kernel of `causal_sdpa_forward`
//! ([`score_range`], [`softmax_range`], [`mix_values`]): scores sum the
//! head dimension from 0, the softmax sums keys from 0, and the value mix
//! adds keys from 0. With `kv_len == Tq` and `H == Hkv` the result is
//! therefore `causal_sdpa_forward`'s (under the same window) whenever that
//! op runs its per-row kernel, which is always under `Numerics::Exact`
//! (Fast switches to a blocked kernel above 256 positions; this op does
//! not).

use std::ops::Range;
use std::sync::Arc;

use ojas_core::{
    cached_attention_dims, kv_cache_write_dims, sdpa_scale, Budget, KvDims, OjasError, Tensor,
};

use crate::attn::{mix_values, score_rows, softmax_range, TASK_WORK};
use crate::pool::Exec;
use crate::validate::{
    all_finite, check_f32, fill_out, nonfinite, product, room_for, shape, Shared,
};

/// The cache slots holding `positions`, as at most two runs of slots
/// (`positions` wraps the ring at most once: it is no longer than
/// `capacity`, which [`cached_attention_dims`] checked).
fn slot_runs(positions: Range<usize>, capacity: usize) -> [Range<usize>; 2] {
    let first = positions.start % capacity;
    let len = positions.len();
    let head = len.min(capacity - first);
    [first..first + head, 0..len - head]
}

/// Check the slots of every position some query reads, in every batch of
/// `cache`, finite in place. A device or strided cache is refused by
/// [`Tensor::f32_slice`] before anything is read.
fn check_read_slots(
    op: &'static str,
    cache: &Tensor,
    dims: &KvDims,
    kv_len: usize,
) -> Result<(), OjasError> {
    let values = cache.f32_slice()?;
    let step = product(op, &[dims.kv_heads, dims.head_dim])?;
    let per_batch = product(op, &[dims.capacity, step])?;
    let read = dims.visible(kv_len, 0).start..kv_len;
    for b in 0..dims.batch {
        for run in slot_runs(read.clone(), dims.capacity) {
            let start = b * per_batch + run.start * step;
            let run = values.get(start..start + run.len() * step).ok_or_else(|| {
                OjasError::OutOfRange {
                    op,
                    detail: format!("cache window {} is shorter than its shape", values.len()),
                }
            })?;
            if !all_finite(run) {
                return Err(nonfinite(op));
            }
        }
    }
    Ok(())
}

/// [`ojas_core::Backend::cached_attention_forward`] on the CPU: the
/// `[B, Tq, H, D]` output as a new tensor. Each head's rows are computed on
/// the pool, reading the cache in place (nothing is repacked), then copied
/// once into the output tensor ([`fill_out`]).
pub(crate) fn cached_attention_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    [q, k_cache, v_cache]: [&Tensor; 3],
    kv_len: usize,
    window: Option<usize>,
) -> Result<Tensor, OjasError> {
    let dims = cached_attention_dims(q, k_cache, v_cache, kv_len, window)?;
    check_f32(op, q)?;
    check_read_slots(op, k_cache, &dims, kv_len)?;
    check_read_slots(op, v_cache, &dims, kv_len)?;
    let KvDims {
        batch,
        new: tq,
        heads,
        head_dim: d,
        ..
    } = dims;
    let scale = sdpa_scale(u32::try_from(d).map_err(|_| OjasError::OutOfRange {
        op,
        detail: format!("head dim {d} does not fit in u32"),
    })?)?;
    let out_len = product(op, &[batch, tq, heads, d])?;
    let tasks = batch * heads;
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        tasks.min(exec.pool.threads())
    };
    // The per-head output parts (as large as the output, live while it is
    // assembled), and two rows as long as the widest query's keys per
    // running task.
    let keys = dims.visible(kv_len, tq - 1).len();
    let work = out_len
        .checked_add(product(op, &[inflight, 2, keys])?)
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "cached attention scratch length overflows".to_string(),
        })?;
    // `q` and the cache prefixes were scanned above; tasks read them in place.
    let operands = [
        Shared::new(op, q)?,
        Shared::new(op, k_cache)?,
        Shared::new(op, v_cache)?,
    ];
    fill_out(op, budget, exec, q.shape(), |out| {
        if out.len() != out_len {
            return Err(shape(
                op,
                format!("cached attention output {} != {out_len}", out.len()),
            ));
        }
        let _scratch = room_for(op, budget, work)?;
        attend(op, exec, operands, dims, kv_len, scale, out)
    })
}

/// The heads of [`cached_attention_forward`], on the pool, each copied
/// into its rows of `out` (`[B, Tq, H, D]`). Key and value `j` of a head
/// are read where the time-major ring holds them, slot `j % Tcap`, rows
/// `kv_heads * D` apart; a query's keys are scored and mixed oldest first,
/// one run of slots at a time.
fn attend(
    op: &'static str,
    exec: Exec<'_>,
    [qv, kv, vv]: [Shared; 3],
    dims: KvDims,
    kv_len: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), OjasError> {
    let KvDims {
        batch,
        new: tq,
        capacity,
        heads,
        kv_heads,
        head_dim: d,
        ..
    } = dims;
    let tasks = batch * heads;
    let group = heads / kv_heads;
    let stride = kv_heads * d;
    let widest = dims.visible(kv_len, tq - 1).len();
    let work_units = tasks
        .saturating_mul(tq)
        .saturating_mul(widest)
        .saturating_mul(d);
    let cancel = exec.pool.cancel_hook();
    let parts = exec.map(tasks, work_units, 2 * TASK_WORK, {
        let cancel = Arc::clone(&cancel);
        move |task| {
            cancel()?;
            let (q, kb, vb) = (qv.values()?, kv.values()?, vv.values()?);
            let (b, h) = (task / heads, task % heads);
            // Head `h`'s keys and values: slot 0's row, then every `stride`.
            let base = (b * capacity * kv_heads + h / group) * d;
            let span = (capacity - 1) * stride + d;
            let (kh, vh) = (&kb[base..base + span], &vb[base..base + span]);
            let mut scores = vec![0.0f32; widest];
            let mut probs = vec![0.0f32; widest];
            let mut out = vec![0.0f32; tq * d];
            for i in 0..tq {
                cancel()?;
                let seen = dims.visible(kv_len, i);
                let n = seen.len();
                let qs = ((b * tq + i) * heads + h) * d;
                let runs = slot_runs(seen, capacity);
                let mut at = 0;
                for run in runs.iter().filter(|r| !r.is_empty()) {
                    score_rows(
                        op,
                        &kh[run.start * stride..],
                        stride,
                        &q[qs..qs + d],
                        &mut scores[at..],
                        0..run.len(),
                        scale,
                    )?;
                    at += run.len();
                }
                softmax_range(op, &scores, &mut probs, 0..n)?;
                let mut at = 0;
                for run in runs.iter().filter(|r| !r.is_empty()) {
                    mix_values(
                        &mut out[i * d..(i + 1) * d],
                        &vh[run.start * stride..],
                        &probs[at..],
                        0..run.len(),
                        d,
                        stride,
                    );
                    at += run.len();
                }
            }
            Ok::<_, OjasError>(out)
        }
    })?;
    for (task, part) in parts.into_iter().enumerate() {
        let part = part?;
        let (b, h) = (task / heads, task % heads);
        for i in 0..tq {
            let dst = ((b * tq + i) * heads + h) * d;
            out[dst..dst + d].copy_from_slice(&part[i * d..(i + 1) * d]);
        }
    }
    Ok(())
}

/// [`ojas_core::Backend::kv_cache_write`] on the CPU: all-or-nothing.
///
/// Every check (shapes, a finite `src`, a uniquely owned host cache) runs
/// before the cache changes; the copy of `src` into its slots, which cannot
/// fail, is last. Position `at + t` goes to slot `(at + t) % Tcap`, in place:
/// each call moves `O(Tn)` values, not `O(Tcap)`. Every other slot keeps
/// its bits, NaN padding included.
pub(crate) fn kv_cache_write(
    op: &'static str,
    cache: &mut Tensor,
    src: &Tensor,
    at: usize,
) -> Result<(), OjasError> {
    let dims = kv_cache_write_dims(cache, src, at)?;
    check_f32(op, src)?;
    let len = cache.num_elements()?;
    cache.ensure_writable_f32(len)?;
    let step = product(op, &[dims.kv_heads, dims.head_dim])?;
    let run = product(op, &[dims.new, step])?;
    let sv = src.f32_slice()?;
    let next = cache.f32_slice_mut()?;
    for b in 0..dims.batch {
        let mut from = b * run;
        for slots in slot_runs(at..at + dims.new, dims.capacity) {
            let dst = (b * dims.capacity + slots.start) * step;
            let n = slots.len() * step;
            next[dst..dst + n].copy_from_slice(&sv[from..from + n]);
            from += n;
        }
    }
    Ok(())
}
