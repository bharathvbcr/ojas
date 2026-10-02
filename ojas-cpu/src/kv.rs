//! KV-cache attention and cache writes, for decode and for prefill onto a
//! non-empty cache.
//!
//! Caches are time-major `[B, Tcap, Hkv, D]`. Only the first `kv_len` time
//! slots of a cache are read: they are checked for NaN and infinity and
//! copied, and the slots past them are never touched, so a cache's unused
//! capacity may hold anything. Query `i` of `Tq` sits at position
//! `kv_len - Tq + i` and attends to keys `0..=` that position; head `h`
//! reads KV head `h / (H / Hkv)`.
//!
//! Each query row runs the per-row kernel of `causal_sdpa_forward`
//! ([`score_prefix`], [`softmax_prefix`], [`mix_values`]): scores sum the
//! head dimension from 0, the softmax sums keys from 0, and the value mix
//! adds keys from 0. With `kv_len == Tq` and `H == Hkv` the result is
//! therefore `causal_sdpa_forward`'s whenever that op runs its per-row
//! kernel, which is always under `Numerics::Exact` (Fast switches to a
//! blocked kernel above 256 positions; this op does not).

use std::sync::Arc;

use ojas_core::{
    cached_attention_dims, kv_cache_write_dims, sdpa_scale, Budget, KvDims, OjasError, Tensor,
};

use crate::attn::{mix_values, score_prefix, softmax_prefix, TASK_WORK};
use crate::pool::Exec;
use crate::validate::{
    all_finite, check_f32, fill_out, nonfinite, product, room_for, shape, Shared,
};

/// The `kv_len` prefix of every batch of `cache`, checked finite in place.
/// Returns the cache's contiguous values. A device or strided cache is
/// refused by [`Tensor::f32_slice`] before anything is read.
fn cache_prefix<'t>(
    op: &'static str,
    cache: &'t Tensor,
    dims: &KvDims,
    kv_len: usize,
) -> Result<&'t [f32], OjasError> {
    let values = cache.f32_slice()?;
    let step = product(op, &[dims.kv_heads, dims.head_dim])?;
    let per_batch = product(op, &[dims.capacity, step])?;
    let used = product(op, &[kv_len, step])?;
    for b in 0..dims.batch {
        let start = b * per_batch;
        let run = values
            .get(start..start + used)
            .ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: format!("cache window {} is shorter than its shape", values.len()),
            })?;
        if !all_finite(run) {
            return Err(nonfinite(op));
        }
    }
    Ok(values)
}

/// [`ojas_core::Backend::cached_attention_forward`] on the CPU: the
/// `[B, Tq, H, D]` output as a new tensor. Each head's rows are computed on
/// the pool, then copied once into the output tensor ([`fill_out`]).
pub(crate) fn cached_attention_forward(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    kv_len: usize,
) -> Result<Tensor, OjasError> {
    let dims = cached_attention_dims(q, k_cache, v_cache, kv_len)?;
    check_f32(op, q)?;
    let kb = cache_prefix(op, k_cache, &dims, kv_len)?;
    let vb = cache_prefix(op, v_cache, &dims, kv_len)?;
    let KvDims {
        batch,
        new: tq,
        heads,
        kv_heads,
        head_dim: d,
        ..
    } = dims;
    let scale = sdpa_scale(u32::try_from(d).map_err(|_| OjasError::OutOfRange {
        op,
        detail: format!("head dim {d} does not fit in u32"),
    })?)?;
    let out_len = product(op, &[batch, tq, heads, d])?;
    let per_kv_head = product(op, &[kv_len, d])?;
    let gathered = product(op, &[2, batch, kv_heads, per_kv_head])?;
    let tasks = batch * heads;
    let inflight = if exec.pool.threads() <= 1 {
        1
    } else {
        tasks.min(exec.pool.threads())
    };
    // Gathered K and V prefixes, the per-head output parts (as large as the
    // output, live while it is assembled), and two length-`kv_len` rows per
    // running task.
    let work = [gathered, out_len, product(op, &[inflight, 2, kv_len])?]
        .into_iter()
        .try_fold(0usize, |sum, n| sum.checked_add(n))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "cached attention scratch length overflows".to_string(),
        })?;
    // `q` was scanned above; its tasks read it in place.
    let qv = Shared::new(op, q)?;
    fill_out(op, budget, exec, q.shape(), |out| {
        if out.len() != out_len {
            return Err(shape(
                op,
                format!("cached attention output {} != {out_len}", out.len()),
            ));
        }
        let _scratch = room_for(op, budget, work)?;
        attend(op, exec, qv, [kb, vb], dims, kv_len, scale, out)
    })
}

/// The heads of [`cached_attention_forward`], on the pool, each copied
/// into its rows of `out` (`[B, Tq, H, D]`).
#[allow(clippy::too_many_arguments)]
fn attend(
    op: &'static str,
    exec: Exec<'_>,
    qv: Shared,
    [kb, vb]: [&[f32]; 2],
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
    } = dims;
    let per_kv_head = kv_len * d;
    let tasks = batch * heads;

    // K packed `[D, kv_len]` and V `[kv_len, D]` per (batch, kv head): the
    // layouts `score_prefix` and `mix_values` read.
    let mut keys = vec![0.0f32; batch * kv_heads * per_kv_head];
    let mut values = vec![0.0f32; batch * kv_heads * per_kv_head];
    for b in 0..batch {
        for j in 0..kv_len {
            for hk in 0..kv_heads {
                let src = ((b * capacity + j) * kv_heads + hk) * d;
                let base = (b * kv_heads + hk) * per_kv_head;
                for dd in 0..d {
                    keys[base + dd * kv_len + j] = kb[src + dd];
                    values[base + j * d + dd] = vb[src + dd];
                }
            }
        }
    }

    let group = heads / kv_heads;
    let work_units = tasks
        .saturating_mul(tq)
        .saturating_mul(kv_len)
        .saturating_mul(d);
    let cancel = exec.pool.cancel_hook();
    let (keys, values) = (Arc::new(keys), Arc::new(values));
    let parts = exec.map(tasks, work_units, 2 * TASK_WORK, {
        let cancel = Arc::clone(&cancel);
        move |task| {
            cancel()?;
            let q = qv.values()?;
            let (b, h) = (task / heads, task % heads);
            let base = (b * kv_heads + h / group) * per_kv_head;
            let kh = &keys[base..base + per_kv_head];
            let vh = &values[base..base + per_kv_head];
            let mut scores = vec![0.0f32; kv_len];
            let mut probs = vec![0.0f32; kv_len];
            let mut out = vec![0.0f32; tq * d];
            for i in 0..tq {
                cancel()?;
                let visible = kv_len - tq + i + 1;
                let qs = ((b * tq + i) * heads + h) * d;
                score_prefix(op, kh, &q[qs..qs + d], &mut scores, visible, kv_len, scale)?;
                softmax_prefix(op, &scores, &mut probs, visible)?;
                mix_values(&mut out[i * d..(i + 1) * d], vh, &probs, visible, d);
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
/// Every check (shapes, a finite `src`, a uniquely owned host cache, room for
/// the scratch) runs before the cache changes, and the one write is last.
/// `Tensor` has no in-place sub-range write, so the new contents are built
/// in a cache-sized buffer and written whole: each call costs `O(Tcap)`, not
/// `O(Tn)`. Slots outside `at..at + Tn` keep their bits, NaN padding included.
pub(crate) fn kv_cache_write(
    op: &'static str,
    budget: &Budget,
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
    let _hold = room_for(op, budget, len)?;
    let mut next = cache.to_f32_vec()?;
    let sv = src.f32_slice()?;
    for b in 0..dims.batch {
        let dst = (b * dims.capacity + at) * step;
        next[dst..dst + run].copy_from_slice(&sv[b * run..(b + 1) * run]);
    }
    cache.write_f32(&next)
}
