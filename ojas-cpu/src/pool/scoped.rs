//! Scoped parallel loops for ops that read tensors in place or fill their
//! output buffers in place.
//!
//! [`crate::pool::Pool`] runs `'static` tasks on persistent workers, so a task
//! cannot borrow a tensor's bytes or write into a slice of one output buffer
//! (the crate forbids `unsafe`); every pool result is a separate vector that
//! the caller joins with a copy. Here the tasks borrow: they run on the
//! calling thread and on up to `exec.pool.threads() - 1` threads spawned with
//! [`std::thread::scope`] for this call (about 37 µs for five threads on an
//! M5 Pro), so they read [`ojas_core::Tensor::f32_slice`] in place and
//! [`fill`] writes each block of the output where it belongs.
//!
//! The partition is the caller's: task `i` always covers the same items, and
//! results come back in task order, so which thread ran a task changes no
//! bit. One task, or one thread, runs inline with no spawn. A task that panics
//! is [`OjasError::Backend`] after every thread has finished.
//!
//! A thread that cannot be spawned is not an error, unlike a pool worker
//! ([`crate::pool::Pool`] refuses every later batch): its share falls to the
//! threads that did start, the calling thread always among them, so every
//! task still runs exactly once and the result is the same. The in-place
//! optimizer and clip steps run passes after their first tensor write, and a
//! refusal there would leave the step half applied.
//!
//! The spawn stays, by decision (2026-10-08). Re-measured that day at 34-37
//! µs minimum and about 70 µs median for five threads under load 26
//! (`bench/results/2026-10-08-cpu-hot-paths/spawn.txt`), it is about the
//! gap that keeps the 0.1 ms mul and add forwards behind torch
//! (`docs/bench-cpu-vs-torch.md`). Removing it means persistent workers
//! writing into one borrowed output, which takes either `unsafe` (a lifetime
//! erased across the worker handoff) here, where the crate forbids it, or a
//! new dependency whose scope API does that soundly (for example rayon's
//! `scope`). Both need the owner's approval, and neither has it, so every
//! split keeps paying this spawn and those two rows stay open.

use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ojas_core::{BackendId, OjasError};

use super::{ranges, Exec, ROW_MIN_ELEMS, WORKER_STACK_BYTES};

fn backend(detail: String) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail,
    }
}

/// `task(i)` for `i` in `0..count`, results in index order; the first error
/// in index order wins.
pub(crate) fn map<T, F>(exec: Exec<'_>, count: usize, task: F) -> Result<Vec<T>, OjasError>
where
    T: Send,
    F: Fn(usize) -> Result<T, OjasError> + Sync,
{
    let threads = exec.pool.threads().min(count);
    if threads <= 1 {
        return (0..count).map(task).collect();
    }
    let next = AtomicUsize::new(0);
    let work = || {
        let mut done = Vec::new();
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= count {
                return done;
            }
            done.push((i, task(i)));
        }
    };
    let mut slots: Vec<Option<Result<T, OjasError>>> = (0..count).map(|_| None).collect();
    let mut panicked = false;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads - 1);
        for _ in 1..threads {
            match std::thread::Builder::new()
                .stack_size(WORKER_STACK_BYTES)
                .spawn_scoped(scope, work)
            {
                Ok(handle) => handles.push(handle),
                // The threads already started and this one take its share.
                Err(_) => break,
            }
        }
        let mut parts = vec![catch_unwind(AssertUnwindSafe(work))];
        for handle in handles {
            parts.push(handle.join());
        }
        for part in parts {
            match part {
                Ok(done) => {
                    for (i, result) in done {
                        slots[i] = Some(result);
                    }
                }
                Err(_) => panicked = true,
            }
        }
    });
    if panicked {
        return Err(backend("scoped worker task panicked".to_string()));
    }
    let mut out = Vec::with_capacity(count);
    for slot in slots {
        match slot {
            Some(result) => out.push(result?),
            None => return Err(backend("scoped task finished without a result".to_string())),
        }
    }
    Ok(out)
}

/// `out` cut into consecutive blocks of `block` items (the last may be
/// shorter); `task(i, block_i)` fills block `i` in place. Each task's
/// result comes back in block order, as [`map`] returns them.
pub(crate) fn fill<T, R, F>(
    exec: Exec<'_>,
    out: &mut [T],
    block: usize,
    task: F,
) -> Result<Vec<R>, OjasError>
where
    T: Send,
    R: Send,
    F: Fn(usize, &mut [T]) -> Result<R, OjasError> + Sync,
{
    fill_parts(exec, out.chunks_mut(block.max(1)).collect(), task)
}

/// `task(i, parts[i])` for every part, each filled in place; results in
/// part order. The caller cuts the parts (disjoint slices of the outputs, of
/// any lengths, or an array of slices of several outputs) and orders them:
/// tasks are claimed in part order, so a part listed first starts first.
pub(crate) fn fill_parts<P, R, F>(
    exec: Exec<'_>,
    parts: Vec<P>,
    task: F,
) -> Result<Vec<R>, OjasError>
where
    P: Send,
    R: Send,
    F: Fn(usize, P) -> Result<R, OjasError> + Sync,
{
    let parts: Vec<Mutex<Option<P>>> = parts
        .into_iter()
        .map(|part| Mutex::new(Some(part)))
        .collect();
    map(exec, parts.len(), |i| {
        let part = parts[i]
            .lock()
            .map_err(|_| backend("fill part lock poisoned".to_string()))?
            .take()
            .ok_or_else(|| backend(format!("fill part {i} claimed twice")))?;
        task(i, part)
    })
}

/// Items `0..len` cut into contiguous pieces of at least `min_chunk` items,
/// at most two a thread so fast cores take over work from slow ones,
/// item `i` owning values `i * width..(i + 1) * width` of `out`, and
/// `task(range, part)` filling each piece's values. One piece, or one
/// thread, runs on the calling thread. Results come back in piece order;
/// the first error in piece order wins.
pub(crate) fn chunks_into<R, F>(
    exec: Exec<'_>,
    out: &mut [f32],
    len: usize,
    width: usize,
    min_chunk: usize,
    task: F,
) -> Result<Vec<R>, OjasError>
where
    R: Send,
    F: Fn(Range<usize>, &mut [f32]) -> Result<R, OjasError> + Sync,
{
    chunks_into_n(exec, [out], len, [width], min_chunk, |range, [part]| {
        task(range, part)
    })
}

/// [`chunks_into`] over `N` outputs at once: item `i` owns values
/// `i * widths[j]..(i + 1) * widths[j]` of `outs[j]`, and each piece's task
/// gets its values of every output. The last item of an output may be
/// short (fixed-size blocks over a length that is not a multiple of them),
/// so `outs[j]` holds `len * widths[j]` values or fewer, but more than
/// `(len - 1) * widths[j]`.
pub(crate) fn chunks_into_n<const N: usize, R, F>(
    exec: Exec<'_>,
    outs: [&mut [f32]; N],
    len: usize,
    widths: [usize; N],
    min_chunk: usize,
    task: F,
) -> Result<Vec<R>, OjasError>
where
    R: Send,
    F: Fn(Range<usize>, [&mut [f32]; N]) -> Result<R, OjasError> + Sync,
{
    for (out, &width) in outs.iter().zip(&widths) {
        let full = len.checked_mul(width);
        let fits = full.is_some_and(|full| {
            out.len() == full || (out.len() < full && out.len() > full - width)
        });
        if !fits {
            return Err(backend(format!(
                "chunks_into: {len} items of {width} != output {}",
                out.len()
            )));
        }
    }
    let parts = pieces(exec, len, min_chunk);
    let mut cuts = Vec::with_capacity(N);
    for (out, width) in outs.into_iter().zip(widths) {
        let end = out.len();
        let lens: Vec<usize> = parts
            .iter()
            .map(|r| (r.end * width).min(end) - (r.start * width).min(end))
            .collect();
        cuts.push(cut(out, &lens)?.into_iter());
    }
    let mut slices = Vec::with_capacity(parts.len());
    for _ in &parts {
        let piece: Vec<&mut [f32]> = cuts.iter_mut().filter_map(Iterator::next).collect();
        let piece: [&mut [f32]; N] = piece
            .try_into()
            .map_err(|_| backend("chunks_into: a piece lacks an output".to_string()))?;
        slices.push(piece);
    }
    if parts.len() == 1 || exec.pool.threads() <= 1 {
        return parts
            .into_iter()
            .zip(slices)
            .map(|(range, part)| task(range, part))
            .collect();
    }
    fill_parts(exec, slices, |i, part| task(parts[i].clone(), part))
}

/// The pieces [`chunks_into_n`] cuts items `0..len` into: at least
/// `min_chunk` items each, at most two a thread, sizes differing by at most
/// one. A read-only pass over the same items uses this cut to match a later
/// write pass piece for piece.
pub(crate) fn pieces(exec: Exec<'_>, len: usize, min_chunk: usize) -> Vec<Range<usize>> {
    let count = (len / min_chunk.max(1)).clamp(1, exec.pool.threads().saturating_mul(2));
    ranges(len, count)
}

/// Rows `0..rows` of `out` (`[rows, width]`), each task filling a chunk of
/// whole rows in place: chunks of about [`ROW_MIN_ELEMS`] values
/// ([`chunks_into`]).
pub(crate) fn rows_into<R, F>(
    exec: Exec<'_>,
    out: &mut [f32],
    rows: usize,
    width: usize,
    task: F,
) -> Result<Vec<R>, OjasError>
where
    R: Send,
    F: Fn(Range<usize>, &mut [f32]) -> Result<R, OjasError> + Sync,
{
    chunks_into(exec, out, rows, width, min_rows(width), task)
}

/// Whole rows of `width` values per chunk of about [`ROW_MIN_ELEMS`]
/// values, at least one.
pub(crate) fn min_rows(width: usize) -> usize {
    (ROW_MIN_ELEMS / width.max(1)).max(1)
}

/// `out` cut into consecutive parts of `lens` items, in order. The lengths
/// must add up to `out.len()`, or this is [`OjasError::Backend`].
pub(crate) fn cut<'a, T>(out: &'a mut [T], lens: &[usize]) -> Result<Vec<&'a mut [T]>, OjasError> {
    let mut parts = Vec::with_capacity(lens.len());
    let mut rest = out;
    for &n in lens {
        if n > rest.len() {
            return Err(backend(format!("cut of {n} items past the output end")));
        }
        let (part, tail) = rest.split_at_mut(n);
        parts.push(part);
        rest = tail;
    }
    if !rest.is_empty() {
        return Err(backend(format!("cut leaves {} items unfilled", rest.len())));
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ojas_core::Numerics;

    use super::*;
    use crate::pool::Pool;

    fn with<R>(threads: usize, f: impl FnOnce(Exec<'_>) -> R) -> R {
        let pool = Arc::new(Pool::new(threads).unwrap());
        f(Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        })
    }

    #[test]
    fn map_returns_index_order_and_the_first_error() {
        for threads in [1usize, 2, 7, 18] {
            let got = with(threads, |e| map(e, 100, |i| Ok(i * i))).unwrap();
            assert_eq!(got, (0..100).map(|i| i * i).collect::<Vec<_>>());
            let err = with(threads, |e| {
                map(e, 50, |i| {
                    if i == 7 || i == 30 {
                        Err(OjasError::NonFinite {
                            op: if i == 7 { "seven" } else { "thirty" },
                        })
                    } else {
                        Ok(i)
                    }
                })
            })
            .unwrap_err();
            assert!(
                matches!(err, OjasError::NonFinite { op: "seven" }),
                "{err:?}"
            );
            assert!(with(threads, |e| map(e, 0, Ok)).unwrap().is_empty());
        }
    }

    #[test]
    fn fill_writes_every_block_in_place() {
        for threads in [1usize, 3, 18] {
            let mut out = vec![0usize; 1003];
            with(threads, |e| {
                fill(e, &mut out, 10, |b, chunk| {
                    for (k, v) in chunk.iter_mut().enumerate() {
                        *v = b * 10 + k;
                    }
                    Ok(())
                })
            })
            .unwrap();
            assert_eq!(out, (0..1003).collect::<Vec<_>>());
        }
    }

    /// Two outputs of different widths, the second's last item short: every
    /// value is written once by the piece that owns its item, and the pieces
    /// cut as `ranges` cuts them.
    #[test]
    fn chunks_into_n_fills_each_output_by_item_with_a_short_last_item() {
        for threads in [1usize, 3, 18] {
            let (len, min_chunk) = (37usize, 4usize);
            let mut a = vec![0.0f32; len * 3];
            let mut b = vec![0.0f32; (len - 1) * 5 + 2];
            let pieces = with(threads, |e| {
                chunks_into_n(
                    e,
                    [&mut a, &mut b],
                    len,
                    [3, 5],
                    min_chunk,
                    |range, [pa, pb]| {
                        assert_eq!(pa.len(), range.len() * 3);
                        for (k, v) in pa.iter_mut().enumerate() {
                            *v = (range.start * 3 + k) as f32;
                        }
                        for (k, v) in pb.iter_mut().enumerate() {
                            *v = (range.start * 5 + k) as f32;
                        }
                        Ok(range)
                    },
                )
            })
            .unwrap();
            assert_eq!(a, (0..a.len()).map(|i| i as f32).collect::<Vec<_>>());
            assert_eq!(b, (0..b.len()).map(|i| i as f32).collect::<Vec<_>>());
            let pieces_wanted = (len / min_chunk).clamp(1, threads * 2);
            assert_eq!(pieces, ranges(len, pieces_wanted), "threads {threads}");
        }
    }

    #[test]
    fn chunks_into_n_refuses_an_output_of_the_wrong_length() {
        // 4 items of 3: 12 values, or a short last item of 1 or 2.
        for (len_b, ok) in [
            (12usize, true),
            (11, true),
            (10, true),
            (9, false),
            (13, false),
        ] {
            let mut a = vec![0.0f32; 4];
            let mut b = vec![0.0f32; len_b];
            let got = with(2, |e| {
                chunks_into_n(e, [&mut a, &mut b], 4, [1, 3], 1, |_, _| Ok(()))
            });
            assert_eq!(got.is_ok(), ok, "second output of {len_b} for 4 items of 3");
        }
    }

    #[test]
    fn a_panicking_task_is_a_backend_error() {
        for threads in [1usize, 4] {
            let got = with(threads, |e| {
                std::panic::catch_unwind(AssertUnwindSafe(|| {
                    map(e, 8, |i| if i == 5 { panic!("task five") } else { Ok(i) })
                }))
            });
            match (threads, got) {
                // One thread runs inline, like the pool's caller: the panic
                // propagates.
                (1, Err(_)) => {}
                (_, Ok(Err(OjasError::Backend { .. }))) => {}
                (t, other) => panic!("threads {t}: {:?}", other.map(|r| r.map(drop))),
            }
        }
    }
}
