//! Scoped parallel loops for ops that read tensors in place or fill one
//! output buffer.
//!
//! [`crate::pool::Pool`] runs `'static` tasks on persistent workers, so a task
//! cannot borrow a tensor's bytes or write into a slice of one output buffer
//! (the crate forbids `unsafe`); every pool result is a separate vector that
//! the caller joins with a copy. Here the tasks borrow: they run on the
//! calling thread and on up to `exec.pool.threads() - 1` threads spawned with
//! [`std::thread::scope`] for this call (about 37 µs for five threads on an
//! M5 Pro), so they read [`ojas_core::Tensor::contiguous_bytes`] in place and
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

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ojas_core::{BackendId, OjasError};

use super::{Exec, WORKER_STACK_BYTES};

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
    let blocks: Vec<Mutex<Option<&mut [T]>>> = out
        .chunks_mut(block.max(1))
        .map(|chunk| Mutex::new(Some(chunk)))
        .collect();
    map(exec, blocks.len(), |i| {
        let chunk = blocks[i]
            .lock()
            .map_err(|_| backend("fill block lock poisoned".to_string()))?
            .take()
            .ok_or_else(|| backend(format!("fill block {i} claimed twice")))?;
        task(i, chunk)
    })
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
