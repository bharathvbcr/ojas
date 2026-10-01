//! Persistent worker pool, `std` only.
//!
//! A pool spawns its workers once, on the first batch that can use them, and
//! keeps them until the last [`crate::CpuBackend`] clone sharing it drops. [`Pool::run`] publishes a batch of independent
//! tasks, runs unclaimed tasks of that batch on the calling thread, and waits
//! only for tasks another thread has already started. A task that calls
//! `run` again, or many threads calling `run` at once, therefore always make
//! progress: nothing waits on a task that has not started.
//!
//! The crate forbids `unsafe`, so a worker cannot borrow the caller's stack.
//! Tasks are `'static` closures over `Arc`-shared inputs, and each task
//! returns its own output, which `run` hands back in task order. A task that
//! panics becomes [`OjasError::Backend`]; the worker survives and the pool
//! stays usable.

use std::collections::VecDeque;
use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ojas_core::{BackendId, Numerics, OjasError, CPU_THREAD_CEILING};

/// Ceiling shared with [`ojas_core::CPU_THREAD_CEILING`]. Refused, not clamped.
pub(crate) const MAX_THREADS: usize = CPU_THREAD_CEILING as usize;

/// Explicit stack for every worker this crate spawns.
pub(crate) const WORKER_STACK_BYTES: usize = 8 << 20;

pub(crate) type CancelHook = Arc<dyn Fn() -> Result<(), OjasError> + Send + Sync>;

pub(crate) fn noop_cancel() -> CancelHook {
    Arc::new(|| Ok(()))
}

/// A worker that finds no batch spins this long before it sleeps, so a
/// burst of small parallel ops does not pay a condvar wake for each one.
const SPIN: Duration = Duration::from_micros(40);

trait Work: Send + Sync {
    /// Claim and run one task. `false` when every task is already claimed.
    fn run_one(&self) -> bool;
}

struct BatchState<T> {
    results: Vec<Option<T>>,
    finished: usize,
    panic: Option<String>,
}

type Task<T> = Box<dyn Fn(usize) -> T + Send + Sync>;

struct Batch<T> {
    next: AtomicUsize,
    count: usize,
    task: Task<T>,
    state: Mutex<BatchState<T>>,
    done: Condvar,
}

impl<T: Send> Work for Batch<T> {
    fn run_one(&self) -> bool {
        let index = self.next.fetch_add(1, Ordering::Relaxed);
        if index >= self.count {
            return false;
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| (self.task)(index)));
        let mut state = lock(&self.state);
        match outcome {
            Ok(value) => {
                if let Some(slot) = state.results.get_mut(index) {
                    *slot = Some(value);
                }
            }
            Err(payload) => {
                if state.panic.is_none() {
                    state.panic = Some(panic_text(payload.as_ref()));
                }
            }
        }
        state.finished += 1;
        if state.finished == self.count {
            self.done.notify_all();
        }
        true
    }
}

struct Queue {
    batches: VecDeque<Arc<dyn Work>>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    /// Bumped on every publish; a spinning worker watches it without the lock.
    posted: AtomicUsize,
}

pub(crate) struct Pool {
    shared: Arc<Shared>,
    threads: usize,
    spawned: AtomicBool,
    workers: Mutex<Workers>,
    cancel: Mutex<CancelHook>,
}

struct Workers {
    handles: Vec<JoinHandle<()>>,
    failed: Option<String>,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("threads", &self.threads())
            .finish()
    }
}

impl Pool {
    /// `threads` counts the caller, so up to `threads - 1` workers are
    /// spawned. `0` and values above [`MAX_THREADS`] are refused.
    pub(crate) fn new(threads: usize) -> Result<Self, OjasError> {
        if threads == 0 {
            return Err(OjasError::OutOfRange {
                op: "CpuBackend::with_threads",
                detail: "thread count is 0".to_string(),
            });
        }
        if threads > MAX_THREADS {
            return Err(OjasError::OutOfRange {
                op: "CpuBackend::with_threads",
                detail: format!("thread count {threads} exceeds {MAX_THREADS}"),
            });
        }
        Ok(Self::sized(threads))
    }

    /// The caller only; never spawns.
    pub(crate) fn serial() -> Self {
        Self::sized(1)
    }

    fn sized(threads: usize) -> Self {
        Pool {
            shared: Arc::new(Shared {
                queue: Mutex::new(Queue {
                    batches: VecDeque::new(),
                    shutdown: false,
                }),
                wake: Condvar::new(),
                posted: AtomicUsize::new(0),
            }),
            threads,
            spawned: AtomicBool::new(threads == 1),
            workers: Mutex::new(Workers {
                handles: Vec::new(),
                failed: None,
            }),
            cancel: Mutex::new(noop_cancel()),
        }
    }

    pub(crate) fn set_cancel(&self, hook: CancelHook) {
        *lock(&self.cancel) = hook;
    }

    pub(crate) fn cancel_hook(&self) -> CancelHook {
        Arc::clone(&*lock(&self.cancel))
    }

    /// Workers plus the calling thread.
    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    /// Spawn the workers once. A spawn failure is kept and returned to every
    /// later batch rather than running with fewer threads than configured.
    fn ensure_workers(&self) -> Result<(), OjasError> {
        if self.spawned.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut workers = lock(&self.workers);
        if let Some(text) = &workers.failed {
            return Err(spawn_error(text));
        }
        while workers.handles.len() + 1 < self.threads {
            let index = workers.handles.len() + 1;
            let shared = Arc::clone(&self.shared);
            match std::thread::Builder::new()
                .name(format!("ojas-cpu-w{index}"))
                .stack_size(WORKER_STACK_BYTES)
                .spawn(move || worker(shared))
            {
                Ok(handle) => workers.handles.push(handle),
                Err(err) => {
                    let text = format!("spawning worker {index} failed: {err}");
                    let out = spawn_error(&text);
                    workers.failed = Some(text);
                    return Err(out);
                }
            }
        }
        self.spawned.store(true, Ordering::Release);
        Ok(())
    }

    /// Run `task(0..count)` and return the outputs in index order.
    ///
    /// The calling thread runs tasks too. A panicking task is reported as
    /// [`OjasError::Backend`] after every task of the batch has finished.
    pub(crate) fn run<T, F>(&self, count: usize, task: F) -> Result<Vec<T>, OjasError>
    where
        T: Send + 'static,
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        if count == 0 {
            return Ok(Vec::new());
        }
        let batch = Arc::new(Batch {
            next: AtomicUsize::new(0),
            count,
            task: Box::new(task),
            state: Mutex::new(BatchState {
                results: (0..count).map(|_| None).collect(),
                finished: 0,
                panic: None,
            }),
            done: Condvar::new(),
        });
        let published = count > 1 && self.threads > 1;
        if published {
            self.ensure_workers()?;
            let work: Arc<dyn Work> = batch.clone();
            let mut queue = lock(&self.shared.queue);
            queue.batches.push_back(work);
            self.shared.posted.fetch_add(1, Ordering::Release);
            drop(queue);
            if count >= self.threads {
                self.shared.wake.notify_all();
            } else {
                for _ in 0..count - 1 {
                    self.shared.wake.notify_one();
                }
            }
        }
        while batch.run_one() {}
        if published {
            unpublish(&self.shared, Arc::as_ptr(&batch) as *const ());
        }
        let mut state = lock(&batch.state);
        while state.finished < count {
            state = match batch.done.wait(state) {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        if let Some(text) = state.panic.take() {
            return Err(OjasError::Backend {
                id: BackendId::Cpu,
                detail: format!("worker task panicked: {text}"),
            });
        }
        let mut out = Vec::with_capacity(count);
        for slot in state.results.iter_mut() {
            match slot.take() {
                Some(value) => out.push(value),
                None => {
                    return Err(OjasError::Backend {
                        id: BackendId::Cpu,
                        detail: "worker task finished without a result".to_string(),
                    })
                }
            }
        }
        Ok(out)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut queue = lock(&self.shared.queue);
            queue.shutdown = true;
        }
        self.shared.posted.fetch_add(1, Ordering::Release);
        self.shared.wake.notify_all();
        let me = std::thread::current().id();
        let handles = std::mem::take(&mut lock(&self.workers).handles);
        for handle in handles {
            // The last backend clone can drop inside a task on a worker; that
            // worker exits on its own once it sees `shutdown`.
            if handle.thread().id() != me {
                let _ = handle.join();
            }
        }
    }
}

fn worker(shared: Arc<Shared>) {
    loop {
        let seen = shared.posted.load(Ordering::Acquire);
        let job = {
            let queue = lock(&shared.queue);
            if queue.shutdown {
                return;
            }
            queue.batches.front().cloned()
        };
        match job {
            Some(job) => {
                while job.run_one() {}
                unpublish(&shared, Arc::as_ptr(&job) as *const ());
            }
            None => {
                let start = Instant::now();
                let mut spins = 0u32;
                while shared.posted.load(Ordering::Acquire) == seen {
                    std::hint::spin_loop();
                    spins = spins.wrapping_add(1);
                    if spins.is_multiple_of(64) && start.elapsed() >= SPIN {
                        break;
                    }
                }
                let mut queue = lock(&shared.queue);
                while queue.batches.is_empty() && !queue.shutdown {
                    queue = match shared.wake.wait(queue) {
                        Ok(guard) => guard,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                }
            }
        }
    }
}

fn spawn_error(text: &str) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail: text.to_string(),
    }
}

/// Take a finished batch off the queue. The queue's handle is dropped after
/// the lock is released, so freeing a batch whose task owned the last
/// backend clone cannot re-enter [`Pool::drop`] under the queue lock.
fn unpublish(shared: &Shared, batch: *const ()) {
    let mut removed = Vec::new();
    {
        let mut queue = lock(&shared.queue);
        let mut kept = VecDeque::with_capacity(queue.batches.len());
        for other in queue.batches.drain(..) {
            if Arc::as_ptr(&other) as *const () == batch {
                removed.push(other);
            } else {
                kept.push_back(other);
            }
        }
        queue.batches = kept;
    }
    drop(removed);
}

/// Tasks never panic while holding a pool lock (they run under
/// `catch_unwind` before the result lock is taken), so a poisoned lock still
/// guards consistent data.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Pool and arithmetic contract one op runs under.
#[derive(Clone, Copy)]
pub(crate) struct Exec<'a> {
    pub pool: &'a Arc<Pool>,
    pub numerics: Numerics,
}

impl Exec<'_> {
    /// Threads worth using for `work` units, where `min_work` units is the
    /// smallest piece worth a task. `1` keeps the op on the caller.
    pub(crate) fn split(&self, work: usize, min_work: usize) -> usize {
        let pieces = work / min_work.max(1);
        pieces.clamp(1, self.pool.threads())
    }

    /// `task(0..count)` in index order: on the pool when `work >= min_work`
    /// and there is more than one task and thread, otherwise inline.
    pub(crate) fn map<T, F>(
        &self,
        count: usize,
        work: usize,
        min_work: usize,
        task: F,
    ) -> Result<Vec<T>, OjasError>
    where
        T: Send + 'static,
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        if count > 1 && self.pool.threads() > 1 && work >= min_work {
            self.pool.run(count, task)
        } else {
            Ok((0..count).map(task).collect())
        }
    }

    /// `[0, len)` cut into contiguous chunks of at least `min_chunk` items,
    /// one task per chunk, results in chunk order. Up to two chunks per
    /// thread let fast cores take over work from slow ones.
    pub(crate) fn chunks<T, F>(
        &self,
        len: usize,
        min_chunk: usize,
        task: F,
    ) -> Result<Vec<T>, OjasError>
    where
        T: Send + 'static,
        F: Fn(Range<usize>) -> T + Send + Sync + 'static,
    {
        let pieces = (len / min_chunk.max(1)).clamp(1, self.pool.threads().saturating_mul(2));
        let parts = ranges(len, pieces);
        self.map(parts.len(), len, min_chunk, move |i| task(parts[i].clone()))
    }

    /// Rows `0..rows` of a `[rows, width]` output, computed in row chunks of
    /// about [`ROW_MIN_ELEMS`] values and concatenated in row order. The
    /// first error in row order wins.
    pub(crate) fn rows<F>(&self, rows: usize, width: usize, task: F) -> Result<Vec<f32>, OjasError>
    where
        F: Fn(Range<usize>) -> Result<Vec<f32>, OjasError> + Send + Sync + 'static,
    {
        let min_rows = (ROW_MIN_ELEMS / width.max(1)).max(1);
        let parts = self.chunks(rows, min_rows, task)?;
        if parts.len() == 1 {
            return parts.into_iter().next().unwrap_or_else(|| Ok(Vec::new()));
        }
        let mut out = Vec::with_capacity(rows.saturating_mul(width));
        for part in parts {
            out.extend_from_slice(&part?);
        }
        Ok(out)
    }
}

/// Elementwise and row-wise work below this many values per chunk is not
/// worth a hand-off to a worker.
pub(crate) const ROW_MIN_ELEMS: usize = 1 << 15;

/// `[0, len)` cut into `pieces` contiguous ranges whose sizes differ by at
/// most one. Which range a value falls in does not change any arithmetic.
pub(crate) fn ranges(len: usize, pieces: usize) -> Vec<Range<usize>> {
    let pieces = pieces.clamp(1, len.max(1));
    let base = len / pieces;
    let extra = len % pieces;
    let mut out = Vec::with_capacity(pieces);
    let mut start = 0;
    for i in 0..pieces {
        let size = base + usize::from(i < extra);
        out.push(start..start + size);
        start += size;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_in_task_order() {
        let pool = Pool::new(4).unwrap();
        let out = pool.run(100, |i| i * 3).unwrap();
        assert_eq!(out, (0..100).map(|i| i * 3).collect::<Vec<_>>());
        let one = Pool::new(1).unwrap();
        assert_eq!(one.run(5, |i| i).unwrap(), vec![0, 1, 2, 3, 4]);
        assert!(one.run(0, |i| i).unwrap().is_empty());
    }

    #[test]
    fn zero_and_oversized_thread_counts_are_refused() {
        assert!(matches!(Pool::new(0), Err(OjasError::OutOfRange { .. })));
        assert!(matches!(
            Pool::new(MAX_THREADS + 1),
            Err(OjasError::OutOfRange { .. })
        ));
    }

    #[test]
    fn a_panicking_task_is_an_error_and_the_pool_survives() {
        for threads in [1usize, 2, 7] {
            let pool = Pool::new(threads).unwrap();
            let err = pool
                .run(16, |i| {
                    if i == 5 {
                        panic!("task five failed");
                    }
                    i
                })
                .unwrap_err();
            match err {
                OjasError::Backend { id, detail } => {
                    assert_eq!(id, BackendId::Cpu);
                    assert!(detail.contains("task five failed"), "{detail}");
                }
                other => panic!("expected Backend, got {other:?}"),
            }
            // Every worker is still alive and serves the next batches.
            for _ in 0..20 {
                assert_eq!(pool.run(64, |i| i + 1).unwrap().len(), 64);
            }
            let string_payload = pool
                .run(4, |i| {
                    if i == 3 {
                        std::panic::panic_any(String::from("owned payload"));
                    }
                    i
                })
                .unwrap_err();
            assert!(string_payload.to_string().contains("owned payload"));
        }
    }

    #[test]
    fn nested_runs_on_the_same_pool_do_not_deadlock() {
        let pool = Arc::new(Pool::new(3).unwrap());
        let inner = Arc::clone(&pool);
        let out = pool
            .run(8, move |i| {
                let deeper = Arc::clone(&inner);
                inner
                    .run(8, move |j| {
                        deeper
                            .run(4, move |k| i * 100 + j * 10 + k)
                            .unwrap()
                            .iter()
                            .sum::<usize>()
                    })
                    .unwrap()
                    .iter()
                    .sum::<usize>()
            })
            .unwrap();
        let expect: Vec<usize> = (0..8)
            .map(|i| {
                (0..8)
                    .map(|j| (0..4).map(|k| i * 100 + j * 10 + k).sum::<usize>())
                    .sum()
            })
            .collect();
        assert_eq!(out, expect);
    }

    #[test]
    fn many_callers_share_one_pool() {
        let pool = Arc::new(Pool::new(4).unwrap());
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for caller in 0..32usize {
                let pool = Arc::clone(&pool);
                handles.push(
                    std::thread::Builder::new()
                        .stack_size(WORKER_STACK_BYTES)
                        .spawn_scoped(scope, move || {
                            for round in 0..50usize {
                                let out = pool.run(9, move |i| caller * 1000 + round + i).unwrap();
                                assert_eq!(out[8], caller * 1000 + round + 8);
                            }
                        })
                        .expect("worker spawn"),
                );
            }
            for handle in handles {
                handle.join().expect("worker join");
            }
        });
    }

    #[test]
    fn dropping_the_last_handle_inside_a_task_does_not_hang() {
        for _ in 0..50 {
            let pool = Arc::new(Pool::new(3).unwrap());
            let captured = Arc::clone(&pool);
            let out = pool.run(6, move |i| captured.threads() + i).unwrap();
            assert_eq!(out[0], 3);
            // A worker may still hold the batch, and with it `captured`;
            // whichever thread frees it last runs `Pool::drop`.
            drop(pool);
        }
    }

    /// Run `body` on its own thread and fail, rather than hang the test
    /// binary, if it does not finish within `secs`.
    fn within(secs: u64, what: &str, body: impl FnOnce() + Send + 'static) {
        let (done, wait) = std::sync::mpsc::channel();
        let runner = std::thread::Builder::new()
            .stack_size(WORKER_STACK_BYTES)
            .spawn(move || {
                body();
                let _ = done.send(());
            })
            .expect("watchdog spawn");
        match wait.recv_timeout(Duration::from_secs(secs)) {
            Ok(()) => runner.join().expect("stress body"),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                // The body panicked before sending; surface its panic.
                runner.join().expect("stress body");
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                panic!("{what}: did not finish in {secs}s (deadlock or lost wake-up)")
            }
        }
    }

    /// Sum over `j < 5`, `k < 3` of `salt + 100 i + 10 j + k`, computed on
    /// the pool three levels deep.
    fn nested_sum(pool: &Arc<Pool>, salt: usize) -> Vec<usize> {
        let inner = Arc::clone(pool);
        pool.run(7, move |i| {
            let deeper = Arc::clone(&inner);
            inner
                .run(5, move |j| {
                    deeper
                        .run(3, move |k| salt + i * 100 + j * 10 + k)
                        .unwrap()
                        .iter()
                        .sum::<usize>()
                })
                .unwrap()
                .iter()
                .sum::<usize>()
        })
        .unwrap()
    }

    fn nested_sum_serial(salt: usize) -> Vec<usize> {
        (0..7)
            .map(|i| {
                (0..5)
                    .map(|j| (0..3).map(|k| salt + i * 100 + j * 10 + k).sum::<usize>())
                    .sum()
            })
            .collect()
    }

    /// One stress round on a `threads`-wide pool: `callers` OS threads
    /// submit nested batches at once, some batches carry a panicking task
    /// (top level and nested), some tasks own the last handle to a
    /// short-lived pool, and the test's own handle to the shared pool is
    /// dropped while every caller is mid-flight.
    fn stress_round(threads: usize, callers: usize, iters: usize, round: usize) {
        let pool = Arc::new(Pool::new(threads).unwrap());
        let start = Arc::new(std::sync::Barrier::new(callers + 1));
        let mut handles = Vec::new();
        for caller in 0..callers {
            let pool = Arc::clone(&pool);
            let start = Arc::clone(&start);
            handles.push(
                std::thread::Builder::new()
                    .stack_size(WORKER_STACK_BYTES)
                    .spawn(move || {
                        start.wait();
                        for iter in 0..iters {
                            let salt = round * 1_000_000 + caller * 1000 + iter;
                            match (caller + iter) % 4 {
                                0 => {
                                    let text = format!("stress panic {salt}");
                                    let msg = text.clone();
                                    let err = pool
                                        .run(9, move |i| {
                                            if i == 4 {
                                                panic!("{msg}");
                                            }
                                            i
                                        })
                                        .unwrap_err();
                                    match err {
                                        OjasError::Backend { id, detail } => {
                                            assert_eq!(id, BackendId::Cpu);
                                            assert!(detail.contains(&text), "{detail}");
                                        }
                                        other => panic!("expected Backend, got {other:?}"),
                                    }
                                }
                                1 => {
                                    // A nested batch panics; only its parent task sees it.
                                    let inner = Arc::clone(&pool);
                                    let failed = pool
                                        .run(4, move |i| {
                                            inner
                                                .run(3, move |j| {
                                                    if i == 2 && j == 1 {
                                                        panic!("nested stress panic");
                                                    }
                                                    i * 10 + j
                                                })
                                                .is_err()
                                        })
                                        .unwrap();
                                    assert_eq!(failed, vec![false, false, true, false]);
                                }
                                2 => {
                                    // The last handle to this pool is freed by whichever
                                    // thread drops the batch last, possibly a worker.
                                    let short = Arc::new(Pool::new(threads).unwrap());
                                    let captured = Arc::clone(&short);
                                    let out =
                                        short.run(6, move |i| captured.threads() + i).unwrap();
                                    assert_eq!(out[5], threads + 5);
                                    drop(short);
                                }
                                _ => {}
                            }
                            assert_eq!(nested_sum(&pool, salt), nested_sum_serial(salt));
                        }
                        // The pool still serves after every panic above.
                        assert_eq!(pool.run(33, |i| i * 2).unwrap()[32], 64);
                    })
                    .expect("caller spawn"),
            );
        }
        start.wait();
        drop(pool);
        for handle in handles {
            handle.join().expect("stress caller failed");
        }
    }

    #[test]
    fn pool_survives_concurrent_nested_panicking_and_dropped_work_at_every_width() {
        within(120, "pool stress", || {
            for threads in 1..=18 {
                stress_round(threads, 6, 8, 0);
            }
        });
    }

    /// 50 rounds of the stress above at every width from 1 to 18.
    /// `cargo test -p ojas-cpu --release --lib pool_soak -- --ignored`.
    #[test]
    #[ignore = "soak: about 10 s in release; run explicitly"]
    fn pool_soak() {
        within(1800, "pool soak", || {
            for round in 1..=50 {
                for threads in 1..=18 {
                    stress_round(threads, 8, 16, round);
                }
            }
        });
    }

    #[test]
    fn ranges_cover_without_overlap() {
        for len in [0usize, 1, 5, 17, 100] {
            for pieces in [1usize, 2, 3, 7, 18, 200] {
                let parts = ranges(len, pieces);
                let mut next = 0;
                for part in &parts {
                    assert_eq!(part.start, next);
                    next = part.end;
                }
                assert_eq!(next, len);
            }
        }
    }
}
