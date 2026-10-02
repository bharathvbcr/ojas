//! Red-team gate for linear budget accounting, cancellation and concurrent
//! callers on one shared `CpuBackend`.
//!
//! Budget boundaries are measured, not assumed: a filler reservation leaves
//! exactly `c` bytes of room on a live backend, and a binary search finds the
//! smallest `c` the op accepts. That peak must sit between the bytes of the
//! outputs (which every implementation must charge) and the peak the current
//! code charges (`inputs + outputs + gemm scratch`), so a change that shrinks
//! the peak passes and one that grows it fails. `c - 1` must refuse with
//! `CapacityExceeded` and leave the budget exactly where it was.
//!
//! Expensive variants are `#[ignore]`d; run them with
//! `cargo test -p ojas-cpu --release --test redteam_linear_budget -- --ignored`.

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use common::SplitMix64;
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const HUGE: u64 = 1 << 40;

fn inputs_budget() -> Budget {
    Budget::new(u64::MAX)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &inputs_budget()).unwrap()
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn cpu(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(HUGE), threads)
        .unwrap()
        .with_numerics(numerics)
}

/// Run on a watchdog thread so a deadlock fails the test instead of hanging CI.
fn within<T: Send + 'static>(secs: u64, body: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, wait) = mpsc::channel();
    let runner = std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            let out = body();
            let _ = done.send(());
            out
        })
        .expect("watchdog spawn");
    match wait.recv_timeout(Duration::from_secs(secs)) {
        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => runner.join().expect("test body"),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("did not finish in {secs}s (deadlock or lost wake-up)")
        }
    }
}

// ---------------------------------------------------------------- current peak formula
//
// A replica of what `ojas-cpu` charges today (validate.rs `f32_inputs`,
// linalg.rs `linear_forward`/`linear_backward`, gemm.rs `scratch`/`plan`).
// Used only as an upper bound.

const MR: usize = 6;
const NR: usize = 16;
const TASK_MACS: usize = 1 << 20;

fn round_up(value: usize, unit: usize) -> usize {
    value.div_ceil(unit).max(1) * unit
}

fn plan_count(threads: usize, fast: bool, m: usize, n: usize, k: usize) -> usize {
    let (mc, nc) = if fast { (144, 512) } else { (72, 256) };
    let macs = m.saturating_mul(n).saturating_mul(k);
    let target = (macs / TASK_MACS).min(threads.saturating_mul(2));
    if threads <= 1 || target < 2 {
        return 1;
    }
    let (mut tm, mut tn) = (mc.min(round_up(m, MR)), nc.min(round_up(n, NR)));
    let count = |tm: usize, tn: usize| m.div_ceil(tm) * n.div_ceil(tn);
    while count(tm, tn) < target {
        if tm >= tn && tm > MR {
            tm = round_up(tm / 2, MR);
        } else if tn > NR {
            tn = round_up(tn / 2, NR);
        } else if tm > MR {
            tm = round_up(tm / 2, MR);
        } else {
            break;
        }
    }
    count(tm, tn).max(1)
}

/// gemm.rs `scratch(m, k, n)` in floats.
fn scratch(threads: usize, fast: bool, m: usize, k: usize, n: usize) -> usize {
    let a = m.div_ceil(MR) * MR * k;
    let b = n.div_ceil(NR) * NR * k;
    let tiles = if plan_count(threads, fast, m, n, k) > 1 { m * n } else { 0 };
    a + b + tiles
}

fn whole_call(fast: bool, m: usize, k: usize, n: usize) -> bool {
    fast && m * n * k >= 2 * TASK_MACS
}

/// Peak bytes charged today, and bytes of the outputs.
fn forward_bounds(threads: usize, fast: bool, rows: usize, kin: usize, nout: usize) -> (u64, u64) {
    let inputs = rows * kin + nout * kin;
    let y = rows * nout;
    let peak = inputs + y + scratch(threads, fast, rows, kin, nout);
    (4 * peak as u64, 4 * y as u64)
}

fn backward_bounds(threads: usize, fast: bool, rows: usize, kin: usize, nout: usize) -> (u64, u64) {
    let inputs = rows * kin + nout * kin + rows * nout;
    let (gx, gw) = (rows * kin, nout * kin);
    let single = |m, k, n| plan_count(threads, fast, m, n, k) == 1;
    let pair = threads > 1
        && single(rows, kin, nout)
        && single(nout, kin, rows)
        && !whole_call(fast, rows, nout, kin)
        && rows * nout * kin >= TASK_MACS / 2;
    let (sx, sw) = (
        scratch(threads, fast, rows, nout, kin),
        scratch(threads, fast, nout, rows, kin),
    );
    let work = if pair { sx + sw } else { sx.max(sw) };
    (4 * (inputs + gx + gw + work) as u64, 4 * (gx + gw) as u64)
}

// ---------------------------------------------------------------- budget boundary

type Op<'a> = &'a dyn Fn(&CpuBackend) -> Result<Vec<Tensor>, OjasError>;

/// Run `op` with exactly `room` bytes free on `be`'s budget. Returns the
/// outputs' bits, or the error; asserts the budget is balanced either way.
fn with_room(be: &CpuBackend, room: u64, op: Op<'_>, what: &str) -> Result<Vec<Vec<u32>>, OjasError> {
    let budget = be.budget();
    assert_eq!(budget.live_bytes().unwrap(), 0, "{what}: budget not idle");
    let filler = budget.try_reserve(HUGE - room).unwrap();
    let held = HUGE - room;
    let out = op(be);
    let result = match out {
        Ok(tensors) => {
            let out_bytes: u64 = tensors.iter().map(|t| 4 * t.num_elements().unwrap() as u64).sum();
            assert_eq!(
                budget.live_bytes().unwrap(),
                held + out_bytes,
                "{what}: room {room}: success must hold exactly the outputs"
            );
            let b = tensors.iter().map(|t| bits(&vals(t))).collect();
            drop(tensors);
            Ok(b)
        }
        Err(err) => Err(err),
    };
    assert_eq!(budget.live_bytes().unwrap(), held, "{what}: room {room}: budget not released");
    drop(filler);
    assert_eq!(budget.live_bytes().unwrap(), 0);
    result
}

/// Smallest room the op accepts; asserts the boundary is sharp and bounded.
fn boundary(be: &CpuBackend, op: Op<'_>, upper: u64, lower: u64, what: &str) -> u64 {
    let full = with_room(be, upper, op, what)
        .unwrap_or_else(|err| panic!("{what}: refused at the current peak {upper}: {err:?}"));
    let (mut lo, mut hi) = (0u64, upper); // lo refuses (or is 0), hi accepts
    if with_room(be, 0, op, what).is_ok() {
        panic!("{what}: accepted with 0 bytes of room");
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match with_room(be, mid, op, what) {
            Ok(_) => hi = mid,
            Err(OjasError::CapacityExceeded { .. }) => lo = mid,
            Err(err) => panic!("{what}: room {mid}: expected CapacityExceeded, got {err:?}"),
        }
    }
    let peak = hi;
    match with_room(be, peak - 1, op, what) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        Err(err) => panic!("{what}: peak - 1 = {} expected CapacityExceeded, got {err:?}", peak - 1),
        Ok(_) => panic!("{what}: peak - 1 = {} expected CapacityExceeded, got Ok", peak - 1),
    }
    for room in [peak, peak + 1, peak + 4, peak + 4096] {
        let got = with_room(be, room, op, what)
            .unwrap_or_else(|err| panic!("{what}: refused at {room} >= peak {peak}: {err:?}"));
        assert_eq!(got, full, "{what}: bits depend on the budget");
    }
    assert!(peak <= upper, "{what}: peak {peak} above the current formula {upper}");
    assert!(peak >= lower, "{what}: peak {peak} below the output bytes {lower}");
    println!("{what}: measured peak {peak} bytes (current formula {upper}, outputs {lower})");
    peak
}

fn budget_sweep(threads_list: &[usize], shapes: &[(usize, usize, usize)]) {
    let mut rng = SplitMix64(0xb0d9e7);
    for &(rows, kin, nout) in shapes {
        let x = t(&rng.vec(rows * kin, 1.0), &[rows, kin]);
        let w = t(&rng.vec(nout * kin, 1.0), &[nout, kin]);
        let g = t(&rng.vec(rows * nout, 1.0), &[rows, nout]);
        for &threads in threads_list {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let fast = numerics == Numerics::Fast;
                let be = cpu(threads, numerics);
                let what = format!("{rows}x{kin}x{nout} threads {threads} {numerics:?}");
                let fwd = |be: &CpuBackend| be.linear_forward(&x, &w).map(|y| vec![y]);
                let (upper, lower) = forward_bounds(threads, fast, rows, kin, nout);
                boundary(&be, &fwd, upper, lower, &format!("forward {what}"));
                let bwd = |be: &CpuBackend| be.linear_backward(&x, &w, &g).map(|(a, b)| vec![a, b]);
                let (upper, lower) = backward_bounds(threads, fast, rows, kin, nout);
                boundary(&be, &bwd, upper, lower, &format!("backward {what}"));
            }
        }
    }
}

/// Tiny, single-tile, pair-path (64x96x128 backward with threads > 1),
/// multi-tile, rows=1, and Fast whole-call (128^3) shapes.
#[test]
fn budget_boundary_is_sharp_balanced_and_not_above_the_current_peak() {
    within(300, || {
        budget_sweep(
            &[1, 2, 7, 18],
            &[(5, 7, 3), (64, 96, 128), (73, 257, 257), (1, 768, 768), (128, 128, 128)],
        );
    });
}

#[test]
#[ignore = "slow: cargo test -p ojas-cpu --release --test redteam_linear_budget -- --ignored"]
fn budget_boundary_every_thread_count_and_large_shapes() {
    within(1200, || {
        budget_sweep(
            &[1, 2, 3, 5, 6, 7, 16, 18],
            &[(5, 7, 3), (64, 96, 128), (73, 257, 257), (1, 768, 768), (128, 128, 128), (512, 768, 768)],
        );
    });
}

// ---------------------------------------------------------------- cancellation

fn cancelled() -> OjasError {
    OjasError::Unsupported {
        op: "redteam-cancel",
        detail: "cancelled by test hook".to_string(),
    }
}

/// Installs a hook that passes `allow` calls and then fails every call;
/// returns the call counter.
fn arm(be: &CpuBackend, allow: usize) -> Arc<AtomicUsize> {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&calls);
    be.set_cancel(move || {
        if seen.fetch_add(1, Ordering::SeqCst) >= allow {
            Err(cancelled())
        } else {
            Ok(())
        }
    });
    calls
}

fn assert_cancelled<T>(r: Result<T, OjasError>, what: &str) {
    match r {
        Err(OjasError::Unsupported {
            op: "redteam-cancel",
            ..
        }) => {}
        Err(err) => panic!("{what}: expected the cancel error, got {err:?}"),
        Ok(_) => panic!("{what}: expected the cancel error, got Ok (the op ran to completion)"),
    }
}

/// Exact single-tile (1 thread), Exact multi-tile, the backward pair path
/// and Fast below the whole-call cutoff all return the hook's error when it
/// fails part-way, leave the budget at 0, and compute the same bits once the
/// hook is reset.
#[test]
fn cancel_mid_gemm_returns_the_hook_error_and_leaves_the_backend_usable() {
    within(300, || {
        let mut rng = SplitMix64(0xca9ce1);
        let cases = [
            (1usize, Numerics::Exact, (512usize, 768usize, 768usize)),
            (7, Numerics::Exact, (512, 768, 768)),
            (18, Numerics::Exact, (257, 257, 257)),
            (7, Numerics::Exact, (64, 96, 128)),
            (2, Numerics::Exact, (64, 96, 128)),
            (7, Numerics::Fast, (127, 128, 128)),
            (7, Numerics::Fast, (64, 96, 128)),
        ];
        for (threads, numerics, (rows, kin, nout)) in cases {
            let be = cpu(threads, numerics);
            let x = t(&rng.vec(rows * kin, 1.0), &[rows, kin]);
            let w = t(&rng.vec(nout * kin, 1.0), &[nout, kin]);
            let g = t(&rng.vec(rows * nout, 1.0), &[rows, nout]);
            let want_y = bits(&vals(&be.linear_forward(&x, &w).unwrap()));
            let (gx, gw) = be.linear_backward(&x, &w, &g).unwrap();
            let want_b = (bits(&vals(&gx)), bits(&vals(&gw)));
            drop((gx, gw));
            // At least 8 row panels of MC=72 (or many tiles): the hook must be
            // consulted more than 3 times, so a cancel part-way is certain.
            let many_panels = rows >= 257;
            for allow in [0usize, 1, 3] {
                let what = format!("{rows}x{kin}x{nout} threads {threads} {numerics:?} allow {allow}");
                let calls = arm(&be, allow);
                let r = be.linear_forward(&x, &w);
                if many_panels {
                    assert!(calls.load(Ordering::SeqCst) > allow, "forward {what}: hook never fired mid-op");
                }
                // Only shapes with more than `allow` panels can be cancelled.
                if calls.load(Ordering::SeqCst) > allow {
                    assert_cancelled(r, &format!("forward {what}"));
                } else {
                    assert_eq!(bits(&vals(&r.unwrap())), want_y, "forward {what} not cancelled");
                }
                assert_eq!(be.budget().live_bytes().unwrap(), 0, "forward {what}");
                let calls = arm(&be, allow);
                let r = be.linear_backward(&x, &w, &g);
                if many_panels {
                    assert!(calls.load(Ordering::SeqCst) > allow, "backward {what}: hook never fired mid-op");
                }
                if calls.load(Ordering::SeqCst) > allow {
                    assert_cancelled(r, &format!("backward {what}"));
                } else {
                    let (gx, gw) = r.unwrap();
                    assert_eq!((bits(&vals(&gx)), bits(&vals(&gw))), want_b, "{what}");
                }
                assert_eq!(be.budget().live_bytes().unwrap(), 0, "backward {what}");
                be.set_cancel(|| Ok(()));
                assert_eq!(bits(&vals(&be.linear_forward(&x, &w).unwrap())), want_y, "{what} after");
                let (gx, gw) = be.linear_backward(&x, &w, &g).unwrap();
                assert_eq!((bits(&vals(&gx)), bits(&vals(&gw))), want_b, "{what} after");
            }
            // allow = 0 must always cancel: every packed GEMM consults the hook.
            let calls = arm(&be, 0);
            assert_cancelled(be.linear_forward(&x, &w), "allow 0 forward");
            assert!(calls.load(Ordering::SeqCst) >= 1);
            be.set_cancel(|| Ok(()));
        }
    });
}

/// A Fast GEMM at or above 2^21 multiply-adds on macOS is one Accelerate
/// call (gemm.rs `accelerate`). It cannot be interrupted, so the hook is
/// consulted once before it; an always-failing hook stops the op there.
#[test]
fn fast_whole_call_honors_an_always_failing_cancel_hook() {
    let be = cpu(7, Numerics::Fast);
    let mut rng = SplitMix64(0xacce);
    let x = t(&rng.vec(128 * 128, 1.0), &[128, 128]);
    let w = t(&rng.vec(128 * 128, 1.0), &[128, 128]);
    let calls = arm(&be, 0);
    let r = be.linear_forward(&x, &w);
    assert_cancelled(r, "fast 128^3 forward");
    assert!(calls.load(Ordering::SeqCst) >= 1);
}

// ---------------------------------------------------------------- concurrency

struct Job {
    x: Tensor,
    w: Tensor,
    g: Tensor,
    numerics: Numerics,
    want: (Vec<u32>, Vec<u32>, Vec<u32>),
}

fn jobs(exact: &CpuBackend, fast: &CpuBackend) -> Vec<Job> {
    let mut rng = SplitMix64(0xc0c0);
    // Fast shapes stay below 2^21 multiply-adds, where the bits do not depend
    // on the thread count or on other callers.
    let shapes = [
        (Numerics::Exact, vec![5usize, 3], 7usize),
        (Numerics::Exact, vec![64, 96], 128),
        (Numerics::Exact, vec![73, 257], 257),
        (Numerics::Exact, vec![2, 37, 45], 29),
        (Numerics::Exact, vec![1, 768], 768),
        (Numerics::Fast, vec![127, 128], 128),
        (Numerics::Fast, vec![64, 96], 128),
        (Numerics::Fast, vec![3, 7, 11], 13),
    ];
    let serial_exact = cpu(1, Numerics::Exact);
    let serial_fast = cpu(1, Numerics::Fast);
    shapes
        .into_iter()
        .map(|(numerics, x_shape, nout)| {
            let kin = *x_shape.last().unwrap();
            let rows: usize = x_shape[..x_shape.len() - 1].iter().product();
            let x = t(&rng.vec(rows * kin, 1.0), &x_shape);
            let w = t(&rng.vec(nout * kin, 1.0), &[nout, kin]);
            let mut y_shape = x_shape[..x_shape.len() - 1].to_vec();
            y_shape.push(nout);
            let g = t(&rng.vec(rows * nout, 1.0), &y_shape);
            // Serial one-thread results: the bits every concurrent call must give.
            let serial = if numerics == Numerics::Exact { &serial_exact } else { &serial_fast };
            let y = bits(&vals(&serial.linear_forward(&x, &w).unwrap()));
            let (gx, gw) = serial.linear_backward(&x, &w, &g).unwrap();
            let want = (y, bits(&vals(&gx)), bits(&vals(&gw)));
            // And the shared backend agrees when uncontended.
            let shared = if numerics == Numerics::Exact { exact } else { fast };
            assert_eq!(bits(&vals(&shared.linear_forward(&x, &w).unwrap())), want.0);
            Job { x, w, g, numerics, want }
        })
        .collect()
}

fn storm(threads: usize, callers: usize, iters: usize) {
    let exact = cpu(threads, Numerics::Exact);
    let fast = exact.clone().with_numerics(Numerics::Fast);
    let jobs = Arc::new(jobs(&exact, &fast));
    let start = Arc::new(std::sync::Barrier::new(callers));
    let handles: Vec<_> = (0..callers)
        .map(|caller| {
            let (exact, fast, jobs, start) = (exact.clone(), fast.clone(), Arc::clone(&jobs), Arc::clone(&start));
            std::thread::spawn(move || {
                let mut rng = SplitMix64(0x5707 + caller as u64);
                start.wait();
                for i in 0..iters {
                    let job = &jobs[rng.below(jobs.len())];
                    let be = if job.numerics == Numerics::Exact { &exact } else { &fast };
                    let what = format!("caller {caller} iter {i} threads {threads} {:?} {:?}", job.numerics, job.x.shape());
                    if rng.below(2) == 0 {
                        let y = be.linear_forward(&job.x, &job.w).unwrap();
                        assert_eq!(bits(&vals(&y)), job.want.0, "{what} forward");
                    } else {
                        let (gx, gw) = be.linear_backward(&job.x, &job.w, &job.g).unwrap();
                        assert_eq!(bits(&vals(&gx)), job.want.1, "{what} grad_x");
                        assert_eq!(bits(&vals(&gw)), job.want.2, "{what} grad_w");
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("caller panicked");
    }
    assert_eq!(exact.budget().live_bytes().unwrap(), 0);
}

/// Eight callers share one pooled backend (and a Fast clone of it that shares
/// the pool), issuing mixed shapes, directions and numerics; every result
/// equals the serial one-thread bits, nothing deadlocks.
#[test]
fn concurrent_mixed_linear_calls_match_serial_bits() {
    within(240, || {
        for threads in [2usize, 7, 18] {
            storm(threads, 8, 40);
        }
    });
}

#[test]
#[ignore = "soak: cargo test -p ojas-cpu --release --test redteam_linear_budget -- --ignored"]
fn concurrent_mixed_linear_calls_soak() {
    within(1800, || {
        for threads in [1usize, 2, 3, 5, 6, 7, 16, 18] {
            storm(threads, 16, 200);
        }
    });
}

/// Callers race for a shared budget that fits about two ops at once. Every
/// call either succeeds with the serial bits or is `CapacityExceeded`; the
/// budget ends at exactly 0.
#[test]
fn concurrent_calls_under_a_tight_shared_budget_stay_balanced() {
    within(240, || {
        let (rows, kin, nout) = (73usize, 257usize, 257usize);
        let threads = 7;
        let (peak, _) = backward_bounds(threads, false, rows, kin, nout);
        let be = CpuBackend::with_threads(Budget::new(peak * 5 / 2), threads).unwrap();
        let mut rng = SplitMix64(0x7197);
        let x = t(&rng.vec(rows * kin, 1.0), &[rows, kin]);
        let w = t(&rng.vec(nout * kin, 1.0), &[nout, kin]);
        let g = t(&rng.vec(rows * nout, 1.0), &[rows, nout]);
        let want_y = bits(&vals(&be.linear_forward(&x, &w).unwrap()));
        let (gx, gw) = be.linear_backward(&x, &w, &g).unwrap();
        let want_b = (bits(&vals(&gx)), bits(&vals(&gw)));
        drop((gx, gw));
        let shared = Arc::new((x, w, g, want_y, want_b));
        let refused = Arc::new(AtomicUsize::new(0));
        let ok = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(std::sync::Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|caller| {
                let (be, shared, refused, ok, start) = (
                    be.clone(),
                    Arc::clone(&shared),
                    Arc::clone(&refused),
                    Arc::clone(&ok),
                    Arc::clone(&start),
                );
                std::thread::spawn(move || {
                    let (x, w, g, want_y, want_b) = &*shared;
                    start.wait();
                    for i in 0..60 {
                        let r = if (caller + i) % 2 == 0 {
                            be.linear_forward(x, w).map(|y| assert_eq!(&bits(&vals(&y)), want_y))
                        } else {
                            be.linear_backward(x, w, g).map(|(gx, gw)| {
                                assert_eq!(&(bits(&vals(&gx)), bits(&vals(&gw))), want_b)
                            })
                        };
                        match r {
                            Ok(()) => ok.fetch_add(1, Ordering::Relaxed),
                            Err(OjasError::CapacityExceeded { .. }) => refused.fetch_add(1, Ordering::Relaxed),
                            Err(err) => panic!("caller {caller} iter {i}: {err:?}"),
                        };
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("caller panicked");
        }
        assert_eq!(be.budget().live_bytes().unwrap(), 0, "budget leaked under contention");
        assert!(ok.load(Ordering::Relaxed) > 0, "no call fit");
        println!(
            "tight shared budget: {} ok, {} refused",
            ok.load(Ordering::Relaxed),
            refused.load(Ordering::Relaxed)
        );
    });
}
