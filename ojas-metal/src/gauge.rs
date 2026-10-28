//! The device-wide allocation figure, for the lib tests.
//!
//! tessl's `current_allocated_bytes` is `MTLDevice.currentAllocatedSize` on
//! the system default device, one object per process. A before/after
//! comparison of it is exact only while no other test in this binary holds
//! a runtime: a sibling's buffers move the figure between the two reads.
//!
//! Two guards, one lock:
//!
//! - [`Lease`]: held by every runtime a test opens, from `Worker::open` and
//!   `Session::open` until the runtime (and anything that shares it) drops,
//!   so every test that can allocate on the device holds one without writing
//!   it out. Leases share.
//! - [`Measuring`]: held by every test that reads the figure. It waits for
//!   every lease to drop and admits no new one while it is held, and
//!   [`live_bytes`] takes it by reference, so a figure read cannot compile
//!   without it. Take it before opening a runtime; a test that already holds
//!   a lease is refused at once, since it would wait on itself.
//!
//! A lease belongs to the test thread that asked for it, even when a device
//! thread takes it: [`inherit`] hands the test's identity to a thread it
//! starts. The measuring test and its device threads open runtimes under
//! their own guard. Those runtimes are counted, so a later measurement waits
//! for them.
//!
//! A measurement that is waiting does not hold new leases out. Doing so
//! bounds its wait by the longest running holder, but it would also stall
//! every test that opens a runtime inside a `watchdog(60)` or `watchdog(120)`
//! for that long, charging one test's wait to another's budget. Without it a
//! measurement waits for a moment with no runtime anywhere, which a finite
//! suite reaches at the latest when its queue empties (the lib suite takes
//! 12 to 24 s at 8 threads under load 8 to 25, measured; the two gdn tests
//! behind the 600 s watchdogs take 1 to 2 s alone), and a measurement
//! blocks leases only while it runs, which is milliseconds. The cost is that
//! the measuring tests run late, not that they can be starved.
//!
//! The pattern is `heap_limit::measuring()` in ojas-device's bandwidth tests
//! (D7/C17), but those few tests can share one mutex. Here a plain mutex
//! would run every device test one at a time, and deadlock a test that opens
//! two runtimes.

use std::cell::Cell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use tessl::runtime::GpuRuntime;

struct State {
    /// Live runtimes by the test that owns them.
    leases: HashMap<ThreadId, usize>,
    /// The test reading the figure.
    measuring: Option<ThreadId>,
    /// Tests waiting to read it.
    pending: usize,
}

impl State {
    fn held(&self, owner: ThreadId) -> usize {
        self.leases.get(&owner).copied().unwrap_or(0)
    }
}

static GAUGE: LazyLock<(Mutex<State>, Condvar)> = LazyLock::new(|| {
    (
        Mutex::new(State {
            leases: HashMap::new(),
            measuring: None,
            pending: 0,
        }),
        Condvar::new(),
    )
});

/// The longest a test may legitimately hold a runtime is its `watchdog`,
/// 600 s at most (the two gdn tests in backend.rs). A wait twice that is not
/// a measurement queued behind a slow test: it is a thread the test started
/// without [`inherit`], or a runtime kept past its test. Fail it with the
/// state.
const WAIT_LIMIT: Duration = Duration::from_secs(1200);

thread_local! {
    /// The test this thread works for, when it is not the thread itself.
    static ACTING_FOR: Cell<Option<ThreadId>> = const { Cell::new(None) };
}

fn owner() -> ThreadId {
    ACTING_FOR
        .with(Cell::get)
        .unwrap_or_else(|| thread::current().id())
}

/// A panicking holder must not fail the tests after it.
fn state() -> MutexGuard<'static, State> {
    GAUGE.0.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait_until(
    what: &str,
    queued: bool,
    ready: impl Fn(&State) -> bool,
) -> MutexGuard<'static, State> {
    let deadline = Instant::now() + WAIT_LIMIT;
    let mut g = state();
    while !ready(&g) {
        let Some(left) = deadline.checked_duration_since(Instant::now()) else {
            if queued {
                g.pending -= 1;
            }
            let leases = g.leases.values().sum::<usize>();
            let (measuring, pending) = (g.measuring.is_some(), g.pending);
            drop(g);
            GAUGE.1.notify_all();
            panic!(
                "gauge: {what} waited {WAIT_LIMIT:?} (leases {leases}, measuring {measuring}, \
                 pending {pending}); the longest legitimate holder is a 600 s watchdog, so a \
                 thread started without `gauge::inherit` or a runtime kept past its test \
                 is holding the lock"
            );
        };
        g = GAUGE
            .1
            .wait_timeout(g, left)
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
    g
}

/// One live runtime. Taken before the device is touched, dropped after it
/// is released.
pub(crate) struct Lease(ThreadId);

pub(crate) fn lease() -> Lease {
    let me = owner();
    // The measuring test's own runtime goes through; any other waits only
    // while a measurement runs.
    let mut g = wait_until("a runtime open", false, |s| {
        s.measuring.is_none() || s.measuring == Some(me)
    });
    *g.leases.entry(me).or_insert(0) += 1;
    Lease(me)
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut g = state();
        if let Some(n) = g.leases.get_mut(&self.0) {
            *n -= 1;
            if *n == 0 {
                g.leases.remove(&self.0);
            }
        }
        drop(g);
        GAUGE.1.notify_all();
    }
}

/// Exclusive use of the device's figure. Not `Send`: the thread that took
/// it is the one that works for it.
pub(crate) struct Measuring(PhantomData<*const ()>);

/// Waits until no runtime is open anywhere. A runtime stays open until its
/// device thread has exited, which is a moment after its `MetalBackend` drops
/// (`spawn_with` detaches it and the thread ends when its channel closes), so
/// a test that opens and drops a backend and only then asks to measure can be
/// refused here, at random. Measure first; open runtimes after.
pub(crate) fn measuring() -> Measuring {
    let me = owner();
    {
        let mut g = state();
        assert!(
            g.measuring != Some(me),
            "gauge: this test is already measuring"
        );
        assert!(
            g.held(me) == 0,
            "gauge: this test holds a runtime, so a measurement would wait on it; \
             measure before opening one"
        );
        g.pending += 1;
    }
    let mut g = wait_until("a measurement", true, |s| {
        s.measuring.is_none() && s.leases.is_empty()
    });
    g.pending -= 1;
    g.measuring = Some(me);
    Measuring(PhantomData)
}

impl Drop for Measuring {
    fn drop(&mut self) {
        state().measuring = None;
        GAUGE.1.notify_all();
    }
}

/// The test a thread works for, to hand to a thread it starts.
pub(crate) struct Inherit(ThreadId);

pub(crate) fn inherit() -> Inherit {
    Inherit(owner())
}

impl Inherit {
    /// Call first on the new thread, before it opens a runtime.
    pub(crate) fn enter(self) {
        ACTING_FOR.with(|a| a.set(Some(self.0)));
    }
}

/// Runtimes this test holds, for the tests of who holds a lease.
pub(crate) fn held_here() -> usize {
    state().held(owner())
}

/// The device-wide figure, read under [`Measuring`].
pub(crate) fn live_bytes(rt: &GpuRuntime, _measuring: &Measuring) -> u64 {
    rt.current_allocated_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;

    const SETTLE: Duration = Duration::from_millis(250);

    /// Until some measurement is waiting.
    fn queued() {
        let deadline = Instant::now() + Duration::from_secs(60);
        while state().pending == 0 {
            assert!(Instant::now() < deadline, "no measurement queued");
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn a_measurement_holds_every_lease_out_until_it_drops() {
        let held = measuring();
        let opened = Arc::new(AtomicBool::new(false));
        let sibling = {
            let opened = Arc::clone(&opened);
            thread::spawn(move || {
                let _lease = lease();
                opened.store(true, Ordering::SeqCst);
            })
        };
        thread::sleep(SETTLE);
        assert!(
            !opened.load(Ordering::SeqCst),
            "a sibling opened a runtime inside the measurement"
        );
        drop(held);
        sibling.join().unwrap();
        assert!(opened.load(Ordering::SeqCst));
    }

    #[test]
    fn a_measurement_waits_for_every_lease_without_holding_new_ones_out() {
        let (taken_tx, taken_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let holder = thread::spawn(move || {
            let _lease = lease();
            taken_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        taken_rx.recv().unwrap();
        let measured = Arc::new(AtomicBool::new(false));
        let reader = {
            let measured = Arc::clone(&measured);
            thread::spawn(move || {
                let _held = measuring();
                measured.store(true, Ordering::SeqCst);
            })
        };
        queued();
        // A runtime opened while the measurement waits is not held out: its
        // wait would be charged to whatever watchdog its test runs under.
        let (opened_tx, opened_rx) = mpsc::channel::<()>();
        let fresh = thread::spawn(move || {
            let _lease = lease();
            opened_tx.send(()).unwrap();
        });
        let opened = opened_rx.recv_timeout(Duration::from_secs(60));
        let began = measured.load(Ordering::SeqCst);
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        reader.join().unwrap();
        fresh.join().unwrap();
        opened.expect("a runtime waited on a measurement that was only queued");
        assert!(!began, "a measurement began while a runtime was open");
        assert!(measured.load(Ordering::SeqCst));
    }

    #[test]
    fn leases_share() {
        // Two threads each hold a lease until the other has taken one: a
        // lock that excluded leases from each other would never get there.
        // Taken before the thread starts: once this thread holds a lease no
        // measurement can begin, so a real one cannot hold `other` out.
        let _lease = lease();
        let (a_tx, a_rx) = mpsc::channel::<()>();
        let (b_tx, b_rx) = mpsc::channel::<()>();
        let other = thread::spawn(move || {
            let _lease = lease();
            b_tx.send(()).unwrap();
            a_rx.recv_timeout(Duration::from_secs(60)).unwrap();
        });
        a_tx.send(()).unwrap();
        b_rx.recv_timeout(Duration::from_secs(60)).unwrap();
        other.join().unwrap();
    }

    #[test]
    fn a_measuring_test_opens_runtimes_under_its_own_guard_and_keeps_them_counted() {
        let held = measuring();
        // The same thread, and a device thread it starts.
        let first = lease();
        let acting = inherit();
        let device = thread::spawn(move || {
            acting.enter();
            lease()
        });
        let second = device.join().unwrap();
        drop(held);
        // Both runtimes outlive the measurement, so the next one waits.
        let (started_tx, started_rx) = mpsc::channel();
        let next = thread::spawn(move || {
            started_tx.send(()).unwrap();
            drop(measuring());
        });
        started_rx.recv().unwrap();
        thread::sleep(SETTLE);
        assert!(
            !next.is_finished(),
            "a measurement began over a live runtime"
        );
        drop(first);
        drop(second);
        next.join().unwrap();
    }

    #[test]
    fn a_second_device_thread_of_a_lease_holder_opens_while_a_measurement_waits() {
        // A backend's lease is taken on its device thread, for its test. The
        // test's second backend must open while a measurement waits on its
        // first.
        let acting = inherit();
        let first = thread::spawn(move || {
            acting.enter();
            lease()
        })
        .join()
        .unwrap();
        let (queued_tx, queued_rx) = mpsc::channel::<()>();
        let reader = thread::spawn(move || {
            queued_tx.send(()).unwrap();
            drop(measuring());
        });
        queued_rx.recv().unwrap();
        queued();
        let acting = inherit();
        let (opened_tx, opened_rx) = mpsc::channel::<()>();
        let device = thread::spawn(move || {
            acting.enter();
            let _second = lease();
            opened_tx.send(()).unwrap();
        });
        let opened = opened_rx.recv_timeout(Duration::from_secs(60));
        drop(first);
        reader.join().unwrap();
        device.join().unwrap();
        opened.expect("a second runtime waited on a measurement that waits on its test");
    }

    #[test]
    fn a_lease_taken_by_a_device_thread_is_released_for_its_test() {
        let acting = inherit();
        let taken = thread::spawn(move || {
            acting.enter();
            lease()
        })
        .join()
        .unwrap();
        assert_eq!(state().held(owner()), 1);
        drop(taken);
        assert_eq!(state().held(owner()), 0);
    }
}
