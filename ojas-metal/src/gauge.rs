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
//!   `Session::open` until the runtime drops, so every test that can
//!   allocate on the device holds one without writing it out. Leases share.
//! - [`Measuring`]: held by every test that reads the figure. It waits for
//!   every lease to drop and admits no new one until it drops, and
//!   [`live_bytes`] takes it by reference, so a figure read cannot compile
//!   without it.
//!
//! The thread that holds `Measuring` opens its own runtimes under it, and a
//! device thread it starts ([`inherit`]) does too. A runtime opened that way
//! is still counted, so a later measurement waits for it. The pattern is
//! `heap_limit::measuring()` in ojas-device's bandwidth tests (D7/C17), but
//! those few tests can share one mutex. Here a plain mutex would run every
//! device test one at a time, and deadlock a test that opens two runtimes.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use tessl::runtime::GpuRuntime;

struct State {
    /// Runtimes alive in this process.
    leases: usize,
    /// A test is reading the figure.
    measuring: bool,
}

static GAUGE: (Mutex<State>, Condvar) = (
    Mutex::new(State {
        leases: 0,
        measuring: false,
    }),
    Condvar::new(),
);

/// A wait that outlasts every test in the suite is a deadlock: a test that
/// holds a runtime and then asks to measure, say. Fail it with the state.
const WAIT_LIMIT: Duration = Duration::from_secs(900);

thread_local! {
    /// This thread holds [`Measuring`], or was started by a thread that does.
    static UNDER_MEASURING: Cell<bool> = const { Cell::new(false) };
}

/// A panicking holder must not fail the tests after it.
fn state() -> MutexGuard<'static, State> {
    GAUGE.0.lock().unwrap_or_else(PoisonError::into_inner)
}

fn wait_until(what: &str, ready: impl Fn(&State) -> bool) -> MutexGuard<'static, State> {
    let deadline = Instant::now() + WAIT_LIMIT;
    let mut s = state();
    while !ready(&s) {
        let Some(left) = deadline.checked_duration_since(Instant::now()) else {
            let (leases, measuring) = (s.leases, s.measuring);
            drop(s);
            panic!(
                "gauge: {what} waited {WAIT_LIMIT:?} (leases {leases}, measuring {measuring}); \
                 a test that holds a runtime must not ask to measure"
            );
        };
        s = GAUGE
            .1
            .wait_timeout(s, left)
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
    s
}

/// One live runtime. Taken before the device is touched, dropped after it
/// is released.
pub(crate) struct Lease(());

pub(crate) fn lease() -> Lease {
    let under = UNDER_MEASURING.with(Cell::get);
    let mut s = if under {
        // The measuring test's own runtime: no wait, but still counted.
        state()
    } else {
        wait_until("a runtime open", |s| !s.measuring)
    };
    s.leases += 1;
    Lease(())
}

impl Drop for Lease {
    fn drop(&mut self) {
        state().leases -= 1;
        GAUGE.1.notify_all();
    }
}

/// Exclusive use of the device's figure. Not `Send`: the thread that took
/// it is the one marked as measuring.
pub(crate) struct Measuring(PhantomData<*const ()>);

pub(crate) fn measuring() -> Measuring {
    assert!(
        !UNDER_MEASURING.with(Cell::get),
        "gauge: this thread is already measuring"
    );
    let mut s = wait_until("a measurement", |s| !s.measuring && s.leases == 0);
    s.measuring = true;
    drop(s);
    UNDER_MEASURING.with(|u| u.set(true));
    Measuring(PhantomData)
}

impl Drop for Measuring {
    fn drop(&mut self) {
        UNDER_MEASURING.with(|u| u.set(false));
        state().measuring = false;
        GAUGE.1.notify_all();
    }
}

/// What this thread's measuring status is, to hand to a thread it starts.
pub(crate) struct Inherit(bool);

pub(crate) fn inherit() -> Inherit {
    Inherit(UNDER_MEASURING.with(Cell::get))
}

impl Inherit {
    /// Call first on the new thread, before it opens a runtime.
    pub(crate) fn enter(self) {
        if self.0 {
            UNDER_MEASURING.with(|u| u.set(true));
        }
    }
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
    use std::thread;

    const SETTLE: Duration = Duration::from_millis(250);

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
    fn a_measurement_waits_for_every_lease() {
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
        thread::sleep(SETTLE);
        assert!(
            !measured.load(Ordering::SeqCst),
            "a measurement began while a runtime was open"
        );
        release_tx.send(()).unwrap();
        holder.join().unwrap();
        reader.join().unwrap();
        assert!(measured.load(Ordering::SeqCst));
    }

    #[test]
    fn leases_share() {
        // Two threads each hold a lease until the other has taken one: a
        // lock that excluded leases from each other would never get there.
        let (a_tx, a_rx) = mpsc::channel::<()>();
        let (b_tx, b_rx) = mpsc::channel::<()>();
        let other = thread::spawn(move || {
            let _lease = lease();
            b_tx.send(()).unwrap();
            a_rx.recv_timeout(Duration::from_secs(60)).unwrap();
        });
        let _lease = lease();
        a_tx.send(()).unwrap();
        b_rx.recv_timeout(Duration::from_secs(60)).unwrap();
        other.join().unwrap();
    }

    #[test]
    fn a_measuring_test_opens_runtimes_under_its_own_guard_and_keeps_them_counted() {
        let held = measuring();
        // The same thread, and a device thread it starts.
        let first = lease();
        let hand_over = inherit();
        let device = thread::spawn(move || {
            hand_over.enter();
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
}
