//! `MetalBackend::with_runtime`: a provider closure runs on the backend's
//! device thread with that thread's one tessl runtime (queue, buffer pool,
//! pipeline cache, residency set), the same runtime for every call and every
//! clone, and a different one for a different backend. The closure's error
//! comes back as its own; a panic poisons the backend like any device-thread
//! panic; the backend serves its ordinary ops before and after.
#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::{mpsc, Arc};
use std::thread::{self, ThreadId};

use common::*;
use ojas_core::{BackendCore, BackendFloat, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

/// The address of the runtime the closure was handed, and the thread it ran
/// on. The runtime itself is `!Send`, so only these two leave the closure.
fn seen_by_closure(m: &MetalBackend) -> (usize, ThreadId) {
    let (tx, rx) = mpsc::channel();
    ok(
        "with_runtime",
        m.with_runtime(move |rt| {
            let _ = tx.send((Arc::as_ptr(rt) as usize, thread::current().id()));
            Ok(())
        }),
    );
    rx.recv().expect("the closure ran and reported")
}

fn probe(m: &MetalBackend, values: &[f32]) -> Tensor {
    let host = ok(
        "from_f32",
        Tensor::from_f32(values, &[values.len()], m.budget()),
    );
    up(m, &host)
}

#[test]
fn every_call_and_clone_sees_one_runtime_on_the_device_thread() {
    let m = metal();
    let before = probe(&m, &[1.0, 2.0, 3.0]);

    let (first, thread_first) = seen_by_closure(&m);
    let (second, thread_second) = seen_by_closure(&m.clone());
    assert_eq!(first, second, "a second call got a second runtime");
    assert_eq!(thread_first, thread_second);
    assert_ne!(
        thread_first,
        thread::current().id(),
        "the closure ran on the caller's thread"
    );

    // The worker serves its own ops between provider runs, and the buffer
    // uploaded before the first run is intact.
    ok("sync", m.sync());
    assert_eq!(down(&before), vec![1.0, 2.0, 3.0]);
    let after = probe(&m, &[4.0, 5.0]);
    assert_eq!(down(&after), vec![4.0, 5.0]);
    assert_eq!(seen_by_closure(&m).0, first);

    // Another backend has its own device thread and so its own runtime.
    let other = metal();
    assert_ne!(seen_by_closure(&other).0, first);
}

#[test]
fn the_closures_error_is_returned_and_the_backend_stays_healthy() {
    let m = metal();
    let kept = probe(&m, &[7.0, 8.0, 9.0]);

    let err = m
        .with_runtime(|_| {
            Err(OjasError::Unsupported {
                op: "probe",
                detail: "the closure refused".into(),
            })
        })
        .unwrap_err();
    assert!(
        matches!(err, OjasError::Unsupported { op: "probe", .. }),
        "got {err:?}"
    );

    ok("sync", m.sync());
    assert_eq!(down(&kept), vec![7.0, 8.0, 9.0]);
    // The runtime's flags were restored: the next provider run still works.
    ok("with_runtime", m.with_runtime(|_| Ok(())));
}

#[test]
fn a_panicking_closure_poisons_the_backend() {
    let m = metal();
    let kept = probe(&m, &[1.0]);

    let err = m
        .with_runtime(|_| panic!("provider panic, injected by the test"))
        .unwrap_err();
    assert!(matches!(err, OjasError::Backend { .. }), "got {err:?}");

    assert!(matches!(m.sync(), Err(OjasError::Poisoned)));
    assert!(matches!(m.download(&kept), Err(OjasError::Poisoned)));
    assert!(matches!(
        m.with_runtime(|_| Ok(())),
        Err(OjasError::Poisoned)
    ));
}

/// A closure that calls back into its own backend (`sync`, an op, `trim_pool`)
/// gets an error, not a wait for a reply only its own thread could send
/// (ft-44871303).
#[test]
fn a_closure_that_calls_back_into_its_backend_is_refused_not_hung() {
    let m = metal();
    let t = probe(&m, &[1.0, 2.0]);
    let inner = m.clone();
    let seen = bounded("with_runtime re-entry", move || {
        let (tx, rx) = mpsc::channel();
        let ran = m.with_runtime(move |_| {
            let _ = tx.send((
                inner.sync(),
                inner.silu_forward(&t).map(|_| ()),
                inner.trim_pool(),
            ));
            Ok(())
        });
        (ran, rx.recv().expect("the closure ran"))
    });
    let (ran, (sync, op, trim)) = seen;
    ok("with_runtime", ran);
    for (name, r) in [("sync", sync), ("silu_forward", op), ("trim_pool", trim)] {
        match r {
            Err(OjasError::Backend { detail, .. }) => {
                assert!(detail.contains("re-entry"), "{name}: {detail}")
            }
            other => panic!("{name} from the device thread: {other:?}"),
        }
    }
}

/// The refusal leaves the backend serving: the next op from a host thread works.
#[test]
fn the_backend_serves_after_a_refused_re_entry() {
    let m = metal();
    let inner = m.clone();
    bounded("re-entry", move || {
        ok(
            "with_runtime",
            m.with_runtime(move |_| {
                assert!(inner.sync().is_err());
                Ok(())
            }),
        );
        ok("sync", m.sync());
        let t = probe(&m, &[3.0]);
        assert_eq!(down(&t), vec![3.0]);
    });
}

/// A reservation that does not fit, made inside the closure on a budget that
/// holds this backend's trim handler, and on a child that inherits it, is
/// `CapacityExceeded`: the handler's trim would call the link from the device
/// thread.
#[test]
fn a_reservation_inside_the_closure_that_does_not_fit_is_capacity_exceeded() {
    let budget = Budget::new(CAP);
    let m = ok("new", MetalBackend::new(budget.clone()));
    let _held = ok("reserve", budget.try_reserve(CAP - 64));
    let child = budget.child(CAP);
    let (b, c) = (budget.clone(), child.clone());
    let (own, inherited) = bounded("reserve in with_runtime", move || {
        let (tx, rx) = mpsc::channel();
        ok(
            "with_runtime",
            m.with_runtime(move |_| {
                let _ = tx.send((
                    b.try_reserve(4096).map(|_| ()),
                    c.try_reserve(4096).map(|_| ()),
                ));
                Ok(())
            }),
        );
        rx.recv().expect("the closure ran")
    });
    assert!(
        matches!(own, Err(OjasError::CapacityExceeded { .. })),
        "{own:?}"
    );
    assert!(
        matches!(inherited, Err(OjasError::CapacityExceeded { .. })),
        "{inherited:?}"
    );
}

/// Pressure inside the closure on the budget `with_budget` was given is
/// `CapacityExceeded` without blocking.
#[test]
fn pressure_on_a_with_budget_budget_inside_the_closure_does_not_block() {
    let m = metal();
    let other = Budget::new(CAP);
    let shared = m.with_budget(other.clone());

    let _held = ok("reserve", other.try_reserve(CAP - 64));
    let o = other.clone();
    let r = bounded("with_budget pressure", move || {
        let (tx, rx) = mpsc::channel();
        ok(
            "with_runtime",
            shared.with_runtime(move |_| {
                let _ = tx.send(o.try_reserve(4096).map(|_| ()));
                Ok(())
            }),
        );
        rx.recv().expect("the closure ran")
    });
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
}

/// After a closure panic the backend is poisoned and the next op is refused
/// before it dispatches: nothing the device does can run on a runtime left
/// with `async_encode` off.
#[test]
fn an_op_after_a_closure_panic_is_poisoned() {
    let m = metal();
    let t = probe(&m, &[1.0, 2.0]);
    let _ = m.with_runtime(|_| panic!("provider panic, injected by the test"));
    assert!(matches!(m.silu_forward(&t), Err(OjasError::Poisoned)));
    assert!(matches!(m.trim_pool(), Err(OjasError::Poisoned)));
}
