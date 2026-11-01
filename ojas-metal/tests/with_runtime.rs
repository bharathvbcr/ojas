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
use ojas_core::{Backend, OjasError, Tensor};
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
