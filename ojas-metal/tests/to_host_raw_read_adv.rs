//! Adversarial pins for the raw-read doc on `DeviceBuffer::read_bytes` and
//! `Tensor::to_host` (ft-99045487, lane rawadv). Each test attacks one
//! sentence of the Metal half of that contract and asserts what the code
//! really does.
//!
//! - `clip_grad_norm` reports a fault a raw read left pending, naming the
//!   faulted op (not `clip_grad_norm`), and leaves the gradients alone.
//! - `adamw_step` does NOT take or clear the pending fault.
//! - `MetalBackend::with_runtime` DOES take it (a fourth reporting boundary
//!   that the doc's list, "sync, download or clip_grad_norm", does not name).
//! - The fault is one pending slot for the backend: a raw read on one thread
//!   leaves it for a sync on another.
//! - A mapped (zero-copy) tensor's `to_host` decodes from the lease, returns
//!   `Ok`, and neither takes nor clears a pending fault.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::Arc;
use std::thread;

use common::*;
use ojas_core::{
    AdamWConfig, BackendCore, BackendFloat, BackendId, BackendModule, Budget, DType, OjasError,
    Tensor,
};
use tessl::shared_buf::SharedAlloc;

fn nonfinite_op(r: Result<impl std::fmt::Debug, OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

/// A faulted `silu_forward` output, already raw-read once (`Ok`, NaN at 1).
fn faulted_and_raw_read(m: &ojas_metal::MetalBackend) -> Tensor {
    let x = up(m, &host(&[1.0, f32::NAN, 2.0], &[3]));
    let y = ok("silu records", m.silu_forward(&x));
    let raw = ok("raw read", y.to_host(&Budget::new(GIB)));
    assert!(ok("decode", raw.to_f32_vec())[1].is_nan(), "fault raised");
    y
}

/// Attack 1a: the raw read leaves the fault for `clip_grad_norm`, which
/// names the op that faulted, refuses before scaling, and takes it once.
#[test]
fn clip_grad_norm_reports_the_fault_a_raw_read_left_pending() {
    let m = metal();
    let _y = faulted_and_raw_read(&m);
    let clean = host(&[3.0, 4.0], &[2]);
    let mut grads = vec![up(&m, &clean)];
    // max_norm 1 < the norm 5, so a clean clip would scale: it must not.
    assert_eq!(
        nonfinite_op(m.clip_grad_norm(&mut grads, 1.0)),
        "silu_forward"
    );
    assert_eq!(
        bits(&down(&grads[0])),
        bits(&ok("clean", clean.to_f32_vec())),
        "clip refused before it scaled anything"
    );
    m.sync().expect("clip_grad_norm took it, once");
    // And the clip works again once the fault is gone.
    let norm = ok("clip", m.clip_grad_norm(&mut grads, 1.0));
    assert!((norm - 5.0).abs() < 1e-5, "{norm}");
}

/// Attack 1b: `adamw_step` is not a reporting boundary on Metal. It neither
/// refuses on a pending fault nor clears it: the next `sync` still names the
/// op that faulted first.
#[test]
fn adamw_step_neither_reports_nor_clears_a_pending_fault() {
    let m = metal();
    let _y = faulted_and_raw_read(&m);
    let mut p = up(&m, &host(&[1.0; 8], &[8]));
    let g = up(&m, &host(&[0.5; 8], &[8]));
    let mut m1 = up(&m, &host(&[0.0; 8], &[8]));
    let mut m2 = up(&m, &host(&[0.0; 8], &[8]));
    ok(
        "adamw_step runs past a pending fault",
        m.adamw_step(&mut p, &g, &mut m1, &mut m2, 1, AdamWConfig::nanolab(1e-2, 0.1)),
    );
    assert_ne!(
        bits(&down(&p)),
        bits(&[1.0; 8]),
        "the step was applied, not refused"
    );
    assert_eq!(nonfinite_op(m.sync()), "silu_forward", "still pending");
    m.sync().expect("reported once");
}

/// Attack 1c: `with_runtime` (the device thread's `run_provider`) reports
/// and clears the pending fault before it runs the closure. The doc's list
/// of Metal's reporting boundaries leaves it out.
#[test]
fn with_runtime_takes_the_fault_a_raw_read_left_pending() {
    let m = metal();
    let _y = faulted_and_raw_read(&m);
    let ran = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&ran);
    let r = m.with_runtime(move |_| {
        flag.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    });
    assert_eq!(nonfinite_op(r), "silu_forward");
    assert!(
        !ran.load(std::sync::atomic::Ordering::SeqCst),
        "the closure did not run"
    );
    m.sync().expect("with_runtime took it, once");
}

/// Attack 3b: the pending slot belongs to the backend, not the reading
/// thread: thread A raises and raw-reads, thread B's `sync` reports.
#[test]
fn the_fault_a_raw_read_left_is_reported_to_a_sync_on_another_thread() {
    let m = metal();
    let raised = {
        let m = m.clone();
        thread::spawn(move || {
            let y = faulted_and_raw_read(&m);
            drop(y);
        })
    };
    raised.join().expect("thread A");
    let reporter = {
        let m = m.clone();
        thread::spawn(move || nonfinite_op(m.sync()))
    };
    assert_eq!(reporter.join().expect("thread B"), "silu_forward");
    m.sync().expect("taken once, by thread B");
}

/// Attack 3a: a mapped tensor shares memory with a device tensor (the
/// upload adopts the buffer). `to_host` of the mapped tensor decodes from
/// the lease and is `Ok`; it neither checks nor takes the pending fault of
/// an in-place op on the adopted buffer, and the fault is still reported at
/// the next `sync`.
#[test]
fn a_mapped_tensor_to_host_is_a_raw_read_and_leaves_the_fault_pending() {
    let m = metal();
    let budget = Budget::new(GIB);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    ok(
        "with_runtime",
        m.with_runtime(move |rt| {
            let alloc = SharedAlloc::new(rt, 16).map_err(|e| OjasError::Backend {
                id: BackendId::Metal,
                detail: e,
            })?;
            tx.send(alloc.into_shared_buf()).unwrap();
            Ok(())
        }),
    );
    let shared = rx.recv().expect("shared buf");
    {
        let mut w = shared.write().expect("write lease");
        w.as_f32_mut().expect("f32").copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }
    let mapped = ok(
        "from_mapped",
        Tensor::from_mapped(Arc::new(shared), &[4], DType::F32, &budget),
    );
    let mut acc = up(&m, &mapped);
    let bad = up(&m, &host(&[0.0, f32::NAN, 0.0, 0.0], &[4]));
    ok(
        "accumulate_grad records, it does not refuse",
        m.accumulate_grad(&mut acc, &bad),
    );

    let h = ok("mapped to_host is Ok", mapped.to_host(&budget));
    assert_eq!(
        ok("decode", h.to_f32_vec()),
        vec![1.0, 2.0, 3.0, 4.0],
        "the in-place add is withheld when the sum is non-finite"
    );
    // Reading it again changes nothing either.
    ok("second mapped read", mapped.to_host(&budget));
    assert_eq!(nonfinite_op(m.sync()), "accumulate_grad");
    m.sync().expect("reported once");
}
