//! The raw-read contract (`DeviceBuffer::read_bytes`, `Tensor::to_host`): on
//! a wgpu tensor both are a byte read that checks no fault. An op that
//! detects a non-finite value returns `Ok`, `to_host` of its output returns
//! `Ok` with the bytes as the kernel wrote them (the NaN included), and the
//! fault the read observed is deferred to the context's shared pair, which
//! the next `sync`, `download`, `clip_grad_norm` or `adamw_step` on any
//! thread takes, naming the op, once. A caller that needs checked bytes syncs
//! or downloads first.
//!
//! A missing adapter fails these tests (`fresh()` expects one); there is no
//! skip path.

mod common;

use std::thread;

use common::*;
use ojas_core::{BackendCore, BackendFloat, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

fn nonfinite_op(r: Result<impl std::fmt::Debug, OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

/// An uploaded tensor of `values`, and `silu_forward` of it on `g`.
fn faulted(g: &WgpuBackend, values: &[f32]) -> Tensor {
    let x = g
        .upload(&Tensor::from_f32(values, &[values.len()], host_budget()).unwrap())
        .unwrap();
    g.silu_forward(&x)
        .expect("silu records the fault, it does not refuse")
}

/// A raw read of `y`: `Ok`, whatever the bytes are.
fn raw_read(g: &WgpuBackend, y: &Tensor) -> Vec<f32> {
    match y.to_host(g.budget()) {
        Ok(h) => h.to_f32_vec().unwrap(),
        Err(e) => panic!("to_host must be a raw read, got {e:?}"),
    }
}

#[test]
fn to_host_of_a_faulted_output_is_a_raw_read_and_sync_reports_the_op() {
    let g = &fresh();
    g.sync().unwrap();
    let y = faulted(g, &[1.0, f32::NAN, 2.0]);

    let raw = raw_read(g, &y);
    assert_eq!(raw.len(), 3);
    assert!(raw[0].is_finite() && raw[2].is_finite(), "{raw:?}");
    assert!(raw[1].is_nan(), "the NaN is returned as written: {raw:?}");

    // The read took no fault: a second raw read is still Ok, and the fault
    // is still pending for the next sync, which names the op once.
    raw_read(g, &y);
    assert_eq!(nonfinite_op(g.sync()), "silu_forward");
    g.sync().expect("reported once");
}

#[test]
fn download_reports_the_fault_a_raw_read_left_pending() {
    let g = &fresh();
    g.sync().unwrap();
    let y = faulted(g, &[f32::INFINITY, 1.0]);
    assert!(raw_read(g, &y).iter().any(|v| !v.is_finite()));
    assert_eq!(nonfinite_op(g.download(&y)), "silu_forward");
    g.sync().expect("download reported it once");
}

/// Thread A's raw read returns `Ok`; the fault it deferred is taken by the
/// next `sync` on thread B. Joined between steps: no race.
#[test]
fn cross_thread_raw_read_defers_the_fault_to_a_sync_on_another_thread() {
    let g = &fresh();
    g.sync().unwrap();
    let y = faulted(g, &[1.0, f32::NAN, 2.0]);

    let raw = thread::scope(|s| s.spawn(|| raw_read(g, &y)).join().unwrap());
    assert!(
        raw[1].is_nan(),
        "thread A read the bytes as written: {raw:?}"
    );

    let op = thread::scope(|s| s.spawn(|| nonfinite_op(g.sync())).join().unwrap());
    assert_eq!(op, "silu_forward");
    g.sync().expect("taken once, by thread B");
}

/// The same hand-off with a `download` of a clean tensor as the reporting
/// boundary on thread B.
#[test]
fn cross_thread_raw_read_defers_the_fault_to_a_download_on_another_thread() {
    let g = &fresh();
    g.sync().unwrap();
    let y = faulted(g, &[f32::INFINITY, 1.0]);
    let clean = g
        .upload(&Tensor::from_f32(&[1.0, 2.0], &[2], host_budget()).unwrap())
        .unwrap();

    thread::scope(|s| s.spawn(|| raw_read(g, &y)).join().unwrap());

    let op = thread::scope(|s| s.spawn(|| nonfinite_op(g.download(&clean))).join().unwrap());
    assert_eq!(op, "silu_forward");
    g.sync().expect("taken once, by thread B");
}
