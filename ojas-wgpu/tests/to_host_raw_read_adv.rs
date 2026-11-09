//! Adversarial pins for the wgpu half of the raw-read doc on
//! `DeviceBuffer::read_bytes` and `Tensor::to_host` (ft-99045487, lane
//! rawadv). The doc says the fault the read observed is deferred to one pair
//! shared by every thread of the context, which the next `sync`, `download`,
//! `clip_grad_norm` or `adamw_step` takes once.
//!
//! The pin tests of the branch under attack cannot tell "deferred to the
//! pair" from "left on the device": a `sync` finds either. `adamw_step` reads
//! nothing from the device and refuses only on the pair, so it is the probe
//! that proves the deferral. These tests use it, and attack each boundary,
//! the once-ness, the public `WgpuContext::sync`, and the contexts' isolation.
//!
//! A missing adapter fails these tests (`fresh()` expects one).

mod common;

use std::sync::{Arc, Barrier};
use std::thread;

use common::*;
use ojas_core::{AdamWConfig, BackendCore, BackendFloat, BackendModule, DType, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

fn nonfinite_op(r: Result<impl std::fmt::Debug, OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

fn f32s(g: &WgpuBackend, values: &[f32]) -> Tensor {
    g.upload(&Tensor::from_f32(values, &[values.len()], host_budget()).unwrap())
        .unwrap()
}

/// `silu_forward` of `[1, NaN, 2]`, raw-read once. The read returned `Ok`
/// with the NaN as written, so the fault it observed is now in the pair.
fn faulted_and_raw_read(g: &WgpuBackend) -> Tensor {
    g.sync().unwrap();
    let y = g
        .silu_forward(&f32s(g, &[1.0, f32::NAN, 2.0]))
        .expect("silu records");
    let h = y.to_host(g.budget()).expect("raw read is Ok");
    assert!(h.to_f32_vec().unwrap()[1].is_nan(), "fault raised");
    y
}

fn adamw_state(g: &WgpuBackend) -> (Tensor, Tensor, Tensor, Tensor) {
    let zeros = || {
        g.upload(&Tensor::zeros(&[8], DType::F32, host_budget()).unwrap())
            .unwrap()
    };
    (f32s(g, &[1.0; 8]), f32s(g, &[0.5; 8]), zeros(), zeros())
}

const CFG: fn() -> AdamWConfig = || AdamWConfig::nanolab(1e-2, 0.1);

/// Attack 2a: `adamw_step` takes the deferred fault: refused, naming the op
/// that faulted, the parameter untouched, and the very next step proceeds.
/// Only the pair can make it refuse, so this is the proof of the deferral.
#[test]
fn adamw_step_takes_the_pair_once_and_the_next_step_runs() {
    let g = &fresh();
    let _y = faulted_and_raw_read(g);
    let (mut p, grad, mut m1, mut m2) = adamw_state(g);
    assert_eq!(
        nonfinite_op(g.adamw_step(&mut p, &grad, &mut m1, &mut m2, 1, CFG())),
        "silu_forward"
    );
    assert_eq!(
        g.download(&p).unwrap().to_f32_vec().unwrap(),
        vec![1.0; 8],
        "a refused step changes nothing"
    );
    g.adamw_step(&mut p, &grad, &mut m1, &mut m2, 1, CFG())
        .expect("taken once: the next step is not refused");
    g.sync().expect("nothing left to report");
}

/// Attack 2b: the fault is NOT on the device after the raw read: only the
/// pair holds it. An `adamw_step` taken before the raw read of a still
/// unobserved fault is not refused (it reads nothing), and the later sync
/// reports it: `adamw_step` takes what a read observed, not what is merely
/// pending on the device.
#[test]
fn adamw_step_does_not_observe_a_fault_nobody_read() {
    let g = &fresh();
    g.sync().unwrap();
    let _y = g
        .silu_forward(&f32s(g, &[1.0, f32::NAN, 2.0]))
        .expect("silu records");
    let (mut p, grad, mut m1, mut m2) = adamw_state(g);
    g.adamw_step(&mut p, &grad, &mut m1, &mut m2, 1, CFG())
        .expect("no read has observed the fault, so there is nothing to refuse on");
    assert_eq!(nonfinite_op(g.sync()), "silu_forward");
    g.sync().unwrap();
}

/// Attack 2c: `clip_grad_norm` takes the pair, refuses before scaling.
#[test]
fn clip_grad_norm_takes_the_pair_once() {
    let g = &fresh();
    let _y = faulted_and_raw_read(g);
    let mut grads = vec![f32s(g, &[3.0, 4.0])];
    assert_eq!(
        nonfinite_op(g.clip_grad_norm(&mut grads, 1.0)),
        "silu_forward"
    );
    assert_eq!(
        g.download(&grads[0]).unwrap().to_f32_vec().unwrap(),
        vec![3.0, 4.0],
        "refused before it scaled"
    );
    let norm = g.clip_grad_norm(&mut grads, 1.0).expect("taken once");
    assert!((norm - 5.0).abs() < 1e-5, "{norm}");
}

/// Attack 2d: after thread A's raw read, thread B's `sync` takes the pair
/// and neither A, B nor a third thread C sees it again.
#[test]
fn a_second_sync_on_any_thread_is_clean_after_one_takes_it() {
    let g = &fresh();
    let _y = thread::scope(|s| s.spawn(|| faulted_and_raw_read(g)).join().unwrap());
    let b = thread::scope(|s| s.spawn(|| nonfinite_op(g.sync())).join().unwrap());
    assert_eq!(b, "silu_forward");
    g.sync().expect("thread A (the reader) after B took it");
    thread::scope(|s| s.spawn(|| g.sync().expect("thread C")).join().unwrap());
    thread::scope(|s| {
        s.spawn(|| g.sync().expect("thread B again"))
            .join()
            .unwrap()
    });
}

/// Attack 2e: eight threads race to take one deferred fault: exactly one
/// gets it. (The raw read has returned and joined before the race starts,
/// and its read cleared the device word, so the pair is the only holder.)
#[test]
fn exactly_one_of_many_concurrent_syncs_takes_the_deferred_fault() {
    for round in 0..20 {
        let g = &fresh();
        let _y = faulted_and_raw_read(g);
        let gate = Arc::new(Barrier::new(8));
        let reports: Vec<Result<(), OjasError>> = thread::scope(|s| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let gate = Arc::clone(&gate);
                    s.spawn(move || {
                        gate.wait();
                        g.sync()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let errs: Vec<_> = reports.iter().filter(|r| r.is_err()).collect();
        assert_eq!(errs.len(), 1, "round {round}: {reports:?}");
        assert!(
            matches!(errs[0], Err(OjasError::NonFinite { op: "silu_forward" })),
            "round {round}: {errs:?}"
        );
    }
}

/// Attack 2f: "the next sync" is `BackendCore::sync`. The public
/// `WgpuContext::sync` is a sync that does NOT take the pair (it defers what
/// it observes), so a reader who calls it first still gets `Ok` and the
/// fault is still pending afterwards.
#[test]
fn the_public_context_sync_does_not_take_the_deferred_fault() {
    let g = &fresh();
    let _y = faulted_and_raw_read(g);
    g.context()
        .sync()
        .expect("WgpuContext::sync returns no fault");
    assert_eq!(
        nonfinite_op(g.sync()),
        "silu_forward",
        "still pending after WgpuContext::sync"
    );
    g.sync().unwrap();
}

/// Attack 2g: the pair belongs to the context. A raw read on one context
/// leaves nothing for a sync on another.
#[test]
fn the_pair_is_per_context() {
    let a = &fresh();
    let b = &fresh();
    let _y = faulted_and_raw_read(a);
    b.sync().expect("another context sees no fault");
    assert_eq!(nonfinite_op(a.sync()), "silu_forward");
}

/// Attack 4: no path on wgpu makes `to_host` itself return `Err` for a
/// fault: not a repeated read, not a read of a narrowed view, not a read
/// after another thread's raw read deferred one.
#[test]
fn to_host_never_errs_for_a_fault() {
    let g = &fresh();
    let y = faulted_and_raw_read(g);
    for _ in 0..3 {
        y.to_host(g.budget()).expect("repeat raw read");
    }
    let view = y.narrow(4, &[2], &[1]).expect("narrow view");
    let h = view.to_host(g.budget()).expect("view raw read");
    assert!(h.to_f32_vec().unwrap()[0].is_nan());
    thread::scope(|s| {
        s.spawn(|| y.to_host(g.budget()).expect("other thread raw read"))
            .join()
            .unwrap()
    });
    assert_eq!(nonfinite_op(g.sync()), "silu_forward");
}

/// Attack 2h (stress, `--ignored`): the module docs of `ojas_wgpu::backend`
/// concede that two overlapping reads of a held fault "both report it" and
/// that "a reported once guarantee is not given" (ft-84e2b74e). The
/// rustdoc under attack says the pair is taken "once". A raw read on thread
/// A racing a `sync` on thread B, then a final `sync`, must deliver the one
/// fault exactly once in total. The assertion is the doc's claim; a run
/// that reports it twice proves the claim needs the same caveat.
#[test]
#[ignore = "stress: a race, run explicitly with --ignored"]
fn a_raw_read_racing_a_sync_reports_the_fault_exactly_once() {
    let g = &fresh();
    let mut doubles = 0;
    let mut lost = 0;
    for round in 0..400 {
        g.sync().unwrap();
        let y = g
            .silu_forward(&f32s(g, &[1.0, f32::NAN, 2.0]))
            .expect("silu records");
        let gate = Arc::new(Barrier::new(2));
        let (a_gate, b_gate) = (Arc::clone(&gate), gate);
        let b_result = thread::scope(|s| {
            let b = s.spawn(move || {
                b_gate.wait();
                g.sync()
            });
            let a = s.spawn(|| {
                a_gate.wait();
                y.to_host(g.budget()).expect("raw read is Ok")
            });
            a.join().unwrap();
            b.join().unwrap()
        });
        let tail = g.sync();
        let total = usize::from(b_result.is_err()) + usize::from(tail.is_err());
        match total {
            0 => lost += 1,
            1 => {}
            _ => {
                doubles += 1;
                eprintln!("round {round}: reported twice: B {b_result:?}, tail {tail:?}");
            }
        }
    }
    eprintln!("RAWADV-STRESS doubles={doubles} lost={lost} of 400");
    assert_eq!(
        (doubles, lost),
        (0, 0),
        "once is not guaranteed under overlap"
    );
}
