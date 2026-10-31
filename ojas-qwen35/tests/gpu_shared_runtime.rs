//! `Qwen35Step` on the runtime of an ojas `MetalBackend`'s device thread,
//! not on a `GpuRuntime` of its own (`MetalBackend::with_runtime`,
//! `Qwen35Step::open_on`). Opens a Metal runtime: `#[ignore]`, run in the
//! agreed GPU window.
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test gpu_shared_runtime -- --ignored --test-threads=1
//! ```
//!
//! What it pins:
//! - the provider's runtime is the device thread's runtime (`Arc::ptr_eq`),
//!   while a standalone `Qwen35Step::open` has a different one;
//! - the one `Budget` is charged the provider's 16 bytes per parameter while
//!   it is open and released when it drops, and a budget that cannot hold it
//!   refuses before any device byte;
//! - one micro-step (forward, backward, AdamW) on the shared runtime leaves
//!   the same parameter bits as the same micro-step on a standalone runtime;
//! - the backend still serves its own ops before and after the provider ran.

#![cfg(target_os = "macos")]

mod support;

use std::sync::{mpsc, Arc};

use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;
use ojas_qwen35::names::tower_tensors;
use ojas_qwen35::{Numerics, Precision, Qwen35Error, Qwen35Step, Snapshot, Which};
use support::{backward_one, open_tiny, table_bits, take_step, tiny_dir};

const GIB: u64 = 1 << 30;

type Bits = Vec<(String, Vec<u32>)>;

/// What the closure on the device thread reports; every field is `Send`.
struct Seen {
    same_runtime: bool,
    live_open: u64,
    live_dropped: u64,
    bits: Bits,
}

fn device_bytes_of_tiny() -> u64 {
    let snap = Snapshot::from_dir(&tiny_dir()).unwrap();
    tower_tensors(&snap.config)
        .iter()
        .map(|t| t.shape.iter().map(|&d| d as u64).product::<u64>())
        .sum::<u64>()
        * 16
}

/// One micro-step on the backend's own runtime, reported over a channel.
fn micro_step_on(backend: &MetalBackend, budget: &Budget) -> Seen {
    let (tx, rx) = mpsc::channel::<Result<Seen, String>>();
    let budget = budget.clone();
    backend
        .with_runtime(move |rt| {
            let run = || -> Result<Seen, Qwen35Error> {
                let snap = Snapshot::from_dir(&tiny_dir())?;
                let mut s = Qwen35Step::open_on(
                    Arc::clone(rt),
                    &snap,
                    Numerics::ExactF32,
                    Precision::F32,
                    budget.clone(),
                )?;
                let same_runtime = Arc::ptr_eq(s.runtime(), rt);
                let live_open = budget.live_bytes()?;
                backward_one(&mut s, 0);
                take_step(&mut s);
                let bits = table_bits(&s, Which::Parameters);
                drop(s);
                Ok(Seen {
                    same_runtime,
                    live_open,
                    live_dropped: budget.live_bytes()?,
                    bits,
                })
            };
            let _ = tx.send(run().map_err(|e| e.to_string()));
            Ok(())
        })
        .unwrap();
    rx.recv().unwrap().unwrap()
}

#[test]
#[ignore = "opens a Metal runtime; run in the GPU window with --release -- --ignored --test-threads=1"]
fn gpu_the_step_runs_on_the_backends_runtime_and_charges_its_budget() {
    let backend = MetalBackend::new(Budget::new(4 * GIB)).unwrap();
    let budget = backend.budget().clone();

    // The backend's own op before, to prove the same worker serves both.
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], &budget).unwrap();
    let dev = backend.upload(&x).unwrap();

    let seen = micro_step_on(&backend, &budget);
    assert!(seen.same_runtime, "the provider opened its own runtime");
    let device = device_bytes_of_tiny();
    assert!(device > 0);
    assert!(
        seen.live_open >= device,
        "budget live {} while open must include the provider's {device} device bytes",
        seen.live_open
    );
    assert!(
        seen.live_dropped < seen.live_open,
        "the reservation must release with the provider ({} -> {})",
        seen.live_open,
        seen.live_dropped
    );

    // The worker is idle and healthy after the provider: sync, memory, and
    // the buffer uploaded before it reads back unchanged.
    backend.sync().unwrap();
    backend.memory().unwrap();
    let back = backend.download(&dev).unwrap();
    assert_eq!(back.to_f32_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    // A second provider run on the same worker works too (the first left it
    // settled, with its pool buffers recycled).
    let again = micro_step_on(&backend, &budget);
    assert!(again.same_runtime);
}

#[test]
#[ignore = "opens a Metal runtime; run in the GPU window with --release -- --ignored --test-threads=1"]
fn gpu_a_standalone_open_has_its_own_runtime() {
    let a = open_tiny();
    let b = open_tiny();
    assert!(
        !Arc::ptr_eq(a.runtime(), b.runtime()),
        "two `open`s must still be two runtimes: only `open_on` shares"
    );
}

#[test]
#[ignore = "opens a Metal runtime; run in the GPU window with --release -- --ignored --test-threads=1"]
fn gpu_shared_runtime_matches_a_standalone_runtime_bit_for_bit() {
    let backend = MetalBackend::new(Budget::new(4 * GIB)).unwrap();
    let shared = micro_step_on(&backend, &backend.budget().clone());

    let mut alone = open_tiny();
    backward_one(&mut alone, 0);
    take_step(&mut alone);
    let alone = table_bits(&alone, Which::Parameters);

    assert_eq!(shared.bits.len(), alone.len());
    for ((sn, sb), (an, ab)) in shared.bits.iter().zip(&alone) {
        assert_eq!(sn, an);
        assert_eq!(sb, ab, "{sn}: parameter bits differ on the shared runtime");
    }
}

#[test]
#[ignore = "opens a Metal runtime; run in the GPU window with --release -- --ignored --test-threads=1"]
fn gpu_a_budget_that_cannot_hold_the_step_refuses_before_loading() {
    let backend = MetalBackend::new(Budget::new(4 * GIB)).unwrap();
    let (tx, rx) = mpsc::channel::<Result<(), String>>();
    let need = device_bytes_of_tiny();
    backend
        .with_runtime(move |rt| {
            let small = Budget::new(need - 1);
            let snap = Snapshot::from_dir(&tiny_dir()).unwrap();
            let out = match Qwen35Step::open_on(
                Arc::clone(rt),
                &snap,
                Numerics::ExactF32,
                Precision::F32,
                small.clone(),
            ) {
                Err(Qwen35Error::Ojas(OjasError::CapacityExceeded { .. })) => {
                    if small.live_bytes().unwrap() == 0 {
                        Ok(())
                    } else {
                        Err("a refused open left bytes charged".to_string())
                    }
                }
                Err(e) => Err(format!("expected CapacityExceeded, got {e}")),
                Ok(_) => Err("a budget one byte short opened the step".to_string()),
            };
            let _ = tx.send(out);
            Ok(())
        })
        .unwrap();
    rx.recv().unwrap().unwrap();
}
