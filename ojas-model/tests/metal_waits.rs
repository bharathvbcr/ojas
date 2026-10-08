//! Waited GPU commits per Metal trainer step, counted by trigger
//! (`MetalBackend::wait_counts`), with small uploads behind recorded work
//! carried inline (the default) and with that turned off, which is how
//! every upload behaved before inline uploads existed.
//!
//! The tiny spec at batch 2, sequence 32, two micro-batches. Measured on
//! an M-series GPU (2026-10-07, `docs/metal-deferred-faults.md` §9.3):
//! with inline uploads off, 11 uploads per step wait behind recorded work.
//! With them on, none do: every one of those uploads is within the inline
//! cap (64 KiB). The step's sync and clip-norm waits (one each) are the
//! same either way.
//!
//! A missing device fails unless `OJAS_ALLOW_NO_GPU=1`.

#![cfg(target_os = "macos")]

mod common;

use std::time::{Duration, Instant};

use common::token_bin;
use ojas_core::{Backend, Budget};
use ojas_cpu::{CosineSchedule, LrSchedule};
use ojas_metal::{MetalBackend, WaitCounts};
use ojas_model::{init_params, ModelSpec, TrainConfig, Trainer};

const STEPS: usize = 3;
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

/// One measured trainer configuration.
struct Run {
    spec: ModelSpec,
    batch: usize,
    seq: usize,
    accum: usize,
    /// The backend's byte budget. Metal's memory-cap commit threshold is
    /// `min(1 GiB, budget / 4)`, so this sets how often those commits fire.
    budget: u64,
    tokens: usize,
}

/// One measured step: waits by trigger, the budget's peak bytes (the
/// trainer resets the peak at each step's start) and the wall time.
struct Step {
    waits: WaitCounts,
    peak_bytes: u64,
    time: Duration,
}

/// The tiny spec at batch 2, sequence 32, two micro-batches.
fn tiny() -> Run {
    Run {
        spec: ModelSpec::tiny(),
        batch: 2,
        seq: 32,
        accum: 2,
        budget: 1 << 30,
        tokens: 20_000,
    }
}

/// Per-step waits by trigger, after a warm-up step, with inline uploads
/// `inline`.
fn step_waits(inline: bool) -> Option<Vec<WaitCounts>> {
    let steps = measure(&tiny(), inline)?;
    Some(steps.into_iter().map(|s| s.waits).collect())
}

/// [`STEPS`] measured steps of `run` after one warm-up step, or `None` when
/// there is no Metal device and `OJAS_ALLOW_NO_GPU=1`.
fn measure(run: &Run, inline: bool) -> Option<Vec<Step>> {
    let backend = match MetalBackend::new(Budget::new(run.budget)) {
        Ok(m) => m,
        Err(err) => {
            if std::env::var(ALLOW_NO_GPU).as_deref() == Ok("1") {
                eprintln!("SKIP ({ALLOW_NO_GPU}=1): MetalBackend::new: {err}");
                return None;
            }
            panic!("MetalBackend::new failed: {err}. Set {ALLOW_NO_GPU}=1 to skip explicitly");
        }
    };
    backend.set_inline_uploads(inline).unwrap();
    let spec = run.spec;
    let params = init_params(&spec, 5, &Budget::new(run.budget)).unwrap();
    let (_tmp, bin) = token_bin(run.tokens, spec.vocab as u32);
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    let config = TrainConfig::nanolab(run.batch, run.seq, run.accum, 1337, schedule);
    let mut t = Trainer::new(backend, spec, &params, bin, config).unwrap();
    t.step().unwrap();
    let per_step = (0..STEPS)
        .map(|_| {
            let before = t.backend().wait_counts();
            let start = Instant::now();
            t.step().unwrap();
            Step {
                time: start.elapsed(),
                waits: t.backend().wait_counts().since(&before),
                peak_bytes: t.backend().budget().peak_bytes(),
            }
        })
        .collect();
    Some(per_step)
}

#[test]
fn inline_uploads_remove_the_upload_waits_of_a_metal_trainer_step() {
    let Some(before) = step_waits(false) else {
        return;
    };
    let Some(after) = step_waits(true) else {
        return;
    };
    for (n, (b, a)) in before.iter().zip(&after).enumerate() {
        eprintln!("step {n}: inline off {b:?}");
        eprintln!("step {n}: inline on  {a:?}");
    }
    for (n, (b, a)) in before.iter().zip(&after).enumerate() {
        // Without inline uploads, every upload made behind recorded work
        // waits; with them, the step's uploads all fit inline.
        assert!(b.upload > 0, "step {n}: no upload waits to remove: {b:?}");
        assert_eq!(a.upload, 0, "step {n}: upload waits remain: {a:?}");
        // Nothing else moves: the step's sync points wait as before.
        assert_eq!(
            WaitCounts { upload: 0, ..*b },
            WaitCounts { upload: 0, ..*a },
            "step {n}: a trigger other than upload changed"
        );
        assert!(a.total() < b.total(), "step {n}: {b:?} -> {a:?}");
    }
}

/// The 124M split by trigger (`docs/metal-deferred-faults.md` §9.3), which
/// was once inferred by subtraction: nanolab 124M at batch 4, sequence
/// 1024, four micro-batches, on a 16 GiB budget (so the memory-cap commit
/// threshold is its 1 GiB ceiling). About 5 s a step.
///
/// `cargo test --release -p ojas-model --test metal_waits -- --ignored
/// --nocapture nanolab_124m`
#[test]
#[ignore = "124M on Metal, about 20 s and several GiB; run with --ignored in release"]
fn nanolab_124m_step_waits_by_trigger() {
    let run = Run {
        spec: ModelSpec::nanolab_124m(),
        batch: 4,
        seq: 1024,
        accum: 4,
        budget: 16 << 30,
        tokens: 200_000,
    };
    let Some(steps) = measure(&run, true) else {
        return;
    };
    for (n, s) in steps.iter().enumerate() {
        eprintln!(
            "step {n}: {:?} total {}, peak {} MiB, {:.2} s",
            s.waits,
            s.waits.total(),
            s.peak_bytes >> 20,
            s.time.as_secs_f64()
        );
    }
    for (n, s) in steps.iter().enumerate() {
        let w = &s.waits;
        // The step's two sync points wait once each, as at tiny.
        assert_eq!((w.sync, w.clip_norm), (1, 1), "step {n}: {w:?}");
        // Token ids and targets are 16 KiB per micro-batch: inline.
        assert_eq!(w.upload, 0, "step {n}: {w:?}");
        assert_eq!(
            (w.read, w.slab_full, w.recycle),
            (0, 0, 0),
            "step {n}: {w:?}"
        );
        // At this size the step allocates past the 1 GiB cap.
        assert!(w.mem_cap > 0, "step {n}: no memory-cap commit: {w:?}");
        // Every step does the same work, so it waits the same way.
        assert_eq!(*w, steps[0].waits, "step {n} differs from step 0");
    }
}
