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

use common::token_bin;
use ojas_core::Budget;
use ojas_cpu::{CosineSchedule, LrSchedule};
use ojas_metal::{MetalBackend, WaitCounts};
use ojas_model::{init_params, ModelSpec, TrainConfig, Trainer};

const STEPS: usize = 3;
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

/// Per-step waits by trigger, after a warm-up step, with inline uploads
/// `inline`.
fn step_waits(inline: bool) -> Option<Vec<WaitCounts>> {
    let backend = match MetalBackend::new(Budget::new(1 << 30)) {
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
    let spec = ModelSpec::tiny();
    let params = init_params(&spec, 5, &Budget::new(1 << 30)).unwrap();
    let (_tmp, bin) = token_bin(20_000, 256);
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    let config = TrainConfig::nanolab(2, 32, 2, 1337, schedule);
    let mut t = Trainer::new(backend, spec, &params, bin, config).unwrap();
    t.step().unwrap();
    let per_step = (0..STEPS)
        .map(|_| {
            let before = t.backend().wait_counts();
            t.step().unwrap();
            t.backend().wait_counts().since(&before)
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
