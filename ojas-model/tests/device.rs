//! The tiny spec trained for 5 steps on wgpu and on Metal against
//! `CpuBackend` (Exact): every step's mean loss within 1e-4, and exactly
//! one device readback per step on the backend's own budget (G10).
//!
//! G9 on each device: 40 steps straight against 20, save, drop, resume on
//! a newly opened backend, 20 more, within the same tolerance.
//!
//! A missing device fails unless `OJAS_ALLOW_NO_GPU=1`, so it is never
//! counted as a pass. The tests take one lock, so they run one at a time.

mod common;

use std::sync::{Mutex, MutexGuard};

use common::{bits, token_bin, TempDir};
use ojas_core::{Backend, Budget, Numerics};
use ojas_cpu::{CosineSchedule, CpuBackend, LrSchedule};
use ojas_data::TokenBin;
use ojas_model::{init_params, ModelSpec, MomentsRef, TrainConfig, Trainer};

const STEPS: usize = 5;
const TOL: f32 = 1e-4;
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

fn config() -> TrainConfig {
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    TrainConfig::nanolab(2, 32, 2, 1337, schedule)
}

/// Per-step `(loss, readbacks)` of the tiny spec on `backend`.
fn run<B: Backend>(backend: B) -> Vec<(f32, u64)> {
    let spec = ModelSpec::tiny();
    let params = init_params(&spec, 5, &Budget::new(1 << 30)).unwrap();
    let (_tmp, bin) = token_bin(20_000, 256);
    let mut t = Trainer::new(backend, spec, &params, bin, config()).unwrap();
    (0..STEPS)
        .map(|_| {
            let before = t.backend().budget().device_readbacks().0;
            let r = t.step().unwrap();
            let after = t.backend().budget().device_readbacks().0;
            (r.loss, after - before)
        })
        .collect()
}

fn check(name: &str, device: &[(f32, u64)]) {
    let cpu = run(CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact));
    for (step, ((dl, reads), (cl, _))) in device.iter().zip(&cpu).enumerate() {
        eprintln!(
            "{name} step {step}: device {dl:.7} cpu {cl:.7} |d| {:.2e}",
            (dl - cl).abs()
        );
        assert!((dl - cl).abs() <= TOL, "{name} step {step}: {dl} vs {cl}");
        assert_eq!(*reads, 1, "{name} step {step}: readbacks");
    }
}

fn skip_or_fail(what: &str, err: &dyn std::fmt::Display) {
    if std::env::var(ALLOW_NO_GPU).as_deref() == Ok("1") {
        eprintln!("SKIP ({ALLOW_NO_GPU}=1): {what}: {err}");
    } else {
        panic!("{what} failed: {err}. Set {ALLOW_NO_GPU}=1 to skip explicitly");
    }
}

#[test]
fn wgpu_tiny_five_steps_match_cpu() {
    let _guard = serial();
    match ojas_wgpu::WgpuBackend::open(Budget::new(1 << 30)) {
        Ok(gpu) => check("wgpu", &run(gpu)),
        Err(err) => skip_or_fail("WgpuBackend::open", &err),
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tiny_five_steps_match_cpu() {
    let _guard = serial();
    match ojas_metal::MetalBackend::new(Budget::new(1 << 30)) {
        Ok(gpu) => check("metal", &run(gpu)),
        Err(err) => skip_or_fail("MetalBackend::new", &err),
    }
}

/// Every parameter's and moment's values, read back through the trainer's
/// backend.
fn state<B: Backend>(t: &Trainer<B>) -> Vec<Vec<u32>> {
    let b = t.backend();
    let mut out = Vec::new();
    for (info, value) in t.params() {
        out.push(bits(b, value));
        match t.moments(&info.name).unwrap() {
            MomentsRef::Muon { momentum } => out.push(bits(b, momentum)),
            MomentsRef::AdamW { m, v } => out.extend([bits(b, m), bits(b, v)]),
            MomentsRef::Frozen => {}
        }
    }
    out
}

/// G9 on one device: 40 steps straight against 20 steps, save, drop,
/// resume on a newly opened backend, and 20 more. Losses, parameters and
/// moments within `TOL`; step and cursor equal. Prints whether every bit
/// matched.
fn g9<B: Backend>(name: &str, open: impl Fn() -> B) {
    let spec = ModelSpec::tiny();
    let params = init_params(&spec, 5, &Budget::new(1 << 30)).unwrap();
    let (tmp, bin) = token_bin(20_000, 256);
    let reopen = || TokenBin::open_headerless(&tmp.path).unwrap();
    let mut straight = Trainer::new(open(), spec, &params, bin, config()).unwrap();
    let want: Vec<f32> = (0..40).map(|_| straight.step().unwrap().loss).collect();

    let dir = TempDir::new();
    let mut first = Trainer::new(open(), spec, &params, reopen(), config()).unwrap();
    let mut got: Vec<f32> = (0..20).map(|_| first.step().unwrap().loss).collect();
    first.save(&dir.ckpt()).unwrap();
    drop(first);
    let mut resumed = Trainer::resume_from(open(), &dir.ckpt(), reopen(), config()).unwrap();
    got.extend((0..20).map(|_| resumed.step().unwrap().loss));

    let loss_d = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max);
    let loss_bitwise = got
        .iter()
        .zip(&want)
        .all(|(g, w)| g.to_bits() == w.to_bits());
    let (a, b) = (state(&resumed), state(&straight));
    assert_eq!(a.len(), b.len());
    let mut state_d = 0.0f32;
    let mut differing = 0usize;
    let mut total = 0usize;
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.len(), y.len());
        for (&p, &q) in x.iter().zip(y) {
            total += 1;
            if p != q {
                differing += 1;
                state_d = state_d.max((f32::from_bits(p) - f32::from_bits(q)).abs());
            }
        }
    }
    eprintln!(
        "{name} G9: loss max |d| {loss_d:.2e} (bitwise {loss_bitwise}); \
         params+moments max |d| {state_d:.2e}, {differing} of {total} elements differ"
    );
    assert!(loss_d <= TOL, "{name} G9 loss |d| {loss_d}");
    assert!(state_d <= TOL, "{name} G9 state |d| {state_d}");
    assert_eq!(resumed.step_count(), 40);
    assert_eq!(resumed.cursor(), straight.cursor());
}

#[test]
fn wgpu_g9_resume_matches_the_straight_run() {
    let _guard = serial();
    let open = || ojas_wgpu::WgpuBackend::open(Budget::new(1 << 30));
    match open() {
        Ok(first) => {
            drop(first);
            g9("wgpu", || open().unwrap());
        }
        Err(err) => skip_or_fail("WgpuBackend::open", &err),
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_g9_resume_matches_the_straight_run() {
    let _guard = serial();
    let open = || ojas_metal::MetalBackend::new(Budget::new(1 << 30));
    match open() {
        Ok(first) => {
            drop(first);
            g9("metal", || open().unwrap());
        }
        Err(err) => skip_or_fail("MetalBackend::new", &err),
    }
}
