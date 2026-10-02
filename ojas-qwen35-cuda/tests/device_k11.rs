//! K11 on the device: AdamW in place and the squared gradient norm. Every
//! test needs an sm_90 NVIDIA GPU, so each is `#[ignore]` and NOT RUN on the
//! Mac. On the box: `./device_k11-<hash> --ignored --test-threads=1`.
//!
//! The tiers (`src/k11_golden.rs`): the device equals the f32 emulation bit
//! for bit and repeats itself; on L-oracle's decay-sensitive golden it is
//! within the pre-registered 1e-6 of torch; on the adamw_f float64 golden it
//! is within the sanity bounds. The f64 reference of `tests/reference/adamw.rs`
//! is held to that golden by `tests/reference_adamw.rs`; here it judges the
//! device directly as well, at the golden's own bound for the f32 path.
#![cfg(feature = "cuda")]

mod reference;

use ojas_qwen35_cuda::check::{Check, Status};
use ojas_qwen35_cuda::k11_golden::{AdamwF, DecaySensitive, ADAMW_F_PARAM_BOUND};
use ojas_qwen35_cuda::k11_host::{replay_f32, synthetic_cases, BankCase};
use ojas_qwen35_cuda::k11_smoke::{k11_checks, run_device};
use ojas_qwen35_cuda::runtime::{CudaRuntime, RuntimeConfig};

use reference::adamw::{adamw_step_f64, Hyper, State};

fn runtime() -> CudaRuntime {
    CudaRuntime::open(RuntimeConfig::default()).unwrap_or_else(|e| panic!("CudaRuntime::open: {e}"))
}

fn assert_all_pass(checks: &[Check]) {
    assert!(!checks.is_empty(), "no checks ran");
    let bad: Vec<String> = checks
        .iter()
        .filter(|c| c.status != Status::Pass)
        .map(|c| format!("{} {}: {}", c.status.name(), c.name, c.detail))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k11_device_checks_pass() {
    let rt = runtime();
    let checks = k11_checks(&rt);
    for c in &checks {
        println!("{:<8} {}  {}", c.status.name(), c.name, c.detail);
    }
    assert_all_pass(&checks);
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k11_decay_sensitive_device_run_is_within_the_preregistered_bound_of_torch() {
    let rt = runtime();
    let ds = DecaySensitive::embedded().expect("golden");
    let case = BankCase {
        label: "decay_sensitive".to_string(),
        table: ds.table.clone(),
        hyper: ds.hyper,
        init: ds.init.clone(),
        grads: ds.grads.clone(),
    };
    let run = run_device(&rt, &case).expect("device run");
    let gap = ds.measure(&run.w).expect("measure");
    println!(
        "K11_DEVICE_DECAY max_abs={:e} at_step={} entry={:?} bound={:e}",
        gap.max_abs, gap.at_step, gap.entry, ds.bound
    );
    assert!(gap.max_abs <= ds.bound, "{gap:?}");
    let torch: Vec<u64> = ds.entries.iter().map(|e| e.adamw_steps_taken).collect();
    assert_eq!(run.steps, torch, "per-entry step counts (D7)");
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k11_device_equals_the_emulation_on_the_synthetic_banks_over_five_runs() {
    let rt = runtime();
    for case in synthetic_cases().expect("cases") {
        let want =
            replay_f32(&case.table, &case.hyper, &case.init, &case.grads).expect("emulation");
        for rep in 0..5 {
            let got = run_device(&rt, &case).expect("device run");
            for (t, (g, w)) in got.w.iter().zip(&want.w).enumerate() {
                let first = g
                    .iter()
                    .zip(w)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    first.is_none(),
                    "{} run {rep} step {}: first mismatch at {first:?}",
                    case.label,
                    t + 1
                );
            }
            assert_eq!(
                got.partials, want.partials,
                "{} run {rep}: sq partials",
                case.label
            );
            assert_eq!(
                got.steps, want.steps,
                "{} run {rep}: step counts",
                case.label
            );
        }
    }
}

/// The f64 reference (`tests/reference/adamw.rs`, held to torch's float64 by
/// `tests/reference_adamw.rs`) against the device on adamw_f's inputs, at
/// the same sanity bound the f64 golden is judged by.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k11_device_tracks_the_f64_reference_on_adamw_f() {
    let rt = runtime();
    let f = AdamwF::embedded().expect("adamw_f");
    let case = BankCase {
        label: "adamw_f".to_string(),
        table: f.table.clone(),
        hyper: f.hyper,
        init: f.p0_f32(),
        grads: f.grads_with_flags(),
    };
    let run = run_device(&rt, &case).expect("device run");
    let sizes: Vec<usize> = f.table.entries().iter().map(|e| e.len).collect();
    let split = |flat: &[f64]| -> Vec<Vec<f64>> {
        f.table
            .entries()
            .iter()
            .map(|e| flat[e.offset..e.offset + e.len].to_vec())
            .collect()
    };
    let mut params = split(&f.p0);
    let mut state = State::new(&sizes);
    let h = Hyper {
        lr: f.hyper.lr,
        beta1: f.hyper.beta1,
        beta2: f.hyper.beta2,
        eps: f.hyper.eps,
    };
    let wd: Vec<f64> = f.table.entries().iter().map(|e| e.weight_decay).collect();
    let scale: Vec<f64> = f.table.entries().iter().map(|e| e.lr_scale).collect();
    for t in 0..f.steps {
        adamw_step_f64(&mut params, &split(&f.grads[t]), &mut state, h, &wd, &scale);
        let want: Vec<f64> = params.iter().flatten().copied().collect();
        let max = want.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        let worst = run.w[t]
            .iter()
            .zip(&want)
            .map(|(&g, &w)| (f64::from(g) - w).abs() / max)
            .fold(0.0f64, f64::max);
        println!("K11_DEVICE_VS_F64_REF step {} rel {worst:e}", t + 1);
        assert!(worst <= ADAMW_F_PARAM_BOUND, "step {}: {worst:e}", t + 1);
    }
}
