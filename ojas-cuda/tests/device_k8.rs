//! K8 on the device: SwiGLU forward (f32 and bf16 out) and backward (one or
//! two output buffers), the residual add, and the activation sweep, each bit
//! for bit against its host reference (`src/k8_plan.rs`, `src/k8_act.rs`).
//! Every test needs an sm_90 NVIDIA GPU, so each is `#[ignore]` and NOT RUN
//! on the Mac. On the box: `./device_k8-<hash> --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

use ojas_cuda::check::{Check, Status};
use ojas_cuda::k8_plan::k8_cases;
use ojas_cuda::k8_smoke::{act_sweep_checks, case_checks, k8_checks};
use ojas_cuda::runtime::{CudaRuntime, RuntimeConfig};

fn runtime() -> CudaRuntime {
    CudaRuntime::open(RuntimeConfig::default()).unwrap_or_else(|e| panic!("CudaRuntime::open: {e}"))
}

fn assert_all_pass(checks: &[Check]) {
    assert!(!checks.is_empty(), "no checks ran");
    for c in checks {
        println!("{:<8} {}  {}", c.status.name(), c.name, c.detail);
    }
    let bad: Vec<String> = checks
        .iter()
        .filter(|c| c.status != Status::Pass)
        .map(|c| format!("{} {}: {}", c.status.name(), c.name, c.detail))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k8_device_checks_pass() {
    assert_all_pass(&k8_checks(&runtime()));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k8_activation_sweep_is_bitwise_the_host_emulation() {
    let checks = act_sweep_checks(&runtime());
    assert_eq!(checks.len(), 8, "seven functions and the repeat");
    assert_all_pass(&checks);
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k8_swiglu_and_residual_are_bitwise_and_repeat_five_times() {
    let rt = runtime();
    for c in k8_cases() {
        for _ in 0..5 {
            assert_all_pass(&case_checks(&rt, &c));
        }
    }
}
