//! K1 GEMM on the device: `#[ignore]`, NOT RUN on the Mac (no NVIDIA GPU).
//! On the box: `LD_LIBRARY_PATH=<cublas>:<nvrtc> ./device_k1-<hash> --ignored --test-threads=1`.
//!
//! Bounds and shapes are tessl's, cited in `src/smoke.rs`.
#![cfg(feature = "cuda")]

use ojas_qwen35_cuda::check::{Check, Status};
use ojas_qwen35_cuda::gemm::{Accumulate, Bf16Engine, GemmSpec, Operands};
use ojas_qwen35_cuda::gemm_plan::{GemmLayout, GemmShape};
use ojas_qwen35_cuda::runtime::{CudaRuntime, RuntimeConfig};
use ojas_qwen35_cuda::smoke::{self, GemmCase};

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

fn cases(operands: Operands, engine: Bf16Engine) -> Vec<GemmCase> {
    smoke::gemm_cases()
        .unwrap()
        .into_iter()
        .filter(|c| c.spec.operands == operands && c.engine == engine)
        .collect()
}

fn run(cases: &[GemmCase]) {
    assert!(!cases.is_empty());
    let rt = runtime();
    let checks: Vec<Check> = cases
        .iter()
        .flat_map(|c| smoke::gemm_case_checks(&rt, c))
        .collect();
    assert_all_pass(&checks);
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn cublas_gemm_ex_accepts_bf16_operands_with_an_f32_c() {
    let rt = runtime();
    assert_all_pass(&smoke::cublas_probe(&rt));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn exact_f32_ffma_matches_the_host_bitwise_and_f64_within_tessls_bound() {
    run(&cases(Operands::ExactF32, Bf16Engine::Ffma));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn bf16_ffma_matches_the_host_bitwise_and_f64_within_tessls_bound() {
    run(&cases(Operands::Bf16, Bf16Engine::Ffma));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn bf16_cublas_matches_f64_within_tessls_bound_and_repeats() {
    run(&cases(Operands::Bf16, Bf16Engine::Cublas));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn bf16_cublas_agrees_with_the_bf16_ffma_oracle() {
    let rt = runtime();
    let checks: Vec<Check> = GemmLayout::ALL
        .iter()
        .flat_map(|&l| smoke::bf16_engines_agree(&rt, l, (130, 70, 260)))
        .collect();
    assert_all_pass(&checks);
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn every_engine_and_layout_is_bit_identical_over_25_runs() {
    let rt = runtime();
    let shape = GemmShape::new(130, 70, 260).unwrap();
    let mut checks = Vec::new();
    for layout in GemmLayout::ALL {
        for (operands, engine) in [
            (Operands::ExactF32, Bf16Engine::Ffma),
            (Operands::Bf16, Bf16Engine::Ffma),
            (Operands::Bf16, Bf16Engine::Cublas),
        ] {
            let case = GemmCase {
                spec: GemmSpec {
                    operands,
                    layout,
                    shape,
                    acc: Accumulate::Add,
                },
                engine,
            };
            checks.extend(smoke::gemm_determinism(&rt, &case, 25));
        }
    }
    assert_all_pass(&checks);
}
