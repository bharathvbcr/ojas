//! M0 and K0 on the device. Every test here needs an sm_90 NVIDIA GPU, so
//! each is `#[ignore]`: on the Mac they compile and are reported NOT RUN. On
//! the box (README, "Device tests on the box"):
//! `LD_LIBRARY_PATH=<cublas>:<nvrtc> ./device_k0-<hash> --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

use ojas_qwen35_cuda::check::{Check, Status};
use ojas_qwen35_cuda::runtime::{CudaRuntime, RuntimeConfig, REQUIRED_CC};
use ojas_qwen35_cuda::smoke;
use ojas_qwen35_cuda::CudaError;

fn runtime(config: RuntimeConfig) -> CudaRuntime {
    CudaRuntime::open(config).unwrap_or_else(|e| panic!("CudaRuntime::open: {e}"))
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
fn the_runtime_opens_an_sm90_device_with_cublas_in_default_math_and_no_atomics() {
    let rt = runtime(RuntimeConfig::default());
    let info = rt.info();
    assert_eq!(info.compute_capability, REQUIRED_CC);
    assert!(info.sm_count > 0);
    assert_eq!(info.cublas_math_mode, 0, "CUBLAS_DEFAULT_MATH");
    assert_eq!(info.cublas_atomics_mode, 0, "CUBLAS_ATOMICS_NOT_ALLOWED");
    assert_eq!(info.cublas_version.0, 12, "the box's venv cuBLAS is 12.x");
    assert!(
        info.driver_version >= 12080,
        "driver API {}",
        info.driver_version
    );
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn an_allocation_past_the_budget_is_refused_before_the_driver() {
    let rt = runtime(RuntimeConfig {
        budget_bytes: 64 << 20,
        ..RuntimeConfig::default()
    });
    // The 32 MiB cuBLAS workspace is already reserved.
    let used = rt.budget().used();
    assert_eq!(used, 32 << 20);
    let err = rt
        .alloc_zeros::<f32>(9 << 20, "too big")
        .expect_err("36 MiB more must not fit a 64 MiB budget holding 32 MiB");
    assert!(matches!(err, CudaError::Capacity { .. }), "{err}");
    assert_eq!(
        rt.budget().used(),
        used,
        "a refused allocation reserved bytes"
    );
    let ok = rt.alloc_zeros::<f32>(1 << 20, "fits").unwrap();
    assert_eq!(rt.budget().used(), used + (4 << 20));
    drop(ok);
    assert_eq!(rt.budget().used(), used);
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn uploads_and_downloads_are_length_checked_and_round_trip() {
    let rt = runtime(RuntimeConfig::default());
    let data: Vec<f32> = (0..1000u16).map(f32::from).collect();
    let mut buf = rt.upload(&data, "round trip").unwrap();
    assert_eq!(rt.download(&buf).unwrap(), data);
    let err = rt.write(&mut buf, &data[..999]).unwrap_err();
    assert!(matches!(err, CudaError::Invalid { .. }), "{err}");
    assert!(rt.alloc_zeros::<f32>(0, "empty").is_err());
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn every_kernel_module_compiles_through_nvrtc_and_the_cache_hits_on_reuse() {
    let rt = runtime(RuntimeConfig::default());
    assert_all_pass(&smoke::compile_checks(&rt));
    let before = rt.cache_stats();
    assert_all_pass(&smoke::compile_checks(&rt));
    let after = rt.cache_stats();
    assert_eq!(
        after.misses, before.misses,
        "a second compile of the same modules missed"
    );
    assert!(after.hits > before.hits);
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn k0_kernels_match_the_host_bitwise_and_repeat_bit_identically() {
    let rt = runtime(RuntimeConfig::default());
    assert_all_pass(&smoke::k0_checks(&rt));
}
