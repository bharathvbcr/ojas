//! The library probe on a host with no CUDA libraries: the Mac. Runs here
//! (not `#[ignore]`), with `--features cuda`, and proves the runtime refuses
//! by name instead of letting cudarc panic (`cudarc/src/lib.rs:199-201`).
//! On Linux the libraries may be present, so the test is macOS-only.
#![cfg(all(feature = "cuda", target_os = "macos"))]

use ojas_cuda::runtime::{probe_libraries, CudaRuntime, RuntimeConfig};
use ojas_cuda::CudaError;

#[test]
fn a_host_without_cuda_is_refused_by_library_name_not_by_a_panic() {
    let err = probe_libraries().expect_err("this Mac has no libcuda");
    match &err {
        CudaError::LibraryMissing { libraries, detail } => {
            for lib in ["libcuda", "libnvrtc", "libcublas"] {
                assert!(
                    libraries.iter().any(|l| l == lib),
                    "{lib} not named in {err}"
                );
            }
            assert!(detail.contains("searched"), "{detail}");
            assert!(detail.contains("LD_LIBRARY_PATH="), "{detail}");
        }
        other => panic!("expected LibraryMissing, got {other}"),
    }
    let open = CudaRuntime::open(RuntimeConfig::default());
    assert!(
        matches!(open, Err(CudaError::LibraryMissing { .. })),
        "open must refuse before any cudarc call"
    );
}
