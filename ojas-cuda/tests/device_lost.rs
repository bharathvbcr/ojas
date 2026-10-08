//! A lost CUDA context surfaces as the typed `OjasError::DeviceLost`, and
//! stays lost. A one-thread kernel runs `__trap()`, which leaves the
//! context in a sticky error state (a `STICKY_DRIVER_CODES` code). The
//! backend's next sync, a later upload, and a second sync must each be
//! `DeviceLost { backend: Cuda }`, not `Backend` or a timeout.
//!
//! Needs an sm_90 NVIDIA GPU, so the test is `#[ignore]`: on the Mac it
//! compiles and is reported NOT RUN. The trap poisons the whole process's
//! context, so this file holds one test and its binary runs alone (README,
//! "Device tests on the box"):
//! `LD_LIBRARY_PATH=<cublas>:<nvrtc> ./device_lost-<hash> --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

use std::rc::Rc;
use std::time::Duration;

use cudarc::driver::LaunchConfig;
use ojas_core::{Backend, BackendId, Budget, OjasError, Tensor};
use ojas_cuda::error::STICKY_DRIVER_CODES;
use ojas_cuda::kernels::{KernelModule, STRICT_SM90};
use ojas_cuda::runtime::driver_error;
use ojas_cuda::{CudaBackend, CudaError, CudaRuntime, RuntimeConfig};

static TRAP: KernelModule = KernelModule {
    name: "device_lost_trap",
    source: "extern \"C\" __global__ void qd_trap() { __trap(); }\n",
    entries: &["qd_trap"],
};

fn lost(what: &str, r: Result<impl std::fmt::Debug, OjasError>) {
    match r {
        Err(OjasError::DeviceLost {
            backend: BackendId::Cuda,
            detail,
        }) => eprintln!("{what}: DeviceLost: {detail}"),
        other => panic!("{what}: expected DeviceLost {{ backend: Cuda }}, got {other:?}"),
    }
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn a_trapped_kernel_leaves_the_backend_with_the_typed_device_lost_error() {
    // A short cap, so a fault that never surfaces ends as a Timeout (a
    // clear failure) rather than a 60 s wait.
    let config = RuntimeConfig {
        sync_timeout: Duration::from_secs(10),
        ..RuntimeConfig::default()
    };
    let rt = Rc::new(CudaRuntime::open(config).unwrap_or_else(|e| panic!("open: {e}")));
    let backend = CudaBackend::with_runtime(Rc::clone(&rt), Budget::new(1 << 30));
    let host = Budget::new(1 << 20);
    let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], &host).unwrap();

    // The context works before the trap.
    backend.upload(&x).expect("upload before the trap");
    backend.sync().expect("sync before the trap");

    let f = rt
        .function(&TRAP, &STRICT_SM90, "qd_trap")
        .unwrap_or_else(|e| panic!("compile qd_trap: {e}"));
    let mut b = rt.stream().launch_builder(&f);
    // SAFETY: qd_trap takes no arguments and touches no memory; it traps,
    // which is the fault under test.
    let launched = unsafe { b.launch(LaunchConfig::for_num_elems(1)) }
        .map_err(|e| driver_error("launch qd_trap", e));
    // The launch itself is asynchronous; the trap is reported by the next
    // wait. Some drivers report it at launch already, which must be sticky.
    if let Err(e) = &launched {
        assert!(
            e.is_device_lost(),
            "launch failed with a non-sticky error: {e}"
        );
    }

    lost("sync after the trap", backend.sync());

    // The raw driver code is one the crate lists as sticky.
    match rt.sync("probe") {
        Err(CudaError::Driver { code, .. }) => assert!(
            STICKY_DRIVER_CODES.contains(&code),
            "driver code {code} is not in STICKY_DRIVER_CODES"
        ),
        other => panic!("raw sync after the trap: expected a driver error, got {other:?}"),
    }

    // Loss is permanent: later calls fail the same way.
    lost("upload after the trap", backend.upload(&x));
    lost("second sync after the trap", backend.sync());
}
