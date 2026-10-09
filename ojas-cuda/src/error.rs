//! The crate's one error type.
//!
//! Every fallible call returns [`CudaError`]. A variant names what failed and
//! carries enough detail to act on without a debugger: the library and its
//! searched names, the driver op and its code, the NVRTC log, the cuBLAS
//! status name.

use std::fmt;

/// Everything that can go wrong in this crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaError {
    /// The crate was built without the `cuda` feature, so no device path exists.
    NotCompiled,
    /// One or more shared libraries cudarc would `dlopen` are not loadable.
    /// cudarc panics on a missing library, so this is found by probing first.
    LibraryMissing {
        /// The libraries that did not load, by name (`libcuda`, `libnvrtc`, `libcublas`).
        libraries: Vec<String>,
        /// The names searched for each, and the `LD_LIBRARY_PATH` the process started with.
        detail: String,
    },
    /// The device is not the architecture this crate targets.
    Arch {
        /// `(major, minor)` compute capability the device reports.
        found: (i32, i32),
        /// `(major, minor)` this crate requires.
        required: (i32, i32),
        /// The device name the driver reports.
        name: String,
    },
    /// A CUDA driver call failed with a status other than out-of-memory.
    Driver {
        /// The call, e.g. `cuMemAlloc` or `launch qd_zero_f32`.
        op: String,
        /// The raw `CUresult` value.
        code: u32,
        /// The driver error's own description.
        detail: String,
    },
    /// Out of device memory, or over this runtime's allocation budget.
    Capacity {
        /// What was being allocated.
        op: String,
        /// Requested and available bytes.
        detail: String,
    },
    /// NVRTC failed to compile, or the driver failed to load the result.
    Compile {
        /// The kernel module.
        module: String,
        /// The NVRTC log or the load error.
        detail: String,
    },
    /// cuBLAS returned a status other than `CUBLAS_STATUS_SUCCESS`.
    Cublas {
        /// The cuBLAS call.
        op: String,
        /// The status name, e.g. `CUBLAS_STATUS_NOT_SUPPORTED`.
        status: String,
        /// The raw `cublasStatus_t` value.
        code: u32,
    },
    /// Arguments failed validation before any device work was queued.
    Invalid {
        /// The operation being validated.
        op: String,
        /// What was wrong.
        detail: String,
    },
    /// A bounded wait ran out before the device finished.
    Timeout {
        /// What was being waited for.
        op: String,
        /// How long the wait lasted, in milliseconds.
        waited_ms: u128,
    },
}

impl CudaError {
    /// Shorthand for [`CudaError::Invalid`].
    pub fn invalid(op: impl Into<String>, detail: impl Into<String>) -> Self {
        CudaError::Invalid {
            op: op.into(),
            detail: detail.into(),
        }
    }

    /// Shorthand for [`CudaError::Capacity`].
    pub fn capacity(op: impl Into<String>, detail: impl Into<String>) -> Self {
        CudaError::Capacity {
            op: op.into(),
            detail: detail.into(),
        }
    }

    /// `CUDA_ERROR_OUT_OF_MEMORY` is 2: that status is [`CudaError::Capacity`],
    /// every other failing status is [`CudaError::Driver`]. The same split as
    /// `ojas-cuda/src/lib.rs` `cuda_status`.
    pub fn from_driver_code(op: impl Into<String>, code: u32, detail: impl Into<String>) -> Self {
        if code == CUDA_ERROR_OUT_OF_MEMORY {
            CudaError::Capacity {
                op: op.into(),
                detail: detail.into(),
            }
        } else {
            CudaError::Driver {
                op: op.into(),
                code,
                detail: detail.into(),
            }
        }
    }

    /// A short, stable name for the variant, for reports.
    pub fn kind(&self) -> &'static str {
        match self {
            CudaError::NotCompiled => "not_compiled",
            CudaError::LibraryMissing { .. } => "library_missing",
            CudaError::Arch { .. } => "arch",
            CudaError::Driver { .. } => "driver",
            CudaError::Capacity { .. } => "capacity",
            CudaError::Compile { .. } => "compile",
            CudaError::Cublas { .. } => "cublas",
            CudaError::Invalid { .. } => "invalid",
            CudaError::Timeout { .. } => "timeout",
        }
    }
}

/// The CUDA driver's `CUDA_ERROR_OUT_OF_MEMORY`.
pub const CUDA_ERROR_OUT_OF_MEMORY: u32 = 2;

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::NotCompiled => write!(
                f,
                "ojas-cuda was built without the `cuda` feature; no device path exists"
            ),
            CudaError::LibraryMissing { libraries, detail } => write!(
                f,
                "refusing to start: CUDA libraries not loadable: {}. {detail}",
                libraries.join(", ")
            ),
            CudaError::Arch {
                found,
                required,
                name,
            } => write!(
                f,
                "device {name:?} is compute capability {}.{}; this crate requires {}.{}",
                found.0, found.1, required.0, required.1
            ),
            CudaError::Driver { op, code, detail } => {
                write!(f, "CUDA driver call {op} failed with code {code}: {detail}")
            }
            CudaError::Capacity { op, detail } => write!(f, "{op}: out of memory: {detail}"),
            CudaError::Compile { module, detail } => {
                write!(f, "compiling kernel module {module} failed: {detail}")
            }
            CudaError::Cublas { op, status, code } => {
                write!(f, "cuBLAS {op} returned {status} ({code})")
            }
            CudaError::Invalid { op, detail } => write!(f, "{op}: {detail}"),
            CudaError::Timeout { op, waited_ms } => {
                write!(f, "{op}: the device did not finish within {waited_ms} ms")
            }
        }
    }
}

impl std::error::Error for CudaError {}

/// `CUresult` codes after which the CUDA context is unusable: every later
/// call fails the same way, and only a new process recovers. The list is
/// what NVIDIA's `cuda.h` text says, not a guess: every code documented
/// "This leaves the process in an inconsistent state and any further CUDA
/// work will return the same error" (CUDA 12.9 and 13.0 headers, as
/// NVIDIA/cuda-python's `driver_cu_result_explanations.py` carries them),
/// plus `CUDA_ERROR_ASSERT`, documented "The context cannot be used anymore".
/// Codes whose text says nothing about it (`ECC_UNCORRECTABLE` 214,
/// `UNKNOWN` 999, ...) stay [`OjasError::Backend`](ojas_core::OjasError).
/// Values from cudarc 0.19.10's `CUresult`; 226 is named only in its newer
/// header versions, and a newer driver may return it.
pub const STICKY_DRIVER_CODES: &[u32] = &[
    226, // CUDA_ERROR_CONTAINED
    700, // CUDA_ERROR_ILLEGAL_ADDRESS
    702, // CUDA_ERROR_LAUNCH_TIMEOUT
    710, // CUDA_ERROR_ASSERT
    714, // CUDA_ERROR_HARDWARE_STACK_ERROR
    715, // CUDA_ERROR_ILLEGAL_INSTRUCTION
    716, // CUDA_ERROR_MISALIGNED_ADDRESS
    717, // CUDA_ERROR_INVALID_ADDRESS_SPACE
    718, // CUDA_ERROR_INVALID_PC
    719, // CUDA_ERROR_LAUNCH_FAILED
    911, // CUDA_ERROR_EXTERNAL_DEVICE
];

impl CudaError {
    /// A driver failure that leaves the context unusable
    /// ([`STICKY_DRIVER_CODES`]).
    pub fn is_device_lost(&self) -> bool {
        matches!(self, CudaError::Driver { code, .. } if STICKY_DRIVER_CODES.contains(code))
    }
}

/// The one mapping from this crate's errors to ojas's: a sticky driver
/// failure is [`OjasError::DeviceLost`](ojas_core::OjasError::DeviceLost),
/// so a caller stops instead of retrying on a dead context; everything else
/// is [`OjasError::Backend`](ojas_core::OjasError::Backend) with this error's
/// text.
impl From<CudaError> for ojas_core::OjasError {
    fn from(e: CudaError) -> Self {
        if e.is_device_lost() {
            ojas_core::OjasError::DeviceLost {
                backend: ojas_core::BackendId::Cuda,
                detail: e.to_string(),
            }
        } else {
            ojas_core::OjasError::Backend {
                id: ojas_core::BackendId::Cuda,
                detail: e.to_string(),
            }
        }
    }
}

/// `CUresult` codes that mean no usable device answered: the driver is not
/// (or no longer) initialised (3, 4), the loaded `libcuda` is the toolkit's
/// stub (34), the device is busy in exclusive mode (46), or there is no such
/// device (100, 101). Values from cudarc 0.19.10's `CUresult`. Every other
/// failing code came from a device that answered.
pub const NO_DEVICE_DRIVER_CODES: &[u32] = &[3, 4, 34, 46, 100, 101];

/// The one mapping from this crate's errors to a device open's
/// [`ojas_device::DeviceError`], so an open names what failed instead of
/// reporting every failure as `NoDevice`:
///
/// | `CudaError` | `DeviceError` |
/// |---|---|
/// | `NotCompiled` | `NotCompiled` |
/// | `LibraryMissing`, `Driver` with a [`NO_DEVICE_DRIVER_CODES`] code | `NoDevice` |
/// | `Arch` | `Unsupported` |
/// | `Capacity` (the budget, the cuBLAS workspace, device memory) | `Capacity` |
/// | `Compile` (NVRTC) | `Compile` |
/// | `Cublas`, `Invalid`, `Timeout`, any other `Driver` | `Init` |
impl From<CudaError> for ojas_device::DeviceError {
    fn from(e: CudaError) -> Self {
        use ojas_device::{Device, DeviceError};
        let kind = Device::Cuda;
        let detail = e.to_string();
        match e {
            CudaError::NotCompiled => DeviceError::NotCompiled { kind },
            CudaError::LibraryMissing { .. } => DeviceError::NoDevice { kind, detail },
            CudaError::Driver { code, .. } if NO_DEVICE_DRIVER_CODES.contains(&code) => {
                DeviceError::NoDevice { kind, detail }
            }
            CudaError::Arch { .. } => DeviceError::Unsupported { kind, detail },
            CudaError::Capacity { .. } => DeviceError::Capacity { kind, detail },
            CudaError::Compile { .. } => DeviceError::Compile { kind, detail },
            CudaError::Driver { .. }
            | CudaError::Cublas { .. }
            | CudaError::Invalid { .. }
            | CudaError::Timeout { .. } => DeviceError::Init { kind, detail },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::{BackendId, OjasError};

    /// A sticky driver code (the context is gone) is `DeviceLost`; any other
    /// failure, out-of-memory and a non-sticky driver code included, stays
    /// `Backend`. Before this mapping every CUDA failure was `Backend`.
    #[test]
    fn sticky_driver_codes_map_to_device_lost_and_the_rest_to_backend() {
        for &code in STICKY_DRIVER_CODES {
            let e = CudaError::from_driver_code("cuStreamSynchronize", code, "sticky");
            assert!(e.is_device_lost(), "{code}");
            match OjasError::from(e) {
                OjasError::DeviceLost { backend, detail } => {
                    assert_eq!(backend, BackendId::Cuda);
                    assert!(detail.contains(&format!("code {code}")), "{detail}");
                }
                other => panic!("code {code} mapped to {other}"),
            }
        }
        for e in [
            CudaError::from_driver_code("cuMemAlloc", CUDA_ERROR_OUT_OF_MEMORY, "oom"),
            // CUDA_ERROR_INVALID_VALUE: a bad argument, the context is fine.
            CudaError::from_driver_code("cuMemcpyHtoD", 1, "invalid value"),
            // Documented without the sticky sentence: not assumed lost.
            CudaError::from_driver_code("cuMemcpyDtoH", 214, "ecc uncorrectable"),
            CudaError::from_driver_code("cuCtxSynchronize", 999, "unknown"),
            CudaError::Timeout {
                op: "sync".to_string(),
                waited_ms: 5,
            },
        ] {
            assert!(!e.is_device_lost(), "{e}");
            assert!(
                matches!(
                    OjasError::from(e),
                    OjasError::Backend {
                        id: BackendId::Cuda,
                        ..
                    }
                ),
                "not Backend"
            );
        }
    }

    #[test]
    fn out_of_memory_maps_to_capacity_and_other_codes_to_driver() {
        let oom = CudaError::from_driver_code("cuMemAlloc", 2, "CUDA_ERROR_OUT_OF_MEMORY");
        assert_eq!(oom.kind(), "capacity", "{oom}");
        let other = CudaError::from_driver_code("cuLaunchKernel", 700, "illegal address");
        assert_eq!(
            other,
            CudaError::Driver {
                op: "cuLaunchKernel".to_string(),
                code: 700,
                detail: "illegal address".to_string()
            }
        );
    }

    #[test]
    fn library_missing_names_every_library() {
        let err = CudaError::LibraryMissing {
            libraries: vec!["libnvrtc".to_string(), "libcublas".to_string()],
            detail: "LD_LIBRARY_PATH unset".to_string(),
        };
        let text = err.to_string();
        assert!(text.contains("libnvrtc, libcublas"), "{text}");
        assert!(text.starts_with("refusing to start"), "{text}");
    }

    /// Before this mapping `CudaBackend::open` reported every failure as
    /// `NoDevice`. Each variant now lands on its own kind, and the text is kept.
    #[test]
    fn open_failures_map_to_distinct_device_error_kinds() {
        use ojas_device::{Device, DeviceError};
        let cases: Vec<(CudaError, &str)> = vec![
            (CudaError::NotCompiled, "not_compiled"),
            (
                CudaError::LibraryMissing {
                    libraries: vec!["libcublas".to_string()],
                    detail: "searched libcublas.so.12".to_string(),
                },
                "no_device",
            ),
            (
                CudaError::from_driver_code("CudaContext::new(0)", 100, "no device"),
                "no_device",
            ),
            (
                CudaError::from_driver_code("CudaContext::new(0)", 34, "stub"),
                "no_device",
            ),
            (
                CudaError::Arch {
                    found: (8, 0),
                    required: (9, 0),
                    name: "A100".to_string(),
                },
                "unsupported",
            ),
            (
                CudaError::capacity("cuBLAS workspace", "33554432 bytes requested"),
                "capacity",
            ),
            (
                CudaError::from_driver_code("cuMemAlloc", CUDA_ERROR_OUT_OF_MEMORY, "oom"),
                "capacity",
            ),
            (
                CudaError::Compile {
                    module: "nvrtcVersion".to_string(),
                    detail: "symbol missing".to_string(),
                },
                "compile",
            ),
            (
                CudaError::Cublas {
                    op: "cublasCreate".to_string(),
                    status: "CUBLAS_STATUS_NOT_INITIALIZED".to_string(),
                    code: 1,
                },
                "init",
            ),
            (
                CudaError::from_driver_code("cuStreamCreate", 999, "unknown"),
                "init",
            ),
            (CudaError::invalid("NvrtcCache::new", "0 entries"), "init"),
            (
                CudaError::Timeout {
                    op: "open".to_string(),
                    waited_ms: 60_000,
                },
                "init",
            ),
        ];
        for (err, want) in cases {
            let text = err.to_string();
            let got = DeviceError::from(err);
            let (kind, detail) = match &got {
                DeviceError::NotCompiled { kind } => ("not_compiled", (*kind, None)),
                DeviceError::NoDevice { kind, detail } => ("no_device", (*kind, Some(detail))),
                DeviceError::Unsupported { kind, detail } => ("unsupported", (*kind, Some(detail))),
                DeviceError::Capacity { kind, detail } => ("capacity", (*kind, Some(detail))),
                DeviceError::Compile { kind, detail } => ("compile", (*kind, Some(detail))),
                DeviceError::Init { kind, detail } => ("init", (*kind, Some(detail))),
                other => panic!("{text} mapped to {other}"),
            };
            assert_eq!(kind, want, "{text} -> {got}");
            assert_eq!(detail.0, Device::Cuda, "{got}");
            if let Some(detail) = detail.1 {
                assert_eq!(detail, &text, "{got}");
            }
        }
    }

    /// Every driver code from 0 to 1023, and the extremes, maps without a
    /// panic onto exactly one kind: out-of-memory is `Capacity`, the
    /// no-device codes `NoDevice`, everything else from a device that
    /// answered `Init`. The text always survives.
    #[test]
    fn stress_every_driver_code_maps_to_one_kind_at_open() {
        use ojas_device::{Device, DeviceError};
        for code in (0..1024u32).chain([u32::MAX - 1, u32::MAX]) {
            let err =
                CudaError::from_driver_code("CudaContext::new(0)", code, format!("code {code}"));
            let text = err.to_string();
            match DeviceError::from(err) {
                DeviceError::Capacity { kind, detail } => {
                    assert_eq!(code, CUDA_ERROR_OUT_OF_MEMORY);
                    assert_eq!((kind, detail), (Device::Cuda, text));
                }
                DeviceError::NoDevice { kind, detail } => {
                    assert!(NO_DEVICE_DRIVER_CODES.contains(&code), "{code}");
                    assert_eq!((kind, detail), (Device::Cuda, text));
                }
                DeviceError::Init { kind, detail } => {
                    assert!(
                        code != CUDA_ERROR_OUT_OF_MEMORY && !NO_DEVICE_DRIVER_CODES.contains(&code),
                        "{code}"
                    );
                    assert_eq!((kind, detail), (Device::Cuda, text));
                }
                other => panic!("code {code} mapped to {other}"),
            }
        }
        // The no-device list and the sticky (device-lost) list are disjoint:
        // a lost context is never reported as a missing device.
        for code in NO_DEVICE_DRIVER_CODES {
            assert!(!STICKY_DRIVER_CODES.contains(code), "{code}");
            assert_ne!(*code, CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
}
