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
                "ojas-qwen35-cuda was built without the `cuda` feature; no device path exists"
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
