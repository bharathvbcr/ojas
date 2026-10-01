//! Device kinds for ojas.
//!
//! Callers name the device. This crate does not open a GPU runtime and does
//! not substitute [`Device::Cpu`] when a GPU kind was requested.
//! GPU probes live in the crates that link those runtimes.

#![forbid(unsafe_code)]

use std::fmt;

/// Where a call is allowed to run.
///
/// `Metal` is the Apple training path (tessl). `Vulkan` is the portable GPU
/// path (wgpu). On Apple that path uses wgpu's Metal HAL; it is still
/// `Vulkan` here so it does not replace tessl.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
    Hip,
    Vulkan,
}

/// A device a compiled-in probe actually saw.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub backend: Device,
}

/// Failure to use the device the caller named.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceError {
    /// The runtime for `kind` was not compiled into this build.
    NotCompiled { kind: Device },
    /// The runtime is compiled, and no usable device answered.
    NoDevice { kind: Device, detail: String },
    /// The caller asked for one kind and the call is bound to another.
    DeviceMismatch { expected: Device, actual: Device },
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::NotCompiled { kind } => {
                write!(f, "{kind:?} runtime is not compiled into this build")
            }
            DeviceError::NoDevice { kind, detail } => {
                write!(f, "no {kind:?} device: {detail}")
            }
            DeviceError::DeviceMismatch { expected, actual } => {
                write!(
                    f,
                    "device mismatch: expected {expected:?}, actual {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for DeviceError {}

/// Refuse a call whose kind is not the kind of the runtime that will execute it.
pub fn require_kind(expected: Device, actual: Device) -> Result<(), DeviceError> {
    if expected == actual {
        Ok(())
    } else {
        Err(DeviceError::DeviceMismatch { expected, actual })
    }
}

/// Devices this crate can see without linking a GPU runtime.
///
/// That set is the host CPU only. CUDA, HIP, Vulkan, and Metal are probed
/// by the crates that link those runtimes. An empty GPU list is not rewritten
/// into a CPU device.
pub fn probe() -> Vec<DeviceInfo> {
    vec![DeviceInfo {
        name: "host".to_string(),
        vendor: std::env::consts::ARCH.to_string(),
        backend: Device::Cpu,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_lists_cpu_and_nothing_else() {
        let found = probe();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].backend, Device::Cpu);
        assert!(found.iter().all(|d| d.backend != Device::Cuda));
        assert!(found.iter().all(|d| d.backend != Device::Hip));
        assert!(found.iter().all(|d| d.backend != Device::Vulkan));
        assert!(found.iter().all(|d| d.backend != Device::Metal));
    }

    #[test]
    fn mismatch_is_not_success() {
        let err = require_kind(Device::Vulkan, Device::Cpu).unwrap_err();
        assert_eq!(
            err,
            DeviceError::DeviceMismatch {
                expected: Device::Vulkan,
                actual: Device::Cpu,
            }
        );
    }

    #[test]
    fn not_compiled_displays() {
        let err = DeviceError::NotCompiled { kind: Device::Cuda };
        let text = err.to_string();
        assert!(text.contains("Cuda"));
        assert!(text.contains("not compiled"));
    }

    const ALL: [Device; 5] = [
        Device::Cpu,
        Device::Metal,
        Device::Cuda,
        Device::Hip,
        Device::Vulkan,
    ];

    #[test]
    fn require_kind_accepts_only_the_same_kind() {
        for expected in ALL {
            for actual in ALL {
                let got = require_kind(expected, actual);
                if expected == actual {
                    assert_eq!(got, Ok(()));
                } else {
                    assert_eq!(got, Err(DeviceError::DeviceMismatch { expected, actual }));
                }
            }
        }
    }

    #[test]
    fn errors_name_the_requested_kind() {
        for kind in ALL {
            let name = format!("{kind:?}");
            let no_device = DeviceError::NoDevice {
                kind,
                detail: "probe returned 0 adapters".to_string(),
            };
            let text = no_device.to_string();
            assert!(
                text.contains(&name) && text.contains("0 adapters"),
                "{text}"
            );
            assert!(DeviceError::NotCompiled { kind }
                .to_string()
                .contains(&name));
        }
    }

    #[test]
    fn cuda_and_hip_mismatch_is_constructible_and_not_success() {
        for (expected, actual) in [
            (Device::Cuda, Device::Hip),
            (Device::Hip, Device::Cuda),
            (Device::Cuda, Device::Vulkan),
            (Device::Hip, Device::Vulkan),
            (Device::Vulkan, Device::Cuda),
        ] {
            let err = require_kind(expected, actual).unwrap_err();
            assert_eq!(err, DeviceError::DeviceMismatch { expected, actual });
            let not_compiled = DeviceError::NotCompiled { kind: expected };
            assert_ne!(err, not_compiled);
            assert!(not_compiled.to_string().contains("not compiled"));
        }
        assert!(probe().iter().all(|d| d.backend == Device::Cpu));
    }
}
