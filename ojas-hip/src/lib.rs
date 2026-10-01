//! AMD ROCm through `hip-runtime-sys`.
//!
//! With the `hip` feature off, [`HipDevice::open`] returns
//! [`DeviceError::NotCompiled`]. It does not copy the buffer on the CPU.
//! With the feature on, open calls the HIP runtime functions documented for
//! `hip-runtime-sys` 0.1.2: `hipInit`, `hipGetDeviceCount`, `hipMalloc`,
//! `hipMemcpy`, and `hipFree`. A missing device is [`DeviceError::NoDevice`].

#![cfg_attr(not(feature = "hip"), forbid(unsafe_code))]

use ojas_device::{require_kind, Device, DeviceError};

/// Proof that HIP opened. The allocation from the startup copy is freed
/// before this value is returned.
#[derive(Debug)]
pub struct HipDevice {
    devices: i32,
}

impl HipDevice {
    /// Open the HIP runtime.
    ///
    /// When `hip` is off this is [`DeviceError::NotCompiled`].
    pub fn open() -> Result<Self, DeviceError> {
        #[cfg(not(feature = "hip"))]
        {
            Err(DeviceError::NotCompiled { kind: Device::Hip })
        }
        #[cfg(feature = "hip")]
        {
            open_runtime()
        }
    }

    /// Number of HIP devices seen at open. Zero is never stored: that open fails.
    pub fn device_count(&self) -> i32 {
        self.devices
    }

    /// Copy `input` through HIP. `kind` must be [`Device::Hip`].
    pub fn copy_roundtrip(&self, kind: Device, input: &[f32]) -> Result<Vec<f32>, DeviceError> {
        require_kind(Device::Hip, kind)?;
        #[cfg(not(feature = "hip"))]
        {
            let _ = (self, input);
            Err(DeviceError::NotCompiled { kind: Device::Hip })
        }
        #[cfg(feature = "hip")]
        {
            let _ = self;
            copy_through_hip(input)
        }
    }
}

#[cfg(feature = "hip")]
fn hip_error(op: &str, status: hip_runtime_sys::hipError_t) -> Result<(), DeviceError> {
    if status == hip_runtime_sys::hipError_t::hipSuccess {
        Ok(())
    } else {
        Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("{op} returned {status:?}"),
        })
    }
}

#[cfg(feature = "hip")]
fn open_runtime() -> Result<HipDevice, DeviceError> {
    hip_error("hipInit", unsafe { hip_runtime_sys::hipInit(0) })?;
    let mut count: std::os::raw::c_int = 0;
    hip_error("hipGetDeviceCount", unsafe {
        hip_runtime_sys::hipGetDeviceCount(&mut count)
    })?;
    if count <= 0 {
        return Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("hipGetDeviceCount returned {count}"),
        });
    }
    let sample = [1.0f32, -2.0, 0.5, 4.0];
    let back = copy_through_hip(&sample)?;
    if back != sample {
        return Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("HIP copy returned {back:?}, sent {sample:?}"),
        });
    }
    Ok(HipDevice { devices: count })
}

#[cfg(feature = "hip")]
fn copy_through_hip(input: &[f32]) -> Result<Vec<f32>, DeviceError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let nbytes = std::mem::size_of_val(input);
    // Pointer type is `*mut libc::c_void` in the published bindings. Inference
    // keeps that type without naming `libc` here.
    let mut dev = std::ptr::null_mut();
    hip_error("hipMalloc", unsafe {
        hip_runtime_sys::hipMalloc(&mut dev, nbytes)
    })?;
    if dev.is_null() {
        return Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("hipMalloc({nbytes}) returned success and a null pointer"),
        });
    }
    let mut back = vec![0.0f32; input.len()];
    let copied = hip_error("hipMemcpy host to device", unsafe {
        hip_runtime_sys::hipMemcpy(
            dev,
            input.as_ptr().cast(),
            nbytes,
            hip_runtime_sys::hipMemcpyKind::hipMemcpyHostToDevice,
        )
    })
    .and_then(|()| {
        hip_error("hipMemcpy device to host", unsafe {
            hip_runtime_sys::hipMemcpy(
                back.as_mut_ptr().cast(),
                dev,
                nbytes,
                hip_runtime_sys::hipMemcpyKind::hipMemcpyDeviceToHost,
            )
        })
    });
    let freed = hip_error("hipFree", unsafe { hip_runtime_sys::hipFree(dev) });
    copied?;
    freed?;
    Ok(back)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the feature the kind check and the `NotCompiled` arm are
    /// reachable without ROCm. With the feature a missing device fails.
    fn device() -> HipDevice {
        #[cfg(not(feature = "hip"))]
        {
            HipDevice { devices: 1 }
        }
        #[cfg(feature = "hip")]
        {
            HipDevice::open().expect("hip feature on: a missing HIP device fails the test")
        }
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn default_build_reports_not_compiled() {
        match HipDevice::open() {
            Err(DeviceError::NotCompiled { kind: Device::Hip }) => {}
            Err(other) => panic!("open returned {other}"),
            Ok(_) => panic!("open succeeded without the hip feature"),
        }
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn default_build_does_not_copy_on_the_cpu() {
        let err = device()
            .copy_roundtrip(Device::Hip, &[1.0, 2.0])
            .unwrap_err();
        assert!(
            matches!(err, DeviceError::NotCompiled { kind: Device::Hip }),
            "{err}"
        );
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn default_build_refuses_hip_without_running_a_copy() {
        let caught = std::panic::catch_unwind(HipDevice::open);
        let opened = caught.expect("HipDevice::open panicked on the default build");
        match opened {
            Err(DeviceError::NotCompiled { kind: Device::Hip }) => {}
            Err(other) => panic!("open returned {other}, not NotCompiled"),
            Ok(_) => panic!("open succeeded without the hip feature"),
        }
        let err = device()
            .copy_roundtrip(Device::Cuda, &[1.0, 2.0, 3.0])
            .unwrap_err();
        assert!(
            matches!(
                err,
                DeviceError::DeviceMismatch {
                    expected: Device::Hip,
                    actual: Device::Cuda,
                }
            ),
            "{err}"
        );
        let same = device()
            .copy_roundtrip(Device::Hip, &[4.0, 5.0])
            .unwrap_err();
        assert!(
            matches!(same, DeviceError::NotCompiled { kind: Device::Hip }),
            "{same}"
        );
    }

    #[test]
    fn hip_is_not_accepted_as_cpu() {
        let err = device().copy_roundtrip(Device::Cpu, &[1.0]).unwrap_err();
        assert!(matches!(err, DeviceError::DeviceMismatch { .. }));
    }

    #[cfg(feature = "hip")]
    #[test]
    fn roundtrip_is_exact_at_odd_lengths() {
        let device = device();
        assert!(device.copy_roundtrip(Device::Hip, &[]).unwrap().is_empty());
        for n in [1usize, 63, 65, 1023, 1025, 100_003] {
            let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 7.0).collect();
            assert_eq!(
                device.copy_roundtrip(Device::Hip, &input).unwrap(),
                input,
                "len {n}"
            );
        }
    }
}
