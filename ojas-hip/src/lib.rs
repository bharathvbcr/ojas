//! AMD ROCm through `hip-runtime-sys`.
//!
//! With the `hip` feature off, [`HipDevice::open`] returns
//! [`DeviceError::NotCompiled`]. It does not copy the buffer on the CPU.
//! With the feature on, open calls the HIP runtime functions documented for
//! `hip-runtime-sys` 0.1.2: `hipInit`, `hipGetDeviceCount`, `hipMalloc`,
//! `hipMemcpyAsync`, and `hipFree`. A missing device is [`DeviceError::NoDevice`].
//! The feature stays off by default: that crate's build script panics when
//! the HIP headers are missing.

#![cfg_attr(not(feature = "hip"), forbid(unsafe_code))]

use ojas_device::{require_kind, Device, DeviceError};

/// Proof that HIP opened. The allocation from the startup copy is freed
/// before this value is returned.
#[derive(Debug)]
pub struct HipDevice {
    devices: i32,
}

/// Copy larger than this is refused before `hipMalloc`. 1 GiB.
pub const MAX_COPY_BYTES: usize = 1 << 30;

/// Byte length of an f32 copy, or [`DeviceError::Capacity`] when it does not fit.
pub fn copy_bytes(elems: usize) -> Result<usize, DeviceError> {
    let bytes = elems
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| DeviceError::Capacity {
            kind: Device::Hip,
            detail: format!("{elems} f32 values overflow the byte length"),
        })?;
    if bytes > MAX_COPY_BYTES {
        return Err(DeviceError::Capacity {
            kind: Device::Hip,
            detail: format!("{bytes} bytes exceeds the {MAX_COPY_BYTES} byte ceiling"),
        });
    }
    Ok(bytes)
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

/// `hipErrorOutOfMemory` is 2. Pre-fix, every non-success status, including
/// that one, became [`DeviceError::NoDevice`].
#[cfg_attr(not(feature = "hip"), allow(dead_code))]
fn hip_status(op: &str, code: u32) -> Result<(), DeviceError> {
    if code == 0 {
        Ok(())
    } else if code == 2 {
        Err(DeviceError::Capacity {
            kind: Device::Hip,
            detail: format!("{op} ran out of memory"),
        })
    } else {
        Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("{op} returned status {code}"),
        })
    }
}

#[cfg_attr(not(feature = "hip"), allow(dead_code))]
fn hip_free_covers(free: usize, need: usize) -> Result<(), DeviceError> {
    if need > free {
        Err(DeviceError::Capacity {
            kind: Device::Hip,
            detail: format!("hipMemGetInfo free {free} bytes, copy needs {need}"),
        })
    } else {
        Ok(())
    }
}

#[cfg_attr(not(feature = "hip"), allow(dead_code))]
fn poll_until(
    deadline: std::time::Instant,
    mut ready: impl FnMut() -> Result<bool, DeviceError>,
) -> Result<(), DeviceError> {
    loop {
        if ready()? {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(DeviceError::Launch {
                kind: Device::Hip,
                detail: "HIP copy did not finish within 30s".to_string(),
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[cfg(feature = "hip")]
fn hip_error(op: &str, status: hip_runtime_sys::hipError_t) -> Result<(), DeviceError> {
    hip_status(op, status as u32)
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
    let mut free = 0usize;
    let mut total = 0usize;
    hip_error("hipMemGetInfo", unsafe {
        hip_runtime_sys::hipMemGetInfo(&mut free, &mut total)
    })?;
    if total == 0 {
        return Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("hipMemGetInfo returned total 0, free {free}"),
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
    let nbytes = copy_bytes(input.len())?;
    let mut free = 0usize;
    let mut total = 0usize;
    hip_error("hipMemGetInfo", unsafe {
        hip_runtime_sys::hipMemGetInfo(&mut free, &mut total)
    })?;
    let _ = total;
    hip_free_covers(free, nbytes)?;
    // Freed on every return, including a panic in `vec!`.
    struct Dev(*mut std::ffi::c_void);
    impl Drop for Dev {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let _ = hip_runtime_sys::hipFree(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    struct Host(*mut std::ffi::c_void);
    impl Drop for Host {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let _ = hip_runtime_sys::hipHostFree(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    struct Stream(hip_runtime_sys::hipStream_t);
    impl Drop for Stream {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let _ = hip_runtime_sys::hipStreamDestroy(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    struct Event(hip_runtime_sys::hipEvent_t);
    impl Drop for Event {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let _ = hip_runtime_sys::hipEventDestroy(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    let mut dev = std::ptr::null_mut();
    hip_error("hipMalloc", unsafe {
        hip_runtime_sys::hipMalloc(&mut dev, nbytes)
    })?;
    if dev.is_null() {
        return Err(DeviceError::Launch {
            kind: Device::Hip,
            detail: format!("hipMalloc({nbytes}) returned success and a null pointer"),
        });
    }
    let dev = Dev(dev);
    let mut host = std::ptr::null_mut();
    hip_error("hipHostMalloc", unsafe {
        hip_runtime_sys::hipHostMalloc(&mut host, nbytes, hip_runtime_sys::hipHostMallocDefault)
    })?;
    if host.is_null() {
        return Err(DeviceError::Capacity {
            kind: Device::Hip,
            detail: format!("hipHostMalloc({nbytes}) returned success and a null pointer"),
        });
    }
    let host = Host(host);
    unsafe {
        std::ptr::copy_nonoverlapping(input.as_ptr(), host.0.cast::<f32>(), input.len());
    }
    let mut stream = std::ptr::null_mut();
    hip_error("hipStreamCreateWithFlags", unsafe {
        hip_runtime_sys::hipStreamCreateWithFlags(
            &mut stream,
            hip_runtime_sys::hipStreamNonBlocking,
        )
    })?;
    let stream = Stream(stream);
    let mut event = std::ptr::null_mut();
    hip_error("hipEventCreateWithFlags", unsafe {
        hip_runtime_sys::hipEventCreateWithFlags(&mut event, hip_runtime_sys::hipEventDisableTiming)
    })?;
    let event = Event(event);
    hip_error("hipMemcpyAsync host to device", unsafe {
        hip_runtime_sys::hipMemcpyAsync(
            dev.0,
            host.0,
            nbytes,
            hip_runtime_sys::hipMemcpyKind::hipMemcpyHostToDevice,
            stream.0,
        )
    })?;
    hip_error("hipMemcpyAsync device to host", unsafe {
        hip_runtime_sys::hipMemcpyAsync(
            host.0,
            dev.0,
            nbytes,
            hip_runtime_sys::hipMemcpyKind::hipMemcpyDeviceToHost,
            stream.0,
        )
    })?;
    hip_error("hipEventRecord", unsafe {
        hip_runtime_sys::hipEventRecord(event.0, stream.0)
    })?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    poll_until(deadline, || {
        let status = unsafe { hip_runtime_sys::hipEventQuery(event.0) };
        if status == hip_runtime_sys::hipError_t::hipSuccess {
            Ok(true)
        } else if status == hip_runtime_sys::hipError_t::hipErrorNotReady {
            Ok(false)
        } else {
            hip_error("hipEventQuery", status).map(|_| false)
        }
    })?;
    let mut back = vec![0.0f32; input.len()];
    unsafe {
        std::ptr::copy_nonoverlapping(host.0.cast::<f32>(), back.as_mut_ptr(), input.len());
    }
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

    #[test]
    fn out_of_memory_is_capacity_not_no_device() {
        let oom = hip_status("hipMalloc", 2).unwrap_err();
        assert!(
            matches!(
                oom,
                DeviceError::Capacity {
                    kind: Device::Hip,
                    ..
                }
            ),
            "{oom}"
        );
        let other = hip_status("hipInit", 3).unwrap_err();
        assert!(matches!(other, DeviceError::NoDevice { .. }), "{other}");
        assert!(hip_status("hipInit", 0).is_ok());
    }

    #[test]
    fn memory_probe_refuses_a_copy_past_free_bytes() {
        let err = hip_free_covers(8, 16).unwrap_err();
        assert!(
            matches!(
                err,
                DeviceError::Capacity {
                    kind: Device::Hip,
                    ..
                }
            ),
            "{err}"
        );
        assert!(hip_free_covers(16, 16).is_ok());
    }

    #[test]
    fn copy_wait_returns_when_the_deadline_has_passed() {
        let err = poll_until(std::time::Instant::now(), || Ok(false)).unwrap_err();
        assert!(
            matches!(
                err,
                DeviceError::Launch {
                    kind: Device::Hip,
                    ..
                }
            ),
            "{err}"
        );
        assert!(poll_until(std::time::Instant::now(), || Ok(true)).is_ok());
    }

    #[test]
    fn copy_ceiling_refuses_without_calling_hip() {
        assert_eq!(copy_bytes(0).unwrap(), 0);
        assert!(matches!(
            copy_bytes(usize::MAX),
            Err(DeviceError::Capacity {
                kind: Device::Hip,
                ..
            })
        ));
        let too_big = (MAX_COPY_BYTES / 4) + 1;
        let err = copy_bytes(too_big).unwrap_err();
        assert!(
            matches!(
                err,
                DeviceError::Capacity {
                    kind: Device::Hip,
                    ..
                }
            ),
            "{err}"
        );
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
