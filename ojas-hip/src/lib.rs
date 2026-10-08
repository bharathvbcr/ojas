//! AMD ROCm through `hip-runtime-sys`.
//!
//! With the `hip` feature off, [`HipDevice::open`] returns
//! [`DeviceError::NotCompiled`]. It does not copy the buffer on the CPU.
//! With the feature on, open calls the HIP runtime functions documented for
//! `hip-runtime-sys` 0.1.2: `hipInit`, `hipGetDeviceCount`, `hipMalloc`,
//! `hipMemcpyAsync`, and `hipFree`. A missing device is [`DeviceError::NoDevice`];
//! out of memory is [`DeviceError::Capacity`]; any other failure of a present
//! device (a copy, a stream, an event) is [`DeviceError::Launch`].
//! The feature stays off by default: that crate's build script panics when
//! the HIP headers are missing.
//!
//! This crate is a copy probe today. A full HIP `Backend` is planned in
//! `tasks/gp-hip-backend.md`.

#![cfg_attr(not(feature = "hip"), forbid(unsafe_code))]

use ojas_device::{require_kind, Device, DeviceError};

/// Proof that HIP opened. The allocation from the startup copy is freed
/// before this value is returned.
#[derive(Debug)]
pub struct HipDevice {
    devices: i32,
    ordinal: i32,
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
    /// Open HIP device 0. See [`HipDevice::open_ordinal`].
    pub fn open() -> Result<Self, DeviceError> {
        Self::open_ordinal(0)
    }

    /// Open HIP device `ordinal`.
    ///
    /// When `hip` is off this is [`DeviceError::NotCompiled`]. An ordinal
    /// outside `0..hipGetDeviceCount` is [`DeviceError::NoDevice`].
    pub fn open_ordinal(ordinal: i32) -> Result<Self, DeviceError> {
        #[cfg(not(feature = "hip"))]
        {
            let _ = ordinal;
            Err(DeviceError::NotCompiled { kind: Device::Hip })
        }
        #[cfg(feature = "hip")]
        {
            open_runtime(ordinal)
        }
    }

    /// Number of HIP devices seen at open. Zero is never stored: that open fails.
    pub fn device_count(&self) -> i32 {
        self.devices
    }

    /// The device this value opened and copies through.
    pub fn ordinal(&self) -> i32 {
        self.ordinal
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
            copy_through_hip(self.ordinal, input)
        }
    }
}

/// Status codes from hip-runtime-sys 0.1.2 `hipError_t`. Only a runtime or
/// device that is not there is [`DeviceError::NoDevice`]: `hipErrorNotInitialized`
/// (3), `hipErrorDeinitialized` (4), `hipErrorInsufficientDriver` (35),
/// `hipErrorNoDevice` (100) and `hipErrorInvalidDevice` (101).
/// `hipErrorOutOfMemory` (2) is [`DeviceError::Capacity`]. Every other code,
/// a failed copy, stream or event included, came from a device that answered
/// and is [`DeviceError::Launch`]. Pre-fix, all of those were `NoDevice`.
#[cfg_attr(not(feature = "hip"), allow(dead_code))]
fn hip_status(op: &str, code: i32) -> Result<(), DeviceError> {
    match code {
        0 => Ok(()),
        2 => Err(DeviceError::Capacity {
            kind: Device::Hip,
            detail: format!("{op} ran out of memory"),
        }),
        3 | 4 | 35 | 100 | 101 => Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("{op} returned status {code}"),
        }),
        _ => Err(DeviceError::Launch {
            kind: Device::Hip,
            detail: format!("{op} returned status {code}"),
        }),
    }
}

/// `ordinal` must name one of the `count` devices `hipGetDeviceCount` reported.
#[cfg_attr(not(feature = "hip"), allow(dead_code))]
fn check_ordinal(ordinal: i32, count: i32) -> Result<(), DeviceError> {
    if (0..count).contains(&ordinal) {
        Ok(())
    } else {
        Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!(
                "ordinal {ordinal} is outside the {count} devices hipGetDeviceCount reported"
            ),
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

/// The HIP runtime calls this crate makes, declared to return the raw status.
///
/// hip-runtime-sys 0.1.2 declares each of these to return `hipError_t`, a
/// `#[repr(u32)]` Rust enum. A status the runtime returns that the enum does
/// not list (a newer ROCm, or a negative value) would be an invalid enum
/// value: undefined behaviour before any check could run. Declared here with
/// a `c_int` return, the same ABI, every status is a plain integer that
/// [`hip_status`] classifies. Argument types come from the binding, and the
/// binding's build script supplies the library search path.
#[cfg(feature = "hip")]
mod ffi {
    use hip_runtime_sys::{hipEvent_t, hipMemcpyKind, hipStream_t};
    use std::ffi::c_void;
    use std::os::raw::{c_int, c_uint};

    #[link(name = "amdhip64")]
    extern "C" {
        pub fn hipInit(flags: c_uint) -> c_int;
        pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
        pub fn hipSetDevice(device_id: c_int) -> c_int;
        pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
        pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
        pub fn hipFree(ptr: *mut c_void) -> c_int;
        pub fn hipHostMalloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
        pub fn hipHostFree(ptr: *mut c_void) -> c_int;
        pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> c_int;
        pub fn hipStreamDestroy(stream: hipStream_t) -> c_int;
        pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> c_int;
        pub fn hipEventDestroy(event: hipEvent_t) -> c_int;
        pub fn hipMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            size_bytes: usize,
            kind: hipMemcpyKind,
            stream: hipStream_t,
        ) -> c_int;
        pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> c_int;
        pub fn hipEventQuery(event: hipEvent_t) -> c_int;
    }
}

/// Make `ordinal` the calling thread's current device. HIP keeps the current
/// device per thread, so every entry point sets it before allocating.
#[cfg(feature = "hip")]
fn set_device(ordinal: i32) -> Result<(), DeviceError> {
    // SAFETY: takes the ordinal by value; an out-of-range one is a returned
    // status (hipErrorInvalidDevice), not undefined behaviour.
    hip_status("hipSetDevice", unsafe { ffi::hipSetDevice(ordinal) })
}

#[cfg(feature = "hip")]
fn open_runtime(ordinal: i32) -> Result<HipDevice, DeviceError> {
    // SAFETY: flags must be 0, as documented; no pointers are passed.
    hip_status("hipInit", unsafe { ffi::hipInit(0) })?;
    let mut count: std::os::raw::c_int = 0;
    // SAFETY: `count` is a live, aligned c_int the call writes once.
    hip_status("hipGetDeviceCount", unsafe {
        ffi::hipGetDeviceCount(&mut count)
    })?;
    if count <= 0 {
        return Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("hipGetDeviceCount returned {count}"),
        });
    }
    check_ordinal(ordinal, count)?;
    set_device(ordinal)?;
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: `free` and `total` are live, aligned usizes (size_t) the call writes.
    hip_status("hipMemGetInfo", unsafe {
        ffi::hipMemGetInfo(&mut free, &mut total)
    })?;
    if total == 0 {
        return Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("hipMemGetInfo returned total 0, free {free}"),
        });
    }
    let sample = [1.0f32, -2.0, 0.5, 4.0];
    let back = copy_through_hip(ordinal, &sample)?;
    if back != sample {
        return Err(DeviceError::NoDevice {
            kind: Device::Hip,
            detail: format!("HIP copy returned {back:?}, sent {sample:?}"),
        });
    }
    Ok(HipDevice {
        devices: count,
        ordinal,
    })
}

#[cfg(feature = "hip")]
fn copy_through_hip(ordinal: i32, input: &[f32]) -> Result<Vec<f32>, DeviceError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let nbytes = copy_bytes(input.len())?;
    set_device(ordinal)?;
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: `free` and `total` are live, aligned usizes (size_t) the call writes.
    hip_status("hipMemGetInfo", unsafe {
        ffi::hipMemGetInfo(&mut free, &mut total)
    })?;
    let _ = total;
    hip_free_covers(free, nbytes)?;
    // Freed on every return, including a panic in `vec!`.
    struct Dev(*mut std::ffi::c_void);
    impl Drop for Dev {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: a non-null pointer here came from a successful
                // hipMalloc and is freed once: the field is nulled after.
                unsafe {
                    let _ = ffi::hipFree(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    struct Host(*mut std::ffi::c_void);
    impl Drop for Host {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: a non-null pointer here came from a successful
                // hipHostMalloc and is freed once: the field is nulled after.
                unsafe {
                    let _ = ffi::hipHostFree(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    struct Stream(hip_runtime_sys::hipStream_t);
    impl Drop for Stream {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: a non-null handle here came from a successful
                // hipStreamCreateWithFlags and is destroyed once. It drops
                // before `host` and `dev` (reverse declaration order), and
                // hipStreamDestroy waits for its queued work first.
                unsafe {
                    let _ = ffi::hipStreamDestroy(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    struct Event(hip_runtime_sys::hipEvent_t);
    impl Drop for Event {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: a non-null handle here came from a successful
                // hipEventCreateWithFlags and is destroyed once.
                unsafe {
                    let _ = ffi::hipEventDestroy(self.0);
                }
                self.0 = std::ptr::null_mut();
            }
        }
    }
    let mut dev = std::ptr::null_mut();
    // SAFETY: `dev` is a live out-pointer; `nbytes` is non-zero and at most
    // MAX_COPY_BYTES (checked by `copy_bytes`).
    hip_status("hipMalloc", unsafe { ffi::hipMalloc(&mut dev, nbytes) })?;
    if dev.is_null() {
        return Err(DeviceError::Launch {
            kind: Device::Hip,
            detail: format!("hipMalloc({nbytes}) returned success and a null pointer"),
        });
    }
    let dev = Dev(dev);
    let mut host = std::ptr::null_mut();
    // SAFETY: `host` is a live out-pointer; `nbytes` as for hipMalloc.
    hip_status("hipHostMalloc", unsafe {
        ffi::hipHostMalloc(&mut host, nbytes, hip_runtime_sys::hipHostMallocDefault)
    })?;
    if host.is_null() {
        return Err(DeviceError::Capacity {
            kind: Device::Hip,
            detail: format!("hipHostMalloc({nbytes}) returned success and a null pointer"),
        });
    }
    let host = Host(host);
    // SAFETY: `host.0` is a fresh, non-null pinned allocation of `nbytes` =
    // `input.len() * 4` bytes, aligned for f32 (HIP allocations are at least
    // 256-byte aligned), and cannot overlap the borrowed `input`.
    unsafe {
        std::ptr::copy_nonoverlapping(input.as_ptr(), host.0.cast::<f32>(), input.len());
    }
    let mut stream = std::ptr::null_mut();
    // SAFETY: `stream` is a live out-pointer; the flag is a binding constant.
    hip_status("hipStreamCreateWithFlags", unsafe {
        ffi::hipStreamCreateWithFlags(&mut stream, hip_runtime_sys::hipStreamNonBlocking)
    })?;
    let stream = Stream(stream);
    let mut event = std::ptr::null_mut();
    // SAFETY: `event` is a live out-pointer; the flag is a binding constant.
    hip_status("hipEventCreateWithFlags", unsafe {
        ffi::hipEventCreateWithFlags(&mut event, hip_runtime_sys::hipEventDisableTiming)
    })?;
    let event = Event(event);
    // SAFETY: both buffers hold `nbytes` and outlive the stream (they drop
    // after it, and it drains before destroy). Nothing else touches `host`
    // until the event below completes.
    hip_status("hipMemcpyAsync host to device", unsafe {
        ffi::hipMemcpyAsync(
            dev.0,
            host.0,
            nbytes,
            hip_runtime_sys::hipMemcpyKind::hipMemcpyHostToDevice,
            stream.0,
        )
    })?;
    // SAFETY: as above; same-stream order puts this after the upload.
    hip_status("hipMemcpyAsync device to host", unsafe {
        ffi::hipMemcpyAsync(
            host.0,
            dev.0,
            nbytes,
            hip_runtime_sys::hipMemcpyKind::hipMemcpyDeviceToHost,
            stream.0,
        )
    })?;
    // SAFETY: `event` and `stream` are live handles created above.
    hip_status("hipEventRecord", unsafe {
        ffi::hipEventRecord(event.0, stream.0)
    })?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    poll_until(deadline, || {
        // SAFETY: `event` is a live handle recorded above.
        let status = unsafe { ffi::hipEventQuery(event.0) };
        if status == hip_runtime_sys::hipError_t::hipErrorNotReady as i32 {
            Ok(false)
        } else {
            hip_status("hipEventQuery", status).map(|()| true)
        }
    })?;
    let mut back = vec![0.0f32; input.len()];
    // SAFETY: the event completed, so both copies have finished and `host.0`
    // holds `input.len()` f32 values; `back` is a distinct allocation of that length.
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
            HipDevice {
                devices: 1,
                ordinal: 0,
            }
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

    /// Codes from hip-runtime-sys 0.1.2 `hipError_t`.
    #[test]
    fn only_a_missing_device_is_no_device() {
        // NotInitialized, Deinitialized, InsufficientDriver, NoDevice, InvalidDevice.
        for code in [3, 4, 35, 100, 101] {
            let err = hip_status("hipInit", code).unwrap_err();
            assert!(
                matches!(
                    err,
                    DeviceError::NoDevice {
                        kind: Device::Hip,
                        ..
                    }
                ),
                "{code}: {err}"
            );
        }
        // InvalidValue, InvalidMemcpyDirection, InvalidHandle, IllegalAddress,
        // LaunchFailure, NotSupported, Unknown, and codes the binding's enum
        // lacks: the FFI returns a raw c_int, so these are values, not UB.
        for code in [1, 21, 400, 700, 719, 801, 999, 12345, -1, i32::MIN] {
            let err = hip_status("hipMemcpyAsync host to device", code).unwrap_err();
            assert!(
                matches!(
                    err,
                    DeviceError::Launch {
                        kind: Device::Hip,
                        ..
                    }
                ),
                "{code}: {err}"
            );
            assert!(err.to_string().contains(&format!("status {code}")), "{err}");
        }
    }

    #[test]
    fn an_ordinal_past_the_device_count_is_no_device() {
        assert!(check_ordinal(0, 1).is_ok());
        assert!(check_ordinal(3, 4).is_ok());
        for (ordinal, count) in [(1, 1), (4, 4), (-1, 4), (0, 0)] {
            let err = check_ordinal(ordinal, count).unwrap_err();
            assert!(
                matches!(
                    err,
                    DeviceError::NoDevice {
                        kind: Device::Hip,
                        ..
                    }
                ),
                "{ordinal}/{count}: {err}"
            );
        }
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn default_build_refuses_every_ordinal_as_not_compiled() {
        for ordinal in [0, 1, -1] {
            match HipDevice::open_ordinal(ordinal) {
                Err(DeviceError::NotCompiled { kind: Device::Hip }) => {}
                Err(other) => panic!("open_ordinal({ordinal}) returned {other}"),
                Ok(_) => panic!("open_ordinal({ordinal}) succeeded without the hip feature"),
            }
        }
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
