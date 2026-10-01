//! NVIDIA CUDA through cudarc.
//!
//! With the `cuda` feature off, [`CudaDevice::open`] returns
//! [`DeviceError::NotCompiled`]. It does not run the buffer on the CPU.
//! With the feature on, open loads the driver and launches this crate's
//! affine kernel (`y = x * scale + bias`) through cudarc. A missing driver
//! is [`DeviceError::NoDevice`]. The default build does not compile that path.

#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]

use ojas_device::{require_kind, Device, DeviceError};

/// An open CUDA context. The affine module is compiled once, with `fmad` off.
pub struct CudaDevice {
    #[cfg(feature = "cuda")]
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    #[cfg(feature = "cuda")]
    affine: std::sync::Mutex<Option<AffineGpu>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CudaDevice")
    }
}

impl CudaDevice {
    /// Open CUDA ordinal 0.
    ///
    /// When `cuda` is off this is [`DeviceError::NotCompiled`].
    /// When `cuda` is on, a small affine kernel is launched before `Ok`.
    /// A driver failure is [`DeviceError::NoDevice`].
    pub fn open() -> Result<Self, DeviceError> {
        #[cfg(not(feature = "cuda"))]
        {
            Err(DeviceError::NotCompiled { kind: Device::Cuda })
        }
        #[cfg(feature = "cuda")]
        {
            require_libraries()?;
            let ctx = cudarc::driver::CudaContext::new(0).map_err(|err| DeviceError::NoDevice {
                kind: Device::Cuda,
                detail: format!("CudaContext::new(0): {err}"),
            })?;
            let device = CudaDevice {
                ctx,
                affine: std::sync::Mutex::new(None),
            };
            let sample = [1.0f32, -2.0, 0.5, 4.0];
            let back = device.affine_f32(Device::Cuda, &sample, 2.0, 0.5)?;
            let expect = [2.5f32, -3.5, 1.5, 8.5];
            if back != expect {
                return Err(DeviceError::NoDevice {
                    kind: Device::Cuda,
                    detail: format!("kernel returned {back:?}, expected {expect:?}"),
                });
            }
            Ok(device)
        }
    }

    /// `y = x * scale + bias` on CUDA. `kind` must be [`Device::Cuda`].
    pub fn affine_f32(
        &self,
        kind: Device,
        input: &[f32],
        scale: f32,
        bias: f32,
    ) -> Result<Vec<f32>, DeviceError> {
        require_kind(Device::Cuda, kind)?;
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (self, input, scale, bias);
            Err(DeviceError::NotCompiled { kind: Device::Cuda })
        }
        #[cfg(feature = "cuda")]
        {
            self.launch_affine(input, scale, bias)
        }
    }
}

#[cfg(feature = "cuda")]
const AFFINE_CUDA: &str = ojas_kernels::affine_cuda();

/// cudarc panics when it cannot load `libcuda` or `libnvrtc`. Probe both
/// first so a host without the driver gets [`DeviceError::NoDevice`].
#[cfg(feature = "cuda")]
fn require_libraries() -> Result<(), DeviceError> {
    // SAFETY: both probes only try `dlopen` on cudarc's candidate names. A
    // library's initialisers are the same ones the first cudarc call would run.
    let driver = unsafe { cudarc::driver::sys::is_culib_present() };
    let nvrtc = unsafe { cudarc::nvrtc::sys::is_culib_present() };
    if driver && nvrtc {
        return Ok(());
    }
    Err(DeviceError::NoDevice {
        kind: Device::Cuda,
        detail: format!("CUDA libraries not loadable: driver {driver}, nvrtc {nvrtc}"),
    })
}

#[cfg(feature = "cuda")]
struct AffineGpu {
    func: cudarc::driver::CudaFunction,
    max_grid_x: u32,
    len: usize,
    inp: cudarc::driver::CudaSlice<f32>,
    out: cudarc::driver::CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
fn compile_err(detail: impl std::fmt::Display) -> DeviceError {
    DeviceError::Compile {
        kind: Device::Cuda,
        detail: detail.to_string(),
    }
}

#[cfg(feature = "cuda")]
fn launch_err(detail: impl std::fmt::Display) -> DeviceError {
    DeviceError::Launch {
        kind: Device::Cuda,
        detail: detail.to_string(),
    }
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    fn launch_affine(&self, input: &[f32], scale: f32, bias: f32) -> Result<Vec<f32>, DeviceError> {
        use cudarc::driver::PushKernelArg;
        let n = u32::try_from(input.len()).map_err(|_| DeviceError::Capacity {
            kind: Device::Cuda,
            detail: format!("input length {} does not fit in u32", input.len()),
        })?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut slot = self.affine.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let opts = cudarc::nvrtc::CompileOptions {
                fmad: Some(false),
                ..cudarc::nvrtc::CompileOptions::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(AFFINE_CUDA, opts)
                .map_err(|err| compile_err(format!("compile_ptx_with_opts: {err}")))?;
            let module = self
                .ctx
                .load_module(ptx)
                .map_err(|err| compile_err(format!("load_module: {err}")))?;
            let func = module
                .load_function("affine_f32")
                .map_err(|err| compile_err(format!("load_function: {err}")))?;
            let max_grid = self
                .ctx
                .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)
                .map_err(|err| launch_err(format!("attribute MAX_GRID_DIM_X: {err}")))?;
            if max_grid <= 0 {
                return Err(DeviceError::Capacity {
                    kind: Device::Cuda,
                    detail: format!("MAX_GRID_DIM_X is {max_grid}"),
                });
            }
            let stream = self.ctx.default_stream();
            let inp = stream
                .clone_htod(input)
                .map_err(|err| launch_err(format!("clone_htod: {err}")))?;
            let out = stream
                .alloc_zeros::<f32>(input.len())
                .map_err(|err| launch_err(format!("alloc_zeros: {err}")))?;
            *slot = Some(AffineGpu {
                func,
                max_grid_x: max_grid as u32,
                len: input.len(),
                inp,
                out,
            });
        }
        let gpu = slot.as_mut().unwrap();
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(n);
        if cfg.grid_dim.0 > gpu.max_grid_x {
            return Err(DeviceError::Capacity {
                kind: Device::Cuda,
                detail: format!(
                    "grid {} exceeds CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X {}",
                    cfg.grid_dim.0, gpu.max_grid_x
                ),
            });
        }
        let stream = self.ctx.default_stream();
        if gpu.len != input.len() {
            gpu.inp = stream
                .clone_htod(input)
                .map_err(|err| launch_err(format!("clone_htod: {err}")))?;
            gpu.out = stream
                .alloc_zeros::<f32>(input.len())
                .map_err(|err| launch_err(format!("alloc_zeros: {err}")))?;
            gpu.len = input.len();
        } else {
            stream
                .memcpy_htod(input, &mut gpu.inp)
                .map_err(|err| launch_err(format!("memcpy_htod: {err}")))?;
            stream
                .memset_zeros(&mut gpu.out)
                .map_err(|err| launch_err(format!("memset: {err}")))?;
        }
        unsafe {
            stream
                .launch_builder(&gpu.func)
                .arg(&mut gpu.out)
                .arg(&gpu.inp)
                .arg(&scale)
                .arg(&bias)
                .arg(&n)
                .launch(cfg)
        }
        .map_err(|err| launch_err(format!("launch: {err}")))?;
        // SAFETY: flags 0 is the portable pinned allocation, not write-combined.
        let mut pinned = unsafe { self.ctx.alloc_pinned_with_flags::<f32>(input.len(), 0) }
            .map_err(|err| launch_err(format!("alloc_pinned_with_flags: {err}")))?;
        stream
            .memcpy_dtoh(&gpu.out, &mut pinned)
            .map_err(|err| launch_err(format!("memcpy_dtoh: {err}")))?;
        stream
            .synchronize()
            .map_err(|err| launch_err(format!("synchronize: {err}")))?;
        Ok(pinned
            .as_slice()
            .map_err(|err| launch_err(format!("pinned read: {err}")))?
            .to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the feature the struct has no fields, so the kind check and the
    /// `NotCompiled` arm are reachable without a driver. With the feature a
    /// missing device fails the test.
    fn device() -> CudaDevice {
        #[cfg(not(feature = "cuda"))]
        {
            CudaDevice {}
        }
        #[cfg(feature = "cuda")]
        {
            CudaDevice::open().expect("cuda feature on: a missing CUDA device fails the test")
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_reports_not_compiled() {
        match CudaDevice::open() {
            Err(DeviceError::NotCompiled { kind: Device::Cuda }) => {}
            Err(other) => panic!("open returned {other}"),
            Ok(_) => panic!("open succeeded without the cuda feature"),
        }
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_does_not_compute_on_the_cpu() {
        let err = device()
            .affine_f32(Device::Cuda, &[1.0, 2.0], 2.0, 0.5)
            .unwrap_err();
        assert!(
            matches!(err, DeviceError::NotCompiled { kind: Device::Cuda }),
            "{err}"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_refuses_cuda_without_running_a_kernel() {
        let caught = std::panic::catch_unwind(CudaDevice::open);
        let opened = caught.expect("CudaDevice::open panicked on the default build");
        match opened {
            Err(DeviceError::NotCompiled { kind: Device::Cuda }) => {}
            Err(other) => panic!("open returned {other}, not NotCompiled"),
            Ok(_) => panic!("open succeeded without the cuda feature"),
        }
        let err = device()
            .affine_f32(Device::Hip, &[1.0, 2.0, 3.0], 9.0, 9.0)
            .unwrap_err();
        assert!(
            matches!(
                err,
                DeviceError::DeviceMismatch {
                    expected: Device::Cuda,
                    actual: Device::Hip,
                }
            ),
            "{err}"
        );
        let same = device()
            .affine_f32(Device::Cuda, &[4.0, 5.0], 2.0, 1.0)
            .unwrap_err();
        assert!(
            matches!(same, DeviceError::NotCompiled { kind: Device::Cuda }),
            "{same}"
        );
    }

    #[test]
    fn cuda_is_not_accepted_as_cpu() {
        let err = device()
            .affine_f32(Device::Cpu, &[1.0], 1.0, 0.0)
            .unwrap_err();
        assert!(matches!(err, DeviceError::DeviceMismatch { .. }));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn open_returns_instead_of_panicking() {
        match std::panic::catch_unwind(CudaDevice::open) {
            Ok(Ok(_))
            | Ok(Err(DeviceError::NoDevice {
                kind: Device::Cuda, ..
            })) => {}
            Ok(Err(other)) => panic!("open returned {other}"),
            Err(_) => panic!("CudaDevice::open panicked instead of returning NoDevice"),
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn affine_matches_cpu_at_odd_lengths() {
        let device = device();
        assert_eq!(
            device.affine_f32(Device::Cuda, &[], 2.0, 1.0).unwrap(),
            Vec::<f32>::new()
        );
        for n in [1usize, 63, 65, 1023, 1024, 1025, 100_003] {
            let input: Vec<f32> = (0..n).map(|i| (i % 997) as f32 / 500.0 - 1.0).collect();
            let got = device.affine_f32(Device::Cuda, &input, 1.5, -0.25).unwrap();
            assert_eq!(got.len(), n);
            for (i, (g, x)) in got.iter().zip(&input).enumerate() {
                assert!((g - (x * 1.5 - 0.25)).abs() <= 1e-6, "len {n} index {i}");
            }
        }
    }
}
