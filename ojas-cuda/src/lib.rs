//! NVIDIA CUDA for ojas: the device probe ([`CudaDevice`]), the
//! `ojas_core::Backend` implementation ([`CudaBackend`]), and the Qwen3.5
//! whole-step training provider ([`Qwen35Step`]), design (B) in
//! `docs/cuda-backend-scoping.md`, the CUDA counterpart of `ojas-qwen35`.
//!
//! The target is the parity ladder (rungs a–d), not campaign speed. Every
//! kernel is tested against a float64 host reference before it is timed. GDN
//! fixtures name the `published` rule.
//!
//! Everything that touches a device is behind the `cuda` feature. Without it,
//! the crate builds on any host and its host-side tests run there.
//!
//! # What is here
//!
//! Host-side, built and tested everywhere:
//! - [`error`]: the crate's one error enum.
//! - [`bf16`]: round-to-nearest-even f32 to bf16, tessl's algorithm.
//! - [`budget`]: the bounded device-allocation budget.
//! - [`nvrtc_cache`]: the bounded compile cache, keyed by source hash,
//!   options (architecture included) and NVRTC version.
//! - [`kernels`] and the `*_kernels` / small-kernel modules: the CUDA-C
//!   sources and their compile options.
//! - [`k0_plan`], [`gemm_plan`], [`k8_plan`], [`gdn_plan`], [`geometry`]:
//!   validated plans, the cuBLAS row-major mapping, and launch shapes that
//!   never depend on SM count.
//! - [`host_ref`], [`k8_act`], [`k11_host`], [`gdn_host`], [`small_common`]:
//!   host references and bit-identical host emulations of the device code.
//! - [`check`], [`json`], [`libprobe`], [`rung0_cli`], [`report_cli`],
//!   [`inputs`]: what the rung binaries and the device tests share.
//! - [`step`]: the step provider's contract. Every compute method refuses with
//!   `Unsupported` until its kernels are wired and run on a device.
//!
//! Device-side, behind `cuda`: `runtime` (library probe, sm_90 check, one
//! stream, cuBLAS handle), `buffer`, `k0`, `gemm` (ExactF32 FFMA and bf16
//! cuBLAS tiers), K2 `gdn`, K3–K11, and the `smoke` checks the rung binaries
//! run. Device tests live in `tests/device_*.rs`, `#[ignore]`d, for an sm_90
//! GPU.

#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod backend;
pub mod bf16;
pub mod budget;
pub mod ce_rows;
pub mod check;
pub mod conv1d;
pub mod embed;
pub mod error;
pub mod gates_published;
pub mod gdn_host;
pub mod gdn_kernels;
pub mod gdn_plan;
pub mod gemm_plan;
pub mod geometry;
pub mod host_ref;
pub mod inputs;
pub mod json;
pub mod k0_plan;
pub mod k11_golden;
pub mod k11_host;
pub mod k11_kernels;
pub mod k8_act;
pub mod k8_kernels;
pub mod k8_plan;
pub mod kernels;
pub mod libprobe;
pub mod npy;
pub mod nvrtc_cache;
pub mod qk_norm_rope;
pub mod report_cli;
pub mod rmsnorm;
pub mod rung0_cli;
pub mod small_common;
pub mod step;
pub mod tiny_fixture_published;

#[cfg(feature = "cuda")]
pub mod buffer;
#[cfg(feature = "cuda")]
pub mod ce_rows_cuda;
#[cfg(feature = "cuda")]
pub mod conv1d_cuda;
#[cfg(feature = "cuda")]
pub mod embed_cuda;
#[cfg(feature = "cuda")]
pub mod gates_published_cuda;
#[cfg(feature = "cuda")]
pub mod gdn;
#[cfg(feature = "cuda")]
pub mod gdn_smoke;
#[cfg(feature = "cuda")]
pub mod gemm;
#[cfg(feature = "cuda")]
pub mod k0;
#[cfg(feature = "cuda")]
pub mod k11;
#[cfg(feature = "cuda")]
pub mod k11_smoke;
#[cfg(feature = "cuda")]
pub mod k8;
#[cfg(feature = "cuda")]
pub mod k8_smoke;
#[cfg(feature = "cuda")]
pub mod qk_norm_rope_cuda;
#[cfg(feature = "cuda")]
pub mod rmsnorm_cuda;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
pub mod small_common_cuda;
#[cfg(feature = "cuda")]
pub mod small_smoke;
#[cfg(feature = "cuda")]
pub mod smoke;

pub use backend::CudaBackend;
#[cfg(feature = "cuda")]
pub use buffer::{CudaBuffer, CudaDeviceBuffer};
pub use error::CudaError;
#[cfg(feature = "cuda")]
pub use runtime::{CudaRuntime, DeviceInfo, RuntimeConfig};
pub use step::{
    clip_coefficient, validate_external_grad, AdamWHyper, BankState, ExternalGrad, GemmOperands,
    Numerics, Pending, Qwen35Step, Sequence, StepProvider, Supervise,
};

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
    host: cudarc::driver::PinnedHostSlice<f32>,
}

#[cfg(feature = "cuda")]
fn from_driver(err: cudarc::driver::DriverError, op: &str) -> DeviceError {
    cuda_status(err.0 as u32, format!("{op}: {err}"))
}

#[cfg(feature = "cuda")]
fn compile_err(detail: impl std::fmt::Display) -> DeviceError {
    DeviceError::Compile {
        kind: Device::Cuda,
        detail: detail.to_string(),
    }
}

/// `CUDA_ERROR_OUT_OF_MEMORY` is 2. Pre-fix, `launch_err` mapped that code to
/// [`DeviceError::Launch`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn cuda_status(code: u32, detail: String) -> DeviceError {
    if code == 2 {
        DeviceError::Capacity {
            kind: Device::Cuda,
            detail,
        }
    } else {
        DeviceError::Launch {
            kind: Device::Cuda,
            detail,
        }
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn device_bytes_fit(free: usize, need: usize) -> Result<(), DeviceError> {
    if need > free {
        Err(DeviceError::Capacity {
            kind: Device::Cuda,
            detail: format!("{need} device bytes exceed {free} bytes free"),
        })
    } else {
        Ok(())
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
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
                kind: Device::Cuda,
                detail: "CUDA stream did not complete within 30s".to_string(),
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
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
        let (free, _total) = self
            .ctx
            .mem_get_info()
            .map_err(|err| from_driver(err, "mem_get_info"))?;
        let device_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or_else(|| DeviceError::Capacity {
                kind: Device::Cuda,
                detail: format!("device bytes for {} f32 values overflow", input.len()),
            })?;
        let mut slot = self.affine.lock().unwrap_or_else(|e| e.into_inner());
        let resizing = slot
            .as_ref()
            .map(|gpu| gpu.len != input.len())
            .unwrap_or(true);
        if resizing {
            device_bytes_fit(free, device_bytes)?;
        }
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
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                )
                .map_err(|err| from_driver(err, "attribute MAX_GRID_DIM_X"))?;
            if max_grid <= 0 {
                return Err(DeviceError::Capacity {
                    kind: Device::Cuda,
                    detail: format!("MAX_GRID_DIM_X is {max_grid}"),
                });
            }
            let stream = self.ctx.default_stream();
            let inp = stream
                .clone_htod(input)
                .map_err(|err| from_driver(err, "clone_htod"))?;
            let out = stream
                .alloc_zeros::<f32>(input.len())
                .map_err(|err| from_driver(err, "alloc_zeros"))?;
            // SAFETY: flags 0 is the portable pinned allocation, not write-combined.
            let host = unsafe { self.ctx.alloc_pinned_with_flags::<f32>(input.len(), 0) }
                .map_err(|err| from_driver(err, "alloc_pinned_with_flags"))?;
            *slot = Some(AffineGpu {
                func,
                max_grid_x: max_grid as u32,
                len: input.len(),
                inp,
                out,
                host,
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
            let new_inp = stream
                .clone_htod(input)
                .map_err(|err| from_driver(err, "clone_htod"))?;
            let new_out = stream
                .alloc_zeros::<f32>(input.len())
                .map_err(|err| from_driver(err, "alloc_zeros"))?;
            // SAFETY: flags 0 is the portable pinned allocation, not write-combined.
            let new_host = unsafe { self.ctx.alloc_pinned_with_flags::<f32>(input.len(), 0) }
                .map_err(|err| from_driver(err, "alloc_pinned_with_flags"))?;
            gpu.inp = new_inp;
            gpu.out = new_out;
            gpu.host = new_host;
            gpu.len = input.len();
        } else {
            stream
                .memcpy_htod(input, &mut gpu.inp)
                .map_err(|err| from_driver(err, "memcpy_htod"))?;
            stream
                .memset_zeros(&mut gpu.out)
                .map_err(|err| from_driver(err, "memset"))?;
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
        .map_err(|err| from_driver(err, "launch"))?;
        stream
            .memcpy_dtoh(&gpu.out, &mut gpu.host)
            .map_err(|err| from_driver(err, "memcpy_dtoh"))?;
        let event = self
            .ctx
            .new_event(Some(
                cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
            ))
            .map_err(|err| from_driver(err, "new_event"))?;
        event
            .record(&stream)
            .map_err(|err| from_driver(err, "event.record"))?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        poll_until(deadline, || {
            event
                .try_is_complete()
                .map_err(|err| from_driver(err, "try_is_complete"))
        })?;
        Ok(gpu
            .host
            .as_slice()
            .map_err(|err| from_driver(err, "pinned read"))?
            .to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_memory_is_capacity_not_launch() {
        let oom = cuda_status(2, "cuMemAlloc".to_string());
        assert!(
            matches!(
                oom,
                DeviceError::Capacity {
                    kind: Device::Cuda,
                    ..
                }
            ),
            "{oom}"
        );
        let other = cuda_status(1, "invalid".to_string());
        assert!(matches!(other, DeviceError::Launch { .. }), "{other}");
    }

    #[test]
    fn device_probe_refuses_a_copy_larger_than_free_memory() {
        let err = device_bytes_fit(32, 64).unwrap_err();
        assert!(
            matches!(
                err,
                DeviceError::Capacity {
                    kind: Device::Cuda,
                    ..
                }
            ),
            "{err}"
        );
        assert!(device_bytes_fit(64, 64).is_ok());
    }

    #[test]
    fn stream_wait_returns_when_the_deadline_has_passed() {
        let err = poll_until(std::time::Instant::now(), || Ok(false)).unwrap_err();
        assert!(
            matches!(
                err,
                DeviceError::Launch {
                    kind: Device::Cuda,
                    ..
                }
            ),
            "{err}"
        );
        assert!(poll_until(std::time::Instant::now(), || Ok(true)).is_ok());
    }

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
