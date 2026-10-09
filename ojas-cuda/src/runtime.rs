//! [`CudaRuntime`]: the device, one stream, the allocation budget, the NVRTC
//! cache and the cuBLAS handle.
//!
//! [`CudaRuntime::open`] in order:
//! 1. probes `libcuda`, `libnvrtc` and `libcublas` with cudarc's own
//!    `is_culib_present` **before any other cudarc call**, because cudarc
//!    panics on a library it cannot load (`cudarc/src/lib.rs:199-201`); a
//!    missing library is [`CudaError::LibraryMissing`], by name;
//! 2. opens the device and refuses anything but compute capability 9.0;
//! 3. creates one stream: every kernel, copy and cuBLAS call is ordered on
//!    it (cuBLAS reproducibility does not hold across streams);
//! 4. creates a cuBLAS handle on that stream with a fixed 32 MiB workspace
//!    (`cublasSetWorkspace`; 256-byte aligned, as `cuMemAlloc` returns),
//!    `CUBLAS_DEFAULT_MATH` (TF32 is not enabled: that needs
//!    `CUBLAS_TF32_TENSOR_OP_MATH` or a `*_FAST_TF32` compute type), and the
//!    atomics mode never set; both modes are read back into [`DeviceInfo`].
//!
//! Every wait is bounded ([`RuntimeConfig::sync_timeout`]): an event is
//! recorded and polled, never an unbounded `cuStreamSynchronize`.
//!
//! Every device byte the runtime allocates, the cuBLAS workspace included,
//! is charged to one [`ojas_core::Budget`] ([`CudaRuntime::open_with`]);
//! [`crate::CudaBackend`] charges its tensors to the same one.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr,
    DriverError,
};
use ojas_core::Budget;

use crate::budget::{bytes_for, AllocBudget, Reservation};
use crate::error::CudaError;
use crate::json::{Json, JsonObj};
use crate::kernels::{CompileSpec, KernelModule};
use crate::libprobe;
use crate::nvrtc_cache::{CacheKey, CacheStats, NvrtcCache};
use crate::wait::poll_until;

/// The compute capability this crate is written and tested for (GH200:
/// sm_90), required exactly. Stated once, in [`crate::kernels`], where the
/// NVRTC architectures derive from it.
pub use crate::kernels::REQUIRED_CC;

/// cuBLAS workspace size on sm_90 and later, per NVIDIA's `cublasSetWorkspace`
/// guidance (32 MiB).
pub const CUBLAS_WORKSPACE_BYTES: usize = 32 << 20;

/// How a runtime is opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    /// CUDA device ordinal.
    pub ordinal: usize,
    /// Cap on this runtime's device allocations, workspace included. With
    /// [`CudaRuntime::open_with`] it is the given budget's cap.
    pub budget_bytes: u64,
    /// Longest any single device wait may take.
    pub sync_timeout: Duration,
    /// NVRTC cache bound, in modules.
    pub cache_entries: usize,
    /// NVRTC cache bound, in compiled image bytes.
    pub cache_bytes: usize,
    /// cuBLAS workspace bytes.
    pub cublas_workspace_bytes: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            ordinal: 0,
            budget_bytes: 4 << 30,
            sync_timeout: Duration::from_secs(60),
            cache_entries: 32,
            cache_bytes: 64 << 20,
            cublas_workspace_bytes: CUBLAS_WORKSPACE_BYTES,
        }
    }
}

/// What the device and libraries report; recorded in every rung-0 report,
/// because bits are promised only within (GPU, SM count, driver, NVRTC,
/// cuBLAS) (`cuda-backend-scoping.md` §5.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `cuDeviceGetName`.
    pub name: String,
    /// `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Streaming multiprocessors.
    pub sm_count: i32,
    /// Device memory in bytes.
    pub total_mem: usize,
    /// Opt-in shared memory per block.
    pub smem_per_block_optin: i32,
    /// `cuDriverGetVersion`: 1000 * major + 10 * minor.
    pub driver_version: i32,
    /// `nvrtcVersion`.
    pub nvrtc_version: (i32, i32),
    /// `cublasGetProperty` major, minor, patch.
    pub cublas_version: (i32, i32, i32),
    /// `cublasGetMathMode`, read back after setting `CUBLAS_DEFAULT_MATH` (0).
    pub cublas_math_mode: u32,
    /// `cublasGetAtomicsMode`; never set, so `CUBLAS_ATOMICS_NOT_ALLOWED` (0).
    pub cublas_atomics_mode: u32,
    /// The workspace handed to `cublasSetWorkspace`.
    pub cublas_workspace_bytes: usize,
}

impl DeviceInfo {
    /// The report object.
    pub fn to_json(&self) -> Json {
        JsonObj::new()
            .with("name", self.name.as_str())
            .with(
                "compute_capability",
                format!(
                    "{}.{}",
                    self.compute_capability.0, self.compute_capability.1
                ),
            )
            .with("sm_count", self.sm_count)
            .with("total_mem_bytes", self.total_mem)
            .with("smem_per_block_optin", self.smem_per_block_optin)
            .with("driver_version", self.driver_version)
            .with(
                "nvrtc_version",
                format!("{}.{}", self.nvrtc_version.0, self.nvrtc_version.1),
            )
            .with(
                "cublas_version",
                format!(
                    "{}.{}.{}",
                    self.cublas_version.0, self.cublas_version.1, self.cublas_version.2
                ),
            )
            .with("cublas_math_mode", self.cublas_math_mode)
            .with("cublas_atomics_mode", self.cublas_atomics_mode)
            .with("cublas_workspace_bytes", self.cublas_workspace_bytes)
            .into()
    }
}

/// A driver error as [`CudaError`] (out-of-memory becomes `Capacity`).
pub fn driver_error(op: &str, err: DriverError) -> CudaError {
    // `CUresult` is a `#[repr(u32)]` field-less enum: the cast is its value.
    CudaError::from_driver_code(op, err.0 as u32, err.to_string())
}

/// A cuBLAS status as [`CudaError::Cublas`].
pub fn cublas_error(op: &str, status: cublas_sys::cublasStatus_t) -> CudaError {
    CudaError::Cublas {
        op: op.to_string(),
        status: format!("{status:?}"),
        // `cublasStatus_t` is a `#[repr(u32)]` field-less enum.
        code: status as u32,
    }
}

/// Wait, at most `timeout`, for everything queued on `stream` so far, then
/// report any error cudarc deferred from a drop path (frees, event waits).
/// The crate's one device wait: an event is recorded and polled
/// ([`poll_until`]), never an unbounded `cuStreamSynchronize`.
pub(crate) fn wait_stream(
    stream: &CudaStream,
    timeout: Duration,
    op: &str,
) -> Result<(), CudaError> {
    let ctx = stream.context();
    let event = ctx
        .new_event(Some(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
        ))
        .map_err(|e| driver_error(&format!("{op}: cuEventCreate"), e))?;
    event
        .record(stream)
        .map_err(|e| driver_error(&format!("{op}: cuEventRecord"), e))?;
    poll_until(
        timeout,
        Duration::from_micros(200),
        || {
            event
                .try_is_complete()
                .map_err(|e| driver_error(&format!("{op}: cuEventQuery"), e))
        },
        |waited| CudaError::Timeout {
            op: op.to_string(),
            waited_ms: waited.as_millis(),
        },
    )?;
    ctx.check_err()
        .map_err(|e| driver_error(&format!("{op}: deferred error"), e))
}

/// Load-probe every library the runtime needs, with cudarc's own candidate
/// names, before any cudarc call that would panic on a missing one.
pub fn probe_libraries() -> Result<(), CudaError> {
    probe(&libprobe::REQUIRED)
}

/// Load-probe `specs` (a subset of [`libprobe::REQUIRED`]), refusing by name
/// with every missing library and where it was looked for. A spec this crate
/// does not know how to probe counts as missing.
pub(crate) fn probe(specs: &[libprobe::LibrarySpec]) -> Result<(), CudaError> {
    let ld = std::env::var("LD_LIBRARY_PATH").ok();
    let mut libraries = Vec::new();
    let mut details = Vec::new();
    for &spec in specs {
        // SAFETY (all three): `is_culib_present` only `dlopen`s cudarc's
        // candidate sonames and drops the handle. Loading runs the library's
        // initialisers, which cudarc's first real call would run anyway.
        let present = if spec == libprobe::DRIVER {
            unsafe { cudarc::driver::sys::is_culib_present() }
        } else if spec == libprobe::NVRTC {
            unsafe { cudarc::nvrtc::sys::is_culib_present() }
        } else if spec == libprobe::CUBLAS {
            unsafe { cudarc::cublas::sys::is_culib_present() }
        } else {
            false
        };
        if !present {
            let candidates: Vec<String> = spec
                .cudarc_names
                .iter()
                .flat_map(|n| cudarc::get_lib_name_candidates(n))
                .collect();
            libraries.push(spec.label.to_string());
            details.push(libprobe::missing_detail(&spec, &candidates, ld.as_deref()));
        }
    }
    if libraries.is_empty() {
        Ok(())
    } else {
        Err(CudaError::LibraryMissing {
            libraries,
            detail: details.join(" | "),
        })
    }
}

/// What a new cuBLAS handle reports back.
struct CublasReadback {
    math_mode: u32,
    atomics_mode: u32,
    version: (i32, i32, i32),
}

/// An owned cuBLAS handle bound to the runtime's stream, with its workspace.
/// Dropped before the workspace (field order), and its destroy failure is
/// reported on stderr rather than unwrapped (cudarc's `CudaBlas` unwraps).
struct CublasHandle {
    handle: cublas_sys::cublasHandle_t,
    workspace: Option<(CudaSlice<u8>, Reservation)>,
}

impl CublasHandle {
    fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        budget: &AllocBudget,
        workspace_bytes: usize,
    ) -> Result<(Self, CublasReadback), CudaError> {
        ctx.bind_to_thread()
            .map_err(|e| driver_error("bind_to_thread", e))?;
        let handle = cudarc::cublas::result::create_handle()
            .map_err(|e| cublas_error("cublasCreate", e.0))?;
        let mut this = CublasHandle {
            handle,
            workspace: None,
        };
        // SAFETY: `handle` is live; the driver `CUstream` and cuBLAS's
        // `cudaStream_t` are the same object behind two bindgen pointer types.
        unsafe { cudarc::cublas::result::set_stream(handle, stream.cu_stream() as _) }
            .map_err(|e| cublas_error("cublasSetStream", e.0))?;
        // SAFETY: `handle` is live; the mode is a valid enum value.
        unsafe {
            cublas_sys::cublasSetMathMode(handle, cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH)
        }
        .result()
        .map_err(|e| cublas_error("cublasSetMathMode", e.0))?;

        if workspace_bytes > 0 {
            let reservation = budget.reserve(
                u64::try_from(workspace_bytes).unwrap_or(u64::MAX),
                "cuBLAS workspace",
            )?;
            let ws = stream
                .alloc_zeros::<u8>(workspace_bytes)
                .map_err(|e| driver_error("alloc cuBLAS workspace", e))?;
            {
                let (ptr, _sync) = ws.device_ptr(stream);
                // SAFETY: `ptr` is a live allocation of `workspace_bytes`, from
                // cuMemAlloc(Async), so 256-byte aligned; it outlives the handle
                // (dropped after it).
                unsafe {
                    cublas_sys::cublasSetWorkspace_v2(
                        handle,
                        ptr as *mut std::ffi::c_void,
                        workspace_bytes,
                    )
                }
                .result()
                .map_err(|e| cublas_error("cublasSetWorkspace", e.0))?;
            }
            this.workspace = Some((ws, reservation));
        }

        // Read the modes back as raw u32s: cuBLAS may report a value the
        // bindgen enum does not list, and a raw read never materialises one.
        let mut math: u32 = u32::MAX;
        let mut atomics: u32 = u32::MAX;
        // SAFETY: both enums are `#[repr(u32)]`, so cuBLAS writes 4 bytes into
        // a u32; the value is only ever read as a u32.
        unsafe {
            cublas_sys::cublasGetMathMode(handle, (&mut math as *mut u32).cast())
                .result()
                .map_err(|e| cublas_error("cublasGetMathMode", e.0))?;
            cublas_sys::cublasGetAtomicsMode(handle, (&mut atomics as *mut u32).cast())
                .result()
                .map_err(|e| cublas_error("cublasGetAtomicsMode", e.0))?;
        }
        let mut version = [0i32; 3];
        for (slot, prop) in version.iter_mut().zip([
            cublas_sys::libraryPropertyType::MAJOR_VERSION,
            cublas_sys::libraryPropertyType::MINOR_VERSION,
            cublas_sys::libraryPropertyType::PATCH_LEVEL,
        ]) {
            // SAFETY: `slot` is a valid `int` out-parameter.
            unsafe { cublas_sys::cublasGetProperty(prop, slot) }
                .result()
                .map_err(|e| cublas_error("cublasGetProperty", e.0))?;
        }
        Ok((
            this,
            CublasReadback {
                math_mode: math,
                atomics_mode: atomics,
                version: (version[0], version[1], version[2]),
            },
        ))
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // SAFETY: created by cublasCreate in `new` and destroyed only here.
        if let Err(e) = unsafe { cudarc::cublas::result::destroy_handle(self.handle) } {
            eprintln!("ojas-cuda: cublasDestroy failed: {e}");
        }
    }
}

/// The device runtime. Fields drop in order: the cuBLAS handle (then its
/// workspace), the module cache, the stream, the context.
pub struct CudaRuntime {
    blas: CublasHandle,
    cache: Mutex<NvrtcCache<Arc<CudaModule>>>,
    budget: AllocBudget,
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    info: DeviceInfo,
    config: RuntimeConfig,
}

impl CudaRuntime {
    /// Open the device with a budget of `config.budget_bytes` that nothing
    /// else charges; see the module docs for the order of checks.
    pub fn open(config: RuntimeConfig) -> Result<Self, CudaError> {
        let budget = Budget::new(config.budget_bytes);
        Self::open_with(config, budget)
    }

    /// Open the device charging `budget` for every allocation, the cuBLAS
    /// workspace included, so the caller's own charges on `budget` (or on
    /// its parent) and the runtime's share one cap. `config.budget_bytes` is
    /// replaced by `budget`'s cap. A 0-byte budget, or one too small for the
    /// workspace, is [`CudaError::Capacity`].
    pub fn open_with(mut config: RuntimeConfig, budget: Budget) -> Result<Self, CudaError> {
        config.budget_bytes = budget.cap_bytes();
        probe_libraries()?;
        let ctx = CudaContext::new(config.ordinal)
            .map_err(|e| driver_error(&format!("CudaContext::new({})", config.ordinal), e))?;
        let name = ctx.name().map_err(|e| driver_error("cuDeviceGetName", e))?;
        let cc = ctx
            .compute_capability()
            .map_err(|e| driver_error("compute_capability", e))?;
        if cc != REQUIRED_CC {
            return Err(CudaError::Arch {
                found: cc,
                required: REQUIRED_CC,
                name,
            });
        }
        let attr = |a, what: &str| ctx.attribute(a).map_err(|e| driver_error(what, e));
        use cudarc::driver::sys::CUdevice_attribute as A;
        let sm_count = attr(
            A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            "MULTIPROCESSOR_COUNT",
        )?;
        let smem_optin = attr(
            A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
        )?;
        let total_mem = ctx
            .total_mem()
            .map_err(|e| driver_error("cuDeviceTotalMem", e))?;

        let mut driver_version: i32 = 0;
        // SAFETY: a valid `int` out-parameter.
        unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut driver_version) }
            .result()
            .map_err(|e| driver_error("cuDriverGetVersion", e))?;
        let (mut nv_major, mut nv_minor) = (0i32, 0i32);
        // SAFETY: valid `int` out-parameters.
        unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nv_major, &mut nv_minor) }
            .result()
            .map_err(|e| CudaError::Compile {
                module: "nvrtcVersion".to_string(),
                detail: e.to_string(),
            })?;

        let stream = ctx
            .new_stream()
            .map_err(|e| driver_error("cuStreamCreate", e))?;
        let budget = AllocBudget::over(budget)?;
        let (blas, readback) =
            CublasHandle::new(&ctx, &stream, &budget, config.cublas_workspace_bytes)?;
        let cache = NvrtcCache::new(config.cache_entries, config.cache_bytes)?;

        let info = DeviceInfo {
            name,
            compute_capability: cc,
            sm_count,
            total_mem,
            smem_per_block_optin: smem_optin,
            driver_version,
            nvrtc_version: (nv_major, nv_minor),
            cublas_version: readback.version,
            cublas_math_mode: readback.math_mode,
            cublas_atomics_mode: readback.atomics_mode,
            cublas_workspace_bytes: config.cublas_workspace_bytes,
        };
        let rt = CudaRuntime {
            blas,
            cache: Mutex::new(cache),
            budget,
            stream,
            ctx,
            info,
            config,
        };
        rt.sync("open")?;
        Ok(rt)
    }

    /// What the device and libraries reported at open.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The configuration it was opened with.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// The one stream all work is ordered on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The allocation budget: a view of the one [`Budget`] every device byte
    /// is charged to.
    pub fn budget(&self) -> &AllocBudget {
        &self.budget
    }

    /// The raw cuBLAS handle, bound to [`Self::stream`].
    pub(crate) fn cublas_handle(&self) -> cublas_sys::cublasHandle_t {
        self.blas.handle
    }

    /// NVRTC cache counters.
    pub fn cache_stats(&self) -> CacheStats {
        self.lock_cache().stats()
    }

    /// The cache lock. A panic while it was held can only have come from
    /// inside a compile (cudarc panics on a missing NVRTC symbol), and the
    /// cache inserts only after a compile returns, so a poisoned cache is
    /// still consistent and is used rather than failing every later call.
    fn lock_cache(&self) -> std::sync::MutexGuard<'_, NvrtcCache<Arc<CudaModule>>> {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Wait for all queued work, polling an event until `sync_timeout`
    /// ([`wait_stream`]).
    pub fn sync(&self, op: &str) -> Result<(), CudaError> {
        wait_stream(&self.stream, self.config.sync_timeout, op)
    }

    /// Compile (or fetch from the cache) `module` with `spec` and load `entry`.
    pub fn function(
        &self,
        module: &'static KernelModule,
        spec: &CompileSpec,
        entry: &str,
    ) -> Result<CudaFunction, CudaError> {
        if !module.entries.contains(&entry) {
            return Err(CudaError::invalid(
                "CudaRuntime::function",
                format!("module {} has no entry {entry}", module.name),
            ));
        }
        let key = CacheKey::new(module, spec, self.info.nvrtc_version);
        let loaded = self
            .lock_cache()
            .get_or_compile(key, module.source, || self.compile(module, spec))?;
        loaded.load_function(entry).map_err(|e| CudaError::Compile {
            module: module.name.to_string(),
            detail: format!("cuModuleGetFunction({entry}): {e}"),
        })
    }

    fn compile(
        &self,
        module: &'static KernelModule,
        spec: &CompileSpec,
    ) -> Result<(Arc<CudaModule>, usize), CudaError> {
        let opts = cudarc::nvrtc::CompileOptions {
            options: spec.options(),
            name: Some(format!("{}.cu", module.name)),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(module.source, opts).map_err(|e| {
            CudaError::Compile {
                module: module.name.to_string(),
                detail: e.to_string(),
            }
        })?;
        let bytes = ptx.as_bytes().map_or(0, <[u8]>::len);
        let loaded = self.ctx.load_module(ptx).map_err(|e| CudaError::Compile {
            module: module.name.to_string(),
            detail: format!("cuModuleLoadData: {e}"),
        })?;
        Ok((loaded, bytes))
    }

    /// Reserve budget for `len` elements of `T`, after checking the device has
    /// that much free.
    pub(crate) fn reserve<T>(&self, len: usize, label: &str) -> Result<Reservation, CudaError> {
        if len == 0 {
            return Err(CudaError::invalid(
                label,
                "a 0-element device buffer is refused; skip the call instead",
            ));
        }
        let bytes = bytes_for(len, std::mem::size_of::<T>(), label)?;
        let reservation = self.budget.reserve(bytes, label)?;
        self.check_free(bytes, label)?;
        Ok(reservation)
    }

    /// Refuse `bytes` more than the device has free, before the driver is
    /// asked to allocate them.
    pub(crate) fn check_free(&self, bytes: u64, label: &str) -> Result<(), CudaError> {
        let (free, _total) = self
            .ctx
            .mem_get_info()
            .map_err(|e| driver_error("cuMemGetInfo", e))?;
        if usize::try_from(bytes).map_or(true, |b| b > free) {
            return Err(CudaError::capacity(
                label,
                format!("{bytes} bytes requested, {free} bytes free on the device"),
            ));
        }
        Ok(())
    }

    /// A new device allocation holding a copy of `data`, already reserved by
    /// the caller. `clone_htod` allocates without the `memset` that
    /// `alloc_zeros` queues, since the copy overwrites every byte, and copies
    /// from `data` itself, with no staging copy on the host.
    ///
    /// Returns only once the copy has finished ([`Self::htod_done`]), so the
    /// caller may drop or change `data` at once.
    pub(crate) fn copy_in<T: DeviceRepr>(
        &self,
        data: &[T],
        label: &str,
    ) -> Result<CudaSlice<T>, CudaError> {
        let label = format!("upload {label}");
        let slice = self
            .stream
            .clone_htod(data)
            .map_err(|e| driver_error(&label, e))?;
        self.htod_done(&label)?;
        Ok(slice)
    }

    /// Wait (bounded) for a host-to-device copy from pageable memory just
    /// queued on the stream. NVIDIA documents such an async copy only as one
    /// that "might be synchronous with respect to host" (driver API,
    /// "API synchronization behavior"), and cudarc keeps nothing alive for a
    /// `&[T]` source, so until the copy is known done the source must not
    /// be freed or written. Waiting here makes that hold whatever the driver
    /// does, at the cost of also waiting for work queued before the copy.
    pub(crate) fn htod_done(&self, label: &str) -> Result<(), CudaError> {
        self.sync(label)
    }
}
