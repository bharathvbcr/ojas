//! What a session holds: the model on its device, its trainer once one is
//! open, and its tokenizer (`docs/framework-design.md` §7).
//!
//! [`ModelState`] sits behind the session's `Arc<Mutex<_>>`. A call takes it
//! with `try_lock`; a held lock is `ojas:E_BUSY:`, never a wait.

use std::sync::Arc;

use ojas_core::{Autocast, Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_data::Bpe;
use ojas_device::{
    Device, MemoryArchitecture, MemoryProbe, MemoryReport, ResourcePlan, ResourcePolicy,
    SystemProfile,
};
use ojas_model::{ModelSpec, TrainState, Trainer};

use crate::gate::{CancelSlot, Check, Gated};
use crate::session::{self, DeviceKind, Lease};

/// The model's parameters: resident on the device, or owned by the trainer
/// (`Trainer::new` copies them; the resident copy is dropped once it exists).
pub enum Weights<B: Backend> {
    Resident(Vec<Tensor>),
    /// The trainer's backend is an autocast wrapper. The session engine stays
    /// the unwrapped backend, so inference does not enter a region.
    Training(Box<Trainer<Autocast<B>>>),
}

pub struct Model<B: Backend> {
    pub backend: B,
    pub spec: ModelSpec,
    pub weights: Weights<B>,
}

impl<B: Backend + Clone> Model<B> {
    /// `host` parameters in `param_table` order, uploaded to `backend`.
    pub fn resident(backend: B, spec: ModelSpec, host: Vec<Tensor>) -> Result<Self, OjasError> {
        let mut resident = Vec::with_capacity(host.len());
        for tensor in host {
            resident.push(backend.upload(&tensor)?);
        }
        Ok(Self {
            backend,
            spec,
            weights: Weights::Resident(resident),
        })
    }

    pub fn trainer(&self) -> Option<&Trainer<Autocast<B>>> {
        match &self.weights {
            Weights::Training(t) => Some(t),
            Weights::Resident(_) => None,
        }
    }

    /// The current parameters, shared, in `param_table` order. A poisoned
    /// trainer's are partly updated and are refused.
    pub fn params(&self) -> Result<Vec<Tensor>, OjasError> {
        match &self.weights {
            Weights::Resident(w) => Ok(w.clone()),
            Weights::Training(t) => {
                if t.state() == TrainState::Poisoned {
                    return Err(OjasError::Poisoned);
                }
                Ok(t.params().map(|(_, value)| value.clone()).collect())
            }
        }
    }
}

pub enum Engine {
    Cpu(Model<Gated<CpuBackend>>),
    #[cfg(target_os = "macos")]
    Metal(Model<Gated<ojas_metal::MetalBackend>>),
    Wgpu(Model<Gated<Arc<ojas_wgpu::WgpuBackend>>>),
}

/// Run `$body` with `$m` bound to the engine's `Model<_>`, whatever its
/// backend type.
macro_rules! on_model {
    ($engine:expr, $m:ident => $body:expr) => {
        match $engine {
            $crate::model::Engine::Cpu($m) => $body,
            #[cfg(target_os = "macos")]
            $crate::model::Engine::Metal($m) => $body,
            $crate::model::Engine::Wgpu($m) => $body,
        }
    };
}
pub(crate) use on_model;

impl Engine {
    pub fn spec(&self) -> ModelSpec {
        on_model!(self, m => m.spec)
    }

    pub fn tensors(&self) -> Result<u32, String> {
        let n = ojas_model::param_count(&self.spec()).map_err(|e| crate::ojas_error("load", &e))?;
        u32::try_from(n).map_err(|_| "load: parameter count exceeds u32".to_string())
    }
}

pub struct ModelState {
    pub engine: Engine,
    pub tokenizer: Option<Bpe>,
    pub slot: Arc<CancelSlot>,
    /// Declared last so it drops after the engine has released its charge:
    /// the process ceiling stays fixed until then.
    pub lease: Lease,
}

/// Where a new session computes and what it may hold.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    pub device: DeviceKind,
    pub budget_bytes: u64,
    /// `None` keeps the backend's default. Only the CPU takes one.
    pub numerics: Option<Numerics>,
}

/// A backend opened for a new session, before it is wrapped.
pub enum Opened {
    Cpu(CpuBackend),
    #[cfg(target_os = "macos")]
    Metal(ojas_metal::MetalBackend),
    Wgpu(Arc<ojas_wgpu::WgpuBackend>),
}

/// The memory probe of a GPU session's device, for [`ResourcePlan::derive`].
/// A CPU session plans with none: its memory is the host's, which the
/// system profile already reads.
#[derive(Clone, Copy, Debug)]
pub enum DeviceProbe {
    #[cfg(target_os = "macos")]
    Metal(ojas_metal::MetalMemory),
    Wgpu(ojas_wgpu::WgpuMemory),
}

/// Forwards every reading to the backend's own probe.
macro_rules! on_probe {
    ($probe:expr, $p:ident => $body:expr) => {
        match $probe {
            #[cfg(target_os = "macos")]
            DeviceProbe::Metal($p) => $body,
            DeviceProbe::Wgpu($p) => $body,
        }
    };
}

impl MemoryProbe for DeviceProbe {
    fn kind(&self) -> Device {
        on_probe!(self, p => p.kind())
    }
    fn memory_bytes(&self) -> MemoryReport {
        on_probe!(self, p => p.memory_bytes())
    }
    fn resident_bytes(&self) -> MemoryReport {
        on_probe!(self, p => p.resident_bytes())
    }
    fn pool_cache_bytes(&self) -> MemoryReport {
        on_probe!(self, p => p.pool_cache_bytes())
    }
    fn architecture(&self) -> MemoryArchitecture {
        on_probe!(self, p => p.architecture())
    }
}

/// What `opened`'s device reports about its memory; `None` for the CPU.
pub fn device_probe(opened: &Opened) -> Result<Option<DeviceProbe>, String> {
    match opened {
        Opened::Cpu(_) => Ok(None),
        #[cfg(target_os = "macos")]
        Opened::Metal(b) => b
            .memory()
            .map(|m| Some(DeviceProbe::Metal(m)))
            .map_err(|e| crate::ojas_error("metal", &e)),
        Opened::Wgpu(b) => Ok(Some(DeviceProbe::Wgpu(b.context().memory()))),
    }
}

/// The plan for a session of `caller` bytes on `probe`'s device: the device
/// first, then the CPU, against `profile`.
pub fn device_plan(caller: u64, probe: &DeviceProbe, profile: &SystemProfile) -> ResourcePlan {
    let mut policy = ResourcePolicy::new(caller);
    policy.devices = vec![probe.kind(), Device::Cpu];
    ResourcePlan::derive(&policy, profile, std::slice::from_ref(probe))
}

/// Open `placement`'s backend charging its own budget, a child of the
/// process ceiling ([`session::session_budget`]), and return the lease that
/// pins that ceiling and the budget's size. `check` is polled while a device
/// opens.
///
/// A GPU session's budget is then planned against what its device reports
/// ([`device_plan`], [`ResourcePlan::device_budget`]): a caller budget past
/// the device's room, or on shared memory past the host's limits, is cut to
/// that, and the session charges the smaller budget. A device with no room
/// at all is refused with `E_CAPACITY`. The budget is never raised.
pub fn open(
    placement: &Placement,
    check: impl FnMut() -> Result<(), String>,
) -> Result<(Opened, Lease, u64), String> {
    let caller = placement.budget_bytes;
    let (budget, lease) = session::session_budget(caller)?;
    let opened = open_on(placement, budget, check)?;
    let Some(probe) = device_probe(&opened)? else {
        return Ok((opened, lease, caller));
    };
    let plan = device_plan(caller, &probe, &ojas_device::probe_system());
    let planned = plan.device_budget(probe.kind()).unwrap_or(caller);
    if planned >= caller {
        return Ok((opened, lease, caller));
    }
    if planned == 0 {
        return Err(crate::kinded(
            crate::ErrorKind::Capacity,
            format!(
                "capacity exceeded: load: the {} device has no room for a session (device \
                 memory {:?}, held {:?}, pool cache {:?}, host budget {} bytes)",
                device_label(probe.kind()),
                plan.device_memory[0],
                probe.resident_bytes(),
                plan.device_pool_cache[0],
                plan.budget_bytes
            ),
        ));
    }
    // Nothing has been charged yet; the session takes a smaller child
    // instead, and the unused one and its lease go.
    let (budget, planned_lease) = session::session_budget(planned)?;
    drop(lease);
    Ok((rebudget(opened, budget), planned_lease, planned))
}

/// How messages name a device: wgpu is `Device::Vulkan` in the plan.
pub fn device_label(device: Device) -> &'static str {
    match device {
        Device::Metal => "Metal",
        Device::Vulkan => "wgpu",
        Device::Cpu => "CPU",
        Device::Cuda => "CUDA",
        Device::Hip => "HIP",
    }
}

/// `opened` charging `budget` instead, on the same device.
fn rebudget(opened: Opened, budget: Budget) -> Opened {
    match opened {
        Opened::Cpu(b) => Opened::Cpu(b),
        #[cfg(target_os = "macos")]
        Opened::Metal(b) => Opened::Metal(b.with_budget(budget)),
        Opened::Wgpu(b) => Opened::Wgpu(Arc::new(ojas_wgpu::WgpuBackend::with_context(
            b.context().clone(),
            budget,
        ))),
    }
}

fn open_on(
    placement: &Placement,
    budget: Budget,
    check: impl FnMut() -> Result<(), String>,
) -> Result<Opened, String> {
    match placement.device {
        DeviceKind::Cpu { threads } => {
            let cpu = if threads <= 1 {
                CpuBackend::new(budget)
            } else {
                let cpu = CpuBackend::with_threads(budget, threads)
                    .map_err(|e| crate::ojas_error("load", &e))?;
                // Every worker exists before the session does: a spawn
                // failure refuses the load, not a later step.
                cpu.start_workers()
                    .map_err(|e| crate::ojas_error("load", &e))?;
                cpu
            };
            Ok(Opened::Cpu(match placement.numerics {
                Some(n) => cpu.with_numerics(n),
                None => cpu,
            }))
        }
        #[cfg(target_os = "macos")]
        DeviceKind::Metal => crate::owner::open_metal(budget, check).map(Opened::Metal),
        #[cfg(not(target_os = "macos"))]
        DeviceKind::Metal => {
            crate::owner::open_metal(check).and(Err("metal: Metal requires macOS".to_string()))
        }
        DeviceKind::Wgpu => {
            crate::owner::open_wgpu(budget, check).map(|w| Opened::Wgpu(Arc::new(w)))
        }
    }
}

/// Builds a session's model on whatever backend it was given.
pub trait Build {
    /// The parameters' bytes, when known before the build, so a session
    /// budget planned below them is refused before anything uploads.
    fn param_bytes(&self) -> Option<u64> {
        None
    }

    fn build<B: Backend + Clone>(self, backend: B) -> Result<Model<B>, OjasError>;
}

/// The session state for `opened`, built by `build` with a fresh cancel
/// slot armed with `check` while it runs (uploads poll it). The state keeps
/// `lease` until it is dropped.
pub fn state_on(
    opened: Opened,
    lease: Lease,
    check: Check,
    build: impl Build,
) -> Result<ModelState, String> {
    let slot = Arc::new(CancelSlot::default());
    let armed = slot.arm(check);
    let engine = match opened {
        // The CPU reports every fault from the op that caused it.
        Opened::Cpu(b) => build
            .build(Gated::new(b, Arc::clone(&slot)))
            .map(Engine::Cpu),
        #[cfg(target_os = "macos")]
        Opened::Metal(b) => {
            let backend = Gated::new(b, Arc::clone(&slot));
            let sync = backend.clone();
            let built = build.build(backend).map(Engine::Metal);
            settle_build(&sync, built)
        }
        Opened::Wgpu(b) => {
            let backend = Gated::new(b, Arc::clone(&slot));
            let sync = backend.clone();
            let built = build.build(backend).map(Engine::Wgpu);
            settle_build(&sync, built)
        }
    };
    let engine = engine.map_err(|e| armed.report("load", &e))?;
    drop(armed);
    Ok(ModelState {
        engine,
        tokenizer: None,
        slot,
        lease,
    })
}

/// A device build ends with the backend's deferred faults drained, so a
/// fault from an upload is this call's error, not the session's next.
fn settle_build<B: Backend>(
    backend: &B,
    built: Result<Engine, OjasError>,
) -> Result<Engine, OjasError> {
    match backend.sync() {
        Ok(()) => built,
        Err(fault) => Err(fault),
    }
}
