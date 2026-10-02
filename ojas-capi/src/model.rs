//! What a session holds: the model on its device, its trainer once one is
//! open, and its tokenizer (`docs/framework-design.md` §7).
//!
//! [`ModelState`] sits behind the session's `Arc<Mutex<_>>`. A call takes it
//! with `try_lock`; a held lock is `ojas:E_BUSY:`, never a wait.

use std::sync::Arc;

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_data::Bpe;
use ojas_model::{ModelSpec, TrainState, Trainer};

use crate::gate::{CancelSlot, Check, Gated};
use crate::session::{self, DeviceKind, Lease};

/// The model's parameters: resident on the device, or owned by the trainer
/// (`Trainer::new` copies them; the resident copy is dropped once it exists).
pub enum Weights<B: Backend> {
    Resident(Vec<Tensor>),
    Training(Box<Trainer<B>>),
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

    pub fn trainer(&self) -> Option<&Trainer<B>> {
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

/// Open `placement`'s backend charging its own budget, a child of the
/// process ceiling ([`session::session_budget`]), and return the lease that
/// pins that ceiling. `check` is polled while a device opens.
pub fn open(
    placement: &Placement,
    check: impl FnMut() -> Result<(), String>,
) -> Result<(Opened, Lease), String> {
    let (budget, lease) = session::session_budget(placement.budget_bytes)?;
    let opened = open_on(placement, budget, check)?;
    Ok((opened, lease))
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
                CpuBackend::with_threads(budget, threads)
                    .map_err(|e| crate::ojas_error("load", &e))?
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
