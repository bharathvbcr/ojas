//! From-scratch Metal training stack for arch_02.
//!
//! Phase 0–3: runtime, fwd/bwd, on-device optim.
//! Phase 4: bf16/async command path, FineWeb train harness, BPB eval.

#![allow(dead_code)]

pub mod ab_flags;

// The Metal runtime, tensor, dispatch and GEMM layers live in `tessl` and are
// re-exported here so every `crate::{runtime,tensor,dispatch,gemm}::` path in
// this crate keeps resolving. These were byte-for-byte forks kept in step by a
// static audit; tessl's copies were a strict superset of every item this crate
// used, so the forks were deleted rather than merged.
pub use tessl::{dispatch, gemm, npy, runtime, tensor};
pub mod bpb;
pub mod checkpoint;
pub mod data;

pub mod engine;
pub mod ffi;

pub mod init;
pub mod log;
pub mod mixers;
pub mod model_bwd;
pub mod model_fwd;
pub mod optim;
pub mod optimizer_registry;
pub mod parity;

pub mod research;
pub mod signpost;
pub mod ssm_glue;
pub mod tape;

pub mod weights;

pub use checkpoint::{
    collect_divergence_norms, collect_divergence_norms_device, load_ema_state_python_npy,
    load_muon_momentum_python_npy, load_optim_state_python_npy, read_training_checkpoint_meta,
    save_ema_state_python_npy, save_optim_state_python_npy, save_training_checkpoint,
    save_weights_python_npy, DivergenceNorms, TrainingCheckpointMeta, CHECKPOINT_VERSION,
};
pub use engine::{EngineCreateConfig, EngineStepMetrics, PrecisionModeConfig, TrainingEngine};
pub use optim::{
    clip_grad_norm_device, copy_ema_into_weights, lr_mul_at, optim_step, ClipMode, LrSchedule,
    OptimHyperparams, OptimState,
};
pub use optimizer_registry::OptimizerKind;
pub use runtime::{BufferKind, DeviceMemoryInfo, GpuRuntime, PrecisionMode};
pub use tensor::{DType, GpuBuffer, Tensor};
pub use tessl::gemm::{gemm_f32, GemmBackend};
pub use weights::{ModelConfig, Weights};

/// This crate's GPU runtime.
///
/// `GpuRuntime::new()` loads **tessl's** inference/runtime metallib. arch_02
/// needs its own training kernels too, so build.rs compiles only tessl's
/// canonical GEMM/runtime owner sources together with this crate's training
/// kernels into a single library — so every runtime here must be built from
/// [`metallib_path`]. Calling `GpuRuntime::new()` directly in this crate gets a
/// runtime that is missing every training kernel, and the failure surfaces late,
/// as "kernel 'x' not found in metallib" at first dispatch.
pub fn gpu_runtime() -> Result<std::sync::Arc<GpuRuntime>, String> {
    GpuRuntime::from_metallib_path(std::path::Path::new(metallib_path()))
}

/// Metallib produced by `build.rs` (absolute path baked at compile time).
pub fn metallib_path() -> &'static str {
    env!("METAL_NATIVE_METALLIB")
}
