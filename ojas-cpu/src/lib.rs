//! CPU reference for the nanolab-default ops.
//!
//! Reductions walk the reduction axis in increasing index order and accumulate
//! in `f32`. The exceptions are the AdamW moments (the frozen contract forms
//! those scalars in `f64`) and the clip total norm, whose sum of squares is
//! `f64` so a finite norm is not refused. [`CpuBackend::new`] is one thread.
//! [`CpuBackend::with_threads`] partitions independent output rows with
//! `std::thread::scope` and does not split a reduction, so the bits do not
//! depend on the thread count. There is no `mul_add`. Head dimension above 64
//! is not truncated here; [`metal_head_dim_policy`] is the check the Metal
//! path calls.

#![forbid(unsafe_code)]

mod attn;
mod backend;
mod linalg;
mod norm;
mod optim;
mod pointwise;
mod schedule;
mod train;
mod validate;

pub use backend::CpuBackend;
pub use schedule::{scaled_lr, CosineSchedule, COSINE_FLOOR_FRAC};
pub use train::{
    clip_grads, mean_micrograds, optim_group, GradAccumulator, HybridOptimizer, HybridParam,
    OptimGroup, ADAM_HYBRID_WEIGHT_DECAY, MUON_MOMENTUM, MUON_WEIGHT_DECAY,
};

/// Metal flash-attention refuses `head_dim > 64`.
///
/// [`CpuBackend`] causal attention does not call this and may compute any
/// positive head dimension. A Metal launch should call it and return
/// [`ojas_core::OjasError::UnsupportedHeadDim`] instead of clamping to 64.
pub fn metal_head_dim_policy(head_dim: u32) -> Result<(), ojas_core::OjasError> {
    ojas_core::refuse_unsupported_metal_head_dim(ojas_core::BackendId::Metal, head_dim)
}
