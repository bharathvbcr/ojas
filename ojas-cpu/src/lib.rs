//! CPU reference for the nanolab-default ops.
//!
//! Reductions walk the reduction axis in increasing index order and accumulate
//! in `f32`. The exceptions are the AdamW moments (the frozen contract forms
//! those scalars in `f64`) and the clip total norm, whose sum of squares is
//! `f64` so a finite norm is not refused. [`CpuBackend::new`] is one thread.
//! [`CpuBackend::with_threads`] gives the backend a persistent worker pool.
//! Linear, matmul and Newton-Schulz go through one packed GEMM that splits
//! output tiles, never the reduction; attention splits heads and query rows;
//! norms, RoPE, cross-entropy and the optimizers split rows or elements.
//! Small ops stay on the calling thread. [`ojas_core::Numerics::Fast`] (the
//! default) uses `mul_add`, larger GEMM blocks and blocked attention above
//! 256 positions; its bits do not depend on the thread count, except that on
//! macOS a Fast GEMM of at least 2²¹ multiply-adds is one Accelerate call
//! whose order Apple does not specify. Under [`ojas_core::Numerics::Exact`]
//! (opt in with [`CpuBackend::with_numerics`]) every output reduces from
//! index 0 with no `mul_add`, and the bits do not depend on the thread count.
//! Off macOS those GEMMs use `ojas_simd::sgemm_tile`, which keeps the
//! thread-count guarantee. Any positive head dimension is computed here; the
//! Metal limit is `ojas_core::refuse_unsupported_metal_head_dim`.

#![forbid(unsafe_code)]

mod attn;
mod backend;
mod gemm;
mod layout;
mod linalg;
mod norm;
mod optim;
mod pointwise;
mod pool;
mod schedule;
mod train;
mod validate;

pub use backend::CpuBackend;
pub use schedule::{scaled_lr, CosineSchedule, COSINE_FLOOR_FRAC};
pub use train::{
    clip_grads, mean_micrograds, optim_group, GradAccumulator, HybridOptimizer, HybridParam,
    OptimGroup, ADAM_HYBRID_WEIGHT_DECAY, MUON_MOMENTUM, MUON_WEIGHT_DECAY,
};

