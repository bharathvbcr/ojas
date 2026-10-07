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
//! default) uses `mul_add`, larger GEMM blocks, blocked attention above 256
//! positions, the vector exponential of `exp.rs` with fixed lane sums for
//! SiLU backward and cross-entropy, Accelerate `vvexpf` for macOS Fast SiLU
//! forward and the per-head gate sigmoid (`e^{-|x|}`, then the same sigmoid
//! pair), and fixed-block `f64` partial sums for the clip
//! norm; its bits do not depend on the thread count, except that on
//! macOS a Fast GEMM of at least [`FAST_WHOLE_CALL_MACS`] multiply-adds
//! (2¹³ there, 2²¹ elsewhere) is one Accelerate call whose order Apple does
//! not specify. A one-row linear backward does not send the weight gradient
//! through that call: the product is an outer product of two contiguous
//! vectors, written as one `mul_add` from `+0.0` per element, because
//! Accelerate's `sgemm` spent the call in `memset`. Under [`ojas_core::Numerics::Exact`]
//! (opt in with [`CpuBackend::with_numerics`]) every output reduces from
//! index 0 with no `mul_add`, and the bits do not depend on the thread count.
//! The exception is Muon's Nesterov blend and parameter update, which are
//! torch's fused `add(.., alpha=..)` under both tiers: one correctly rounded
//! `mul_add` per value, the same on every platform.
//! Off macOS those GEMMs use `ojas_simd::sgemm_tile`, which keeps the
//! thread-count guarantee. Any positive head dimension is computed here; the
//! Metal limit is `ojas_core::refuse_unsupported_metal_head_dim`.
//!
//! Where Fast and Exact agree, as the tests assert it. "Bit-equal" means the
//! test compares bits; a tolerance is `max |fast - ref| / max |ref|` per
//! tensor.
//!
//! | Op | Fast against Exact | Test |
//! | :--- | :--- | :--- |
//! | RMSNorm forward and backward | bit-equal | `numerics.rs`, `pointwise_parallel.rs` |
//! | RoPE forward and backward | bit-equal | `numerics.rs` |
//! | Muon momentum | bit-equal | `numerics.rs` |
//! | Permute | bit-equal (moves bits) | `pointwise_parallel.rs` |
//! | Clip scale pass, given the norm | bit-equal (one multiply) | `heavy_ops.rs` |
//! | Clip norm | 1e-7 of the `f64` sum; Exact is that sum rounded | `heavy_ops.rs` |
//! | Causal SDPA, T ≤ 256 | bit-equal (tested at T = 256) | `attention_fast.rs` |
//! | Causal SDPA, T > 256 (blocked) | 2e-6 | `numerics.rs`, `attention_fast.rs` |
//! | Linear, matmul | 2e-6 | `numerics.rs` |
//! | Muon parameter | 1e-6 | `numerics.rs` |
//! | AdamW parameter and both moments | 1e-6 (1.2e-7 measured); Fast does the element arithmetic in f32 and is bit-equal to an f32 scalar reference at every thread count | `numerics.rs`, `heavy_ops.rs` |
//! | Cross-entropy loss and gradient | 1e-6 | `numerics.rs` |
//! | SiLU forward and backward | 1e-6 of `f64`, per element | `pointwise_parallel.rs` |
//! | Value residual, λ gradient | 1e-6 of `f64` | `pointwise_parallel.rs` |
//! | Per-head gate | 1e-5 of `f64` | `pointwise_parallel.rs` |
//! | Gated delta rule forward and backward | bit-equal (one kernel for both) | `ojas-oracle` `gdn.rs` |
//!
//! The last three compare Fast with an `f64` reference, not with Exact.
//! Every Fast output above has the same bits at every thread count those
//! tests try (between 1 and 18). Several of those tests also run GEMMs of
//! at least [`FAST_WHOLE_CALL_MACS`]. On macOS each of those is one Accelerate call
//! whatever the pool size, so there the same bits rest on Accelerate
//! repeating itself, which Apple does not document.

#![forbid(unsafe_code)]

mod accum;
mod attn;
mod backend;
mod exp;
mod fused_ce;
mod gdn;
mod gemm;
mod hybrid;
mod kv;
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
pub use gemm::FAST_WHOLE_CALL_MACS;
pub use schedule::{
    scaled_lr, CosineSchedule, LrSchedule, WsdSchedule, COSINE_FLOOR_FRAC, WSD_DECAY_FRAC,
};
pub use train::{
    clip_grads, mean_micrograds, optim_group, GradAccumulator, HybridOptimizer, HybridParam,
    OptimGroup, ADAM_HYBRID_WEIGHT_DECAY, MUON_MOMENTUM, MUON_WEIGHT_DECAY,
};
