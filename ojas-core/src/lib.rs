//! Frozen ojas API.
//!
//! `Tensor`, [`OjasError`], [`Backend`], the dtype policy constants, and
//! the checkpoint v1 types live here. Numeric kernels do not. This crate
//! depends only on the standard library.
//!
//! [`Backend`]: backend::Backend

#![forbid(unsafe_code)]

mod autocast;
mod backend;
mod budget;
mod checkpoint;
mod dtype;
mod error;
mod exp_exact;
mod limits;
mod shapes;
mod tensor;

pub use autocast::{
    bf16_to_f32, f32_to_bf16, round_f32_to_bf16, Autocast, AutocastGuard, AutocastMode,
};
pub use backend::{
    check_adamw, clip_scale, inverse_permutation, next_step, permute_output_shape, pow_u64,
    refuse_unsupported_metal_head_dim, require_ns5, sdpa_scale, AdamWConfig, Backend, BackendId,
    CeChunk, LinearCe, MuonNs5Config, Numerics, OptimizerKind, PerHeadGateGrad, ValueResidualGrad,
    ADAMW_BETA1, ADAMW_BETA2, ADAMW_EPS, CLIP_GRAD_NORM_EPS, MAX_PERMUTE_RANK, METAL_MAX_HEAD_DIM,
    MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS, RMS_NORM_EPS,
};
pub use budget::{Budget, Reservation, Scratch};
pub use checkpoint::{
    prefix_bytes, read_prefix, CheckpointPrefix, CheckpointV1, DataCursor, NamedBlob,
    OptimizerCheckpoint, CHECKPOINT_MAGIC, CHECKPOINT_VERSION,
};
pub use dtype::DType;
pub use error::OjasError;
pub use exp_exact::exp_exact;
pub use limits::{shape_product, CPU_THREAD_CEILING, GIBIBYTE, KIBIBYTE, MEBIBYTE};
pub use shapes::{
    adamw_step_dims, cached_attention_dims, causal_sdpa_backward_dims, causal_sdpa_forward_dims,
    clip_grad_norm_dims, cross_entropy_mean_backward_dims, cross_entropy_mean_forward_dims,
    embedding_backward_dims, embedding_forward_dims, kv_cache_write_dims, linear_backward_dims,
    linear_ce_dims, linear_forward_dims, mul_backward_dims, mul_forward_dims, muon_ns5_step_dims,
    per_head_sigmoid_gate_backward_dims, per_head_sigmoid_gate_forward_dims,
    residual_add_backward_dims, residual_add_forward_dims, rms_norm_backward_dims,
    rms_norm_forward_dims, rms_qk_norm_backward_dims, rms_qk_norm_forward_dims,
    rope_half_split_backward_dims, rope_half_split_forward_dims, silu_backward_dims,
    silu_forward_dims, value_residual_blend_backward_dims, value_residual_blend_forward_dims,
    CeDims, EmbeddingDims, GateDims, KvDims, LinearCeDims, LinearDims, MuonDims, RmsDims, RopeDims,
    RopeLayout, SdpaDims,
};
pub use tensor::{
    device_readbacks, f32_all_finite, DeviceBuffer, HostElement, Tensor, LE_READ_CHUNK_BYTES,
    READBACK_CHUNK_BYTES,
};
