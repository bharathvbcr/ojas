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
mod rope;
mod shapes;
mod tensor;

pub use autocast::{
    bf16_to_f32, f32_to_bf16, round_f32_to_bf16, Autocast, AutocastGuard, AutocastMode,
};
pub use backend::{
    check_adamw, clip_scale, next_step, pow_u64, refuse_bf16_operands, refuse_unsupported_metal_gdn,
    refuse_unsupported_metal_head_dim, require_ns5, sdpa_scale, AdamWConfig, Backend, BackendId,
    CeChunk, GatedRmsGrad, GdnForward, GdnGrad, GdnInputs, LinearCe, MuonNs5Config, Ns5Precision,
    Numerics, OptimizerKind, PerHeadGateGrad, ValueResidualGrad, ADAMW_BETA1, ADAMW_BETA2,
    ADAMW_EPS, CLIP_GRAD_NORM_EPS, GDN_CHECKPOINT_TOKENS, GDN_L2NORM_EPS, METAL_GDN_KEY_DIM,
    METAL_GDN_VALUE_BLOCK, METAL_MAX_HEAD_DIM, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS,
    RMS_NORM_EPS,
};
pub use budget::{Budget, Reservation, Scratch};
pub use checkpoint::{
    prefix_bytes, read_prefix, CheckpointPrefix, CheckpointV1, DataCursor, NamedBlob,
    OptimizerCheckpoint, CHECKPOINT_MAGIC, CHECKPOINT_VERSION,
};
pub use dtype::DType;
pub use error::OjasError;
pub use exp_exact::{exp_exact, log_sum_exp_exact};
pub use limits::{shape_product, CPU_THREAD_CEILING, GIBIBYTE, KIBIBYTE, MEBIBYTE};
pub use rope::{mrope_tables, mrope_text_tables, MropeSection};
pub use shapes::{
    accumulate_grad_dims, adamw_step_dims, cached_attention_dims, causal_conv1d_silu_backward_dims,
    causal_conv1d_silu_forward_dims, causal_sdpa_backward_dims, causal_sdpa_forward_dims,
    chunked_gdn_backward_dims, chunked_gdn_forward_dims, clip_grad_norm_dims,
    cross_entropy_mean_backward_dims, cross_entropy_mean_forward_dims, embedding_backward_dims,
    embedding_forward_dims, gated_rms_norm_backward_dims, gated_rms_norm_forward_dims,
    inverse_permutation, kv_cache_write_dims, linear_backward_dims, linear_ce_dims,
    linear_forward_dims, mul_backward_dims, mul_forward_dims, muon_ns5_step_dims,
    per_head_sigmoid_gate_backward_dims, per_head_sigmoid_gate_forward_dims, permute_dims,
    permute_output_shape, residual_add_backward_dims, residual_add_forward_dims,
    rms_norm_backward_dims, rms_norm_forward_dims, rms_qk_norm_backward_dims,
    rms_qk_norm_forward_dims, rope_half_split_backward_dims, rope_half_split_forward_dims,
    rope_partial_backward_dims, rope_partial_forward_dims, silu_backward_dims, silu_forward_dims,
    value_residual_blend_backward_dims, value_residual_blend_forward_dims, CeDims, Conv1dDims,
    EmbeddingDims, GateDims, GdnDims, KvDims, LinearCeDims, LinearDims, MuonDims, PartialRopeDims,
    RmsDims, RopeDims, RopeLayout, SdpaDims, MAX_PERMUTE_RANK,
};
pub use tensor::{
    device_readbacks, f32_all_finite, DeviceBuffer, HostElement, Tensor, LE_READ_CHUNK_BYTES,
    READBACK_CHUNK_BYTES,
};
