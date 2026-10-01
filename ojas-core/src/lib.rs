//! Frozen ojas API.
//!
//! `Tensor`, [`OjasError`], [`Backend`], the dtype policy constants, and
//! the checkpoint v1 types live here. Numeric kernels do not. This crate
//! depends only on the standard library.
//!
//! [`Backend`]: backend::Backend

#![forbid(unsafe_code)]

mod backend;
mod budget;
mod checkpoint;
mod dtype;
mod error;
mod tensor;

pub use backend::{
    next_step, refuse_unsupported_metal_head_dim, require_ns5, sdpa_scale, AdamWConfig, Backend,
    BackendId, MuonNs5Config, PerHeadGateGrad, ValueResidualGrad, ADAMW_BETA1, ADAMW_BETA2,
    ADAMW_EPS, CLIP_GRAD_NORM_EPS, METAL_MAX_HEAD_DIM, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C,
    MUON_NS_EPS, RMS_NORM_EPS,
};
pub use budget::{Budget, Reservation};
pub use checkpoint::{
    prefix_bytes, read_prefix, CheckpointPrefix, CheckpointV1, DataCursor, NamedBlob,
    OptimizerCheckpoint, CHECKPOINT_MAGIC, CHECKPOINT_VERSION,
};
pub use dtype::DType;
pub use error::OjasError;
pub use tensor::Tensor;
