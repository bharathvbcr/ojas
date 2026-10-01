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
mod limits;
mod tensor;

pub use backend::{
    check_adamw, clip_scale, inverse_permutation, next_step, permute_output_shape, pow_u64,
    refuse_unsupported_metal_head_dim, require_ns5, sdpa_scale, AdamWConfig, Backend, BackendId,
    MuonNs5Config, Numerics, PerHeadGateGrad, ValueResidualGrad, ADAMW_BETA1, ADAMW_BETA2,
    ADAMW_EPS, CLIP_GRAD_NORM_EPS, MAX_PERMUTE_RANK, METAL_MAX_HEAD_DIM, MUON_NS5_A, MUON_NS5_B,
    MUON_NS5_C, MUON_NS_EPS, RMS_NORM_EPS,
};
pub use budget::{Budget, Reservation, Scratch};
pub use checkpoint::{
    prefix_bytes, read_prefix, CheckpointPrefix, CheckpointV1, DataCursor, NamedBlob,
    OptimizerCheckpoint, CHECKPOINT_MAGIC, CHECKPOINT_VERSION,
};
pub use dtype::DType;
pub use error::OjasError;
pub use limits::{shape_product, CPU_THREAD_CEILING, GIBIBYTE, KIBIBYTE, MEBIBYTE};
pub use tensor::{device_readbacks, DeviceBuffer, Tensor};
