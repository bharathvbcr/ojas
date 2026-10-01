//! Safetensors and checkpoint v1 file IO.
//!
//! The header length cap is [`safetensors::MAX_HEADER_BYTES`] (100_000_000).
//! Tensor ranges must tile the data buffer. [`safetensors::SafeTensors::read_into`]
//! loads one range with a positioned read. Safetensors F32, BF16, F16, I64
//! and U16 are read and written; BF16 and F16 decode to f32 exactly and
//! encode with round-to-nearest-even ([`f32_to_bf16`], [`f32_to_f16`]).

#![forbid(unsafe_code)]

mod checkpoint;
mod error;
mod half;
mod json;
mod replace;
mod safetensors;
#[cfg(test)]
mod test_util;

pub use checkpoint::{decode_checkpoint, encode_checkpoint, read_checkpoint, write_checkpoint};
pub use error::IoError;
pub use half::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
pub use safetensors::{
    encode_f32_as, encode_safetensors, write_safetensors, SafeTensors, StDtype, TensorInfo,
    TensorOut, MAX_HEADER_BYTES,
};

#[doc(hidden)]
pub use ojas_core::CheckpointV1;
