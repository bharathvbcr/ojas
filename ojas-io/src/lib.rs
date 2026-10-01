//! Safetensors and checkpoint v1 file IO.
//!
//! The header length cap is [`safetensors::MAX_HEADER_BYTES`] (100_000_000).
//! Tensor ranges must tile the data buffer. [`safetensors::SafeTensors::read_into`]
//! loads one range with a positioned read.

#![forbid(unsafe_code)]

mod checkpoint;
mod error;
mod json;
mod replace;
mod safetensors;
#[cfg(test)]
mod test_util;

pub use checkpoint::{decode_checkpoint, encode_checkpoint, read_checkpoint, write_checkpoint};
pub use error::IoError;
pub use safetensors::{
    encode_safetensors, write_safetensors, SafeTensors, StDtype, TensorInfo, TensorOut,
    MAX_HEADER_BYTES,
};

#[doc(hidden)]
pub use ojas_core::CheckpointV1;
