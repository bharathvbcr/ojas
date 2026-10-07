//! Safetensors and checkpoint v1 file IO.
//!
//! The header length cap is [`safetensors::MAX_HEADER_BYTES`] (100_000_000).
//! Tensor ranges must tile the data buffer. [`safetensors::SafeTensors::read_into`]
//! loads one range with a positioned read. Safetensors F32, BF16, F16, I64
//! and U16 are read and written; BF16 and F16 decode to f32 exactly and
//! encode with round-to-nearest-even ([`f32_to_bf16`], [`f32_to_f16`]).
//!
//! [`SafeTensorsWriter`] streams a file one tensor at a time after writing
//! the header up front; [`encode_safetensors`] and [`write_safetensors`] are
//! built on it. [`SafeTensors::from_file`] reads a file the caller opened.
//! [`replace_dir_with`] swaps in a whole directory; its docs give what a
//! reader sees if the writer dies at each step.

#![forbid(unsafe_code)]

mod checkpoint;
mod error;
mod half;
pub mod json;
mod open;
mod posread;
mod replace;
mod safetensors;
#[cfg(test)]
mod test_util;

pub use checkpoint::{
    checkpoint_file_len, decode_checkpoint, encode_checkpoint, read_checkpoint,
    read_checkpoint_from, write_checkpoint, MAX_CHECKPOINT_BYTES,
};
pub use error::IoError;
pub use half::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16};
pub use json::{parse_json, parse_json_with, JsonLimits, JsonNumber, JsonValue};
pub use open::open_nofollow;
pub use replace::{recover_replaced_dir, replace_dir_with};
pub use safetensors::{
    encode_f32_as, encode_safetensors, safetensors_file_len, write_safetensors, SafeTensors,
    SafeTensorsWriter, StDtype, TensorInfo, TensorOut, TensorSpec, MAX_HEADER_BYTES,
};

#[doc(hidden)]
pub use ojas_core::CheckpointV1;
