//! Small f32 GPT for greedy decode.
//!
//! Weights come in as [`ojas_core::Tensor`] values and are copied to `f32`
//! at construction. The kernels live in this crate so it does not link
//! `ojas-cpu`. This is a pre-norm attention block plus a SwiGLU MLP and a
//! tied embedding head. It is not a claim of token-for-token torch parity.

#![forbid(unsafe_code)]

mod gpt;
mod kernels;

pub use gpt::{argmax_token, BlockWeights, CpuGpt, GptConfig, GptWeights, KvCache};
pub use kernels::{attend_one, causal_self_attention, embed, rms_norm};

#[doc(hidden)]
pub use ojas_core::OjasError;
