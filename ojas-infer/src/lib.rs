//! Nanolab GPT inference on the CPU.
//!
//! Weights come in as [`ojas_core::Tensor`] values. [`CpuGpt`] is nanolab's
//! default block: pre-RMSNorm, RMS QK-norm then half-split RoPE at the
//! absolute position, GQA causal attention, per-head sigmoid output gate,
//! value residual from layer 0, SwiGLU MLP, tied embedding head. The
//! one-row linears and single-query attention over the [`KvCache`] are
//! local; norms, RoPE, gate, value residual and SwiGLU's pointwise ops run
//! on [`ojas_cpu::CpuBackend`]. [`CpuGpt::forward_sequence`] runs the same
//! model through the trainer's batched [`ojas_core::Backend`] ops, and the
//! tests hold cached decode to it and to an f64 transcription of nanolab.
//!
//! [`CpuGpt::generate`] samples with temperature, top-k and top-p from a
//! seeded [`SplitMix64`]; [`CpuGpt::greedy_decode`] is the greedy case.

#![forbid(unsafe_code)]

mod gpt;
mod kernels;
mod sample;

pub use gpt::{argmax_token, BlockWeights, CpuGpt, GptConfig, GptWeights, KvCache};
pub use kernels::attend_one;
pub use sample::{sample_token, GenerateConfig, SamplingConfig, SplitMix64};

#[doc(hidden)]
pub use ojas_core::OjasError;
