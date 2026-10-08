//! Nanolab GPT inference.
//!
//! The model is defined once, in `ojas-model`: [`GptConfig`] is its
//! [`ojas_model::ModelSpec`] and [`GptWeights`] its
//! [`ojas_model::ModelParams`], under nanolab's `state_dict` names. The
//! whole-sequence forward is `ojas_model::Eval` through
//! [`ojas_model::forward_logits`].
//!
//! - [`CpuGpt`] is the fast host path: one token at a time against a host
//!   [`KvCache`], with local one-row linears and single-query attention;
//!   norms, RoPE, gate, value residual and SwiGLU's pointwise ops run on
//!   [`ojas_cpu::CpuBackend`].
//! - [`DeviceDecoder`] decodes on any [`ojas_core::Backend`] with the KV
//!   cache resident there: prefill and decode through `Eval` with
//!   `kv_cache_write` and `cached_attention_forward`, one logit-row
//!   readback per call, sampling on the host. It runs grouped-query
//!   attention.
//!
//! Both decode with [`argmax_token`] or [`sample_token`] (temperature, top-k
//! and top-p from a seeded [`SplitMix64`]) through one shared loop.

#![forbid(unsafe_code)]

mod decode;
mod device;
mod gpt;
mod kernels;
mod sample;

pub use device::{DeviceDecoder, HostTraffic};
pub use gpt::{argmax_token, BlockWeights, CpuGpt, GptConfig, GptWeights, KvCache};
pub use kernels::attend_one;
pub use ojas_model::{BlockParams, ModelParams, ModelSpec};
pub use sample::{sample_token, GenerateConfig, SamplingConfig, SplitMix64};

#[doc(hidden)]
pub use ojas_core::OjasError;
