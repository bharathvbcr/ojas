//! Nanolab GPT inference.
//!
//! The model is defined once, in `ojas-model`: [`GptConfig`] is its
//! [`ojas_model::ModelSpec`] and [`GptWeights`] its
//! [`ojas_model::ModelParams`], under nanolab's `state_dict` names. The
//! whole-sequence forward is `ojas_model::Eval` through
//! [`ojas_model::forward_logits`].
//!
//! - [`CpuGpt`] is the host path: [`DeviceDecoder`]'s step on an
//!   [`ojas_cpu::CpuBackend`] against a caller-owned host [`KvCache`].
//! - [`DeviceDecoder`] decodes on any [`ojas_core::Backend`] with the KV
//!   cache resident there: prefill and decode through `Eval` with
//!   `kv_cache_write` and `cached_attention_forward`, one logit-row
//!   readback per call, sampling on the host. It runs grouped-query
//!   attention.
//!
//! Both decode with [`argmax_token`] or [`sample_token`] (temperature, top-k
//! and top-p from a seeded [`SplitMix64`]) through one shared loop.

#![forbid(unsafe_code)]

mod cache;
mod decode;
mod device;
mod gpt;
mod sample;

pub use device::{DeviceDecoder, HostTraffic};
pub use gpt::{argmax_token, BlockWeights, CpuGpt, GptConfig, GptWeights, KvCache};
pub use ojas_model::{BlockParams, ModelParams, ModelSpec};
pub use sample::{sample_token, GenerateConfig, SamplingConfig, SplitMix64};

#[doc(hidden)]
pub use ojas_core::OjasError;
