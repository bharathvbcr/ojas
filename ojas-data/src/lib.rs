//! Token bins, a seeded epoch batch sampler, a counter RNG, and a BPE
//! tokenizer.
//!
//! Twenty strings match tiktoken 0.12.0 `gpt2` `encode_ordinary`
//! ([`bpe::TIKTOKEN_GPT2_BYTE_IDENTITY`] is `verified-20-strings`). That check used
//! the Step-Audio-EditX virtualenv and the local Hugging Face table at
//! `MLSystemsLab/Step-Audio-EditX/funasr_detach/models/whisper/utils/assets/gpt2`
//! (`vocab.json`, `merges.txt`); it did not download ranks and it is not a
//! million-line check. System `python3` does not import tiktoken. Tests also read
//! `testdata/gpt2` when those two files are dropped in. [`bpe::fixture_bpe`] is a
//! separate tiny vocabulary.
//!
//! Token files are read with `File` positioned reads. There is no `memmap2`
//! dependency.

#![forbid(unsafe_code)]

mod bpe;
mod error;
mod gpt2_class;
mod rng;
mod sampler;
mod tokens;

pub use bpe::{
    bytes_to_unicode, fixture_bpe, gpt2_split, load_hf_gpt2, Bpe, BpeBuilder,
    TIKTOKEN_GPT2_BYTE_IDENTITY,
};
pub use error::DataError;
pub use rng::CounterRng;
pub use sampler::{Batch, BatchSampler, SamplerConfig};
pub use tokens::{TokenBin, FINEWEB_HEADER_BYTES, FINEWEB_MAGIC, FINEWEB_VERSION};

#[doc(hidden)]
pub use ojas_core::DType;
