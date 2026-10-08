//! Crate-level errors for gemma-metal.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("config: {0}")]
    Config(String),
    #[error("io: {0}")]
    Io(String),
    #[error("weights: {0}")]
    Weights(String),
    #[error("quant: {0}")]
    Quant(String),
    #[error("ple: {0}")]
    Ple(String),
    #[error("kv: {0}")]
    Kv(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    Safetensors(String),
    #[error("metal: {0}")]
    Metal(String),
}
