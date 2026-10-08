//! Gemma 4 Metal inference.
//!
//! Phase 1–3: config, banks, kernels, synthetic host forward + parity stubs.
//! Phase 4: Hot-resident Q4 decode stack (`gpu_model`) + bench tok/s.
//! Phase 5: MTP assistant / cross-KV / clustered head / verify (`mtp`).
//! Phase 6: 31B config deltas + `serve` OpenAI stub.
//!
//! Depends on [`tessl`] for Metal 4 encode / GEMM / MTLTensor prep.

#![allow(dead_code)]

pub use tessl;

pub mod config;
pub mod diag;
pub mod dflash;
pub mod error;
pub mod forward;
pub mod gpu_model;
pub mod kernels;
pub mod kv;
pub mod mtp;
pub mod parity;
pub mod ple;
pub mod quant;
pub mod scratch;
pub mod step_verify;
pub mod thermal;
pub mod tokenizer;
pub mod trace;
pub mod weights;

pub use config::{Gemma4AssistantConfig, Gemma4Config, Gemma4TextConfig, LayerType};
pub use dflash::{
    generate_with_dflash, generate_with_dflash_host, generate_with_dflash_speed, project_context,
    DFlashConfig,
    DFlashGpuConditioner, DFlashGpuDraft, HostDFlashDraft, DFLASH_31B_MASK_TOKEN_ID,
    DFLASH_31B_TARGET_LAYER_IDS, DFLASH_DEFAULT_BLOCK,
};
pub use error::{Error, Result};
pub use forward::{
    greedy_decode_host, host_forward_prefill, ForwardDumps, SyntheticE4bGraph,
};
pub use gpu_model::{
    GpuDecodeSession, GpuSynthModel, HiddenCapture, StepVerifyResult, VERIFY_MAX_M,
};
pub use kernels::{GemmaGpu, HotQuantBanks, KernelId, metallib_path as gemma_metallib_path};
pub use kv::{KvLayout, KvRole, KvRingBuffer, SharedKvBuffer, SharedKvId};
pub use mtp::{
    b31_assistant_preset, e4b_assistant_preset, verify_draft, AdaptiveDraftPolicy,
    ClusteredLmHead, MtpSession, VerifyResult,
};
pub use parity::{compare_activations, ActivationDump, CompareReport, RefBackend};
pub use ple::{PleBanks, METAL_MAX_BUFFER_BYTES};
pub use quant::{QuantMatrix, QuantScheme};
pub use scratch::{ActStorage, ScratchArena, ScratchPlan};
pub use step_verify::{
    accept_block, commit_accepted, compare_token_stream, generate_with_host_stub, host_stub_draft,
    BlockAccept,
};
pub use thermal::{
    evaluate as evaluate_thermal, run_sustained, ThermalGateConfig, ThermalReport, ThermalWindow,
    Verdict as ThermalVerdict,
};
pub use tokenizer::{GemmaTokenizer, TokenizerPaths};
pub use weights::{
    load_from_hf_dir, resolve_default_31b_mlx_cache, resolve_default_dflash_draft_cache,
    resolve_default_e4b_assistant_cache, resolve_default_e4b_mlx_cache, HostWeightBanks,
    LoadOptions,
};

/// Crate version tag.
pub fn version() -> &'static str {
    "gemma-metal-0.1.0-phase6"
}
