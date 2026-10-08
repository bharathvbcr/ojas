#![recursion_limit = "256"]
//! arch02-burn — Rust/Burn port of the Value-Residual GPT (arch_02).
//!
//! Faithful port of `train_gpt_sprint_native.py` with
//! `VALUE_RESIDUAL=1 GATED_ATTENTION=0`, targeting Apple Silicon
//! (Metal via Burn's wgpu backend). Full training loop:
//! Muon (NS5, momentum warmup) + AdamW groups, global grad clip,
//! EMA, sequential shard loader, BPB evaluation.

pub mod bpb;
pub mod config;
pub mod data;
pub mod ema;
pub mod log;
pub mod model;
pub mod optim;
