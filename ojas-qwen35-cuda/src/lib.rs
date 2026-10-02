//! A Qwen3.5 whole-step training provider on NVIDIA CUDA: design (B) in
//! Lappi's `AUDIT/ojas-training-2026-10-01/cuda-backend-scoping.md`, the CUDA
//! counterpart of `ojas-qwen35` (which runs the same step on Metal through
//! tessl).
//!
//! **Not an ojas `Backend`.** The step runs as one provider-level unit:
//! forward, the letter cross-entropy over the tied 248,320-row head, hidden
//! rows for an outside loss, the backward with GDN at the published rule,
//! head-dim-256 grouped attention, the gradient bank, its norm, and AdamW.
//! None of those ops is added to `ojas_core::Backend`; promoting them is the
//! ojas coordinator's trait decision.
//!
//! The target is the parity ladder (rungs a–d), not campaign speed (user
//! decision, 2026-10-01). Every kernel is tested against a float64 host
//! reference before it is timed. GDN fixtures name the `published` rule.
//!
//! Everything that touches a device is behind the `cuda` feature. Without it,
//! the crate builds on any host and its host-side tests run there.
//!
//! # What is here (milestone M0, kernels K0 and K1)
//!
//! Host-side, built and tested everywhere:
//! - [`error`]: the crate's one error enum.
//! - [`bf16`]: round-to-nearest-even f32 to bf16, tessl's algorithm.
//! - [`budget`]: the bounded device-allocation budget.
//! - [`nvrtc_cache`]: the bounded compile cache, keyed by source hash,
//!   options (architecture included) and NVRTC version.
//! - [`kernels`]: the CUDA-C sources and their compile options.
//! - [`k0_plan`], [`gemm_plan`], [`geometry`]: validated plans, the cuBLAS
//!   row-major mapping, and launch shapes that never depend on SM count.
//! - [`host_ref`]: host references (bitwise for K0 and the FFMA GEMM, f64 for
//!   GEMM tolerances).
//! - [`check`], [`json`], [`libprobe`], [`rung0_cli`], [`inputs`]: what the
//!   rung-0 binary and the device tests share.
//!
//! Device-side, behind `cuda`: `runtime` ([`runtime::CudaRuntime`]: library
//! probe, sm_90 check, one stream, cuBLAS handle), `buffer`
//! ([`buffer::CudaBuffer`]), `k0`, `gemm` (ExactF32 FFMA and bf16 cuBLAS
//! tiers), and `smoke` (the checks rung 0 runs).

#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod bf16;
pub mod budget;
pub mod ce_rows;
pub mod check;
pub mod conv1d;
pub mod embed;
pub mod error;
pub mod gates_published;
pub mod gdn_host;
pub mod gdn_kernels;
pub mod gdn_plan;
pub mod gemm_plan;
pub mod geometry;
pub mod host_ref;
pub mod inputs;
pub mod json;
pub mod k0_plan;
pub mod k11_golden;
pub mod k11_host;
pub mod k11_kernels;
pub mod k8_act;
pub mod k8_kernels;
pub mod k8_plan;
pub mod kernels;
pub mod libprobe;
pub mod npy;
pub mod nvrtc_cache;
pub mod qk_norm_rope;
pub mod report_cli;
pub mod rmsnorm;
pub mod rung0_cli;
pub mod small_common;
pub mod tiny_fixture_published;

#[cfg(feature = "cuda")]
pub mod buffer;
#[cfg(feature = "cuda")]
pub mod ce_rows_cuda;
#[cfg(feature = "cuda")]
pub mod conv1d_cuda;
#[cfg(feature = "cuda")]
pub mod embed_cuda;
#[cfg(feature = "cuda")]
pub mod gates_published_cuda;
#[cfg(feature = "cuda")]
pub mod gdn;
#[cfg(feature = "cuda")]
pub mod gdn_smoke;
#[cfg(feature = "cuda")]
pub mod gemm;
#[cfg(feature = "cuda")]
pub mod k0;
#[cfg(feature = "cuda")]
pub mod k11;
#[cfg(feature = "cuda")]
pub mod k11_smoke;
#[cfg(feature = "cuda")]
pub mod k8;
#[cfg(feature = "cuda")]
pub mod k8_smoke;
#[cfg(feature = "cuda")]
pub mod qk_norm_rope_cuda;
#[cfg(feature = "cuda")]
pub mod rmsnorm_cuda;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
pub mod small_common_cuda;
#[cfg(feature = "cuda")]
pub mod small_smoke;
#[cfg(feature = "cuda")]
pub mod smoke;

pub use error::CudaError;
