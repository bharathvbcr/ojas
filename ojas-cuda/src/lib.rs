//! NVIDIA CUDA for ojas: the `ojas_core::Backend` implementation
//! ([`CudaBackend`]) and the Qwen3.5 whole-step training provider
//! ([`Qwen35Step`]), design (B) in `docs/cuda-backend-scoping.md`, the CUDA
//! counterpart of `ojas-qwen35`.
//!
//! The target is the parity ladder (rungs a–d), not campaign speed. Every
//! kernel is tested against a float64 host reference before it is timed. GDN
//! fixtures name the `published` rule.
//!
//! Everything that touches a device is behind the `cuda` feature. Without it,
//! the crate builds on any host and its host-side tests run there.
//!
//! # What is public
//!
//! The default build's API is the backend, the step provider and what they
//! return:
//! - [`backend`]: [`CudaBackend`]. [`CudaBackend::open`] is the device probe:
//!   without `cuda` it is `DeviceError::NotCompiled`, with it the runtime
//!   opens under the caller's budget and every failure keeps its kind.
//! - [`step`]: [`Qwen35Step`] and its contract. Every compute method refuses
//!   with `Unsupported` until its kernels are wired and run on a device.
//! - [`error`]: [`CudaError`], the crate's one error enum, and its one
//!   mapping to `OjasError` and `DeviceError`.
//! - Behind `cuda`, what `CudaBackend::with_runtime` and
//!   `CudaBackend::runtime` hand over: `runtime` (`CudaRuntime`, its
//!   `DeviceInfo` and `RuntimeConfig`) and `buffer` (`CudaBuffer`,
//!   `CudaDeviceBuffer`). They are plain names here because rustdoc without
//!   the feature does not see them.
//!
//! The C ABI (`ojas-capi`) does not link this crate yet: device 5 is refused
//! by name there until the provider lands (`go/api.go`), so it needs nothing
//! here. The runtime's own pieces (the allocation budget, the NVRTC cache, the
//! library probe, launch geometry and the compile specs) are private.
//!
//! # The `harness` feature
//!
//! Off by default. It compiles the parity ladder, which the backend and the
//! step provider do not call yet: the K0–K11 and K2(i) kernel wrappers and
//! their plans, the host references and bit-identical host emulations, and
//! what the rung binaries and the device tests share (the smokes, the report
//! and watchdog CLI, `check`, `json`, `npy`, the goldens, the tiny-fixture
//! loader and the deterministic inputs). The rung binaries require it, and the
//! crate's own tests turn it on through a dev-dependency on this crate, so
//! `cargo test -p ojas-cuda` still runs every host test. A ladder module moves
//! into the default build when the step provider starts calling it.
//!
//! Device tests live in `tests/device_*.rs`, `#[ignore]`d, for an sm_90 GPU.

#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod backend;
pub mod error;
pub mod step;

#[cfg(feature = "cuda")]
pub mod buffer;
#[cfg(feature = "cuda")]
pub mod runtime;

// The runtime's own pieces: private to the `cuda` build, public under
// `harness`, where the ladder, the rung binaries and the tests use them.
#[cfg(feature = "harness")]
pub mod budget;
#[cfg(all(feature = "cuda", not(feature = "harness")))]
mod budget;
#[cfg(feature = "harness")]
pub mod geometry;
#[cfg(all(feature = "cuda", not(feature = "harness")))]
mod geometry;
#[cfg(feature = "harness")]
pub mod kernels;
#[cfg(all(feature = "cuda", not(feature = "harness")))]
mod kernels;
#[cfg(feature = "harness")]
pub mod libprobe;
#[cfg(all(feature = "cuda", not(feature = "harness")))]
mod libprobe;
#[cfg(feature = "harness")]
pub mod nvrtc_cache;
#[cfg(all(feature = "cuda", not(feature = "harness")))]
mod nvrtc_cache;

// The parity ladder and its harness (`harness` feature, see the crate docs).
#[cfg(feature = "harness")]
pub mod bf16;
#[cfg(feature = "harness")]
pub mod ce_rows;
#[cfg(feature = "harness")]
pub mod check;
#[cfg(feature = "harness")]
pub mod conv1d;
#[cfg(feature = "harness")]
pub mod embed;
#[cfg(feature = "harness")]
pub mod gates_published;
#[cfg(feature = "harness")]
pub mod gdn_host;
#[cfg(feature = "harness")]
pub mod gdn_kernels;
#[cfg(feature = "harness")]
pub mod gdn_plan;
#[cfg(feature = "harness")]
pub mod gemm_plan;
#[cfg(feature = "harness")]
pub mod host_ref;
#[cfg(feature = "harness")]
pub mod inputs;
#[cfg(feature = "harness")]
pub mod json;
#[cfg(feature = "harness")]
pub mod k0_plan;
#[cfg(feature = "harness")]
pub mod k11_golden;
#[cfg(feature = "harness")]
pub mod k11_host;
#[cfg(feature = "harness")]
pub mod k11_kernels;
#[cfg(feature = "harness")]
pub mod k8_act;
#[cfg(feature = "harness")]
pub mod k8_kernels;
#[cfg(feature = "harness")]
pub mod k8_plan;
#[cfg(feature = "harness")]
pub mod npy;
#[cfg(feature = "harness")]
pub mod qk_norm_rope;
#[cfg(feature = "harness")]
pub mod report_cli;
#[cfg(feature = "harness")]
pub mod rmsnorm;
#[cfg(feature = "harness")]
pub mod rung0_cli;
#[cfg(feature = "harness")]
pub mod small_common;
#[cfg(feature = "harness")]
pub mod tiny_fixture_published;

#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod ce_rows_cuda;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod conv1d_cuda;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod embed_cuda;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod gates_published_cuda;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod gdn;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod gdn_smoke;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod gemm;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod k0;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod k11;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod k11_smoke;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod k8;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod k8_smoke;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod qk_norm_rope_cuda;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod rmsnorm_cuda;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod small_common_cuda;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod small_smoke;
#[cfg(all(feature = "cuda", feature = "harness"))]
pub mod smoke;

pub use backend::CudaBackend;
#[cfg(feature = "cuda")]
pub use buffer::{CudaBuffer, CudaDeviceBuffer};
pub use error::CudaError;
#[cfg(feature = "cuda")]
pub use runtime::{CudaRuntime, DeviceInfo, RuntimeConfig};
pub use step::{
    clip_coefficient, validate_external_grad, AdamWHyper, BankState, ExternalGrad, GemmOperands,
    Numerics, Pending, Qwen35Step, Sequence, StepProvider, Supervise,
};
