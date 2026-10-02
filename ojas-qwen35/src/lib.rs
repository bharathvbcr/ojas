//! A Qwen3.5 whole-step training provider for ojas, over canonical tessl's
//! Metal kernels.
//!
//! **Metal only, and not an ojas `Backend`.** The step (forward, the letter
//! cross-entropy over a 248,320-row tied head in vocabulary chunks, hidden
//! rows for an outside loss, the backward with GDN, head-dim-256 grouped
//! attention, Qwen's `(1 + w)` and gated RMSNorms and partial RoPE, the
//! gradient bank, its norm, and AdamW) is tessl's, run as one provider-level
//! unit. None of those ops is added to `ojas_core::Backend`; promoting them is
//! a trait decision that has to cover CPU, Metal and wgpu.
//!
//! What this crate adds over tessl, and why it lives in ojas:
//!
//! - [`config`]: a config reader that refuses every field the step does not
//!   implement, by name (tessl ignores MRoPE fields and checks some shapes only
//!   mid-step), cross-checked against tessl's own reader;
//! - [`names`]: the tower's tensor map and a header pre-flight of the snapshot
//!   through ojas-io, before tessl reads a byte;
//! - [`groups`]: per-parameter learning-rate scales and weight decays built
//!   from the caller's rules, with no built-in exclusions;
//! - [`step`]: the provider API in ojas types (`ojas_core::Tensor` for hidden
//!   rows and their gradient, `ojas_core::Budget` for host memory,
//!   ojas-core's `check_adamw` / `next_step` / `clip_scale` for the
//!   optimizer's scalars), with bank bookkeeping that fails closed;
//! - [`state`]: masters and AdamW moments on disk through ojas-io.
//!
//! Nothing here opens a GPU runtime except [`step::Qwen35Step::open`] and the
//! methods of an open provider.
//!
//! Off macOS the crate is empty: tessl builds only there.

#![cfg(target_os = "macos")]

pub mod config;
pub mod error;
pub mod groups;
pub mod names;
pub mod state;
pub mod step;

pub use config::{LayerKind, Mrope, Qwen35TextConfig};
pub use error::{Qwen35Error, Result};
pub use groups::{GroupSpec, GroupSummary, LrRule, OptimizerPlan, Select, WdRule};
pub use names::{
    check_header, check_weights_file, tower_tensors, HeaderReport, Snapshot, TensorSpec,
};
pub use state::{
    check_device_working_set, check_new_state_dir, plan_shards, transpose_2d, write_kind, Kind,
    NamedTensor, StateIndex, Which, SHARD_BYTES, STATE_FORMAT,
};
pub use step::{
    clip_coefficient, validate_external_grad, AdamWHyper, BankState, ExternalGrad, Numerics,
    Pending, Qwen35Step, Sequence,
};
