//! The host-neutral half of the Qwen3.5 training route: config, tensor names,
//! optimizer groups and the state-directory format.
//!
//! **There is no step here.** The tessl whole-step provider that used to live
//! in this crate (`Qwen35Step`, its gradient bank and host AdamW) was retired
//! with decision D5: ojas owns Metal training, through the Tape. A caller that
//! ran `Qwen35Step` builds the tower with
//! [`Qwen35TextConfig::tape_spec`] and steps it through
//! `ojas_model::qwen35::TapeStepProvider` (the `StepProvider` contract of
//! `ojas-train`). tessl stays underneath as the Metal extension: its training
//! kernels and per-op launchers are what `ojas-metal` dispatches.
//!
//! What this crate keeps, and why it lives in ojas:
//!
//! - [`config`]: a config reader that refuses every field the Tape tower does
//!   not implement, by name, cross-checked against tessl's own reader;
//! - [`names`]: the tower's tensor map and a header pre-flight of the snapshot
//!   through ojas-io, before any weight is read;
//! - [`groups`]: per-parameter learning-rate scales and weight decays built
//!   from the caller's rules, with no built-in exclusions;
//! - [`state`]: the masters and AdamW-moment directory format on disk through
//!   ojas-io (shard planner, streaming writer, checked reader).
//!
//! Off macOS the crate is empty: tessl builds only there.

#![cfg(target_os = "macos")]

pub mod config;
pub mod error;
pub mod groups;
pub mod names;
pub mod state;

pub use config::{LayerKind, Mrope, Qwen35TextConfig};
pub use error::{Qwen35Error, Result};
pub use groups::{GroupSpec, GroupSummary, LrRule, OptimizerPlan, Select, WdRule};
pub use names::{
    check_header, check_weights_file, tower_tensors, HeaderReport, Snapshot, TensorSpec,
};
pub use state::{
    check_new_state_dir, plan_shards, transpose_2d, write_kind, Kind, StateIndex, SHARD_BYTES,
    STATE_FORMAT,
};
