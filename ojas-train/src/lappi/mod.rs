//! Lappi's `qd-train` modules, moved in with their history (phase A of card ft-b5e64557).
//!
//! Visibility and names follow Lappi's `crates/qd-train/src/lib.rs` for these eight modules
//! only, which re-exports nothing: every item is reached through its module path.

pub mod adamw;
pub mod mock;
pub mod pyjson;
pub mod recipe;
pub mod run_control;
pub mod schedule;
pub mod step;
pub mod trainer;
