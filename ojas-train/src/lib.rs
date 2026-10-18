//! Training loop, schedules and recipes over ojas.
//!
//! **Here now:** [`sha256`], re-exported from `ojas_io::sha256` (its home, so
//! the checkpoint loader can hash a base file): a hand-rolled FIPS 180-4
//! SHA-256, streaming and one-shot, for artifact and checkpoint digests. ojas
//! has no `sha2` dependency, so this is the one implementation in the
//! workspace.
//!
//! **Moved in (phase A):** Lappi's `qd-train` modules `step`, `trainer`,
//! `schedule`, `recipe`, `adamw`, `mock`, `run_control` and `pyjson`, imported
//! with their history under [`lappi`]. JSON goes through `ojas_io::json` and
//! digests through `ojas_io::sha256`.
//!
//! This crate depends on `ojas-core` and `ojas-io` only. It does not depend on
//! `ojas-qwen35` or tessl, and it builds on every platform.

#![forbid(unsafe_code)]

pub mod lappi;

pub use ojas_io::sha256;
