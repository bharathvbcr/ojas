# Lappi import (phase A, card ft-b5e64557)

History-preserving import of Lappi `crates/qd-train` into `ojas-train` (git-filter-repo, path rewrite).

- Lappi freeze sha: `fecfd5d22003a439c066df1289e1a0ef0a751df8`
- Lappi `crates/qd-train/src/lib.rs` blob sha at the freeze: `962be359fa3f9a577bb3995925e6c09023f760b3` (lib.rs was NOT imported)
- Imported on: 2026-10-18 (UTC)
- Rewritten filtered tip (second parent of the import merge): `48de180b1fdcd41bf9db7c0448768d98af305dbb`

## Files (as they stand in ojas-train)

- `ojas-train/src/lappi/adamw.rs`
- `ojas-train/src/lappi/mock.rs`
- `ojas-train/src/lappi/pyjson.rs`
- `ojas-train/src/lappi/recipe.rs`
- `ojas-train/src/lappi/run_control.rs`
- `ojas-train/src/lappi/schedule.rs`
- `ojas-train/src/lappi/step.rs`
- `ojas-train/src/lappi/trainer.rs`
- `ojas-train/tests/lappi/adamw_decay_sensitive.rs`
- `ojas-train/tests/lappi/adamw_oracle.rs`
- `ojas-train/tests/lappi/common/mod.rs`
- `ojas-train/tests/lappi/common/pins.rs`
- `ojas-train/tests/lappi/fixtures/adamw-decay-sensitive/golden.safetensors`
- `ojas-train/tests/lappi/fixtures/adamw-decay-sensitive/inputs.safetensors`
- `ojas-train/tests/lappi/fixtures/adamw-decay-sensitive/manifest.json`
- `ojas-train/tests/lappi/fixtures/adamw-decay-sensitive/preregistration.json`
- `ojas-train/tests/lappi/fixtures/adamw-decay-sensitive/table.json`
- `ojas-train/tests/lappi/fixtures/trainer-oracle.json`
- `ojas-train/tests/lappi/pyjson_oracle.rs`
- `ojas-train/tests/lappi/schedule_oracle.rs`

## Moving tests and why

- `crates/qd-train/tests/adamw_decay_sensitive.rs`: imports only qd_train::{adamw,recipe,step} (all moving) plus common::{crate_dir,pins}; oracle for the decay-sensitive AdamW fixture
- `crates/qd-train/tests/adamw_oracle.rs`: imports only qd_train::{adamw,recipe,step} plus common::{fhex,fhex_list,trainer_oracle}; Python-oracle parity for AdamW (ruling Q5: adamw_* oracles move with their fixtures)
- `crates/qd-train/tests/schedule_oracle.rs`: imports only qd_train::schedule plus common::{fhex,fhex_list,trainer_oracle}; Python-oracle parity for the LR schedule (ruling Q5)
- `crates/qd-train/tests/pyjson_oracle.rs`: ruling Q5 names pyjson_oracle as moving; imports qd_train::{pyjson,run_control} (moving) BUT ALSO qd_train::ledger::isoformat_utc (NOT moving): UNSURE, phase A must re-point or cut that one import
- `crates/qd-train/tests/common`: helper module the moving tests declare with 'mod common'; mod.rs imports NON-moving qd_train::{files,held_out,shards,objective} and pins.rs is self-contained: UNSURE, phase A must split mod.rs down to crate_dir/fhex/fhex_list/trainer_oracle/ToyBatch-free helpers
- `crates/qd-train/tests/fixtures/trainer-oracle.json`: read by common::trainer_oracle(), used by adamw_oracle, schedule_oracle, pyjson_oracle
- `crates/qd-train/tests/fixtures/adamw-decay-sensitive`: fixture directory (golden/inputs/manifest/preregistration/table) read by adamw_decay_sensitive via common::pins

## State

This branch does not compile. Phase A still has to rewire the `qd_train::` paths in these files
and add `pub mod lappi` to `ojas-train/src/lib.rs`, then move `pyjson` onto `ojas_io::JsonValue`,
add SHA-256 per ruling Q4, and fix the flagged tests (pyjson_oracle imports `qd_train::ledger`;
`tests/common/mod.rs` imports non-moving shard/held_out/objective modules).
