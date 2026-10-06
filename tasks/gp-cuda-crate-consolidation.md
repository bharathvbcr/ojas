---
id: "gp-cuda-crate-consolidation"
title: "Finish merging ojas-qwen35-cuda into ojas-cuda: move its tests, delete the leftover crate"
status: ready
priority: 1
severity: high
type: chore
owner: "unassigned"
due: "none"
labels:
  - "cuda"
  - "tests"
  - "cleanup"
repositories:
  - "ojas"
planned_files:
  - "ojas-cuda/tests/"
  - "ojas-cuda/src/buffer.rs"
  - "ojas-cuda/src/lib.rs"
  - "ojas-cuda/src/runtime.rs"
  - "ojas-qwen35-cuda/"
  - ".github/workflows/test.yml"
acceptance_criteria:
  - "Every test file under ojas-qwen35-cuda/tests is moved to ojas-cuda/tests (or deliberately dropped, with the reason in the commit)"
  - "The buffer.rs, lib.rs and runtime.rs differences between the two crates are reconciled into ojas-cuda"
  - "Host-side CUDA tests run in CI, and the device tests at least build there (cargo test -p ojas-cuda --features cuda --no-run)"
  - "ojas-qwen35-cuda/ is removed from the tree, and no doc or Cargo.toml still refers to it"
---

# Task brief v1

## Title
Finish merging ojas-qwen35-cuda into ojas-cuda: move its tests, delete the leftover crate

Task: gp-cuda-crate-consolidation
Type: chore
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: cuda, tests, cleanup

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Commit 1f26a4b copied the `ojas-qwen35-cuda` sources into `ojas-cuda/src`, but left the old crate in place: 443 tracked files with their own workspace and lock file. It is not a workspace member, and no CI job builds it. `ojas-qwen35-cuda/Cargo.toml:12-16` and its README say the standalone workspace goes away once it is merged into ojas-cuda.

The device tests did not move. `ojas-cuda/tests/` holds only `fixtures/`, while about 278 tests (43 `#[ignore]`d sm_90 device tests in `tests/device_*.rs`, plus host-side `reference_*.rs`, `fixture_pins.rs`, `runtime_refusal.rs`, `fmt_boundary.rs`) still live only in the leftover crate. CI's gpu-compile job runs only `cargo check -p ojas-cuda --features cuda`, so ~15k duplicated lines are compiled with none of their tests.

Per the audit, the shared `src` files are byte-identical except `buffer.rs`, `lib.rs` and `runtime.rs`; reconcile those three before deleting. This blocks gp-cuda-backend-provider's "real GPU tests" criterion.

## Acceptance criteria
- [ ] Every test file under ojas-qwen35-cuda/tests is moved to ojas-cuda/tests (or deliberately dropped, with the reason in the commit)
- [ ] The buffer.rs, lib.rs and runtime.rs differences between the two crates are reconciled into ojas-cuda
- [ ] Host-side CUDA tests run in CI, and the device tests at least build there (cargo test -p ojas-cuda --features cuda --no-run)
- [ ] ojas-qwen35-cuda/ is removed from the tree, and no doc or Cargo.toml still refers to it

## Planned files
- ojas-cuda/tests/
- ojas-cuda/src/buffer.rs
- ojas-cuda/src/lib.rs
- ojas-cuda/src/runtime.rs
- ojas-qwen35-cuda/
- .github/workflows/test.yml
