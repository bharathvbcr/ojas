---
id: "gp-autograd-metal-tape-tests"
title: "Add Metal Backend Dev-Dependency and G6 Seeded Backward Tests to ojas-autograd"
status: ready
priority: 3
severity: low
type: testing
owner: "unassigned"
due: "none"
labels:
  - "autograd"
  - "metal"
  - "testing"
repositories:
  - "ojas"
planned_files:
  - "ojas-autograd/Cargo.toml"
  - "ojas-autograd/tests/device_tape.rs"
  - "ojas-autograd/tests/common/mod.rs"
acceptance_criteria:
  - "Add optional target-gated ojas-metal dev-dependency on macOS targets in ojas-autograd"
  - "Wire MetalBackend into autograd G6 seeded backward and multi-head attention block gradient checks"
  - "De-duplicate test helpers (Resident, Rng, lin, ce) across tests/common/mod.rs, device_tape.rs, and gradcheck_random.rs"
  - "Pass all autograd test suites cleanly across CPU, WGPU, and Metal"
---

# Task Brief: Add Metal Backend Dev-Dependency and G6 Seeded Backward Tests to ojas-autograd

## Context & Problem Statement
Currently, `ojas-autograd` tests the dynamic reverse-mode automatic differentiation tape against CPU, test double backends, and `ojas-wgpu`, but lacks an `ojas-metal` dev-dependency. As documented in `docs/pytorch-parity-plan.md` (§4 open residue), G6 (seeded backward pass) is verified on CPU and wgpu, but not directly on Metal inside `ojas-autograd`. Furthermore, autograd test helpers (`Resident`, `Rng`, `lin`, `ce`) are duplicated across `tests/common/mod.rs`, `device_tape.rs`, and `gradcheck_random.rs`.

## Scope & Implementation Details
1. **Target-Specific Dev-Dependency**:
   - Add `ojas-metal = { path = "../ojas-metal" }` under `[target.'cfg(target_os = "macos")'.dev-dependencies]` in `ojas-autograd/Cargo.toml`.
2. **Unified Test Helpers**:
   - Consolidate common tape fixtures and tensor generators into `tests/common/mod.rs`.
3. **Metal Tape Test Suites**:
   - Add Metal tape gradchecks and seeded backward tests under `tests/metal_tape.rs` or gated inside `tests/device_tape.rs`.
4. **Verification**:
   - Run `cargo test -p ojas-autograd --release` and verify all tests pass on macOS Apple Silicon.
