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

# Task brief v1

## Title
Add Metal Backend Dev-Dependency and G6 Seeded Backward Tests to ojas-autograd

Task: gp-autograd-metal-tape-tests
Type: testing
Status: ready
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: autograd, metal, testing

## Repositories
- ojas

## Description
Currently, `ojas-autograd` tests the dynamic reverse-mode automatic differentiation tape against CPU, test double backends, and `ojas-wgpu`, but lacks an `ojas-metal` dev-dependency. As documented in `docs/pytorch-parity-plan.md` (§4 open residue), G6 (seeded backward pass) is verified on CPU and wgpu, but not directly on Metal inside `ojas-autograd`. Furthermore, autograd test helpers (`Resident`, `Rng`, `lin`, `ce`) are duplicated across `tests/common/mod.rs`, `device_tape.rs`, and `gradcheck_random.rs`.

## Acceptance criteria
- [ ] Add optional target-gated ojas-metal dev-dependency on macOS targets in ojas-autograd
- [ ] Wire MetalBackend into autograd G6 seeded backward and multi-head attention block gradient checks
- [ ] De-duplicate test helpers (Resident, Rng, lin, ce) across tests/common/mod.rs, device_tape.rs, and gradcheck_random.rs
- [ ] Pass all autograd test suites cleanly across CPU, WGPU, and Metal

## Planned files
- ojas-autograd/Cargo.toml
- ojas-autograd/tests/device_tape.rs
- ojas-autograd/tests/common/mod.rs
