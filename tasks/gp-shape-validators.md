---
id: "gp-shape-validators"
title: "Move accumulate_grad and permute onto the shared ojas_core::shapes validators"
status: backlog
priority: 2
severity: low
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "shape-contract"
  - "backend"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/shapes.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/accum.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-wgpu/src/backend.rs"
acceptance_criteria:
  - "shapes.rs owns the accumulate_grad and permute checks, and every backend calls them"
  - "The per-backend copies are deleted"
  - "A cross-backend test pins one answer for a zero-length permute axis on CPU, Metal and wgpu"
---

# Task brief v1

## Title
Move accumulate_grad and permute onto the shared ojas_core::shapes validators

Task: gp-shape-validators
Type: refactor
Status: backlog
Priority: 2 (Normal)
Severity: low
Owner: unassigned
Due: none
Labels: shape-contract, backend

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Every other op's argument shapes are checked by one function in `ojas-core/src/shapes.rs` (`docs/shape-contract.md`). `accumulate_grad` and `permute` are not. `shapes.rs` has no entry for either. CPU and Metal each have their own `accumulate_grad` check (`ojas-cpu/src/accum.rs:8-12`, `ojas-metal/src/backend.rs:519-521`), and `permute` checks inline (`ojas-core/src/backend.rs:143`). `docs/shape-contract.md:78-80` records the gap.

As a consequence (per `docs/pytorch-parity-plan.md:180-182` and `docs/shape-contract.md:69`, D17; reported, not re-verified), CPU accepts a zero-length permute axis that Metal refuses.

## Acceptance criteria
- [ ] shapes.rs owns the accumulate_grad and permute checks, and every backend calls them
- [ ] The per-backend copies are deleted
- [ ] A cross-backend test pins one answer for a zero-length permute axis on CPU, Metal and wgpu

## Planned files
- ojas-core/src/shapes.rs
- ojas-core/src/backend.rs
- ojas-cpu/src/accum.rs
- ojas-metal/src/backend.rs
- ojas-wgpu/src/backend.rs
