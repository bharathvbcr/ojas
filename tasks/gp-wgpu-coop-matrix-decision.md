---
id: "gp-wgpu-coop-matrix-decision"
title: "Decide: allow unsafe in ojas-wgpu for cooperative-matrix GEMM?"
status: backlog
priority: 2
severity: medium
type: spike
owner: "unassigned"
due: "none"
labels:
  - "wgpu"
  - "performance"
  - "needs-your-words"
repositories:
  - "ojas"
planned_files:
  - "docs/pytorch-parity-plan.md"
  - "ojas-wgpu/src/context.rs"
acceptance_criteria:
  - "The decision and its reason are recorded in docs/pytorch-parity-plan.md"
  - "If approved, a follow-up feature task is filed with a GEMM benchmark target"
---

# Task brief v1

## Title
Decide: allow unsafe in ojas-wgpu for cooperative-matrix GEMM?

Task: gp-wgpu-coop-matrix-decision
Type: spike
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: wgpu, performance, needs-your-words

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). wgpu GEMM runs at about 1.6-3.2 TFLOP/s against torch's 5.2 (`docs/pytorch-parity-plan.md:203,300`). Matrix units via wgpu cooperative-matrix would close most of that, but the feature needs `unsafe`, and `ojas-wgpu` forbids `unsafe`. The design is written (`docs/pytorch-parity-plan.md:219-223`); nothing is built, and `ojas-wgpu/src/context.rs:328` sets `ExperimentalFeatures::disabled()`.

The decision is yours: keep the crate `unsafe`-free (accept the gap), allow a small audited `unsafe` module behind a feature, or put matrix-unit GEMM in a separate crate.

## Acceptance criteria
- [ ] The decision and its reason are recorded in docs/pytorch-parity-plan.md
- [ ] If approved, a follow-up feature task is filed with a GEMM benchmark target

## Planned files
- docs/pytorch-parity-plan.md
- ojas-wgpu/src/context.rs
