---
id: "gp-typed-device-lost-error"
title: "Give OjasError a typed device-lost variant instead of matching error text"
status: backlog
priority: 2
severity: medium
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "errors"
  - "capi"
  - "contract"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/error.rs"
  - "ojas-capi/src/lib.rs"
  - "ojas-wgpu/src/"
  - "ojas-metal/src/"
acceptance_criteria:
  - "OjasError has a typed device-lost variant (or a structured field) that Metal, wgpu and CUDA set at the point of detection"
  - "ojas-capi maps the kind from the variant, with no substring matching"
  - "A test per GPU backend proves a lost or poisoned device surfaces as the typed variant"
---

# Task brief v1

## Title
Give OjasError a typed device-lost variant instead of matching error text

Task: gp-typed-device-lost-error
Type: refactor
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: errors, capi, contract

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). The C API classifies device loss by searching the error message: `ojas-capi/src/lib.rs:104-105` maps `OjasError::Backend { detail, .. }` to a kind when `detail.contains("device lost") || detail.contains("runtime poisoned")`. A reworded message in any backend silently changes the kind the Go side sees. `docs/pytorch-parity-plan.md:187` records this. Commit 85d1f5a names a lost wgpu device in every device-call error, which makes a typed variant straightforward.

## Acceptance criteria
- [ ] OjasError has a typed device-lost variant (or a structured field) that Metal, wgpu and CUDA set at the point of detection
- [ ] ojas-capi maps the kind from the variant, with no substring matching
- [ ] A test per GPU backend proves a lost or poisoned device surfaces as the typed variant

## Planned files
- ojas-core/src/error.rs
- ojas-capi/src/lib.rs
- ojas-wgpu/src/
- ojas-metal/src/
