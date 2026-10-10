---
id: "gp-split-oversized-files"
title: "Split the largest non-test source files along op-family seams: ojas-metal device.rs (3,095 non-test lines), ojas-wgpu backend.rs (2,885), ojas-metal backend.rs (2,310), ojas-cpu pointwise.rs (2,250), ojas-wgpu context.rs (1,996)"
status: backlog
priority: 3
severity: low
type: refactor
owner: "unassigned"
due: "none"
labels:
  - "refactor"
  - "metal"
  - "wgpu"
  - "cpu"
repositories:
  - "ojas"
planned_files:
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-wgpu/src/context.rs"
  - "ojas-cpu/src/pointwise.rs"
acceptance_criteria:
  - "Each listed file is split into modules by op family (attention, norms, optimizers, hybrid ops, transfer/pool) with no behaviour change: public items, test names and outputs are identical, and the diff is moves plus visibility"
  - "No resulting non-test module exceeds roughly 1,200 lines, or the exception is recorded"
  - "Done in one commit per file, while no other lane holds that file (checked with GitPulse collision risk first)"
---

# Task brief v1

## Title
Split the largest non-test source files along op-family seams: ojas-metal device.rs (3,095 non-test lines), ojas-wgpu backend.rs (2,885), ojas-metal backend.rs (2,310), ojas-cpu pointwise.rs (2,250), ojas-wgpu context.rs (1,996)

Task: gp-split-oversized-files
Type: refactor
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: refactor, metal, wgpu, cpu

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949). Non-test line counts (up to the first #[cfg(test)]) [A, C] are in the title. pointwise.rs mixes embedding, gate, mul/add, SiLU and dot kernels in one file. Several open briefs edit these files (gp-gpu-step-throughput, gp-structural-dedup, gp-wgpu-hybrid-ops, gp-oom-error-class). This is pure movement, so schedule it after those land or between them, never under a running lane. ojas-metal/src/gpu.rs (3,599 lines) disappears with gp-structural-dedup's tiny-step retirement and is out of scope here.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] Each listed file is split into modules by op family (attention, norms, optimizers, hybrid ops, transfer/pool) with no behaviour change: public items, test names and outputs are identical, and the diff is moves plus visibility
- [ ] No resulting non-test module exceeds roughly 1,200 lines, or the exception is recorded
- [ ] Done in one commit per file, while no other lane holds that file (checked with GitPulse collision risk first)

## Planned files
- ojas-metal/src/device.rs
- ojas-metal/src/backend.rs
- ojas-wgpu/src/backend.rs
- ojas-wgpu/src/context.rs
- ojas-cpu/src/pointwise.rs
