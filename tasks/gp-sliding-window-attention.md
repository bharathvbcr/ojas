---
id: "gp-sliding-window-attention"
title: "Implement Sliding Window Local Causal Attention Kernels"
status: backlog
priority: 2
severity: low
type: feature
owner: "unassigned"
due: "none"
labels:
  - "attention"
  - "kernel"
  - "long-context"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-kernels/src/wgsl/"
acceptance_criteria:
  - "Add sliding window size parameter W to attention forward and backward dispatches"
  - "Ensure attention masks ignore keys beyond the causal sliding window (t - W < j <= t)"
  - "Implement block skipping in tiled attention kernels to avoid loading out-of-window tiles"
  - "Verify mathematical parity against CPU reference implementation"
---

# Task brief v1

## Title
Implement Sliding Window Local Causal Attention Kernels

Task: gp-sliding-window-attention
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: low
Owner: unassigned
Due: none
Labels: attention, kernel, long-context

## Repositories
- ojas

## Description
Long-context models like Mistral and Qwen hybrid variants use sliding-window local causal attention where tokens only attend to the last W positions. Ojas currently only supports full dense causal attention across all past tokens.

This task implements sliding window attention forward and backward kernels with tile skipping for zero compute on out-of-window key blocks.

## Acceptance criteria
- [ ] Add sliding window size parameter W to attention forward and backward dispatches
- [ ] Ensure attention masks ignore keys beyond the causal sliding window (t - W < j <= t)
- [ ] Implement block skipping in tiled attention kernels to avoid loading out-of-window tiles
- [ ] Verify mathematical parity against CPU reference implementation

## Planned files
- ojas-core/src/backend.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-kernels/src/wgsl/
