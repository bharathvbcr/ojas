---
id: "gp-gpu-step-overhead"
title: "Cut per-step GPU overhead: waiting host uploads on Metal, recomputed gate sigmoid on Metal/wgpu"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "metal"
  - "wgpu"
  - "performance"
repositories:
  - "ojas"
planned_files:
  - "ojas-model/src/trainer.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-wgpu/src/backend.rs"
  - "bench/"
acceptance_criteria:
  - "Upload waits and memory-cap commits per step are counted by trigger, not estimated"
  - "Waiting host uploads per Metal trainer step are reduced, with a before/after count"
  - "Metal and wgpu implement the saved-sigmoid gate pair, with a parity test against CPU"
  - "Any speedup claim cites a committed benchmark with interleaved A/B runs"
---

# Task brief v1

## Title
Cut per-step GPU overhead: waiting host uploads on Metal, recomputed gate sigmoid on Metal/wgpu

Task: gp-gpu-step-overhead
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: metal, wgpu, performance

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Two per-step costs the docs name and nobody has removed:

1. **Host uploads that wait (Metal).** `docs/metal-deferred-faults.md` §9.4 and `docs/pytorch-parity-plan.md:237` report 28 waiting host uploads per trainer step as the largest stall per step once faults were deferred. The fix is either an upload in tessl that does not wait for a fresh buffer, or the trainer uploading all micro-batch ids up front (`ojas-model/src/trainer.rs:520,946` show no batching). The same section estimates ~41 memory-cap commits per 124M step but never counted them; add a per-trigger counter before optimizing.
2. **Gate sigmoid recomputed (Metal, wgpu).** Only CPU overrides the `per_head_sigmoid_gate_*_saving` / `_saved` pair (`ojas-core/src/backend.rs:577-603`). The default is correct, but backward recomputes the sigmoid.

Profile first; report any speedup with interleaved A/B, min-of-N, and a committed benchmark.

## Acceptance criteria
- [ ] Upload waits and memory-cap commits per step are counted by trigger, not estimated
- [ ] Waiting host uploads per Metal trainer step are reduced, with a before/after count
- [ ] Metal and wgpu implement the saved-sigmoid gate pair, with a parity test against CPU
- [ ] Any speedup claim cites a committed benchmark with interleaved A/B runs

## Planned files
- ojas-model/src/trainer.rs
- ojas-metal/src/backend.rs
- ojas-wgpu/src/backend.rs
- bench/
