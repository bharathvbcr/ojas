---
id: "gp-wgpu-hybrid-ops"
title: "wgpu kernels for the whole Qwen3.5 hybrid layer: causal conv1d+SiLU, gated RMSNorm, partial RoPE, chunked GDN, sigmoid, GDN log-decay"
status: ready
priority: 3
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "wgpu"
  - "kernel"
  - "qwen35"
  - "parity"
repositories:
  - "ojas"
planned_files:
  - "ojas-wgpu/src/backend.rs"
  - "ojas-kernels/src/wgsl/"
  - "ojas-wgpu/tests/"
  - "docs/op-coverage.md"
acceptance_criteria:
  - "ojas-wgpu overrides the Backend methods for causal conv1d+SiLU, gated RMSNorm, partial RoPE, chunked_gdn, sigmoid and gdn_log_decay (forward and backward) instead of inheriting the refusing trait defaults; wgpu implements none of them today"
  - "Each kernel has a parity test against the CPU backend and the f64 reference, plus shape-first malformed-input and non-finite fault tests matching the existing wgpu suites"
  - "docs/op-coverage.md hybrid rows updated for wgpu"
  - "ojas-model's Qwen3.5 hybrid tower runs through the wgpu Tape and matches the CPU Tape on the tiny fixture, as metal_hybrid_tape.rs does for Metal"
---

# Task brief v1

## Title
wgpu kernels for the whole Qwen3.5 hybrid layer: causal conv1d+SiLU, gated RMSNorm, partial RoPE, chunked GDN, sigmoid, GDN log-decay

Task: gp-wgpu-hybrid-ops
Type: feature
Status: ready
Priority: 3 (Low)
Severity: medium
Owner: unassigned
Due: none
Labels: wgpu, kernel, qwen35, parity

## Repositories
- ojas

## Description
Gap audit 2026-10-07. `gp-autograd-and-model-primitives` (in progress) Phase 3 names Metal only for these ops, and already notes wgpu GDN; the three non-GDN hybrid ops have no wgpu owner. ojas-wgpu inherits the trait defaults, which return `Unsupported` [V by the override list and `docs/op-coverage.md` hybrid rows]. Filed as its own card rather than rewriting the in-progress brief under a running agent.

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- The start gate is met: b363802 landed the Metal hybrid kernels with tests (ojas-metal/tests/hybrid.rs, ojas-model/tests/metal_hybrid_tape.rs), so the trait signatures are settled [A].
- Scope widened to the whole hybrid layer. sigmoid and gdn_log_decay (66c5e52) and chunked_gdn are Metal-only. The wgpu GDN item that gp-autograd-and-model-primitives listed as 'still open' had no owner and is owned here now [A].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Lowered to P3: off the Mac (Metal) training path. Raise it again when CUDA, HIP or wgpu training becomes a goal.

## Acceptance criteria
- [ ] ojas-wgpu overrides the Backend methods for causal conv1d+SiLU, gated RMSNorm, partial RoPE, chunked_gdn, sigmoid and gdn_log_decay (forward and backward) instead of inheriting the refusing trait defaults; wgpu implements none of them today
- [ ] Each kernel has a parity test against the CPU backend and the f64 reference, plus shape-first malformed-input and non-finite fault tests matching the existing wgpu suites
- [ ] docs/op-coverage.md hybrid rows updated for wgpu
- [ ] ojas-model's Qwen3.5 hybrid tower runs through the wgpu Tape and matches the CPU Tape on the tiny fixture, as metal_hybrid_tape.rs does for Metal

## Planned files
- ojas-wgpu/src/backend.rs
- ojas-kernels/src/wgsl/
- ojas-wgpu/tests/
- docs/op-coverage.md
