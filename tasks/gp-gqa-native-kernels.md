---
id: "gp-gqa-native-kernels"
title: "Tape-level GQA gradcheck, then native grouped attention kernels on Metal and wgpu"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "attention"
  - "gqa"
  - "autograd"
  - "performance"
repositories:
  - "ojas"
planned_files:
  - "ojas-autograd/tests/gradcheck.rs"
  - "ojas-model/tests/forward.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/kernels/"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-kernels/src/wgsl/"
acceptance_criteria:
  - "A finite-difference gradcheck runs GQA (H != Hkv) through the Tape on CPU, and through Metal and wgpu against CPU"
  - "Metal and wgpu attention forward and backward index KV heads natively, with no expand/sum-back scratch"
  - "The scratch saving is measured at a Qwen3.5 shape"
---

# Task brief v1

## Title
Tape-level GQA gradcheck, then native grouped attention kernels on Metal and wgpu

Task: gp-gqa-native-kernels
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: attention, gqa, autograd, performance

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Follow-up to gp-head-dim-256-gqa (in review).

1. **Correctness gap.** The only tape-level GQA test (`ojas-model/tests/forward.rs:817`) checks that logits equal eval and that the k_proj gradient has the right shape and is nonzero. The numerical gradcheck (`ojas-autograd/tests/gradcheck.rs:352`) calls `causal_sdpa_backward` directly, not through the Tape. The gradient summing over repeated KV heads is never numerically checked through the Tape.
2. **Cost.** Metal (`ojas-metal/src/backend.rs:551-565,896-960`) and wgpu (`ojas-wgpu/src/backend.rs:985-1000`, `head_repeat.wgsl`) copy K/V to the full head count, then sum the gradients back. That spends scratch memory and bandwidth proportional to H/Hkv. Native grouped kernels index the shared KV head instead.

## Acceptance criteria
- [ ] A finite-difference gradcheck runs GQA (H != Hkv) through the Tape on CPU, and through Metal and wgpu against CPU
- [ ] Metal and wgpu attention forward and backward index KV heads natively, with no expand/sum-back scratch
- [ ] The scratch saving is measured at a Qwen3.5 shape

## Planned files
- ojas-autograd/tests/gradcheck.rs
- ojas-model/tests/forward.rs
- ojas-metal/src/backend.rs
- ojas-metal/kernels/
- ojas-wgpu/src/backend.rs
- ojas-kernels/src/wgsl/
