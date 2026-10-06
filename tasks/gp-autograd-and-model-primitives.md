---
id: "gp-autograd-and-model-primitives"
title: "Scale Autograd Tape with Activation Checkpointing, Hybrid GDN/Conv1D Primitives, and Shape Validators"
status: ready
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "autograd"
  - "memory"
  - "scaling"
  - "qwen35"
  - "backend"
  - "kernel"
  - "shape-contract"
  - "metal"
repositories:
  - "ojas"
planned_files:
  - "ojas-autograd/src/tape.rs"
  - "ojas-model/src/block.rs"
  - "ojas-model/src/trainer.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-core/src/shapes.rs"
  - "ojas-cpu/src/accum.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-qwen35/src/model.rs"
  - "ojas-qwen35/src/groups.rs"
  - "ojas-qwen35/src/step.rs"
  - "ojas-qwen35/README.md"
acceptance_criteria:
  - "Implement block-level activation checkpointing in Tape, releasing internal layer activations during forward and recomputing on demand during backward traversal"
  - "Enable sequence length scaling past T=2048 without exceeding resident memory budgets, with gradient equality tests against standard runs"
  - "Add chunked_gdn_forward and chunked_gdn_backward methods to Backend trait"
  - "Add depthwise causal_conv1d_silu forward and backward methods to Backend trait"
  - "Add elementwise gated RMSNorm and partial RoPE with MRoPE collapse to Backend trait"
  - "Implement native Metal backends utilizing optimized kernels ported from tessl, verifying Qwen3.5 2B hybrid forward and backward execution through Tape"
  - "shapes.rs owns accumulate_grad and permute checks, per-backend copies are deleted, and cross-backend test passes for edge cases"
  - "Once tessl ships per-parameter LR, check_tessl_lr is removed and a test proves per-group LR scales updates"
  - "Measure and record 2B save/load and longer sequence length peak memory headroom on Metal in README"
---

# Task brief v1

## Title
Scale Autograd Tape with Activation Checkpointing, Hybrid GDN/Conv1D Primitives, and Shape Validators

Task: gp-autograd-and-model-primitives
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: autograd, memory, scaling, qwen35, backend, kernel, shape-contract, metal

## Repositories
- ojas

## Description
Consolidated task covering high-level model execution, Autograd Tape scaling, and architecture contract validation:
- Selective activation checkpointing in Autograd Tape (`gp-activation-checkpointing`)
- Standardizing hybrid primitives (GDN, Conv1D) in Backend trait (`gp-hybrid-layer-primitives`)
- Centralizing `accumulate_grad` and `permute` shape validators in `ojas_core::shapes` (`gp-shape-validators`)
- Qwen3.5 Metal follow-ups: per-parameter learning rate and 2B memory headroom (`gp-qwen35-metal-followups`)

### Audit status and progress notes (2026-10-05)
1. **Activation Checkpointing:** Autograd Tape currently retains all intermediate layer activations across all blocks. At context lengths T > 2048, resident memory spikes dramatically. Selective block checkpointing recomputes forward activations during backward adjoint traversal, trading compute for memory.
2. **Hybrid Layer Primitives:** Qwen3.5 mixes linear attention (Gated Delta Networks / GDN) and causal 1D conv with attention layers. Currently, these primitives bypass `ojas_core::Backend` and exist only as isolated Metal kernels in tessl. They need standardized trait signatures and native backends.
3. **Shape Contracts:** Every other op's shapes are validated centrally in `ojas-core/src/shapes.rs` (`docs/shape-contract.md`), except `accumulate_grad` and `permute`, which have separate per-backend copies in CPU, Metal, and wgpu.
4. **Qwen3.5 Metal Follow-ups:** Per-parameter learning rates are currently blocked on upstream tessl (`check_tessl_lr` in `ojas-qwen35/src/groups.rs:278` refuses `lr_scale != 1.0`). In addition, empirical 2B headroom measurements on Metal remain to be documented.

### Execution plan
- **Phase 1 (Shape Validator Centralization):** Move `accumulate_grad` and `permute` validation into `ojas_core::shapes`, delete backend copies, and verify zero-length permute axes cross-backend.
- **Phase 2 (Activation Checkpointing):** Implement block-level activation checkpointing on Tape, verifying exact gradient parity against uncheckpointed runs.
- **Phase 3 (Hybrid Architecture Primitives):** Expose GDN chunk rule, causal conv1d, gated RMSNorm, and MRoPE in `Backend` trait with Metal implementation.
- **Phase 4 (Qwen3.5 Metal Follow-ups):** Remove `check_tessl_lr` once tessl adds per-param LR, and record 2B peak memory headroom in README.

## Acceptance criteria
- [ ] Implement block-level activation checkpointing in Tape, releasing internal layer activations during forward and recomputing on demand during backward traversal
- [ ] Enable sequence length scaling past T=2048 without exceeding resident memory budgets, with gradient equality tests against standard runs
- [ ] Add chunked_gdn_forward and chunked_gdn_backward methods to Backend trait
- [ ] Add depthwise causal_conv1d_silu forward and backward methods to Backend trait
- [ ] Add elementwise gated RMSNorm and partial RoPE with MRoPE collapse to Backend trait
- [ ] Implement native Metal backends utilizing optimized kernels ported from tessl, verifying Qwen3.5 2B hybrid forward and backward execution through Tape
- [ ] shapes.rs owns accumulate_grad and permute checks, per-backend copies are deleted, and cross-backend test passes for edge cases
- [ ] Once tessl ships per-parameter LR, check_tessl_lr is removed and a test proves per-group LR scales updates
- [ ] Measure and record 2B save/load and longer sequence length peak memory headroom on Metal in README

## Planned files
- ojas-autograd/src/tape.rs
- ojas-model/src/block.rs
- ojas-model/src/trainer.rs
- ojas-core/src/backend.rs
- ojas-core/src/shapes.rs
- ojas-cpu/src/accum.rs
- ojas-metal/src/device.rs
- ojas-metal/src/backend.rs
- ojas-wgpu/src/backend.rs
- ojas-qwen35/src/model.rs
- ojas-qwen35/src/groups.rs
- ojas-qwen35/src/step.rs
- ojas-qwen35/README.md
