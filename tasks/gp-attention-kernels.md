---
id: "gp-attention-kernels"
title: "GQA Native Kernels, Head Dim 256, Attention LSE Return, and Sliding Window"
status: ready
priority: 0
severity: critical
type: feature
owner: "unassigned"
due: "none"
labels:
  - "attention"
  - "kernel"
  - "autograd"
  - "qwen35"
  - "gqa"
  - "performance"
  - "long-context"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-kernels/src/geometry.rs"
  - "ojas-kernels/src/wgsl/"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-cpu/src/backend.rs"
  - "ojas-cpu/src/attn.rs"
  - "ojas-autograd/src/tape.rs"
  - "ojas-autograd/tests/gradcheck.rs"
  - "ojas-model/tests/forward.rs"
acceptance_criteria:
  - "Remove d <= 128 head dimension limit assertion across Metal and wgpu kernels (verified: cap raised to 256)"
  - "Implement tiled attention forward and backward kernels supporting head dimension 256"
  - "Generalize Backend::causal_sdpa_forward and backward to support Grouped-Query Attention (mismatched Q and KV head counts)"
  - "Autograd Tape records and backpropagates GQA projections correctly"
  - "Unit tests verifying head dimensions 64, 128, and 256 match analytical and CPU reference oracles"
  - "A finite-difference gradcheck runs GQA (H != Hkv) through the Tape on CPU, and through Metal and wgpu against CPU"
  - "Metal and wgpu attention forward and backward index KV heads natively, with no expand/sum-back scratch"
  - "Measure scratch saving of native GQA kernels at a Qwen3.5 shape"
  - "Update Backend::causal_sdpa_forward trait signature to return (Tensor, Tensor) representing output activations and row-wise LSE statistics"
  - "Modify CPU, Metal, and wgpu attention forward kernels to write out LSE and refactor backward kernels to consume preserved LSE without recomputing softmax"
  - "Benchmark attention backward latency reduction (targeting 10% to 25% speedup on Metal and wgpu) with bit-level equivalence tests"
  - "Add sliding window size parameter W to attention forward and backward dispatches (t - W < j <= t) with tiled block skipping and parity tests"
---

# Task brief v1

## Title
GQA Native Kernels, Head Dim 256, Attention LSE Return, and Sliding Window

Task: gp-attention-kernels
Type: feature
Status: ready
Priority: 0 (Urgent)
Severity: critical
Owner: unassigned
Due: none
Labels: attention, kernel, autograd, qwen35, gqa, performance, long-context

## Repositories
- ojas

## Description
Consolidated task unifying all attention training kernel enhancements:
- Head dimension 256 and GQA support (`gp-head-dim-256-gqa`)
- Tape-level GQA gradcheck and native grouped kernels (`gp-gqa-native-kernels`)
- Attention LogSumExp (LSE) forward return for accelerated backward pass (`gp-attention-lse-return`)
- Sliding window local causal attention kernels (`gp-sliding-window-attention`)

Modern architectures (Qwen3.5, Gemma, Llama 3) require head dimension 256, Grouped-Query Attention (GQA), efficient backward traversal, and sliding-window attention for long sequences. Consolidating these into one coherent owner eliminates merge conflicts across `causal_sdpa_forward` and `causal_sdpa_backward`.

### Audit status and progress notes (2026-10-05)
1. **Head Dim 256 & GQA in Trait:** Landed in 1f26a4b. Caps raised to 256 (`METAL_MAX_HEAD_DIM = 256`, `ATTN_MAX_HEAD_DIM = 256`, `ATTENTION_MAX_HEAD_DIM = 256`). GQA trait supports mismatched head counts (`q: [B,H,T,D]`, `k,v: [B,Hkv,T,D]`). CPU is native; Metal and wgpu currently expand K/V to full heads via `head_repeat.wgsl` and sum gradients back, which incurs extra scratch memory.
2. **GQA Weak Spot:** The only tape-level GQA test (`ojas-model/tests/forward.rs:817`) checks logits equal eval and nonzero k_proj gradient; numerical gradcheck in `ojas-autograd/tests/gradcheck.rs:352` calls `causal_sdpa_backward` directly without the Tape.
3. **LSE Return:** Attention backward currently recomputes row statistics / softmax. Emitting LSE from forward SDPA saves 10–25% backward latency on Metal and wgpu.
4. **Sliding Window:** Local causal attention with window size W is currently missing in Ojas.

### Execution plan
- **Phase 1 (GQA Hardening):** Add tape-level finite-difference GQA gradcheck. Implement native KV head indexing in Metal and wgpu attention kernels to eliminate repeat/sum-back scratch.
- **Phase 2 (LSE Return):** Update `Backend::causal_sdpa_forward` to return `(Tensor, Tensor)` (output + row LSE). Refactor CPU, Metal, and wgpu backward kernels to consume preserved LSE, benchmarking the 10–25% backward speedup.
- **Phase 3 (Sliding Window):** Add window size parameter W to attention forward and backward dispatches. Implement block skipping in tiled attention kernels to avoid loading out-of-window tiles.

## Acceptance criteria
- [x] Remove d <= 128 head dimension limit assertion across Metal and wgpu kernels (verified: cap raised to 256)
- [x] Implement tiled attention forward and backward kernels supporting head dimension 256
- [x] Generalize Backend::causal_sdpa_forward and backward to support Grouped-Query Attention (mismatched Q and KV head counts)
- [x] Autograd Tape records and backpropagates GQA projections correctly
- [x] Unit tests verifying head dimensions 64, 128, and 256 match analytical and CPU reference oracles
- [ ] A finite-difference gradcheck runs GQA (H != Hkv) through the Tape on CPU, and through Metal and wgpu against CPU
- [ ] Metal and wgpu attention forward and backward index KV heads natively, with no expand/sum-back scratch
- [ ] Measure scratch saving of native GQA kernels at a Qwen3.5 shape
- [ ] Update Backend::causal_sdpa_forward trait signature to return (Tensor, Tensor) representing output activations and row-wise LSE statistics
- [ ] Modify CPU, Metal, and wgpu attention forward kernels to write out LSE and refactor backward kernels to consume preserved LSE without recomputing softmax
- [ ] Benchmark attention backward latency reduction (targeting 10% to 25% speedup on Metal and wgpu) with bit-level equivalence tests
- [ ] Add sliding window size parameter W to attention forward and backward dispatches (t - W < j <= t) with tiled block skipping and parity tests

## Planned files
- ojas-core/src/backend.rs
- ojas-kernels/src/geometry.rs
- ojas-kernels/src/wgsl/
- ojas-metal/src/device.rs
- ojas-metal/src/backend.rs
- ojas-metal/kernels/ojas_backend.metal
- ojas-wgpu/src/backend.rs
- ojas-cpu/src/backend.rs
- ojas-cpu/src/attn.rs
- ojas-autograd/src/tape.rs
- ojas-autograd/tests/gradcheck.rs
- ojas-model/tests/forward.rs
