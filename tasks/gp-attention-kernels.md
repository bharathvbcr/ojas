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

### Verification notes (2026-10-08)
The code for phases 1–3 landed in 6a0a926, whose message says its test suites were not rerun. This pass ran them (`target-attn/attn-verify.sh` under `mac_heavy.sh`, `-j 2`, `--test-threads=2`, `OJAS_ALLOW_NO_GPU` unset, so a missing device fails instead of skipping):
- `cargo clippy -p ojas-metal -p ojas-wgpu --all-targets -- -D warnings` is clean.
- Metal `attention_window`, `attention_backward` and `attention_forward` pass 22/22. wgpu `attention_window` passes 5/5, and `parity attention` passes 3/3 (18 filtered out). CPU `sliding_window` and `ops` pass 17/17.
- `ojas-model` `gqa_tape` passes 3/3 (the CPU, Metal and wgpu Tape against f64 central differences, and both GPUs against the CPU tape). `forward` passes 15/15. `ojas-autograd` `gradcheck` and `multihead` pass 14/14.
- New test `grouped_query_charges_no_expanded_heads_at_the_qwen35_shape`, in Metal's and wgpu's `attention_window.rs`, measures the budget charge at Qwen3.5-2B (B1, H8/Hkv2, T2048, D256). The forward charges 16.06 MiB and the backward 24.06 MiB (Metal) or 24.13 MiB (wgpu). The test asserts outputs plus row statistics only. c74f3ba's expand path charged 48 MiB and 88.13 MiB; that figure is inferred from its source.
- Backward A/B against c74f3ba, in `bench/results/2026-10-08-attn-lse-ab/`: 6 paired rounds, with controls at 0.98–1.02. The backward is 19–24% faster on Metal and on wgpu at every shape, including Qwen3.5 GQA, and the forward is unchanged. W=256 at T=2048 runs the backward at 0.25–0.31 of the full-prefix time, which is consistent with block skipping. Bit-level equivalence: `saved_backward_equals_the_recomputing_one_bit_for_bit` passes on Metal and wgpu.
- Follow-up (2026-10-08): the CPU side of the LSE return moved bits the earlier pass had not checked. `exact_golden` re-records the nine Exact `sdpa_g*` digests and adds three `sdpa_lse` digests (the f64 gates in `sliding_window.rs` and `ops.rs` hold). `attention_fast` had regressed under Fast at T=257 (grad_q 6.9e-7 of f64 against Exact's 2.1e-7, passing at c74f3ba): the flash backward now forms `delta` from its own `P·dP`, and passes 6/6.

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
- [x] A finite-difference gradcheck runs GQA (H != Hkv) through the Tape on CPU, and through Metal and wgpu against CPU
- [x] Metal and wgpu attention forward and backward index KV heads natively, with no expand/sum-back scratch
- [x] Measure scratch saving of native GQA kernels at a Qwen3.5 shape
- [x] Update Backend::causal_sdpa_forward trait signature to return (Tensor, Tensor) representing output activations and row-wise LSE statistics
- [x] Modify CPU, Metal, and wgpu attention forward kernels to write out LSE and refactor backward kernels to consume preserved LSE without recomputing softmax
- [x] Benchmark attention backward latency reduction (targeting 10% to 25% speedup on Metal and wgpu) with bit-level equivalence tests
- [x] Add sliding window size parameter W to attention forward and backward dispatches (t - W < j <= t) with tiled block skipping and parity tests

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
