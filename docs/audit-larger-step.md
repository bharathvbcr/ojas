# Larger-step section times

Shape `B=2`, `T=32`, `d=64`, 1 head, `vocab=128`. Release, `CpuBackend::with_threads` with 6 threads, the same count as the wall-time test. The graph is `regression_graph` in `ojas-cpu/tests/torch_ref.rs`.

DevMap for `/Users/bharath/Code/research/ojas` was stale at the time of this audit (`generation_id` 4, `is_fresh` false, `source_freshness` false, delta 97 added and 23 changed). `devmap build` was not run. The line numbers below were read from source.

## Clock

A temporary test summed `std::time::Instant` around each public backend call, then took the median of 32 samples (40 calls, drop the first 8). That test is not in the tree. The clocks include the backend's `f32` copy in and the output allocation, plus the kernel. "Rest" is the step wall minus those clocks: tensor builds, RMSNorm, RoPE, and the host reshapes between kernels.

Each linear figure is the four calls in one step (Q, K, V, and the output projection, or their backwards). Cross-entropy is the mean forward plus the mean backward. AdamW is the `wq` step and the norm-weight step.

## Before the backward rewrite

| Section | Median µs | Kernel | Backend call |
| :--- | ---: | :--- | :--- |
| linear forward | 393.0 | `linear_forward` `ojas-cpu/src/linalg.rs:65` | `ojas-cpu/src/backend.rs:97` |
| linear backward | 717.4 | `linear_backward` `ojas-cpu/src/linalg.rs:94` | `ojas-cpu/src/backend.rs:113` |
| causal attention forward | 409.8 | `causal_sdpa_forward` `ojas-cpu/src/attn.rs:12` | `ojas-cpu/src/backend.rs:232` |
| causal attention backward | 1148.9 | `causal_sdpa_backward` `ojas-cpu/src/attn.rs:44` | `ojas-cpu/src/backend.rs:243` |
| cross-entropy | 52.3 | `cross_entropy` `ojas-cpu/src/pointwise.rs:369` | `ojas-cpu/src/backend.rs:437` and `:457` |
| AdamW | 45.9 | `adamw` `ojas-cpu/src/optim.rs:74` | `ojas-cpu/src/backend.rs:517` |
| rest | 162.6 | | |

The named clocks plus rest sum to about 2.93 ms, which matches the larger wall median recorded before this rewrite.

Causal attention backward was the largest section. A static count that said `linear_backward` dominated was not used.

## What changed

`causal_sdpa_backward` (`ojas-cpu/src/attn.rs:44`) now calls `backward_head` (`ojas-cpu/src/attn.rs:126`) once per head. Each head is a contiguous `[time, dim]` slice. Dots (`dot_up`, `ojas-cpu/src/attn.rs:213`) and row updates (`saxpy_up`, `ojas-cpu/src/attn.rs:222`) walk the contracted index from 0. The per-element `index` / `get` path is gone from the backward. Scratch is three length-`time` vectors per head (scores, probabilities, dprobs), reused across queries. There is no `B×H×T×T` score matrix. `rayon` was not added. The forward kernel was left as it was.

The reduction order matches the previous per-element loop: head dimension upward, then causal key index `0..=t`, and query `t` upward so `grad_k` and `grad_v` accumulate in increasing `t`. `causal_sdpa_scale_mask_and_adversarial` checks a 1×1 head and `B=2`, `H=2`, `T=3`, `D=5` bit-for-bit against that order. Empty tensors are still refused at the `f32` input check before the kernel. The tiny frozen loss error stayed `2.384e-7`. The larger loss stayed finite (`4.852497`).

## After the rewrite, same section clock

Two medians, same harness, before it was removed.

| Section | Median µs, pass 1 | Median µs, pass 2 |
| :--- | ---: | ---: |
| linear forward | 395.4 | 428.2 |
| linear backward | 713.9 | 762.1 |
| causal attention forward | 417.5 | 468.6 |
| causal attention backward | 52.9 | 54.3 |
| cross-entropy | 51.6 | 65.0 |
| AdamW | 46.6 | 48.6 |
| rest | 156.5 | 175.0 |

Linear backward is the largest section on both passes. Causal attention backward is about 53 µs.

## Before the linear rework, same section clock

Release, 6 threads, 40 calls, drop the first 8, median of 32. Measured with `backward_head` in place, before `linear_backward` was edited. Loss `4.852497`.

| Section | Median µs |
| :--- | ---: |
| linear forward | 350.0 |
| linear backward | 674.1 |
| causal attention forward | 467.8 |
| causal attention backward | 59.0 |
| cross-entropy | 86.1 |
| AdamW | 49.4 |
| rest | 155.4 |

The four linear-backward calls alone were 299.9 µs on one thread and 642.2 µs with six `std::thread::scope` workers. The four linear-forward calls were 354.2 µs on one thread and 288.7 µs with six workers.

## What changed in the linear kernels

Products of at least 4096 multiply-adds no longer split across `std::thread::scope`. On this shape that split was slower than the product. `linear_backward` (`ojas-cpu/src/linalg.rs:104`) calls `grad_x_blocked` (`:259`) and `grad_w_blocked` (`:361`). `grad_x` sums output features from index 0, two rows at a time, eight contiguous weight lanes in registers. `grad_w` sums rows from index 0, four output columns at a time. `linear_forward` (`:68`) still packs `W` to `[in, out]`, then `gemm_blocked` (`:226`) sums the inner index from 0 across eight output columns. Below 4096 multiply-adds, including the tiny step, the saxpy and eight-column loop are unchanged. `backward_head` (`ojas-cpu/src/attn.rs:126`) was not rewritten. A 1×1 product and the odd shape `73×11×13` stay bit-identical to the scalar reference, including at thread counts 1, 2, 3, 7, and 16. Empty axes still error. The tiny frozen loss error stayed `2.384e-7`. The larger loss stayed `4.852497`.

## After the linear rework, same section clock

| Section | Median µs |
| :--- | ---: |
| linear forward | 150.3 |
| linear backward | 144.1 |
| causal attention forward | 445.1 |
| causal attention backward | 52.9 |
| cross-entropy | 51.1 |
| AdamW | 48.5 |
| rest | 145.1 |

Linear backward is no longer the largest section. Causal attention forward is, at 445.1 µs. Wall-clock medians are in `docs/bench-cpu-vs-torch.md`.

## Before the attention-forward rewrite, same section clock

Release, 6 threads, 40 calls, drop the first 8, median of 32. Measured with `backward_head` in place, before `causal_sdpa_forward` was edited. Loss `4.852497`.

| Section | Median µs |
| :--- | ---: |
| linear forward | 164.1 |
| linear backward | 155.8 |
| causal attention forward | 447.3 |
| causal attention backward | 50.4 |
| cross-entropy | 56.8 |
| AdamW | 50.6 |
| rest | 209.5 |

Causal attention forward is the largest section, at 447.3 µs. Causal attention backward stayed near 50 µs.

## What changed in the forward kernel

`causal_sdpa_forward` (`ojas-cpu/src/attn.rs:12`) calls `forward_head` (`:91`) once per head. Each head is a contiguous `[time, dim]` slice. `pack_keys` (`:129`) lays keys out as `[dim, time]` once per head. `score_prefix` (`:138`) sums the head dimension from index 0 across eight key columns when the causal prefix is at least eight long, then the same product for the tail. `softmax_prefix` (`:185`) keeps keys `0..=t`. `mix_values` (`:216`) adds those keys from index 0, four at a time, into each output lane. Scratch is two length-`time` vectors and the packed head. There is no `B×H×T×T` matrix. `rayon` was not added. `backward_head` (`:329`) was not rewritten.

The score and the value mix match the scalar order: head dimension upward, then causal key index `0..=t`. `causal_sdpa_scale_mask_and_adversarial` checks a 1×1 head and `B=2`, `H=2`, `T=3`, `D=5` bit-for-bit against that order, and the same bits for `T=9`, `D=8` (one 8-wide tile plus a remainder) and `T=32`, `D=64`. On that larger shape, writing `40` into key index 8 leaves position 0 unchanged and changes the row that can see that key. Empty tensors are still refused at the `f32` input check before the kernel. The tiny frozen loss error stayed `2.384e-7`. The larger loss stayed `4.852497`.

## After the forward rewrite, same section clock

| Section | Median µs |
| :--- | ---: |
| linear forward | 134.0 |
| linear backward | 134.4 |
| causal attention forward | 27.8 |
| causal attention backward | 47.7 |
| cross-entropy | 48.6 |
| AdamW | 45.2 |
| rest | 125.4 |

Causal attention forward is no longer the largest section. Linear backward is, at 134.4 µs. Linear forward is 134.0 µs. Wall-clock medians are in `docs/bench-cpu-vs-torch.md`.
