# Attention backward A/B: saved log-sum-exp and native grouped-query (2026-10-08)

Task `gp-attention-kernels`. Old is c74f3ba: its backward formed the row
statistics itself (`attn_bwd_prep`) and, for grouped-query heads, expanded K
and V to the query heads and summed dK and dV back. New is HEAD (86a1096): the
forward returns the row log-sum-exp, the backward consumes it with
`Dr = dO · O`, and grouped-query heads read their KV head in place. Both
include every other change 6a0a926 made to these kernels, so the ratio is the
change as landed, not the log-sum-exp alone. On the multi-head shapes the
grouped-query path is not involved.

## Method

- Both trees were exported with `git archive` into `ojas/target-attn-ab/ab/{old,new}`
  and built `--release` against the same tessl checkout (4e5faac plus another
  session's uncommitted WIP in `src/attn_train.rs` and others, listed in
  `env.txt`; one build window, so both sides saw the same tessl). Each tree
  had its own `CARGO_TARGET_DIR`. `ojas-capi` and `ojas-gusset-engine` were
  removed from the exported workspaces' member lists, because their gusset
  path dependency does not resolve from there and the bench uses neither.
- `scripts/attn_ab_common.rs` is the one body that both trees compile, through
  `scripts/shim_old.rs` and `scripts/shim_new.rs`. Each process runs every
  shape: 5 warm-ups, then 20 iterations of forward, backward and a control
  `linear_forward` [2048, 1024] x [1024, 1024]. Each iteration times all
  three, with `Backend::sync` after each call, and alternates the order.
- `scripts/run_ab.sh`: 6 rounds. Each round runs old and new back to back per
  backend; odd rounds put old first and even rounds new first. It ran under
  `mac_heavy.sh` at load 3.2 to 6.5 (`ab.txt` records the load before each
  process).
- `scripts/agg.py` takes the min of each process's iterations, then the new/old
  ratio per round. The control's ratio is computed the same way and should
  read about 1.00.

## Result (`tables.md`)

| backend | shape | backward new/old, median of 6 rounds (range) | control |
|:--|:--|:--|:--|
| Metal | B4 H8 T2048 D64 | 0.761 (0.698–0.889) | 0.984 |
| Metal | B2 H8 T1024 D128 | 0.797 (0.586–0.803) | 1.003 |
| Metal | B1 H8 T2048 D256 | 0.796 (0.723–0.854) | 1.016 |
| Metal | Qwen3.5 B1 H8/Hkv2 T2048 D256 | 0.769 (0.695–0.785) | 0.995 |
| wgpu | B4 H8 T2048 D64 | 0.789 (0.772–0.812) | 0.994 |
| wgpu | B2 H8 T1024 D128 | 0.809 (0.778–0.858) | 0.985 |
| wgpu | B1 H8 T2048 D256 | 0.772 (0.763–0.796) | 1.018 |
| wgpu | Qwen3.5 B1 H8/Hkv2 T2048 D256 | 0.765 (0.739–0.787) | 0.995 |

The backward is 19–24% faster on both backends at every shape, inside the
task's 10–25% target. Forward ratios are 0.99–1.03, so writing the
log-sum-exp costs no measurable forward time; the one exception is the Metal
grouped-query forward, which no longer expands K and V and runs at 0.889. The
range is wide on two Metal rows (the 0.586 and 0.889 ends), with the
controls inside 0.96–1.05 in every round.

Sliding window, new tree only: W=256 at T=2048 runs the Metal backward at
0.27–0.31 of the full-prefix time (for example, 3.74 ms against 13.91 ms on
the Qwen3.5 shape) and the wgpu backward at 0.25–0.31. A window of 256 over
2048 positions holds about 23% of the causal score pairs, so the time follows
the work. At T=1024, where W=256 holds about 44% of the pairs, the backward
runs at 0.54 (Metal) and 0.49 (wgpu) of the full prefix. Masking alone, without skipping blocks, would have kept it near the
full-prefix time.

Bit-level equivalence is a test, not part of this bench. The saved-statistics
backward equals `causal_sdpa_backward_recompute` bit for bit in
`saved_backward_equals_the_recomputing_one_bit_for_bit` on Metal and on wgpu
(`attention_window.rs`), and on the CPU in `ojas-cpu/tests/ops.rs`.

## Grouped-query scratch at the Qwen3.5-2B shape (`gqa_scratch.txt`)

B1, 8 query heads over 2 KV heads, T2048, D256. This is the budget charge
beyond the operands, measured by
`grouped_query_charges_no_expanded_heads_at_the_qwen35_shape` (Metal and
wgpu `attention_window.rs`), which asserts that the charge is the outputs
plus the row statistics, and nothing more.

| | forward | backward |
|:--|--:|--:|
| Metal, native (measured) | 16.06 MiB | 24.06 MiB |
| wgpu, native (measured) | 16.06 MiB | 24.13 MiB |
| expand path at c74f3ba (inferred from its source) | 48.00 MiB | 88.13 MiB |

Native indexing saves 32 MiB in the forward and 64 MiB in the backward per
call at this shape: two and four query-sized f32 buffers. The expand-path
figure is computed from c74f3ba's charges (Metal `causal_sdpa_forward` and
`causal_sdpa_backward` scratch, and wgpu `expand_kv` plus the query-sized
dK/dV scratch). That code is deleted, so the figure is inferred from its
source; it was not run.
