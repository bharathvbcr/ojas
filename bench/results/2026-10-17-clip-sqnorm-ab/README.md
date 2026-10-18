# `clip_grad_norm_full` on Metal: main against tessl's `sq_norm` (2026-10-17)

## Why

The 0.45× against torch 2.15 that `docs/bench-gpu-vs-torch.md` quoted
(ojas 18.96 ms, round 5, 2026-10-02) measured the path from before 0e3254f:
roughly five dispatches per gradient. 0e3254f (2026-10-08) replaced it with a
multi-tensor norm (`ojas_norm_multi`, 24 gradients per dispatch, then one
finish), which had never been timed. Branch `gp/clip-sq-norm` replaces that
with tessl's `sq_norm`. This run times both.

## What was run (verified)

- `clip_ab.sh OUT 7 base new`: the `clip_grad_norm_full` row of
  `ojas-metal/examples/metal_vs_torch` (`bench/ojas_rows.rs`: nanolab's
  123.7M-value parameter set, every call scaling), 10 warm-ups and 40 timed
  iterations per round. Seven rounds, alternating which binary goes first
  (odd rounds base then new, even rounds new then base).
- Both binaries built under the w11 queue's `mac_heavy` lock, `-j 2`, release.
  `env.txt` gives the commits:
  - base is ojas `6426c5d` and tessl `f2735e9`
  - new is ojas `6426c5d` and tessl `1b900ec`, plus the branch diffs
  - `1b900ec..f2735e9` changes only `BRANCH_REPORT.md`, so the two builds
    differ by the branch alone.
- Parity, against torch 2.14.1's reference for the same row, passed in every
  round on both: `parity_rel` 3.0e-7, against a gate of 1e-3.
- Apple M5 Pro, macOS 27.0.1. Load average 46–50 throughout (`out/load.jsonl`).
  The GPU read idle before 27 of 28 rows (36% once, before base round 1).
- The torch side was **not** re-run: no 2.15 nightly is installed here.

## Results (`out/*.jsonl`)

| binary | min of mins | median of medians | round medians |
| :-- | --: | --: | :-- |
| base (`ojas_norm_multi`) | 6.60 ms | 7.07 ms | 6.92–7.68 |
| new (tessl `sq_norm`) | 6.49 ms | 7.45 ms | 7.01–8.62 |

Per-round median ratio base/new: 1.03 0.86 0.95 1.01 0.96 0.83 0.94 (median 0.95).

## Reading

- **No speed change is shown.** The minimums agree within 2%. The medians put
  the new binary 5% behind, but its round medians spread over 7.01–8.62 ms
  against base's 6.92–7.68. On a loaded machine that is not a measured
  regression, and it is not a speedup either. The branch's case is correctness
  and ownership, not speed:
  - one norm implementation, shared with tessl's `Qwen35Model::grad_sq_norm`
  - which no longer overflows on huge gradients: tessl's old kernel gave
    +inf at 2^100 (`grad_sq_norm_of_huge_and_tiny_gradients_is_exact` fails
    on main)
  - one binder scope for the norm whatever the gradient count (main: one per
    24 gradients, plus a finish)
  - a sum exact by construction, so the same bits in any order. That is not
    an improvement shown on ojas: the old f32 finish also gave the same bits
    on the 300-gradient order test.
- **The norm kernel is not what this row costs.** Each call reads the 123.7M
  f32 gradients once for the norm and reads and writes them once more to
  scale, about 1.5 GB in all. At 6.5 ms that is about 230 GB/s. Faster would
  mean not making the scale pass: folding the clip coefficient into the
  optimizer, as tessl's `grad_scale` does. That is a trait change, recorded in
  `tasks/gp-gpu-step-throughput.md`.
- **Against torch (unpaired, direction only):** the 2.15 nightly's 8.45 ms
  (`../2026-10-05-torch215-clip`, another day, other load) against ojas's
  7.07 ms on current main. That puts ojas at about 1.2× torch on this row,
  not 0.45×. A paired run with a 2.15 nightly is still needed before that
  ratio is quoted.
