# Metal LM-head GEMM probe (2026-10-02)

`metal_bench <iters> gemm`, tessl `GemmOperands::ExactF32`, one GEMM per
command buffer, GPU timestamps. The GPU read 100% busy from another process
during both runs, so the numbers show direction only.

- `gemm.md`: first run. Its nn and tn rows print the forward's (M, N, K),
  not the GEMM's own; the probe was mislabelled and is fixed.
- `gemm-relabelled.md`: rerun with each row's own (M, N, K) and 32- and
  128-row LM-head cases added.

## What the data says

- **nt slows down as N grows, and M doesn't matter** (verified, both runs).
  At N <= 2304 it runs at 6.5-7 TFLOP/s. At N = 16384 it drops to 2.4, and
  at the LM head (N = 50304) to 2.3. With N = 50304 fixed, 32, 128, 512,
  1024 and 2048 rows all run at 2.3-2.7 TFLOP/s.
- **Each row of 32x32 tiles costs about 1 ms and reads all of `w` once**
  (verified from the 32-row case, inferred for the cause). 32 rows take
  1.007 ms. That is one pass over `w` (50304 x 768 f32 = 154 MB) at
  ~153 GB/s, and the LM head's 128 tile rows make 128 passes. A 32-row tile
  does 16 flop per byte of `w` it loads, so a DRAM-bound pass caps at about
  2.5 TFLOP/s, which is what is measured.
- **The cause is the order in which tiles run** (inferred). All three
  exact-f32 kernels (`matmul2d_tensorops_f32`, `_nt_f32`, `_tn_f32` in
  `tessl/kernels/matmul_tensorops.metal`) walk tiles row-major through
  `tile_from_linear`. Tiles that run at the same time share an A tile and
  each read a different B tile, so a B too large for the cache is
  re-streamed once per tile row. tessl's bf16 coop NN kernel already has a
  column-panel swizzle; the exact-f32 kernels do not. tessl dispatches nt,
  nn and tn identically (`dispatch_tensorops_nn`, `TILE_F32` 32x32, one
  simdgroup; `tessl/src/gemm.rs`). At these shapes tn does not take the
  split-K path (`prefer_tn_splitk` needs M, N <= 384).
- **tn stays fast because its N is small** (verified). The LM head's weight
  gradient is tn with M = 50304, N = 768, K = 4096 and runs at 5.6 TFLOP/s.
  It is not a counterexample.
- **The LM head's input gradient is a different problem** (inferred). nn with
  M = 4096, N = 768, K = 50304 runs at 2.8 TFLOP/s. Its output is small
  (3072 tiles) and its K is long, so a swizzle does not apply; split-K would
  be the lever.
- **Chunking the columns roughly halves nt** (verified, direction only).
  Splitting into 4096-column GEMMs takes 137 ms down to 71 ms (4.45 TFLOP/s),
  and the earlier run went from 129 to 65 ms.

## The fix and its A/Bs

The fix is in tessl: `kernels/matmul_tensorops.metal`, with
`tile_from_linear_panel` and `tile_walk_f32`. `tessl-panel-walk.diff` is the
diff of its first (grid-gated) version against
`matmul_tensorops-before.metal.txt`. The shipped gate is on B's size,
`N·K >= 2^23`. Write-up: `docs/bench-gpu-vs-torch.md`, "Round 5 and the
LM-head GEMM".

| Directory | What it holds |
|---|---|
| `ab-grid-gate/` | GEMM probe, interleaved, 4 rounds per side: panels on any grid of 2048 tiles or more. LM-head NT 0.47×; TN LM head 1.08× (a 12.6 MB B). |
| `ab-footprint-gate/` | The same with the shipped gate. LM-head NT 0.46×, NT N = 16384 0.44×, TN at a 32 MB B 0.77×, the rest not engaged. |
| `ab-rows/` | `bench/ab_metal_rows.sh` with `OLD_BIN` (the round 5 binary), 6 rounds: `linear_lmhead_fwd` 134.3 → 68.2 ms, `linear_lmhead_bwd` 164.3 → 156.8 ms. |
| `ab-rows-ce/` | The same for the linear_ce rows (r2 torch reference): `c4096x50304` 304.8 → 239.4 ms; `c1024x8192` unchanged. |

| `ab-nn-splitk/` | NN split-K, interleaved with alternating order, 4 rounds per side. Old side: the panel-walk binary. NN LM head 99.8 → 45.4 ms, NN K = 16384 21.0 → 14.6 ms, the rest within noise. |
| `splitk-rows/` | The split-K `metal_vs_torch`, 3 rounds, unpaired (load ~350): `linear_lmhead_bwd` 109.1 ms min, `linear_ce_c4096x50304` 166.4 ms. |

`run.sh` and `sum.sh` are the probe A/B scripts. The binaries they ran are in
the ignored `target-baseline/gemm-ab/`.

## Where ojas issues these shapes

- `linear` (`ojas-metal/src/device.rs`, `fn linear`): nt forward, nn and tn
  backward, at full width. This covers `linear_lmhead_fwd` (0.48x torch) and
  `linear_lmhead_bwd` (0.72x).
- `linear_ce` (`fn linear_ce`): one nt per (row tile, column tile), with the
  column tile the caller's `ct`. `linear_ce_c4096x50304` runs the
  full-vocabulary nt (0.78x); `c1024x8192` is at 0.87x.
