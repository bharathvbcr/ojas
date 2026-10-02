# GPU Backends versus PyTorch MPS

Paired A/B of ojas's two GPU backends, `MetalBackend` (`ojas-metal`) and
`WgpuBackend` (`ojas-wgpu`), against PyTorch 2.13 MPS, at nanolab's default
shapes: d 768, 12 heads of 64, SwiGLU hidden 2048, vocabulary 50304, B 4,
T 1024 (4096 rows). Each kernel is timed forward and backward where it has a
backward. One composed nanolab block is timed forward, and forward plus
backward. Everything is f32 on both sides.

The harness, the commands and the row definitions are in
[`bench/README.md`](../bench/README.md). Raw data: round 1 in
`bench/results/2026-10-01/`, round 2 in `bench/results/2026-10-01-r2/`,
round 3 in `bench/results/2026-10-01-r3b/`, and the round 3 regression A/B
in `bench/results/2026-10-01-ab/`.

The document has four dated sections:
- **Round 5 and the LM-head GEMM** (first): round 5's remaining Metal gaps,
  and the fix to tessl's exact-f32 GEMM tile walk that halves the LM head.
  It ends with the gate_bwd profile and its bias-sum fix.
- **Round 3**, after typed host storage, Metal deferred faults and
  device-resident AdamW. It also settles the round 2 Metal "regressions".
- **Round 2**, after the Metal and wgpu optimization rounds landed.
- **Round 1**, kept below unchanged for comparison.

## Round 5 and the LM-head GEMM: 2026-10-02

### Round 5 (`bench/results/2026-10-02-r5/summary.md`)

Round 5 was run after the condvar feedback wait and the fused elementwise
checks. It used 5 rounds, the full row set, and the binaries in its
`env.txt`. As in round 3, the GPU read 100% busy from another process
throughout, so every row is flagged noisy. Ratios give the direction only.

- **At or near parity:** Metal's nanolab block forward + backward is 1.07×
  torch. The elementwise rows are 1.00–1.14×.
- **Slowest Metal rows (torch median / ojas median):**

  | Row | Ratio |
  |---|---:|
  | `linear_lmhead_fwd` | 0.48× |
  | `gate_bwd` (bias sum fixed below; wall time not re-measured) | 0.48× |
  | `accumulate_grad` | 0.57× |
  | `cross_entropy_bwd` | 0.61× |
  | `sdpa` fwd, d64 | 0.65–0.72× |
  | `block_fwd` | 0.66× |
  | `linear_lmhead_bwd` | 0.72× |
  | `linear_ce_c4096x50304` | 0.78× |

The LM-head rows were the largest by time, so they came first.

### What makes the LM head slow: one pass over the weight per tile row (verified)

The `gemm` group of `metal_bench` times tessl's exact-f32 GEMM directly,
outside the backend (`bench/results/2026-10-02-gemm/`):

- **N sets the rate, M doesn't.** NT runs at 6.5–7 TFLOP/s for N ≤ 2304. It
  drops to 2.4 at N = 16384 and to 2.3 at the LM head (4096 × 50304 × 768).
- **Every tile row is DRAM-bound.** At N = 50304, 32 rows (one row of 32×32
  tiles) run at the same ~2.4 TFLOP/s as 2048 rows: 1.0 ms each. That is one
  read of the 154 MB weight at ~150 GB/s. A 32-row tile does 16 flop per byte
  of weight, so a DRAM-bound pass caps near 2.5 TFLOP/s.
- **Cause (inferred from the kernel and confirmed by the fix):** the five
  exact-f32 kernels walked tiles row-major. Threadgroups running at the same
  time share an A tile and each read a different B tile, so B is re-read for
  every tile row. Splitting the vocabulary into ≤ 4096-column GEMMs halves
  the time, which matches.
- **The other two LM-head GEMMs:**
  - The weight gradient (TN, N = 768) is not affected.
  - The input gradient (NN with K = 50304) is a long-K shape, which the
    walk does not address.

### Fix: tessl walks a large B in column panels (verified)

With the user's approval, the change was made in tessl, the owner of the
kernels, rather than by chunking inside ojas. tessl's bf16 coop NN kernel
already had an 8-row column-panel swizzle. Its index math became a shared
helper, `tile_from_linear_panel`, and the exact-f32 kernels use it in
16-row bands. Each tile's arithmetic is unchanged, so outputs are
bit-identical; only the order threadgroups run in changes.

The first gate copied the coop kernel's grid size (≥ 2048 tiles). It
measured ~8% slower on the LM head's weight gradient, where B is only
12.6 MB. The gate was moved to B's size: panels when N·K ≥ 2²³ elements
(32 MiB).

Interleaved A/B of the GEMM probe, min of 4 runs per side
(`ab-grid-gate/`, `ab-footprint-gate/`), time relative to row-major:

| Shape | B | Row-major ms | Panels ms | Time ratio |
|---|---:|---:|---:|---:|
| NT LM head 4096 × 50304 × 768 | 154 MB | 137.9 | 63.6 | 0.46× |
| NT 128–2048 rows × 50304 | 154 MB | | | 0.39–0.47× |
| NT 4096 × 16384 × 768 | 50 MB | 43.5 | 19.2 | 0.44× |
| TN 768 × 2048 × 4096 | 32 MB | 2.62 | 2.02 | 0.77× |
| TN 50304 × 768 × 4096, grid gate | 12.6 MB | 55.0 | 59.3 | 1.08× (why the gate moved) |
| Transformer-width shapes | < 32 MB | | | 0.99–1.02× (not engaged) |

Some rows swing up to ±20% between runs under the outside GPU load. For
example, NT 4096 × 8192, whose code path did not change, read 1.21× in one
round. Only effects that held in every run are claimed above.

**End to end** (`bench/ab_metal_rows.sh` with `OLD_BIN`; 6 interleaved
rounds of 20 iterations; results in `ab-rows/` and `ab-rows-ce/`):

| Row | Before, min ms | After, min ms | Time ratio |
|---|---:|---:|---:|
| `linear_lmhead_fwd` | 134.3 | 68.2 | 0.51× |
| `linear_lmhead_bwd` | 164.3 | 156.8 | 0.95× (within noise) |
| `linear_ce_c4096x50304` | 304.8 | 239.4 | 0.79× |
| `linear_ce_c1024x8192` | 336.7 | 336.3 | 1.00× (25 MB chunks, not engaged) |
| `floor_silu_1` (control) | | | flat |

Parity is identical before and after: 0 on the LM-head forward and 9.5e-7 on
linear_ce.

Against round 5's torch numbers (unpaired, direction only), the LM-head
forward moves from ~0.48× to ~0.8× torch and `linear_ce_c4096x50304` from
0.78× to ~0.86×.

The new side also contained the cached-finite-flag change in ojas-core, made
by another session. The control row is flat, and the GEMM probe, which calls
tessl directly, shows the same halving. So the gain is the tile walk
(inferred from both).

**Tests:**
- `exact_f32_column_panels_cover_every_tile` in tessl's
  `tests/gemm_ragged_shapes.rs`.
- Panel shapes added to the accumulate and interior-branch tests in
  `tests/gemm_flag_paths.rs`.
- All of them use rank-one operands, so the reference stays O(M·N) at
  N·K ≥ 2²³.
- Removing the partial-band clamp makes all three fail, as does tessl's
  existing bf16 swizzle test, which now goes through the shared helper.

### Fix: the LM head's input gradient runs as K partitions (verified)

The input gradient is an NN GEMM, M 4096 × N 768 × K 50304, and it ran at
~3 TFLOP/s. Like the forward NT, it re-read all of B (here the 154 MB weight)
for every 32-row tile row. The panel walk can't fix it: each tile's A slab is
K long as well, so a band of 16 tile rows needs 103 MB of A.

tessl now runs this shape as K partitions (`nn_splitk_k_tile`,
`matmul2d_tensorops_nn_splitk_f32`):
- C is zeroed, then each partition of `2^21 / N` rows of B (2560 at N = 768)
  accumulates into it, in order. Each partition's 8 MiB slice of B stays in
  cache.
- The route is taken only when K·N ≥ 2²³, there is more than one tile row,
  and K is at least four partitions long.
- The TN split-K lanes now share the same partition dispatcher.
- Results are deterministic but not bit-identical to the single dispatch
  (C is rounded once per partition). They stay within the f32 bound.

**GEMM probe**, interleaved, alternating order, min of 4 runs per side
(`bench/results/2026-10-02-gemm/ab-nn-splitk/`):

| Shape | Before ms (per-run range) | After ms (per-run range) | Time ratio |
|---|---|---|---:|
| NN 4096 × 768 × 50304 | 99.8 (100–113) | 45.4 (45–50) | 0.46× |
| NN 4096 × 768 × 16384 | 21.0 (21–23) | 14.6 (15–16) | 0.69× |
| Shapes that don't route | | | within noise |

At 45 ms the input gradient runs at ~7 TFLOP/s, the rate of the best
transformer-width shapes, so the partition width was not swept further.

**End to end**, unpaired (`splitk-rows/`): the panel-only `metal_vs_torch`
was overwritten before a paired run, and the machine's load average was
~350 during this one. So these compare minimums against the panel-walk A/B's
new side:

| Row | Panel walk only, min ms | + NN split-K, min ms | Time ratio |
|---|---:|---:|---:|
| `linear_lmhead_bwd` | 156.8 | 109.1 | ~0.70× |
| `linear_ce_c4096x50304` | 239.4 | 166.4 | ~0.70× |

Parity against torch on `linear_lmhead_bwd` improved from 1.6e-4 to
2.5e-5. `linear_ce` is unchanged at 9.5e-7.

Against round 5's torch minimums (unpaired, direction only):
- `linear_lmhead_bwd` moves from 0.72× to about parity (torch 106.7 ms).
- `linear_ce_c4096x50304` moves to about 1.2× torch (torch 206.2 ms).

**Tests:**
- `long_k_nn_with_a_large_b_takes_k_partitions` pins the routing (LM head at
  nanolab's and Lappi's vocabularies).
- `long_k_nn_partitions_cover_all_of_k` checks two ragged shapes with a
  short last partition on the GPU. B is rank-one, with powers of two in `v`,
  so every k contributes and the reference stays O(M·K).
- It fails with A's partition offset dropped (31.8× its error budget) or a
  partition skipped (14.8×).

**Still open:** the other round 5 rows listed above.

### gate_bwd: where its time went, and the bias sum (2026-10-02)

`metal_bench <iters> gate` times each dispatch `MetalBackend` records for
`per_head_sigmoid_gate_backward` (`fn gate` in `ojas-metal/src/device.rs`)
at the row's shape: 4096 rows, 12 heads of 64, d_model 768. Each row is
the GPU span of a command buffer holding that dispatch alone; the last
holds all of them. Two runs before the fix, min µs, M5 Pro
(`bench/results/2026-10-02-gate/gate-probe-{1,2}.md`):

| Dispatch | min µs | Share of the whole |
|---|---:|---:|
| `ojas_per_head_gate_dbias` (bias gradient) | 803–807 | 50% |
| ten `ojas_check_finite` passes | 230–231 | 14% |
| TN `gw = d_pre^T · x` (12 × 768 × 4096) | 219–224 | 14% |
| `ojas_per_head_gate_bwd` (gate kernel, ~250 GB/s) | 151–154 | 9% |
| NN `gx = d_pre · w` (4096 × 768 × 12) | 110–114 | 7% |
| NT `pre = x · w^T` (4096 × 12 × 768) | 50–53 | 3% |
| whole backward, one command buffer | 1618–1636 | |

So the earlier note that `gate_bwd` "is dominated by its two GEMMs" was
wrong (it was never measured); the GEMMs are ~330 µs of ~1620.

**Fix: the bias sum runs one SIMD-group per head.** The kernel gave each of
the 12 heads one thread, which made 4096 dependent strided loads. Now each
head's 32 lanes load a chunk of rows each (four chunks in flight), and the
chunk is added lane by lane through `simd_shuffle`. The additions are the
CPU reference's ascending-row order, so the result is bit-identical. Both
dispatch sites (`device.rs` and the tiny step in `gpu.rs`) take the grid
from `gate_dbias_threads`.

**A/B**, old and new `metal_bench` binaries interleaved with alternating
order, 4 rounds per side, min µs per round
(`bench/results/2026-10-02-gate/ab-dbias/`):

| | Old | New | Time ratio |
|---|---|---|---:|
| `ojas_per_head_gate_dbias` | 804.5–807.4 | 120.0–120.4 | 0.15× |
| whole backward, GPU span | 1585.8–1632.4 | 938.1–941.7 | 0.59× |

Every round of the new side beat every round of the old. Seven of the eight
runs started with the GPU reading 0% busy (old round 1: 58%), unlike the
earlier rounds.

**Tests:**
- `gate_dbias_sums_rows_in_ascending_order_bit_for_bit` (`gpu.rs`) compares
  the kernel bit for bit with a host loop in ascending row order. It uses
  rows 1 to 4097 around the 32-row chunks and 128-row load groups, heads 1,
  3 and 12, and values of ±3e7 among small ones, so any other order lands on
  other bits.
- `gate_dbias_keeps_the_empty_and_short_length_edges` keeps the serial
  kernel's behaviour: no rows writes zeros, and a length too short for a
  head's last row leaves that head unwritten.
- Both pass on the old kernel too. They pin behaviour across the change and
  do not fail before it; the A/B is the evidence for the speed. Mutants they
  kill: each chunk added in reverse, the final partial chunk dropped, and
  each chunk reduced with `simd_sum`.
- ojas-metal suite: 162 passed. The one failure is
  `backend::tests::memory_probe_tracks_residency_and_plans_one_shared_budget`,
  another lane's new test (reported to that lane). clippy `--all-targets
  --features metal` is clean.

**Fix: the gate backward kernels check their own operands.**
`ojas_per_head_gate_bwd` sets ST_IN for a non-finite `bias`, `attn` or
`grad_output` and ST_OUT for a non-finite `pre` or `d_attn`.
`ojas_per_head_gate_dbias` sets ST_OUT for a non-finite `d_bias`. Four
standalone passes remain: `x` and `w` in, `gx` and `gw` out, which only
tessl GEMMs touch. The forward is unchanged. The tiny step in `gpu.rs`,
which dispatches the same kernels, now binds a status buffer and refuses a
non-finite `pre`, `d_attn`, `d_bias`, `d_input` or `d_weight`. Before, it
returned them.

A/B, the bias-sum binary against this one, interleaved with alternating
order, 4 rounds per side, min µs per round
(`bench/results/2026-10-02-gate/ab-fold/`; one old round started with the
GPU 49% busy, the rest at 0%):

| | Old | New | Time ratio |
|---|---|---|---:|
| standalone finite checks | 230.5–241.6 (ten) | 67.5–77.3 (four) | 0.31× |
| `ojas_per_head_gate_bwd` with the checks inside | 131.3–143.8 | 138.3–150.6 | within noise |
| whole backward, GPU span | 957.0–965.1 | 764.4–774.8 | 0.80× |

**Tests:**
- `tests/fused_checks.rs`,
  `the_gate_backward_reports_non_finite_operands_and_results`:
  - a NaN or ±Inf at the end of an offset view in each of the five
    operands is reported, and a clean call reports nothing;
  - `pre = x · w^T` overflowing is reported. Its gate is then 1, so nothing
    downstream overflows.
  - a `d_bias` row sum overflowing is reported. Every gate is then 0.5 and
    `x` and `w` are zero, so it is the only non-finite value.
- `gpu.rs`:
  - `gate_dbias_flags_a_sum_that_is_not_finite`;
  - `gate_backward_refuses_a_gradient_that_overflows`, for `d_bias` from the
    status word and `d_input` from the host check. Before this change the
    tiny step returned both as `Ok`, so this test fails on the old code.
- Mutants, each killed (`ab-fold/summary.txt`):
  - the kernel's ST_IN store removed. Only a ±Inf bias catches it: its gate
    is exactly 0 or 1, so no output turns non-finite.
  - the kernel's ST_OUT store removed;
  - the bias sum's ST_OUT store removed;
  - `gpu.rs` ignoring the status words;
  - `gpu.rs` skipping the `d_input` check.
- ojas-metal suite: 166 passed, 0 failed. clippy `--all-targets --features
  metal` is clean.

**Fix (in tessl): a small-C TN runs all its K partitions in one dispatch.**
TN `gw` ran at 59 GB/s: 24 output tiles, each walking K = 4096. It missed
the TN split-K lane (N = 768 > 384), and that lane runs its partitions in
sequence, so it would not have added workgroups anyway. tessl's exact-f32
TN now routes any C under 128 tiles, over a K at least two partitions long,
to `matmul2d_tensorops_tn_splitk_par_f32`. That kernel computes every
partition at once into a scratch, and `reduce_partitions_f32` then adds the
partitions in order. The width (`tn_par_k_tile`) aims at ~768 threadgroups in
multiples of 256. The result is deterministic and within the f32 bound, but
not bit-identical to one dispatch.

`metal_bench 20 tn`, one quiet run (load 6.4, nothing else compiling; the
GPU read 52% busy at start, from UI processes), min µs
of 20 GPU spans (`bench/results/2026-10-02-gate/tn-sweep/tn-sweep-2.md`):

| TN shape (M, N, K) | Tiles | One dispatch | Before (route) | Now (parallel, routed width) |
|---|---:|---:|---:|---:|
| 12, 768, 4096 (gate `gw`) | 24 | 223.0 | 223.0 (one dispatch) | 44.6 (256) |
| 12, 768, 1024 | 24 | 59.8 | 59.8 (one dispatch) | 24.1 (256) |
| 128, 128, 4096 (attention dW) | 16 | 129.4 | 185.7 (sequential split-K) | 31.3 (256) |
| 128, 384, 4096 (MLP dW) | 48 | 131.6 | 192.3 (sequential split-K) | 74.3 (256) |
| 64, 64, 4096 | 4 | 127.3 | 185.5 (sequential split-K) | 19.3 (256) |
| 128, 128, 2048 | 16 | 69.3 | 96.2 (sequential split-K) | 23.8 (256) |
| 96, 2048, 4096 | 192 | 269.6 | 269.6 (one dispatch) | not routed (192 tiles) |

- The sequential split-K, which `prefer_tn_splitk` picked for tall-K small
  C, was slower than the single dispatch on every one of its shapes swept.
  The parallel lane replaces it for overwriting f32 TN. bf16 and `C +=`
  (`gemm_tn_accum_train`) keep the sequential kernel, since `C +=` must add
  into C. No ojas code calls `gemm_tn_accum_train`.
- The "Before" column for the first rows is the single dispatch, measured
  in the same run.
- Where the routed width is not the best one: at 12 × 768 × 1024, a width of
  128 was ~15% faster at the median (21.4 vs 25.3 µs); at 128 × 384 × 4096,
  512 was ~6% faster at the min and ~2% at the median. Elsewhere the routed
  width is within ~2% of the best min.
- 96 × 2048 × 4096 (192 tiles) is not routed. A width of 512 ran 11% faster
  than one dispatch there (239.6 vs 269.6 µs) in this one run; widening the
  tile threshold has not been measured beyond it.
- tessl tests: `few_tile_long_k_tn_takes_parallel_partitions` (routing, and
  the scratch cap held over the whole old sequential domain; inferred, not
  run: the four old sequential shapes fail it before the change, where
  `tn_par_k_tile` returned `None` for them),
  `parallel_tn_partitions_cover_all_of_k` (rank-one B so every k counts;
  routed, ragged, padded-slice and one-partition cases),
  `parallel_tn_refuses_bad_widths_and_an_oversized_scratch`,
  `parallel_tn_is_bit_identical_across_100_calls_in_flight`.
- Mutants (`mutate-tn.sh`):
  - two are killed by `parallel_tn_partitions_cover_all_of_k`: A's partition
    offset dropped, and the last partition left out of the sum.
  - one survives: scratch slices packed at M·N without padding to 16 bytes.
    The M5 Pro computes correctly from a 4-byte-aligned slice, so the padding
    stays for tessl's 16-byte rule on GEMM operands, and no test observes it.
- tessl suite: 531 passed, 0 failed, 8 ignored. ojas-metal suite: 166
  passed, 0 failed. tessl clippy `--all-targets -D warnings` fails on one
  `type_complexity` error at `src/runtime.rs:345` (`FeedbackWatch`). That
  line is in tessl's uncommitted Metal 4 commit-feedback hunk, not HEAD and
  not this change. clippy is clean with that lint allowed
  (`bench/results/2026-10-02-gate/ab-tn/summary.txt`).

A/B of the whole gate backward, the fold binary against this one,
interleaved with alternating order, 4 rounds per side, min µs per round
(`bench/results/2026-10-02-gate/ab-tn/`). A non-Claude agent outside the
lock raised the 1-min load from ~7 to ~19 during these runs, and one old
round started with the GPU 55% busy:

| | Old | New | Time ratio |
|---|---|---|---:|
| TN `gw` | 227.4–234.1 | 43.1–46.0 | 0.19× |
| whole backward, GPU span | 790.4–806.4 | 601.7–638.2 | 0.78× |

Every round of the new side beat every round of the old. The old side's
whole backward ran ~25 µs slower than in the fold A/B (764–775), which is
the load.

**Wall time against torch** (`bench/run_paired.sh 5`, gate rows only,
`bench/results/2026-10-02-gate/paired-gate/`): `gate_bwd` medians were ojas
0.862 ms and torch 0.834 ms (0.96×, per-round 0.85–0.99×). Round 5 had
2.0 ms against ~1.0 ms. This run is flagged noisy: ojas's per-round spread
was 17%, and the 1-min load was ~30 throughout from the same outside agent.

Re-run with the outside agents idle and the 1-min load at 8.3–9.1
(`bench/results/2026-10-02-gate/paired-gate-quiet/`). `gate_bwd` medians
were ojas 0.829 ms and torch 0.833 ms (1.02×, per-round 0.96–1.11×). Spread
was 8% for ojas and 15% for torch. The row is still flagged noisy, from
torch's side, and so still not quoted. The GPU read 43–53% busy even at the
quietest sample of every lane, from something outside the benchmark;
XProtect (~89% CPU) and the GitPulse app (~80%) were active. Both runs put
`gate_bwd` at about parity with torch, against 0.5× in round 5. A quotable
number needs a run with the GPU idle between rows.

**Fix: the bias sum stages rows in threadgroup memory.** The SIMD-group
version made each link of its 4096-add chain a `simd_shuffle` and an add,
about 29 ns per row. Now the SIMD-group's lanes load a 2048-row block of a
head into threadgroup memory together, and lane 0 adds it in row order,
reading eight values ahead. The additions are in the same order, so the
sum is still bit-identical to the CPU reference. ojas compiles its kernels
with `-fmetal-math-mode=safe -ffp-contract=off`, so the compiler may not
reassociate them.

A/B, the parallel-TN binary against this one, interleaved with alternating
order, 4 rounds per side, min µs per round
(`bench/results/2026-10-02-gate/ab-stage/`). The 1-min load was 4.2–6.3.
The GPU read 33–55% busy at the start of every run, from the GitPulse app
spawning git processes outside the lock:

| | Old | New | Time ratio |
|---|---|---|---:|
| `ojas_per_head_gate_dbias` | 120.4–123.4 | 53.4–56.4 | 0.45× |
| whole backward, GPU span | 619.9–621.4 | 548.6–561.2 | 0.90× |

Every round of the new side beat every round of the old. The new binary
also carries another session's edit to `metal_bench.rs`. The probe's other
rows match the old side within noise (`pre` 49–53, `gate_bwd` 154–165, `gx`
111–112, `gw` 43–44, checks 81–82 µs), so the timing method is unchanged.

**Tests:**
- `gate_dbias_sums_rows_in_ascending_order_bit_for_bit` now also covers 2047,
  2048, 2049 and 4096 rows, around the new 2048-row blocks. It passes on
  both kernels: it pins the order across the change, and the A/B is the
  evidence for the speed.
- ojas-metal suite: 166 passed, 0 failed.
- Not run:
  - the three mutants in `mutate-stage.sh` (an add pair swapped, the tail
    loop dropped, every block read from row 0). After the 14:24 kernel
    panic, the rules for this Mac bar fork-heavy steps. That the test
    catches these mutants is unverified for this kernel; it caught the
    equivalent mutants of the shuffle kernel.
  - ojas-metal clippy. It stopped in `ojas-cpu`, at a dead function the CPU
    lane was adding at the time, before it reached ojas-metal. The only Rust
    change here is the test's row list.

At ~13 ns per row, 53 µs is still well above what an add chain alone should
cost. A likely factor, unmeasured, is that 12 SIMD-groups on the whole GPU
may not raise its clock.

**Next, by the same probe (not yet done):**
- a paired `gate` run with the GPU idle between rows (both runs above were
  flagged noisy);
- letting the bias sum overlap the `gx` and `gw` GEMMs. It only reads
  `d_pre`, as they do. The whole backward (549–561 µs) is more than its
  parts measured one by one (~493 µs), so no overlap is visible; the ~60 µs
  gap also holds the gaps between dispatches. Unmeasured hypothesis;
- the routed width at short K (12 × 768 × 1024 above).

## Round 3: 2026-10-01, 22:50–23:06 CDT

### What was run (verified, `bench/results/2026-10-01-r3b/env.txt`)

- `bench/run_paired.sh 5`: 5 rounds, 20 iterations, 5 warm-up, every row,
  lane order alternating per round.
- HEAD dab2a12 plus the uncommitted tree; benchmarked-crates diff sha1
  `de67949a36ff`. This is the tree of snapshot `wip/typed-storage-2026-10-01`
  (8b0e365), before the CPU session's fused AdamW landed in `ojas-cpu`.
  `ojas-cpu` is not on either GPU lane's timed path.
- PyTorch 2.13.0 MPS, Python 3.14.7, Apple M5 Pro, macOS 27.0.1.
- Parity passed on every row of both lanes.
- The first attempt (`bench/results/2026-10-01-r3/`) died of a full disk
  after round 1 and is not used.

### Conditions: nothing in this round is quotable

- **Every row of both lanes fails the 10% spread gate** (94 of 94).
- **The GPU read 100% "Device Utilization" on all 180 samples**, including
  before the first lane started. In round 2 the same probe read 0–99%.
  It still read 100% after the run, with no ojas or torch process alive.
  Something outside the benchmark held the GPU. About eight Chrome renderer
  processes sat near 95% CPU each for about 35 hours. Per-process GPU use
  needs root (`powermetrics`), so which process held it is **unverified**.
- **Host load rose from about 15 to 130 during rounds 3–5** (1-minute
  average). Rounds 1–2 ran at 14–19.
- So ratios below give direction only, and only where the change is far
  larger than the spread.

### Round 2's Metal "regressions" are not code (verified)

Round 2 reported four Metal rows slower than round 1:
- `rope_bwd` 0.56x;
- `permute_bthd_bhtd` 0.48x;
- `residual_add_fwd` 0.71x;
- `residual_add_bwd` 0.82x.

Those were medians compared across separate runs.

**Interleaved A/B, [`bench/ab_metal_rows.sh`](../bench/ab_metal_rows.sh):**
- **Old:** ed17286, the snapshot taken between rounds 1 and 2, built with its
  own harness. In it, Metal ops are synchronous and `sync` is a no-op.
- **New:** the round 3 binary, ops deferred until `Backend::sync`.
- **Protocol:** 10 rounds, order alternating, 50 iterations, 10 warm-up.
  `floor_silu_1` is the control, a one-element op that measures only the
  per-op fixed cost.
- **Build caveat:** ed17286's `ojas-cpu` does not compile; it was captured
  mid-edit. In the A/B copy only, one unused import and one uncalled function
  (`adamw_scratch`) were removed. `ojas-cpu` is a dev-dependency of
  `ojas-metal` and the Metal bench never calls it.
- **What is durable:** the old tree was a `git archive` of ed17286 in the
  git-ignored `target-baseline/ab/ojas`, so it is not kept. The raw
  per-round results in `bench/results/2026-10-01-ab/` are the record.

| row | old min / median of mins / median of medians ms | new, same | new min slower in |
| :-- | :-- | :-- | --: |
| floor_silu_1 | 0.106 / 0.118 / 0.369 | 0.102 / 0.135 / 1.389 | 6/10 |
| rope_bwd | 0.323 / 0.348 / 1.497 | 0.330 / 0.359 / 1.598 | 6/10 |
| permute_bthd_bhtd | 0.310 / 0.346 / 1.620 | 0.331 / 0.358 / 1.622 | 6/10 |
| residual_add_fwd | 0.540 / 0.577 / 1.394 | 0.566 / 0.608 / 1.027 | 5/10 |
| residual_add_bwd | 0.519 / 0.566 / 1.836 | 0.542 / 0.567 / 0.943 | 8/10 |

- **The minimums agree within 0–5%.** New is 0–5% higher on the median of
  round minimums for every row (`residual_add_bwd` 0.566 vs 0.567), and
  slower on the paired round minimum in 5–8 of 10 rounds. Round 2 claimed
  22–108% slower medians.
- **The medians are bimodal on both binaries.** A per-round median is either
  low (0.2–0.9 ms) or high, about 1.1–1.3 ms above that (1.1–2.3 ms).
  Rounds with a high median, out of 10 (old / new):
  - `rope_bwd` 6 / 6;
  - `permute_bthd_bhtd` 7 / 6;
  - `residual_add_fwd` 5 / 5;
  - `residual_add_bwd` 6 / 4;
  - the `floor_silu_1` control 3 / 7.

  The control's 3 / 7 split is why its median of medians differs
  (0.369 vs 1.389 ms). With 10 rounds the split is not distinguishable from
  chance, so whether deferred submission lands in the high mode more often is
  **unverified**. Rounds 1, 2 and 3 of the paired runs show the same two
  modes. Medians from separate runs therefore cannot detect a regression of
  this size.
- **Verdict:** no code regression. No bisect is needed. The real problem is
  the bimodal per-op fixed cost (next section).

### What moved since round 2 (direction verified, magnitude approximate)

Unpaired across runs, and round 3 ran under a saturated GPU. These are
quoted only because the change is several times the spread.

- **Metal `adamw_full`: minimum 200.8 -> 21.4 ms, about 9x.** It now leads
  torch (1.55x, faster on the per-round median in 5/5 rounds). Round 2's
  read was one command per tensor, each followed by a device wait and a
  status read. Both rounds ran on HEAD dab2a12, which already contains
  3e8db12 (device-resident gradient), so that commit is not the cause. The
  uncommitted Metal change between the rounds is deferred faults, which
  drops the per-op wait. That it explains the 9x is **inferred**; it was not
  A/B'd.
- **Metal `block_fwd` minimum 35.2 -> 22.5 ms; `block_fwd_bwd` 99.0 -> 71.1
  ms.** `block_fwd_bwd` is now 0.92x torch (mixed). This is consistent with
  removing the per-op device wait from 24 synchronous ops (**inferred**).
- **Metal `accumulate_grad_50304x768` minimum 9.18 -> 3.02 ms.**
- **Metal `linear_ce_c4096x50304` minimum 377.8 -> 298.6 ms** (0.55x ->
  0.81x).
- **No meaningful change:**
  - wgpu rows;
  - Metal GEMMs;
  - Metal SDPA;
  - cross-entropy backward on both backends.

### The remaining Metal bottleneck: the per-op fixed cost

- **The one-element `floor_silu_1` row:**
  - Metal minimum 0.10–0.13 ms, median 0.24–1.74 ms;
  - torch minimum 0.07 ms, median 0.12 ms.
- **Every small Metal op sits on the same floor,** with a median of 1–2 ms
  against a minimum of 0.3–0.6 ms:
  - elementwise ops, norms and permute;
  - `decode_attn_kv1024`, 0.15 / 1.38 ms against torch's 0.09 / 0.13.
- **This is what ranks them 0.2–0.4x torch.** Kernel time is not the cause.
- **wgpu shows a high round too, less often.** Its `floor_silu_1` per-round
  median was high in 1 of 5 rounds in both round 2 (3.02 ms) and round 3
  (2.25 ms). Metal's was high in 4 of 5 rounds in round 2 and 2 of 5 in
  round 3. wgpu does not use tessl, so the cause below does not explain
  wgpu's high rounds (**unexplained**).

### What sets the Metal floor: a 1 ms sleep poll in tessl's synchronize (verified)

**Profile:** `cargo run -p ojas-metal --release --example metal_bench 20
floor`, 500 runs per scenario. Output is in
`bench/results/2026-10-01-floor/floor.md`. The GPU read 100% busy and load
was 42–45, as in round 3.

**One-element op plus sync, means over the fast and the slow (> 1 ms)
runs:**

| scenario | slow runs | fast: op / sync / event wait / GPU span ms | slow: op / sync / event wait / GPU span ms |
| :-- | --: | :-- | :-- |
| tessl directly: `scale_f32_inplace` + `synchronize` | 265/500 | 0.029 / 0.198 / 0.170 / 0.022 | 0.026 / 1.449 / 0.150 / 0.017 |
| tessl directly, 5 ms idle before | 268/500 | 0.055 / 0.257 / 0.215 / 0.022 | 0.059 / 1.694 / 0.375 / 0.020 |
| `MetalBackend`: `silu_forward` + `sync` | 200/500 | 0.056 / 0.223 / 0.163 / - | 0.098 / 1.548 / 0.184 / - |

**Reading:**
- **The two modes exist in tessl alone,** on one thread with no ojas
  device thread or channel. `MetalBackend::sync` with nothing recorded
  takes 0.003 ms, so the ojas channel round trip is not the cost.
- **In slow runs, `synchronize` grows by about 1.25 ms.** The event wait
  inside it (tessl's `infer_trace` sync-wait counter) and the GPU span
  (Metal 4 counter-heap timestamps, about 0.02 ms) do not grow. The extra
  time is host code in `synchronize` outside the wait.
- **An idle gap before the op does not create the slow mode:** 53% slow
  with no gap (265/500), 54% with 5 ms (268/500). So it is not a GPU clock
  ramp after idle.

**Stack sample** (`/usr/bin/sample`, 4 s at 1 ms, during the tessl-direct
scenario; `bench/results/2026-10-01-floor/sample-tessl-sync.txt`):

| where | samples | share of the main thread (2339) |
| :-- | --: | --: |
| `GpuRuntime::commit_m4` | about 2320 | 99% |
| of it, `observe_finished_feedback` | 2072 | 89% |
| of it, `nanosleep` | 2069 | 88% |
| of it, `waitUntilSignaledValue` (the GPU wait) | 221 | 9% |
| encoding the op (`scale_f32_inplace`) | 19 | 1% |

**Cause (before the fix below):**
- `observe_finished_feedback`
  (`tessl/src/runtime.rs`, uncommitted in the tessl checkout) ran after
  every waited commit.
- It looped on `std::thread::sleep(1 ms)` until Metal's commit-feedback
  callback had set the watch's `done` flag, for up to 500 ms.
- When the callback landed after the shared event, the thread slept once. A
  1 ms sleep returns after 1.2–1.4 ms here. That happened on 53–85% of
  runs: 53% in the profile, 65–85% in the A/B's old rounds.
- **Verified** by the sample and the phase split. That the callback
  arriving after the event is what picks the mode is **inferred**: the
  sleep only ran when `done` was still false.
- The tessl runtime was already in this state for rounds 1 and 2 (it is
  dated before them), so every round's Metal medians include it.

### Fix: the feedback wait blocks on a Condvar (verified)

**The change** (`tessl/src/runtime.rs`, with the user's approval):
- `FeedbackState` gets a `Condvar`. The feedback callback sets `done` under
  the gate mutex and notifies.
- `wait_done(limit)` blocks on that notify instead of sleeping 1 ms per
  check.
- The 500 ms cap (`FEEDBACK_WAIT`) and the pending-watch handling are
  unchanged, so a callback that has not run still stays pending as before.
- The fault message is now read through a poisoned lock instead of
  `.ok()`. Previously a poisoned lock dropped the fault, and a lost fault
  reads as success.
- The diff of only these lines is in
  `bench/results/2026-10-01-floor/tessl-condvar.diff`.
  The rest of `runtime.rs`'s uncommitted work is another author's.

**Tests:**
- Four new tests in tessl's `runtime::tests`: `feedback_wait_*`.
- The wake-latency test failed against the poll with a 762 µs median wake,
  and passed 6 of 6 runs with the Condvar.
- tessl's full suite: 524 passed, 0 failed, 8 ignored (59 test binaries).
- ojas-metal: 154 passed, 0 failed.

**Interleaved A/B** (`bench/results/2026-10-01-floor/ab/summary.md`):
- Protocol: `metal_bench 4 floor`, 6 rounds alternating old and new, 100
  runs per scenario per round, load 18–20 on every run.
- Old is the same ojas tree built against tessl with the pre-change
  `runtime.rs`, saved as
  `bench/results/2026-10-01-floor/tessl-runtime-before-condvar.rs.txt`. The
  driver is `target-baseline/floor_ab.sh`, which is git-ignored; the raw
  per-round tables beside the summary are the record.

| scenario | slow runs, old → new (of 600) | p50 ms, median of rounds [range] old → new | p90 ms old → new |
| :-- | :-- | :-- | :-- |
| tessl `scale` + `synchronize` | 388 → 0 | 1.413 [1.361–1.429] → 0.144 [0.135–0.153] | 1.523 → 0.211 |
| tessl, 5 ms idle before | 507 → 6 | 1.468 → 0.232 | 1.558 → 0.335 |
| `MetalBackend` `silu_forward` + `sync` | 404 → 0 | 1.427 [0.199–1.456] → 0.165 [0.122–0.196] | 1.538 → 0.248 |
| `MetalBackend`, 20 ms idle before | 390 → 2 | 1.510 → 0.305 | 1.629 → 0.477 |
| `MetalBackend` 8 ops + one `sync` | 480 → 4 | 1.500 → 0.247 | 1.625 → 0.379 |

- **Minimums do not move** (0.097 vs 0.101 ms). The fix removes the slow
  mode; it does not make the fast path faster.
- **The p90 ranges of the two sides do not overlap** in any scenario.
- **The 0–6 slow runs left per 600 grow in every phase:** the op call,
  the event wait, and the rest of `sync`. That is consistent with the thread
  being descheduled under load (**inferred**). None shows the fixed +1.25 ms
  host step.

**Against torch after the fix** (`bench/results/2026-10-01-r4-smallops/`):
- Protocol: `bench/run_paired.sh 5`, limited by `BENCH_ROWS` to the small
  rows. Load was 18–20 on every row; the GPU still read 100% busy.
- Not interleaved with round 3, whose load reached 130. So the round 3
  column is context, and the A/B above is the paired evidence.
- Only `rms_norm_bwd` (2.77x) and `accumulate_grad` (0.62x) pass the 10%
  spread gate. The rest give direction only, but the spreads tightened from
  100–600% in round 3 to mostly 4–30%.

| Metal row | round 3 min / median ms | now min / median ms | now vs torch (median ratio [range]) |
| :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.114 / 0.384 | 0.095 / 0.189 | 0.76x [0.65–1.00] |
| rope_fwd | 0.380 / 1.625 | 0.316 / 0.384 | **3.45x** [3.30–3.73] |
| rope_bwd | 0.373 / 1.637 | 0.335 / 0.441 | **3.88x** [3.60–4.62] |
| permute_bthd_bhtd | 0.326 / 1.602 | 0.318 / 0.360 | **1.13x** [1.08–1.26] |
| decode_attn_kv1024 | 0.150 / 1.383 | 0.138 / 0.169 | 0.96x [0.67–1.11] |
| vres_fwd / vres_bwd | 0.515 / 1.805; 0.867 / 2.248 | 0.463 / 0.550; 0.837 / 0.939 | 0.94x; 0.99x |
| gate_fwd | 0.554 / 2.085 | 0.512 / 0.597 | 0.84x [0.55–0.96] |
| residual_add_fwd | 0.557 / 1.858 | 0.551 / 0.618 | 0.49x [0.40–0.55] |
| rms_norm_fwd | 0.361 / 1.289 | 0.334 / 0.436 | 0.49x [0.45–0.52] |
| silu_fwd / silu_bwd | 0.813 / 1.149; 1.119 / 2.672 | 0.790 / 0.873; 1.115 / 1.246 | 0.49x; 0.50x |
| mul_fwd / mul_bwd | 1.134 / 2.026; 1.796 / 3.036 | 1.108 / 1.246; 1.835 / 1.945 | 0.47x; 0.54x |
| gate_bwd | 1.951 / 2.697 | 2.015 / 2.120 | 0.47x [0.46–0.52] |
| accumulate_grad_50304x768 | 3.018 / 4.594 | 2.995 / 3.413 | 0.62x [0.58–0.64] (gate passed) |

**The next Metal bottleneck is in the kernels, not the fixed cost.**
- The rows still near 0.5x are bandwidth-bound elementwise and norm ops at
  [4096, 768] to [4096, 2048]: `silu`, `mul`, `residual_add`, `gate_bwd`,
  `rms_norm_fwd`.
- Their **minimums** are 2.0–2.4x torch's: `silu_fwd` 0.790 vs 0.354 ms,
  `mul_fwd` 1.108 vs 0.481. No fixed cost explains a gap in the minimum.

### What the elementwise gap is: standalone finite-check passes (verified)

**Profile:** `cargo run -p ojas-metal --release --example metal_bench 40
kernels`. Output is in `bench/results/2026-10-02-kernels/kernels.md`.
- Each row is the GPU span of command buffers holding only that kernel or
  sequence. 40 buffers per row; min and median are within 3–10%.
- Shapes are `[4096, 2048]`. The GPU read 100% busy, load 58.
- Op-level minimums (Metal op, torch op) are from
  `bench/results/2026-10-01-r4-smallops/`.

| op | kernel alone, min µs (GB/s) | sequence as `MetalBackend` runs it, min µs | checks' share of the sequence | `MetalBackend` op min ms | torch op min ms |
| :-- | :-- | --: | --: | --: | --: |
| silu_fwd | 262 (256) | 686 (check x, kernel, check y) | 62% | 0.790 | 0.354 |
| silu_bwd | 389 (259) | 1013 (3 checks) | 62% | 1.115 | 0.497 |
| mul_fwd | 387 (260) | 1013 (3 checks) | 62% | 1.108 | 0.481 |
| mul_bwd | 639 (262) | 1674 (5 checks) | 62% | 1.835 | 0.890 |

**Reading:**
- **The compute kernels are not slow.** They run at 256–262 GB/s, the same
  as tessl's plain read-and-write reference pass (283 GB/s). Each one alone
  is faster than torch's whole op: 262 µs vs 354 for `silu_fwd`, 387 vs 481
  for `mul_fwd`.
- **The finite checks are 62% of every sequence.**
  - `MetalBackend` runs `ojas_check_finite` as a separate pass over every
    input and every output (`ojas-metal/src/device.rs`, `fn check`).
  - That is 60 call sites across 23 ops.
  - Each check is a full read of the tensor, and the op waits for each one.
- **Each check is also inefficient.** A read-only pass should take about
  half of a read-and-write pass. `ojas_check_finite` takes 222 µs against
  the reference's 237 µs (151 GB/s): one float per thread, an
  `isfinite` test and an atomic store on failure
  (`ojas-metal/kernels/ojas_backend.metal:30`). That one float per thread is
  the cause is **inferred**; not profiled per instruction.
- **What is left between the sequence and the op minimum** is 0.10–0.16 ms:
  the output allocation, the residency flushes and the encode.

**Two fixes, by size:**
1. **Make `ojas_check_finite` read efficiently:** several floats per thread,
   which brings a check toward 120 µs. One kernel; all 60 call sites gain.
   Estimated `silu_fwd` op: about 0.6 ms, 0.6x torch.
2. **Fold the checks into the elementwise kernels.** Each kernel already
   reads its inputs and writes its output, so it can set the same status
   words with no extra memory traffic. Estimated `silu_fwd` op: about 0.36
   ms, roughly torch's. This touches each op's kernel and device code.

Both estimates are **inferred** from the table, not measured.

### Fix 1: the check reads several floats per thread (verified)

**The change:**
- `ojas_check_finite` walks its window with the grid's stride, so
  neighbouring lanes read neighbouring floats on every step. It stores at
  most once per thread, at the end.
- `fn check` in `device.rs` dispatches `ceil(n / CHECK_PER_THREAD)` threads,
  with `CHECK_PER_THREAD = 4`.
- **The sweep** (`metal_bench 30 kernels`, `[4096, 2048]`, min µs): 1 per
  thread 225; 2 per thread 147; 4 per thread 136; 8–32 per thread 144–145;
  64 per thread 152.

**Tests:**
- `ojas-metal/tests/check_finite.rs` places one NaN, +Inf or -Inf at:
  - every index of lengths up to 33;
  - the start, middle, end and stride boundaries of lengths around 2^k,
    up to 2^20 + 3;
  - inside and just outside row views whose byte offsets are not multiples
    of 16.
- **Mutation check:** a kernel that drops the last stride failed both tests.
- **Gates:** ojas-metal 156 passed, 0 failed. clippy --all-targets is clean.

**Interleaved A/B** (`bench/ab_metal_rows.sh 8`; old is the same tree with
the one-float kernel and dispatch; `bench/results/2026-10-02-kernels/ab-check/`).
8 rounds, load 25–34, median of round minimums:

| row | old ms | new ms | change |
| :-- | --: | --: | --: |
| silu_fwd | 0.849 | 0.662 | −22% |
| silu_bwd | 1.149 | 0.885 | −23% |
| mul_fwd | 1.155 | 0.920 | −20% |
| mul_bwd | 1.865 | 1.438 | −23% |
| residual_add_fwd | 0.586 | 0.488 | −17% |
| gate_fwd | 0.566 | 0.439 | −22% |
| gate_bwd | 2.055 | 1.902 | −7% |
| vres_fwd | 0.483 | 0.369 | −24% |
| vres_bwd | 0.914 | 0.722 | −21% |
| rms_norm_fwd | 0.364 | 0.282 | −23% |
| rms_qk_norm_fwd | 0.559 | 0.438 | −22% |
| rope_fwd / rope_bwd | 0.335 / 0.357 | 0.274 / 0.275 | −18% / −23% |
| permute_bthd_bhtd | 0.321 | 0.278 | −13% |
| accumulate_grad_50304x768 | 3.153 | 3.165 | 0% (has its own check kernel) |

- Parity is identical on both sides.
- `cross_entropy_fwd` and `adamw_full` were requested but not timed: the
  reused torch reference directory has no reference for them. The summary
  lists them as `no_ref`.

### Fix 2: the elementwise kernels check their own operands (verified)

**The change:**
- `ojas_silu_fwd`, `ojas_silu_bwd`, `ojas_mul_fwd`, `ojas_mul_bwd`,
  `ojas_vres_fwd` and `ojas_vres_bwd` take the op's status slot. Each thread
  sets `ST_IN` for a non-finite element it read and `ST_OUT` for one it
  wrote, which are the same words the standalone passes set.
- `residual_add_forward` is one kernel, `ojas_add_fwd` (`out = x + y`). It
  replaces a copy, tessl's in-place `residual_add` and three checks.
- `residual_add_backward` is one kernel, `ojas_add_bwd`. It replaces two
  copies and three checks.
- tessl's `residual_add` bounds-checked its `y` window; the new kernels do not
  need to. `Worker::view` refuses any window that does not fit its buffer
  (`device.rs`, `fn view`), and the backend refuses unequal shapes before the
  command reaches the device (`residual_add_forward_dims`,
  `residual_add_backward_dims` in `backend.rs`). Both verified from source.
- The value-residual `lambda` and its gradient are one value each and keep
  their standalone checks.
- The step-1 versions of `device.rs` and the kernel file are kept in
  `bench/results/2026-10-02-kernels/*-step1.*.txt`.

**Not folded: `gate`.**
- Its kernels are shared with the tiny training step in `gpu.rs`, which has
  no status slot.
- `gate_bwd` is dominated by its two GEMMs. (Unmeasured when written, and
  wrong: the bias sum was half its GPU time. See "gate_bwd: where its time
  went" under Round 5.)
- `gate_fwd` is about 0.84x torch; whether to pursue it is a follow-up
  decision.

**Tests (ojas-metal 159 passed, 0 failed; clippy --all-targets clean):**
- `tests/fused_checks.rs`:
  - finite inputs whose result overflows (`MAX * 4`, `MAX + MAX`,
    `MAX * silu'(3)`, each `mul_backward` gradient) are reported. That is
    the output path on its own; every NaN-in test also trips the output
    check.
  - a NaN or ±Inf at the end of an offset view is reported by all eight ops.
  - `add` and `mul`, forward and backward, match the CPU reference bit for
    bit.
- All three tests passed on the step-1 code too: behaviour is unchanged by
  design, so no test fails before the change.
- **Mutation checks:** removing every in-kernel `ST_IN` store failed the
  offset-view test (`residual_add_backward` reads `x` and `y` only to check
  them). Removing every `ST_OUT` store failed the overflow test.
- **Dependents** (ojas-qwen35, ojas-model, ojas-capi, ojas-infer): 206
  passed, 0 failed.

**Kernel GPU time** (`kernels-after-fold.md`, min µs, `[4096, 2048]`):

| kernel | before (no checks inside) | after (checks inside) |
| :-- | --: | --: |
| silu_fwd | 262 | 251 |
| silu_bwd | 389 | 410 |
| mul_fwd | 387 | 420 |
| mul_bwd | 639 | 673 |
| add_fwd | — | 403 |
| add_bwd | — | 652 |

The two columns come from separate runs (load 58, then 19), not paired. Within that, the checks inside change kernel time by −4% to +8%.

**Interleaved A/B** (`bench/ab_metal_rows.sh 8`; old is the step-1 tree;
`bench/results/2026-10-02-kernels/ab-fold/`). The old side is the live tree mirrored, with the two step-1 snapshot files copied back over `ojas-metal/src/device.rs` and `ojas-metal/kernels/ojas_backend.metal`, passed as `OLD_SRC`. Fix 1's old side was the same, with `ojas_check_finite` put back to one float per thread and `fn check` dispatching `n` threads. The driver scripts are in the git-ignored `target-baseline/`; the raw rounds beside each summary are the record. 8 rounds, load 19–20, median
of round minimums. The torch column is round 4's torch minimum, unpaired:

| row | step 1 ms | folded ms | change | vs torch now |
| :-- | --: | --: | --: | :-- |
| silu_fwd | 0.680 | 0.399 | −41% | 0.89x (torch 0.354) |
| silu_bwd | 0.921 | 0.529 | −43% | 0.94x (0.497) |
| mul_fwd | 0.922 | 0.523 | −43% | 0.92x (0.481) |
| mul_bwd | 1.426 | 0.789 | −45% | **1.13x** (0.890) |
| residual_add_fwd | 0.471 | 0.282 | −40% | 0.83x (0.235) |
| residual_add_bwd | 0.488 | 0.366 | −25% | API artifact (torch returns `gy`) |
| vres_fwd | 0.397 | 0.272 | −31% | **1.63x** (0.443) |
| vres_bwd | 0.743 | 0.496 | −33% | **1.44x** (0.714) |
| rms_norm_fwd, gate_fwd, gate_bwd, permute, floor (unchanged ops) | — | — | within ±5% | control |

Against round 4's numbers before both fixes, Metal `silu_fwd` went from
0.790 to 0.399 ms and `mul_bwd` from 1.835 to 0.789.

**Second-order costs, from the same table:**
- **Residency flushes and cold allocations.** Every `MetalBackend` op makes
  two residency-set flushes and one cold allocation (attributed to its
  output buffer, **inferred**). The fast-mode op call is 0.06–0.21 ms against tessl's 0.03 ms.
- **The fast-mode event wait.** It is 0.15–0.2 ms against a 0.02 ms GPU
  span. Under a GPU that read 100% busy this is likely queueing behind
  another process (**unverified**; not re-measured on an idle GPU).
- **torch's whole floor is 0.12 ms median**
  (`floor_silu_1`, round 3).

`residual_add_bwd` is still an API artifact, not a kernel (round 2, rank 1).

## Round 2: 2026-10-01, 19:01–19:13 CDT

### What was run (verified, `bench/results/2026-10-01-r2/env.txt`)

The protocol is the same as round 1's run A:
- `bench/run_paired.sh 5`;
- 5 rounds, alternating lane order;
- 5 warm-ups, then 20 timed iterations of op + sync per row;
- the same 10% spread gate;
- serial, with no other GPU lane active (per the coordinator).

The binaries were rebuilt by the runner from HEAD `dab2a12` plus an
uncommitted working tree. The diff of the benchmarked crates hashes to
`a73d2f0ef611`; it was `c226ffba076d` in round 1. tessl is unchanged
(`cf65d9d` with 20 dirty files). Toolchain and machine are as in round 1.

**New in round 2:**
- **Per-row load.** `uptime` and an `ioreg` GPU sample are recorded before
  and after every row (`_load` records in each round's JSONL; ranges at the
  end of `summary.md`).
- **Four new rows on all three runtimes.** All pass parity on both backends:

| row | ojas | torch twin |
| :-- | :-- | :-- |
| `linear_ce_c1024x8192`, `linear_ce_c4096x50304` | `linear_cross_entropy_mean`: N 4096, d 768, V 50304, loss plus both gradients, logit chunks of 1024x8192 and 4096x50304 | nanolab's own `FusedLinearCrossEntropy`, imported via `fused_linear_cross_entropy`, forward + `backward()`. It chunks rows only (n_chunks = 4096 / rows) and always spans the whole vocabulary. So **the torch twin of the 1024x8192 row uses 1024x50304 chunks**, and of the 4096x50304 row one 4096x50304 chunk |
| `decode_attn_kv1024` | `cached_attention_forward`, q [1,1,12,64], caches [1,1024,12,64] (time-major), kv_len 1024 | `F.scaled_dot_product_attention` without a mask on the cached K/V. torch keeps them in its own [B,H,T,D] layout, made contiguous once outside the timer |
| `accumulate_grad_50304x768` | `accumulate_grad` in place | `acc.add_(g)` |

**Machine state:**
- **CPU busy.** The 1-min load average was 12–23 throughout. Seven Chrome
  renderers were each pinned at about 100% CPU, as observed at 19:00.
- **GPU much quieter than round 1.** The per-block GPU samples were 0–64%
  (round 1: 84–100%).
- **Per-row GPU samples ranged 0–99%** with a median of 0–36% per lane-round.
  An `ioreg` read taken while that lane's own kernel runs shows its own work.

**Gate result: 8 of 96 rows pass** (spread at most 10% on both sides):
- Metal: `linear_up_bwd`, `linear_lmhead_fwd`, `muon_2048x768` (both twins),
  `block_fwd_bwd`;
- wgpu: `linear_up_bwd`, `linear_lmhead_fwd`, `block_fwd_bwd`.

A gate-failed ratio below is **not** a measured speedup. It is the run's
median of 5 paired ratios, with the min-max beside it. Where the
min-max range lies wholly on one side of 1 in all 5 rounds, the direction is
verified. A change of 2x or more between rounds is informative even when
gated out, but the before/after comparison pairs runs taken hours apart.
So the "ojas change" column compares unpaired medians: verified as a
direction when it is 2x or more, unverified in magnitude.

**Rows that got slower on Metal:**

| row | round 1 median (ms) | round 2 median (ms) | round 2 speed vs round 1 |
| :-- | --: | --: | --: |
| `rope_bwd` | 1.18 | 2.09 | 0.56× |
| `permute_bthd_bhtd` | 0.77 | 1.59 | 0.48× |
| `residual_add_fwd` | 1.56 | 2.18 | 0.71× |
| `residual_add_bwd` | 1.88 | 2.29 | 0.82× |

All are under 2x, which is below this comparison's resolution. None of these kernels changed in round 2. Their minimum times barely moved, so the cause is **inferred** to be the per-op sync floor under CPU load. It was not profiled, and no row here should be read as a regression in the kernel.

**Parity:** all 45 rows passed on both backends in all 5 rounds, with the
same gates as round 1. Metal `rms_qk_norm_bwd` went from 4.0e-6 to 1.0e-6
normalized error under its new kernel.

### Before/after: ojas MetalBackend

"before" is round 1 run A (`bench/results/2026-10-01/runA`) and "after" is
round 2. Ratio = torch / ojas per round; below 1 means ojas is slower.

| row | ojas median ms, before -> after | ojas change | ratio before (gate) | ratio after [min-max] (gate) | direction after (5 rounds) | parity after (rel) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 1.756 -> 1.136 | 1.55x faster | 0.06x (fail) | 0.18x [0.11-0.23] (fail) | slower 5/5 | 0.0e+00 |
| linear_qkv_fwd | 1.999 -> 1.834 | 1.09x faster | 0.46x (fail) | 0.53x [0.35-0.80] (fail) | slower 5/5 | 0.0e+00 |
| linear_qkv_bwd | 3.785 -> 2.418 | 1.57x faster | 0.53x (fail) | 0.86x [0.75-0.94] (fail) | slower 5/5 | 0.0e+00 |
| linear_up_fwd | 4.127 -> 2.906 | 1.42x faster | 0.51x (fail) | 0.84x [0.58-0.85] (fail) | slower 5/5 | 0.0e+00 |
| linear_up_bwd | 5.630 -> 4.821 | 1.17x faster | 0.78x (fail) | 0.88x [0.87-1.00] (**pass**) | mixed (4/5 slower) | 0.0e+00 |
| linear_down_fwd | 3.609 -> 2.734 | 1.32x faster | 0.61x (fail) | 0.84x [0.73-2.08] (fail) | mixed (4/5 slower) | 0.0e+00 |
| linear_down_bwd | 6.541 -> 5.229 | 1.25x faster | 0.66x (fail) | 0.85x [0.72-2.00] (fail) | mixed (4/5 slower) | 0.0e+00 |
| linear_lmhead_fwd | 131.9 -> 118.3 | 1.12x faster | 0.52x (fail) | 0.46x [0.44-0.49] (**pass**) | slower 5/5 | 0.0e+00 |
| linear_lmhead_bwd | 174.3 -> 156.2 | 1.12x faster | 0.70x (fail) | 0.70x [0.61-0.84] (fail) | slower 5/5 | 1.3e-05 |
| sdpa_b4h12t1024d64_fwd | 3.081 -> 2.418 | 1.27x faster | 0.55x (fail) | 0.57x [0.36-0.84] (fail) | slower 5/5 | 2.2e-07 |
| sdpa_b4h12t1024d64_bwd | 8.111 -> 6.473 | 1.25x faster | 1.24x (fail) | 1.53x [1.40-1.82] (fail) | faster 5/5 | 2.1e-06 |
| sdpa_b4h8t2048d64_fwd | 6.909 -> 5.146 | 1.34x faster | 0.50x (fail) | 0.65x [0.57-0.72] (fail) | slower 5/5 | 1.8e-07 |
| sdpa_b4h8t2048d64_bwd | 19.41 -> 14.75 | 1.32x faster | 1.38x (fail) | 1.73x [1.44-2.06] (fail) | faster 5/5 | 2.5e-06 |
| sdpa_b2h8t1024d128_fwd | 2.029 -> 1.641 | 1.24x faster | 0.61x (fail) | 0.74x [0.45-1.23] (fail) | mixed (4/5 slower) | 2.1e-07 |
| sdpa_b2h8t1024d128_bwd | 6.417 -> 4.902 | 1.31x faster | 0.81x (fail) | 1.09x [0.93-1.41] (fail) | mixed (2/5 slower) | 2.6e-06 |
| rms_norm_fwd | 1.654 -> 1.135 | 1.46x faster | 0.18x (fail) | 0.22x [0.19-0.32] (fail) | slower 5/5 | 2.0e-07 |
| rms_norm_bwd | 3.389 -> 1.706 | 1.99x faster | 0.74x (fail) | 1.75x [1.18-2.37] (fail) | faster 5/5 | 2.8e-07 |
| rms_qk_norm_fwd | 4.736 -> 2.135 | 2.22x faster | 0.11x (fail) | 0.21x [0.17-0.37] (fail) | slower 5/5 | 1.7e-07 |
| rms_qk_norm_bwd | 53.15 -> 2.579 | **20.6x faster** | 0.10x (pass) | 1.81x [1.48-3.50] (fail) | faster 5/5 | 1.0e-06 |
| rope_fwd | 1.816 -> 0.897 | 2.03x faster | 0.85x (fail) | 1.63x [0.90-2.89] (fail) | mixed (1/5 slower) | 0.0e+00 |
| rope_bwd | 1.182 -> 2.093 | 0.56x (slower) | 1.81x (fail) | 0.88x [0.85-1.69] (fail) | mixed (3/5 slower) | 0.0e+00 |
| silu_fwd | 2.267 -> 2.001 | 1.13x faster | 0.20x (fail) | 0.26x [0.19-0.35] (fail) | slower 5/5 | 6.1e-08 |
| silu_bwd | 3.191 -> 2.292 | 1.39x faster | 0.23x (fail) | 0.37x [0.24-0.48] (fail) | slower 5/5 | 1.1e-07 |
| mul_fwd | 2.899 -> 1.825 | 1.59x faster | 0.21x (fail) | 0.37x [0.20-0.50] (fail) | slower 5/5 | 0.0e+00 |
| mul_bwd | 4.131 -> 2.552 | 1.62x faster | 0.35x (fail) | 0.44x [0.40-0.49] (fail) | slower 5/5 | 0.0e+00 |
| residual_add_fwd | 1.559 -> 2.180 | 0.71x (slower) | 0.18x (fail) | 0.20x [0.14-0.29] (fail) | slower 5/5 | 0.0e+00 |
| residual_add_bwd | 1.878 -> 2.286 | 0.82x (slower) | 0.01x (fail) | 0.01x [0.00-0.02] (fail) | slower 5/5 | 0.0e+00 |
| gate_fwd | 1.773 -> 1.675 | 1.06x faster | 0.36x (fail) | 0.48x [0.20-0.86] (fail) | slower 5/5 | 1.0e-07 |
| gate_bwd | 4.069 -> 3.208 | 1.27x faster | 0.33x (fail) | 0.46x [0.36-0.50] (fail) | slower 5/5 | 2.3e-06 |
| vres_fwd | 1.645 -> 1.475 | 1.12x faster | 0.41x (fail) | 0.41x [0.26-0.90] (fail) | slower 5/5 | 0.0e+00 |
| vres_bwd | 1.943 -> 1.739 | 1.12x faster | 0.64x (fail) | 0.55x [0.35-0.82] (fail) | slower 5/5 | 2.3e-07 |
| permute_bthd_bhtd | 0.771 -> 1.591 | 0.48x (slower) | 0.72x (fail) | 0.31x [0.24-0.78] (fail) | slower 5/5 | 0.0e+00 |
| cross_entropy_fwd | 20.94 -> 3.851 | **5.44x faster** | 0.97x (fail) | 3.93x [3.57-4.63] (fail) | faster 5/5 | 0.0e+00 |
| cross_entropy_bwd | 96.68 -> 38.40 | **2.52x faster** | 0.35x (fail) | 0.72x [0.61-0.77] (fail) | slower 5/5 | 2.6e-10 |
| linear_ce_c1024x8192 | new -> 266.3 | new row | - | 0.88x [0.71-0.91] (fail) | slower 5/5 | 6.7e-06 |
| linear_ce_c4096x50304 | new -> 423.3 | new row | - | 0.55x [0.50-0.57] (fail) | slower 5/5 | 2.3e-05 |
| clip_grad_norm_full | 33.67 -> 17.59 | 1.91x faster | 4.58x (pass) | 7.67x [7.02-7.96] (fail) | faster 5/5 | 3.0e-07 |
| adamw_full | 598.1 -> 244.5 | **2.45x faster** | 0.09x (fail) | 0.16x [0.14-0.17] (fail) | slower 5/5 | 1.2e-07 |
| muon_768x768 | 6.507 -> 3.364 | 1.93x faster | 0.71x (fail) | 1.23x [1.15-1.30] (fail) | faster 5/5 | 0.0e+00 |
| muon_768x768 vs torch bf16 NS5 | 6.507 -> 3.364 | 1.93x faster | 0.44x (fail) | 0.64x [0.58-0.71] (fail) | slower 5/5 | timing only |
| muon_2048x768 | 12.01 -> 6.899 | 1.74x faster | 0.75x (fail) | 1.20x [1.18-1.22] (**pass**) | faster 5/5 | 5.0e-07 |
| muon_2048x768 vs torch bf16 NS5 | 12.01 -> 6.899 | 1.74x faster | 0.45x (fail) | 0.68x [0.66-0.73] (**pass**) | slower 5/5 | timing only |
| muon_768x2048 | 11.73 -> 6.914 | 1.70x faster | 0.77x (fail) | 1.21x [1.16-1.54] (fail) | faster 5/5 | 2.9e-07 |
| muon_768x2048 vs torch bf16 NS5 | 11.73 -> 6.914 | 1.70x faster | 0.45x (fail) | 0.60x [0.58-0.71] (fail) | slower 5/5 | timing only |
| block_fwd | 67.25 -> 49.22 | 1.37x faster | 0.24x (fail) | 0.32x [0.27-0.33] (fail) | slower 5/5 | 1.9e-07 |
| block_fwd_bwd | 230.5 -> 122.3 | 1.88x faster | 0.37x (fail) | 0.59x [0.56-0.64] (**pass**) | slower 5/5 | 2.5e-06 |
| decode_attn_kv1024 | new -> 0.832 | new row | - | 0.37x [0.22-1.16] (fail) | mixed (4/5 slower) | 3.2e-07 |
| accumulate_grad_50304x768 | new -> 9.624 | new row | - | 0.25x [0.21-0.53] (fail) | slower 5/5 | 0.0e+00 |

### Before/after: ojas WgpuBackend (Metal HAL)

| row | ojas median ms, before -> after | ojas change | ratio before (gate) | ratio after [min-max] (gate) | direction after (5 rounds) | parity after (rel) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.244 -> 0.359 | 0.68x (slower) | 0.48x (fail) | 0.52x [0.06-0.77] (fail) | slower 5/5 | 0.0e+00 |
| linear_qkv_fwd | 2.787 -> 2.167 | 1.29x faster | 0.31x (fail) | 0.47x [0.33-0.48] (fail) | slower 5/5 | 0.0e+00 |
| linear_qkv_bwd | 5.620 -> 3.453 | 1.63x faster | 0.38x (fail) | 0.55x [0.53-0.67] (fail) | slower 5/5 | 0.0e+00 |
| linear_up_fwd | 6.697 -> 4.652 | 1.44x faster | 0.32x (fail) | 0.47x [0.46-0.58] (fail) | slower 5/5 | 0.0e+00 |
| linear_up_bwd | 14.04 -> 8.729 | 1.61x faster | 0.31x (fail) | 0.49x [0.48-0.56] (**pass**) | slower 5/5 | 0.0e+00 |
| linear_down_fwd | 7.456 -> 4.601 | 1.62x faster | 0.29x (fail) | 0.49x [0.46-1.30] (fail) | mixed (4/5 slower) | 0.0e+00 |
| linear_down_bwd | 13.84 -> 8.661 | 1.60x faster | 0.32x (fail) | 0.51x [0.48-1.21] (fail) | mixed (4/5 slower) | 0.0e+00 |
| linear_lmhead_fwd | 194.7 -> 136.4 | 1.43x faster | 0.31x (fail) | 0.40x [0.39-0.41] (**pass**) | slower 5/5 | 0.0e+00 |
| linear_lmhead_bwd | 323.7 -> 199.9 | 1.62x faster | 0.37x (fail) | 0.54x [0.50-0.63] (fail) | slower 5/5 | 1.3e-05 |
| sdpa_b4h12t1024d64_fwd | 77.05 -> 11.83 | **6.51x faster** | 0.02x (fail) | 0.12x [0.12-0.14] (fail) | slower 5/5 | 2.4e-07 |
| sdpa_b4h12t1024d64_bwd | 277.8 -> 53.34 | **5.21x faster** | 0.04x (fail) | 0.19x [0.16-0.22] (fail) | slower 5/5 | 2.1e-06 |
| sdpa_b4h8t2048d64_fwd | 191.6 -> 30.40 | **6.30x faster** | 0.02x (fail) | 0.11x [0.09-0.11] (fail) | slower 5/5 | 1.8e-07 |
| sdpa_b4h8t2048d64_bwd | 705.7 -> 144.3 | **4.89x faster** | 0.04x (fail) | 0.18x [0.17-0.22] (fail) | slower 5/5 | 2.6e-06 |
| sdpa_b2h8t1024d128_fwd | 91.63 -> 11.85 | **7.73x faster** | 0.01x (fail) | 0.10x [0.09-0.17] (fail) | slower 5/5 | 1.8e-07 |
| sdpa_b2h8t1024d128_bwd | 280.7 -> 53.07 | **5.29x faster** | 0.02x (fail) | 0.10x [0.09-0.13] (fail) | slower 5/5 | 2.3e-06 |
| rms_norm_fwd | 0.424 -> 0.430 | 0.99x | 0.61x (fail) | 0.67x [0.43-0.82] (fail) | slower 5/5 | 1.3e-07 |
| rms_norm_bwd | 0.896 -> 0.704 | 1.27x faster | 3.42x (fail) | 3.10x [2.19-6.04] (fail) | faster 5/5 | 2.6e-07 |
| rms_qk_norm_fwd | 2.624 -> 2.102 | 1.25x faster | 0.17x (fail) | 0.20x [0.19-0.24] (fail) | slower 5/5 | 1.2e-07 |
| rms_qk_norm_bwd | 5.421 -> 4.303 | 1.26x faster | 0.95x (pass) | 1.11x [1.06-1.27] (fail) | faster 5/5 | 1.3e-06 |
| rope_fwd | 0.470 -> 0.367 | 1.28x faster | 3.67x (fail) | 3.70x [2.22-5.14] (fail) | faster 5/5 | 6.2e-08 |
| rope_bwd | 0.423 -> 0.393 | 1.08x faster | 4.73x (fail) | 4.86x [2.66-6.15] (fail) | faster 5/5 | 6.1e-08 |
| silu_fwd | 0.728 -> 0.769 | 0.95x | 0.75x (fail) | 0.73x [0.50-1.18] (fail) | mixed (4/5 slower) | 1.8e-07 |
| silu_bwd | 1.004 -> 0.892 | 1.12x faster | 0.86x (fail) | 0.85x [0.55-1.12] (fail) | mixed (4/5 slower) | 4.3e-07 |
| mul_fwd | 0.862 -> 0.804 | 1.07x faster | 0.74x (fail) | 0.85x [0.77-1.01] (fail) | mixed (4/5 slower) | 0.0e+00 |
| mul_bwd | 1.263 -> 1.162 | 1.09x faster | 1.06x (fail) | 1.01x [0.97-1.14] (fail) | mixed (1/5 slower) | 0.0e+00 |
| residual_add_fwd | 0.513 -> 0.552 | 0.93x | 0.72x (fail) | 0.62x [0.58-0.81] (fail) | slower 5/5 | 0.0e+00 |
| residual_add_bwd | 0.898 -> 1.062 | 0.85x | 0.02x (fail) | 0.01x [0.01-0.07] (fail) | slower 5/5 | 0.0e+00 |
| gate_fwd | 0.943 -> 1.186 | 0.79x (slower) | 0.74x (fail) | 0.65x [0.58-0.98] (fail) | slower 5/5 | 1.4e-07 |
| gate_bwd | 2.111 -> 2.267 | 0.93x | 0.63x (fail) | 0.64x [0.52-0.69] (fail) | slower 5/5 | 2.3e-06 |
| vres_fwd | 0.458 -> 0.534 | 0.86x | 1.32x (fail) | 1.11x [0.93-1.82] (fail) | mixed (1/5 slower) | 6.0e-08 |
| vres_bwd | 0.720 -> 0.722 | 1.00x | 1.67x (fail) | 1.36x [1.12-1.69] (fail) | faster 5/5 | 2.3e-07 |
| permute_bthd_bhtd | 0.628 -> 0.772 | 0.81x (slower) | 0.89x (fail) | 0.65x [0.56-1.37] (fail) | mixed (4/5 slower) | 0.0e+00 |
| cross_entropy_fwd | 7.139 -> 7.374 | 0.97x | 2.79x (fail) | 2.21x [1.87-2.49] (fail) | faster 5/5 | 0.0e+00 |
| cross_entropy_bwd | 63.71 -> 47.74 | 1.33x faster | 0.53x (fail) | 0.60x [0.55-0.71] (fail) | slower 5/5 | 2.6e-10 |
| linear_ce_c1024x8192 | new -> 474.2 | new row | - | 0.45x [0.42-0.52] (fail) | slower 5/5 | 2.3e-05 |
| linear_ce_c4096x50304 | new -> 385.1 | new row | - | 0.58x [0.56-0.64] (fail) | slower 5/5 | 2.3e-05 |
| clip_grad_norm_full | 24.18 -> 17.18 | 1.41x faster | 6.36x (pass) | 7.81x [6.99-7.86] (fail) | faster 5/5 | 3.8e-07 |
| adamw_full | 75.15 -> 62.46 | 1.20x faster | 0.70x (fail) | 0.63x [0.61-0.64] (fail) | slower 5/5 | 1.2e-07 |
| muon_768x768 | 10.58 -> 6.207 | 1.70x faster | 0.44x (fail) | 0.65x [0.62-0.70] (fail) | slower 5/5 | 4.5e-07 |
| muon_768x768 vs torch bf16 NS5 | 10.58 -> 6.207 | 1.70x faster | 0.27x (fail) | 0.35x [0.32-0.40] (fail) | slower 5/5 | timing only |
| muon_2048x768 | 22.32 -> 13.83 | 1.61x faster | 0.41x (fail) | 0.60x [0.55-0.63] (fail) | slower 5/5 | 5.0e-07 |
| muon_2048x768 vs torch bf16 NS5 | 22.32 -> 13.83 | 1.61x faster | 0.25x (fail) | 0.33x [0.32-0.37] (fail) | slower 5/5 | timing only |
| muon_768x2048 | 21.64 -> 12.77 | 1.70x faster | 0.42x (fail) | 0.64x [0.64-0.77] (fail) | slower 5/5 | 2.9e-07 |
| muon_768x2048 vs torch bf16 NS5 | 21.64 -> 12.77 | 1.70x faster | 0.25x (fail) | 0.33x [0.30-0.40] (fail) | slower 5/5 | timing only |
| block_fwd | 120.0 -> 43.00 | **2.79x faster** | 0.14x (fail) | 0.35x [0.32-0.36] (fail) | slower 5/5 | 1.9e-07 |
| block_fwd_bwd | 475.3 -> 158.1 | **3.01x faster** | 0.18x (pass) | 0.47x [0.43-0.48] (**pass**) | slower 5/5 | 2.3e-06 |
| decode_attn_kv1024 | new -> 0.414 | new row | - | 0.68x [0.48-3.31] (fail) | mixed (4/5 slower) | 2.5e-07 |
| accumulate_grad_50304x768 | new -> 2.412 | new row | - | 0.91x [0.86-1.77] (fail) | mixed (3/5 slower) | 0.0e+00 |

### What the optimization rounds moved (changes of 2x or more)

All of these are unpaired across runs, so the direction is verified and the
magnitude is approximate. The ratio column is paired within round 2.

- **Metal `rms_qk_norm_bwd`: 53.2 -> 2.6 ms, about 20x.** This is the
  two-stage weight-gradient reduction (`ojas_rms_bwd_w_part` /
  `ojas_rms_bwd_w_sum`, `ojas-metal/kernels/ojas_backend.metal:284,308`)
  replacing the 64-thread column loop. It is now faster than torch in 5/5
  rounds.
- **Metal `cross_entropy_fwd` 5.4x and `cross_entropy_bwd` 2.5x faster**
  (fused two-pass kernel). CE forward is now 3.9x ahead of torch (5/5
  rounds). CE backward still trails (0.72x).
- **Metal `adamw_full` 2.45x faster**, from 598 to 245 ms. It still trails
  torch, which takes 40 ms (0.16x, 5/5 rounds slower).
- **Metal `rms_norm_bwd` 2.0x and `rms_qk_norm_fwd` 2.2x faster.**
  `rms_norm_bwd` flipped to faster than torch (1.75x, 5/5).
- **Metal composed block, forward + backward: 230 -> 122 ms.** The ratio
  went from 0.37x to 0.59x (gate passed).
- **wgpu SDPA 4.9–7.7x faster at every shape** (tiled FlashAttention-2).
  It still trails torch at 0.10–0.19x.
- **wgpu composed block 2.8x (forward) and 3.0x (forward + backward)
  faster.** Forward + backward is 0.47x torch (gate passed).
- **wgpu GEMM 1.3–1.6x faster** (128x128 register-blocked tile, used when
  both output sides are at least 128). This is not a 2x change; the
  direction holds across all eight linear rows.
- **Moved the wrong way, under 2x:**
  - Metal `rope_bwd` (0.56x), `permute` (0.48x) and `residual_add_fwd`
    (0.71x). Their minimums barely moved (rope_bwd 0.44 -> 0.68 ms, permute
    0.39 -> 0.42 ms), so the median shift is mostly fixed-cost jitter under
    CPU load (inferred).
  - wgpu `floor` (0.68x) and `gate_fwd` (0.79x). These are small-op rows
    where 0.1–0.3 ms medians move with host load (inferred).

### Worst remaining ojas-vs-torch rows (slower in 5/5 rounds), ranked

| rank | backend | row | ratio, r2 median [min-max] | gate | ojas / torch median ms | bottleneck read |
| --: | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | Metal, wgpu | residual_add_bwd | 0.01x | fail | 2.29 / 0.016; 1.06 / 0.016 | API artifact, not a kernel: two fresh copies of `gy`, while torch returns `gy` with no kernel. A composed backward does not call it (`bench/ojas_rows.rs::block_backward`). **Verified** from the row semantics |
| 2 | wgpu | sdpa, all 6 rows | 0.10–0.19x | fail | e.g. 11.83 / 1.47 fwd, 53.3 / 10.0 bwd at the nanolab shape | The new kernel is FlashAttention-2 "in plain WGSL (no matrix units, no subgroups)" (`ojas-kernels/src/wgsl/attention.wgsl:1-2`): scalar FMAs over shared tiles, row max and sum through shared memory. Metal's matrix-unit version of the same op is 4.9x faster forward and 8.2x faster backward at the nanolab shape (2.42 vs 11.83 ms; 6.47 vs 53.3 ms). Structure **verified**; that matrix units are the gap is **inferred**. wgpu 30 exposes no cooperative-matrix path this kernel uses (**unverified**) |
| 3 | Metal | adamw_full | 0.16x [0.14-0.17] | fail | 244.5 / 39.6 | One command per tensor now (`ojas_adamw_check` then `ojas_adamw_apply`, `ojas-metal/src/device.rs:1749-1762`): two full passes over p, g, m and v, then a device wait and a status read. That is 170 times, about 1.4 ms per call, including 73 tensors of at most 768 values (counted from `param_shapes`) that pay the same per-call floor. torch does one `step()`. The traffic is about 5.4 GB for ojas against about 3.5 GB for torch (counted from passes × 495 MB). The rest is the per-call wait and host round trip. Pass structure **verified**; the split is **inferred** |
| 4 | Metal, wgpu | rms_qk_norm_fwd | 0.21x / 0.20x | fail | 2.14 / 0.42; 2.10 / 0.42 | Metal is now one command (`rms_pair`, `ojas-metal/src/backend.rs:814-828`) with one simdgroup per row. Its min is 0.58 ms against a 2.1 ms median, so the per-op floor under CPU load dominates (**inferred**). wgpu still issues two `rms_norm_forward` calls (`ojas-wgpu/src/backend.rs:1380-1391`), each with one 256-lane workgroup per 64-wide row (`norm.wgsl:1`), so 3 of 4 lanes idle. **Verified** from source |
| 5 | Metal | small elementwise and norm rows: rms_norm_fwd 0.22x, residual_add_fwd 0.20x, silu 0.26x / 0.37x, mul 0.37x / 0.44x, permute 0.31x, vres_fwd 0.41x, gate 0.46–0.48x | 0.20–0.48x | fail | 1–3 ms against 0.3–1.4 | The per-op fixed cost: status buffer, separate finite-check passes, a device wait and a host status read per op (round 1, cause 4; `device.rs:328-393`). The floor row is 0.16 ms at best and 1.14 ms median, against 0.11 / 0.20 for torch. Unchanged by this round. **Inferred** from minimum vs median, with the structure **verified** in round 1 |
| 6 | Metal | accumulate_grad_50304x768 (new) | 0.25x [0.21-0.53] | fail | 9.62 / 2.38 | `ojas_acc_check` then `ojas_acc_apply` (`device.rs:1718-1747`): acc and g are read twice and acc written once, against torch's one read of each and one write. Then a device wait. 9.6 ms for about 0.8 GB of traffic is well below the bandwidth wgpu reaches on the same op (2.41 ms), so more than the extra pass is involved (**unverified**; not profiled). wgpu is 0.91x, mixed |
| 7 | Metal, wgpu | block_fwd | 0.32x / 0.35x | fail | 49.2 / 15.3; 43.0 / 15.3 | Sum of parts. Metal: 24 synchronous ops, each paying the floor. wgpu: SDPA forward (11.8 ms) plus GEMMs at about 0.47x. **Inferred** from row sums |
| 8 | wgpu | linear, all eight rows; muon | linear 0.40–0.55x (lm_head fwd 0.40x and up_bwd 0.49x pass the gate); muon 0.60–0.65x against fp32 NS5 | mixed | lm_head fwd 136.4 / 54.8 | The 128x128 tile is still scalar vec4 FMAs from workgroup memory (`gemm.wgsl:140-146`); there are no matrix units. lm_head forward is 316 GFLOP / 136.4 ms = 2.3 TFLOP/s against torch's 5.8. **Verified** structure; cause **inferred** |
| 9 | Metal, wgpu | linear_ce (new) | Metal 0.88x (1024x8192) / 0.55x (4096x50304); wgpu 0.45x / 0.58x | fail | Metal 266 / 221 and 423 / 222 | Metal recomputes each logit chunk's GEMM in a second pass. Pass 0 is the stats, pass 1 the gradient (`device.rs:1563-1615`). So it does two forward GEMMs plus two backward GEMMs; nanolab's torch version does one forward plus two backward, and its GEMMs are faster (Metal lm_head fwd is 0.46x). **Verified** from source; that it explains the gap is **inferred**. Why Metal is slower with the single 4096x50304 chunk than with 1024x8192 is **unverified** (not profiled) |
| 10 | Metal, wgpu | cross_entropy_bwd | 0.72x / 0.60x | fail | 38.4 / 28.9; 47.7 / 28.9 | The backward recomputes softmax from the logits and writes the 824 MB gradient; torch reuses its saved `log_softmax`. **Inferred** |
| 11 | Metal | decode_attn_kv1024 (new) | 0.37x [0.22-1.16] | fail, mixed 4/5 | 0.83 / 0.21 | A 0.2 ms minimum against a 0.83 ms median: a single small op dominated by the per-op floor. torch is near its own floor (0.11 / 0.21). **Inferred** |
| 12 | Metal, wgpu | muon vs torch bf16 NS5 | 0.60–0.68x / 0.33–0.35x | Metal 2048x768 passes | - | A dtype difference, not like-for-like: ojas NS5 is f32. Against torch's fp32 NS5, Metal is now 1.20–1.23x **faster** (5/5 rounds; `muon_2048x768` passes the gate) |

**Where ojas leads torch in round 2 (5/5 rounds):**
- Metal:
  - `clip_grad_norm` 7.7x;
  - `cross_entropy_fwd` 3.9x;
  - SDPA backward 1.5x and 1.7x;
  - `rms_norm_bwd` 1.75x;
  - `rms_qk_norm_bwd` 1.8x;
  - Muon against fp32 NS5 1.2x.
- wgpu:
  - `clip_grad_norm` 7.8x;
  - rope 3.7x and 4.9x;
  - `rms_norm_bwd` 3.1x;
  - `cross_entropy_fwd` 2.2x;
  - `vres_bwd` 1.4x;
  - `rms_qk_norm_bwd` 1.1x.

All of these rows are gate-failed: direction verified, magnitude as ranged.

## Round 1: 2026-10-01, 16:34–17:23 CDT (kept for comparison)

> [!IMPORTANT]
> **The GPU was shared with other work for most of these runs.** `ioreg`
> "Device Utilization %" was 84–100% before nearly every runtime block of runs
> A and C. At 16:48 `ps` showed an ollama `llama-server` at 143% CPU and
> several Chrome renderer processes at about 100% CPU each. That observation
> is from one moment, not the whole run. The protocol's 10% spread rule
> therefore flags almost every row as "noisy - not quoted": 5 of 88 rows pass
> in run A, 0 in run C. This document does not quote a speedup for a
> flagged row as a precise number.
>
> Read the evidence in two tiers:
> - **Direction, verified.** The verdict holds where every per-round ratio
>   in all three runs (15 rounds, each pairing the two runtimes minutes
>   apart) lies on one side of 1. Pairing cancels contention that hits both
>   sides alike, so many flagged rows still have tight ratio ranges. One
>   example is wgpu linear at 0.27–0.37x across 15 rounds.
> - **Magnitude, unverified beyond the stated range.** The median ratio
>   quoted is run A's. The 15-round min-max beside it is the honest
>   uncertainty.
>
> Metal is likely hurt more by contention than torch or wgpu, because every
> Metal op waits on the device. The evidence:
> - Its per-op floor has a 0.13–0.19 ms minimum but a 1.5–1.8 ms median.
> - One unpaired smoke run after the three runs measured Metal
>   `block_fwd_bwd` at 131 ms median, against run A's 230 ms. That run is not
>   quoted, and the GPU load at the time was not sampled.
>
> Treat Metal's contended ratios as pessimistic (inferred).

## Toolchain and source state (verified, `bench/results/2026-10-01/run*/env.txt`)

| item | value |
| :-- | :-- |
| machine | Apple M5 Pro, 64 GiB, macOS 27.0.1 (26A434) |
| rustc / cargo | 1.98.0 (88d9e12ae 2026-08-18) / 1.98.0 (797e8a9bc 2026-08-05) |
| Python / torch | 3.14.7 (`/opt/homebrew/opt/python@3.14/bin/python3.14`) / 2.13.0, MPS |
| ojas | `dab2a12` **plus an uncommitted working tree**. The diff of ojas-core, ojas-metal, ojas-wgpu, ojas-kernels and ojas-device hashes to `c226ffba076d` in all three runs; other sessions were editing this checkout |
| tessl (ojas-metal's GEMMs) | `cf65d9d` plus 20 dirty files |
| wgpu | 30.0.1. The adapter is "Apple M5 Pro" with HAL **Metal**, recorded by the binary's `_device` line. So on this machine the portable backend runs through wgpu's Metal HAL. The binding cap is 4,294,967,292 bytes, so the 824 MB logits fit |
| build | release, `CARGO_TARGET_DIR=target-lane-bench`. The runner builds before every run; the binary mtimes are in env.txt |
| nanolab | `/Users/bharath/Code/research/MLSystemsLab/nanolab`, real `Block`, `GPT(Config())` 123,699,612 parameters in 170 tensors. torch_rows.py asserts the optimizer rows' shape list against it |

## Runs

| run | rounds x iters (warmup 5) | GPU before blocks | flagged rows (of 88) |
| :-- | :-- | :-- | --: |
| A, `run5` (primary: fewest flagged) | 5 x 20 | 93–100%, except 70% before round-1 wgpu | 83 |
| B, `run5b` | 5 x 30 | 84–95% in rounds 1–2, 0–50% in rounds 3–5 | 88 |
| C, `run5c` | 5 x 30 | 0–57% in round 1, 85–94% after | 88 |

The primary-run rule was fixed before run C finished: take the run with the
fewest flagged rows. The full before/after load tables are at the end of each
`summary.md`. `ioreg` is an instantaneous sample. In run B round 1 it went
from 0 to 94 within 36 s, so it bounds the contention but does not average it.

## Parity (verified)

Every row on both backends passed its gate, max |ojas - torch| / max |torch|
per output (1e-3; 1e-2 for Muon; 0 for permute), in every round of all three
runs. No row was refused. Both binaries' generator checks passed. The worst
normalized error was 1.3e-5, on `linear_lmhead_bwd`. The largest absolute
error was 1.3e-3, on Metal `rms_qk_norm_bwd`'s weight gradient (a sum over
49,152 rows; 4.0e-6 normalized).

Every linear **forward** matched torch MPS **bit for bit** (error 0), at all
four shapes, on both backends. The comparison is not vacuous. The same
harness reports 1.57e-4 on `linear_lmhead_bwd` (the TN `grad_w` product) and
nonzero errors on 30 other rows. That the GEMMs accumulate in the same
k-ascending FMA order is inferred, not checked.

The block rows compare the output and seven gradients (x, q_proj, ffn.down,
gate.weight, vr_lambda, norm1, v0) against nanolab's real `Block` plus torch
autograd. They pass at 2.5e-6 (Metal) and 2.3e-6 (wgpu) normalized. That is
the check on the hand-composed backward.

## Results: ojas MetalBackend vs torch MPS

ratio = torch median / ojas median, paired per round; > 1 means ojas is
faster. Times are run A, min over rounds / median of per-round medians, in ms.

| row | ojas min / median ms | torch min / median ms | ratio, run A median [min-max] | spread ojas / torch | run A flag | parity max abs (rel) | direction over 3 runs |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.191 / 1.756 | 0.082 / 0.107 | 0.06x [0.05-0.07] | 4% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_fwd | 1.153 / 1.999 | 0.802 / 0.918 | 0.46x [0.28-0.48] | 88% / 17% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_bwd | 2.499 / 3.785 | 1.720 / 2.112 | 0.53x [0.47-0.73] | 54% / 12% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_fwd | 2.709 / 4.127 | 1.943 / 2.152 | 0.51x [0.42-0.59] | 42% / 5% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_bwd | 4.899 / 5.630 | 4.176 / 4.497 | 0.78x [0.55-0.83] | 59% / 7% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| linear_down_fwd | 2.600 / 3.609 | 2.013 / 2.201 | 0.61x [0.48-0.65] | 44% / 6% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_down_bwd | 4.592 / 6.541 | 4.008 / 4.458 | 0.66x [0.42-0.73] | 76% / 8% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_fwd | 109.4 / 131.9 | 54.41 / 60.84 | 0.52x [0.42-0.54] | 20% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_bwd | 131.0 / 174.3 | 105.5 / 120.4 | 0.70x [0.57-0.82] | 31% / 39% | noisy | 1.57e-04 (1.3e-05) | slower, 15/15 |
| sdpa_b4h12t1024d64_fwd | 2.045 / 3.081 | 1.367 / 1.600 | 0.55x [0.39-0.61] | 68% / 19% | noisy | 2.24e-07 (2.2e-07) | mixed (14/15 slower) |
| sdpa_b4h12t1024d64_bwd | 5.830 / 8.111 | 9.474 / 10.30 | 1.24x [1.14-1.37] | 29% / 13% | noisy | 1.10e-06 (2.1e-06) | **faster, 15/15** |
| sdpa_b4h8t2048d64_fwd | 4.775 / 6.909 | 3.225 / 3.542 | 0.50x [0.47-0.65] | 23% / 16% | noisy | 1.79e-07 (1.8e-07) | slower, 15/15 |
| sdpa_b4h8t2048d64_bwd | 17.30 / 19.41 | 24.21 / 25.39 | 1.38x [1.01-1.59] | 35% / 25% | noisy | 1.31e-06 (2.5e-06) | **faster, 15/15** |
| sdpa_b2h8t1024d128_fwd | 1.333 / 2.029 | 1.074 / 1.202 | 0.61x [0.37-0.68] | 96% / 17% | noisy | 2.09e-07 (2.1e-07) | mixed (14/15 slower) |
| sdpa_b2h8t1024d128_bwd | 5.210 / 6.417 | 4.395 / 5.318 | 0.81x [0.77-0.92] | 22% / 40% | noisy | 1.10e-06 (2.6e-06) | mixed (14/15 slower) |
| rms_norm_fwd | 0.478 / 1.654 | 0.183 / 0.258 | 0.18x [0.09-0.70] | 142% / 226% | noisy | 3.58e-07 (2.0e-07) | slower, 15/15 |
| rms_norm_bwd | 1.939 / 3.389 | 2.037 / 2.515 | 0.74x [0.53-0.84] | 48% / 49% | noisy | 2.14e-04 (1.5e-06) | mixed (13/15 slower) |
| rms_qk_norm_fwd | 2.001 / 4.736 | 0.304 / 0.534 | 0.11x [0.08-0.19] | 49% / 63% | noisy | 3.58e-07 (1.7e-07) | slower, 15/15 |
| rms_qk_norm_bwd | 50.48 / 53.15 | 4.369 / 5.276 | 0.10x [0.09-0.10] | 2% / 8% | **quotable** | 1.30e-03 (4.0e-06) | slower, 15/15 |
| rope_fwd | 0.586 / 1.816 | 1.220 / 1.513 | 0.85x [0.68-1.51] | 124% / 19% | noisy | 0.00e+00 (0.0e+00) | mixed (9/15 slower) |
| rope_bwd | 0.440 / 1.182 | 1.722 / 2.081 | 1.81x [0.87-2.38] | 154% / 26% | noisy | 0.00e+00 (0.0e+00) | mixed (4/15 slower) |
| silu_fwd | 0.927 / 2.267 | 0.361 / 0.466 | 0.20x [0.13-0.28] | 96% / 51% | noisy | 2.38e-07 (6.1e-08) | slower, 15/15 |
| silu_bwd | 1.533 / 3.191 | 0.506 / 0.691 | 0.23x [0.18-0.30] | 38% / 62% | noisy | 1.19e-07 (1.1e-07) | slower, 15/15 |
| mul_fwd | 1.291 / 2.899 | 0.496 / 0.615 | 0.21x [0.21-0.24] | 26% / 31% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| mul_bwd | 2.328 / 4.131 | 0.866 / 1.387 | 0.35x [0.26-0.38] | 26% / 49% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| residual_add_fwd | 0.641 / 1.559 | 0.223 / 0.345 | 0.18x [0.13-0.25] | 78% / 93% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| residual_add_bwd | 0.593 / 1.878 | 0.012 / 0.016 | 0.01x [0.01-0.01] | 75% / 135% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| gate_fwd | 0.646 / 1.773 | 0.476 / 0.681 | 0.36x [0.26-0.98] | 75% / 124% | noisy | 8.94e-08 (1.0e-07) | slower, 15/15 |
| gate_bwd | 2.928 / 4.069 | 0.957 / 1.350 | 0.33x [0.26-0.64] | 32% / 104% | noisy | 2.25e-04 (2.3e-06) | slower, 15/15 |
| vres_fwd | 0.590 / 1.645 | 0.481 / 0.650 | 0.41x [0.23-0.57] | 121% / 23% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| vres_bwd | 0.946 / 1.943 | 0.840 / 1.192 | 0.64x [0.32-0.79] | 89% / 38% | noisy | 3.05e-05 (2.3e-07) | slower, 15/15 |
| permute_bthd_bhtd | 0.391 / 0.771 | 0.407 / 0.547 | 0.72x [0.23-1.27] | 214% / 79% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| cross_entropy_fwd | 13.31 / 20.94 | 14.75 / 19.60 | 0.97x [0.76-1.06] | 13% / 35% | noisy | 0.00e+00 (0.0e+00) | mixed (7/15 slower) |
| cross_entropy_bwd | 78.49 / 96.68 | 28.31 / 34.30 | 0.35x [0.33-0.38] | 7% / 14% | noisy | 6.39e-14 (2.6e-10) | slower, 15/15 |
| clip_grad_norm_full | 16.71 / 33.67 | 137.4 / 154.2 | 4.58x [4.46-4.75] | 7% / 1% | **quotable** | 3.05e-05 (3.0e-07) | **faster, 15/15** |
| adamw_full | 515.8 / 598.1 | 44.33 / 53.52 | 0.09x [0.08-0.10] | 7% / 11% | noisy | 3.73e-09 (1.2e-07) | slower, 15/15 |
| muon_768x768 (torch fp32 NS5) | 4.565 / 6.507 | 4.005 / 4.759 | 0.71x [0.62-0.76] | 34% / 19% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| muon_768x768 vs torch bf16 NS5 | 4.565 / 6.507 | 2.236 / 2.943 | 0.44x [0.41-0.47] | 34% / 27% | noisy | (timing only) | slower, 15/15 |
| muon_2048x768 (torch fp32 NS5) | 9.782 / 12.01 | 8.145 / 9.158 | 0.75x [0.67-0.81] | 24% / 12% | noisy | 1.68e-08 (5.0e-07) | mixed (13/15 slower) |
| muon_2048x768 vs torch bf16 NS5 | 9.782 / 12.01 | 4.831 / 5.721 | 0.45x [0.43-0.48] | 24% / 19% | noisy | (timing only) | slower, 15/15 |
| muon_768x2048 (torch fp32 NS5) | 9.908 / 11.73 | 8.615 / 9.080 | 0.77x [0.40-0.80] | 114% / 11% | noisy | 9.31e-09 (2.9e-07) | mixed (13/15 slower) |
| muon_768x2048 vs torch bf16 NS5 | 9.908 / 11.73 | 4.598 / 5.343 | 0.45x [0.23-0.49] | 114% / 10% | noisy | (timing only) | slower, 15/15 |
| block_fwd | 48.08 / 67.25 | 15.74 / 16.92 | 0.24x [0.24-0.27] | 24% / 9% | noisy | 2.38e-07 (1.9e-07) | slower, 15/15 |
| block_fwd_bwd | 179.1 / 230.5 | 77.08 / 83.64 | 0.37x [0.34-0.38] | 11% / 7% | noisy | 9.06e-06 (2.5e-06) | slower, 15/15 |

## Results: ojas WgpuBackend (Metal HAL) vs torch MPS

| row | ojas min / median ms | torch min / median ms | ratio, run A median [min-max] | spread ojas / torch | run A flag | parity max abs (rel) | direction over 3 runs |
| :-- | --: | --: | :-- | :-- | :-- | :-- | :-- |
| floor_silu_1 | 0.170 / 0.244 | 0.082 / 0.107 | 0.48x [0.21-0.50] | 147% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_fwd | 2.370 / 2.787 | 0.802 / 0.918 | 0.31x [0.30-0.33] | 14% / 17% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_qkv_bwd | 5.072 / 5.620 | 1.720 / 2.112 | 0.38x [0.32-0.40] | 14% / 12% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_fwd | 6.080 / 6.697 | 1.943 / 2.152 | 0.32x [0.27-0.35] | 21% / 5% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_up_bwd | 12.97 / 14.04 | 4.176 / 4.497 | 0.31x [0.29-0.34] | 15% / 7% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_down_fwd | 6.693 / 7.456 | 2.013 / 2.201 | 0.29x [0.24-0.31] | 31% / 6% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_down_bwd | 13.07 / 13.84 | 4.008 / 4.458 | 0.32x [0.28-0.34] | 13% / 8% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_fwd | 186.1 / 194.7 | 54.41 / 60.84 | 0.31x [0.28-0.36] | 5% / 26% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| linear_lmhead_bwd | 307.1 / 323.7 | 105.5 / 120.4 | 0.37x [0.34-0.43] | 10% / 39% | noisy | 1.57e-04 (1.3e-05) | slower, 15/15 |
| sdpa_b4h12t1024d64_fwd | 72.67 / 77.05 | 1.367 / 1.600 | 0.02x [0.02-0.02] | 12% / 19% | noisy | 2.24e-07 (2.2e-07) | slower, 15/15 |
| sdpa_b4h12t1024d64_bwd | 270.8 / 277.8 | 9.474 / 10.30 | 0.04x [0.04-0.04] | 3% / 13% | noisy | 1.25e-06 (2.4e-06) | slower, 15/15 |
| sdpa_b4h8t2048d64_fwd | 183.2 / 191.6 | 3.225 / 3.542 | 0.02x [0.02-0.02] | 6% / 16% | noisy | 1.79e-07 (1.8e-07) | slower, 15/15 |
| sdpa_b4h8t2048d64_bwd | 694.0 / 705.7 | 24.21 / 25.39 | 0.04x [0.04-0.04] | 7% / 25% | noisy | 1.37e-06 (2.6e-06) | slower, 15/15 |
| sdpa_b2h8t1024d128_fwd | 68.78 / 91.63 | 1.074 / 1.202 | 0.01x [0.01-0.01] | 6% / 17% | noisy | 2.09e-07 (2.1e-07) | slower, 15/15 |
| sdpa_b2h8t1024d128_bwd | 267.7 / 280.7 | 4.395 / 5.318 | 0.02x [0.02-0.02] | 7% / 40% | noisy | 1.07e-06 (2.3e-06) | slower, 15/15 |
| rms_norm_fwd | 0.304 / 0.424 | 0.183 / 0.258 | 0.61x [0.46-1.07] | 321% / 226% | noisy | 2.38e-07 (1.3e-07) | mixed (13/15 slower) |
| rms_norm_bwd | 0.584 / 0.896 | 2.037 / 2.515 | 3.42x [1.67-3.91] | 132% / 49% | noisy | 3.34e-05 (2.6e-07) | **faster, 15/15** |
| rms_qk_norm_fwd | 1.948 / 2.624 | 0.304 / 0.534 | 0.17x [0.14-0.22] | 58% / 63% | noisy | 2.38e-07 (1.2e-07) | slower, 15/15 |
| rms_qk_norm_bwd | 4.744 / 5.421 | 4.369 / 5.276 | 0.95x [0.88-1.00] | 7% / 8% | quotable | 4.58e-04 (1.3e-06) | mixed (8/15 slower) |
| rope_fwd | 0.243 / 0.470 | 1.220 / 1.513 | 3.67x [3.15-4.45] | 47% / 19% | noisy | 1.19e-07 (6.2e-08) | **faster, 15/15** |
| rope_bwd | 0.250 / 0.423 | 1.722 / 2.081 | 4.73x [3.88-6.78] | 66% / 26% | noisy | 1.19e-07 (6.1e-08) | **faster, 15/15** |
| silu_fwd | 0.488 / 0.728 | 0.361 / 0.466 | 0.75x [0.55-0.79] | 45% / 51% | noisy | 7.15e-07 (1.8e-07) | mixed (14/15 slower) |
| silu_bwd | 0.691 / 1.004 | 0.506 / 0.691 | 0.86x [0.47-1.00] | 60% / 62% | noisy | 4.77e-07 (4.3e-07) | mixed (13/15 slower) |
| mul_fwd | 0.598 / 0.862 | 0.496 / 0.615 | 0.74x [0.56-0.78] | 43% / 31% | noisy | 0.00e+00 (0.0e+00) | mixed (14/15 slower) |
| mul_bwd | 0.906 / 1.263 | 0.866 / 1.387 | 1.06x [0.65-1.16] | 34% / 49% | noisy | 0.00e+00 (0.0e+00) | mixed (8/15 slower) |
| residual_add_fwd | 0.328 / 0.513 | 0.223 / 0.345 | 0.72x [0.43-0.83] | 76% / 93% | noisy | 0.00e+00 (0.0e+00) | mixed (13/15 slower) |
| residual_add_bwd | 0.628 / 0.898 | 0.012 / 0.016 | 0.02x [0.01-0.03] | 26% / 135% | noisy | 0.00e+00 (0.0e+00) | slower, 15/15 |
| gate_fwd | 0.635 / 0.943 | 0.476 / 0.681 | 0.74x [0.67-1.62] | 35% / 124% | noisy | 1.19e-07 (1.4e-07) | mixed (13/15 slower) |
| gate_bwd | 1.813 / 2.111 | 0.957 / 1.350 | 0.63x [0.61-1.21] | 6% / 104% | noisy | 2.21e-04 (2.3e-06) | mixed (14/15 slower) |
| vres_fwd | 0.358 / 0.458 | 0.481 / 0.650 | 1.32x [1.14-1.48] | 17% / 23% | noisy | 5.96e-08 (6.0e-08) | **faster, 15/15** |
| vres_bwd | 0.573 / 0.720 | 0.840 / 1.192 | 1.67x [1.56-2.17] | 20% / 38% | noisy | 3.05e-05 (2.3e-07) | **faster, 15/15** |
| permute_bthd_bhtd | 0.411 / 0.628 | 0.407 / 0.547 | 0.89x [0.60-1.53] | 61% / 79% | noisy | 0.00e+00 (0.0e+00) | mixed (11/15 slower) |
| cross_entropy_fwd | 6.317 / 7.139 | 14.75 / 19.60 | 2.79x [2.14-2.99] | 18% / 35% | noisy | 0.00e+00 (0.0e+00) | **faster, 15/15** |
| cross_entropy_bwd | 57.75 / 63.71 | 28.31 / 34.30 | 0.53x [0.50-0.54] | 12% / 14% | noisy | 6.39e-14 (2.6e-10) | slower, 15/15 |
| clip_grad_norm_full | 21.34 / 24.18 | 137.4 / 154.2 | 6.36x [6.14-6.38] | 5% / 1% | **quotable** | 3.81e-05 (3.8e-07) | **faster, 15/15** |
| adamw_full | 66.35 / 75.15 | 44.33 / 53.52 | 0.70x [0.69-0.75] | 13% / 11% | noisy | 3.73e-09 (1.2e-07) | mixed (14/15 slower) |
| muon_768x768 (torch fp32 NS5) | 9.453 / 10.58 | 4.005 / 4.759 | 0.44x [0.42-0.47] | 22% / 19% | noisy | 1.49e-08 (4.5e-07) | slower, 15/15 |
| muon_768x768 vs torch bf16 NS5 | 9.453 / 10.58 | 2.236 / 2.943 | 0.27x [0.27-0.28] | 22% / 27% | noisy | (timing only) | slower, 15/15 |
| muon_2048x768 (torch fp32 NS5) | 20.14 / 22.32 | 8.145 / 9.158 | 0.41x [0.41-0.44] | 8% / 12% | noisy | 1.68e-08 (5.0e-07) | slower, 15/15 |
| muon_2048x768 vs torch bf16 NS5 | 20.14 / 22.32 | 4.831 / 5.721 | 0.25x [0.24-0.28] | 8% / 19% | noisy | (timing only) | slower, 15/15 |
| muon_768x2048 (torch fp32 NS5) | 19.77 / 21.64 | 8.615 / 9.080 | 0.42x [0.41-0.46] | 14% / 11% | noisy | 9.31e-09 (2.9e-07) | slower, 15/15 |
| muon_768x2048 vs torch bf16 NS5 | 19.77 / 21.64 | 4.598 / 5.343 | 0.25x [0.24-0.26] | 14% / 10% | noisy | (timing only) | slower, 15/15 |
| block_fwd | 115.3 / 120.0 | 15.74 / 16.92 | 0.14x [0.13-0.14] | 16% / 9% | noisy | 2.38e-07 (1.9e-07) | slower, 15/15 |
| block_fwd_bwd | 465.8 / 475.3 | 77.08 / 83.64 | 0.18x [0.17-0.18] | 3% / 7% | **quotable** | 8.58e-06 (2.3e-06) | slower, 15/15 |

Quotable rows (both spreads at most 10% in run A), verified:
- Metal `clip_grad_norm_full`: 4.58x faster than torch.
- wgpu `clip_grad_norm_full`: 6.36x faster.
- Metal `rms_qk_norm_bwd`: 0.10x, about 10 times slower.
- wgpu `block_fwd_bwd`: 0.18x, about 5.6 times slower.
- wgpu `rms_qk_norm_bwd`: 0.95x. Within noise of parity; its direction is mixed across runs.

## Where ojas is slower than torch (plainly)

Rows where ojas was slower in **all 15 rounds** (verified direction), ranked
by run A's median ratio. The `floor` row is excluded, since it measures the
per-op fixed cost and is not work. Ranking by each row's best round out of
15 instead (the round most favorable to ojas) gives the same top 10, so the
order is not an artifact of contention.

| rank | backend | row | run A median ratio [min-max] | best of 15 rounds | ojas / torch median ms |
| --: | :-- | :-- | :-- | --: | :-- |
| 1 | Metal | residual_add_bwd | 0.01x [0.01-0.01] | 0.04x | 1.878 / 0.016 |
| 2 | wgpu | sdpa_b2h8t1024d128_fwd | 0.01x [0.01-0.01] | 0.03x | 91.63 / 1.202 |
| 3 | wgpu | residual_add_bwd | 0.02x [0.01-0.03] | 0.09x | 0.898 / 0.016 |
| 4 | wgpu | sdpa_b4h8t2048d64_fwd | 0.02x [0.02-0.02] | 0.02x | 191.6 / 3.542 |
| 5 | wgpu | sdpa_b2h8t1024d128_bwd | 0.02x [0.02-0.02] | 0.02x | 280.7 / 5.318 |
| 6 | wgpu | sdpa_b4h12t1024d64_fwd | 0.02x [0.02-0.02] | 0.06x | 77.05 / 1.600 |
| 7 | wgpu | sdpa_b4h8t2048d64_bwd | 0.04x [0.04-0.04] | 0.04x | 705.7 / 25.39 |
| 8 | wgpu | sdpa_b4h12t1024d64_bwd | 0.04x [0.04-0.04] | 0.04x | 277.8 / 10.30 |
| 9 | Metal | adamw_full | 0.09x [0.08-0.10] | 0.11x | 598.1 / 53.52 |
| 10 | Metal | rms_qk_norm_bwd | 0.10x [0.09-0.10] | 0.15x | 53.15 / 5.276 |
| 11 | Metal | rms_qk_norm_fwd | 0.11x [0.08-0.19] | 0.33x | 4.736 / 0.534 |
| 12 | wgpu | block_fwd | 0.14x [0.13-0.14] | 0.18x | 120.0 / 16.92 |
| 13 | wgpu | rms_qk_norm_fwd | 0.17x [0.14-0.22] | 0.58x | 2.624 / 0.534 |
| 14 | wgpu | block_fwd_bwd | 0.18x [0.17-0.18] | 0.21x | 475.3 / 83.64 |
| 15 | Metal | rms_norm_fwd | 0.18x [0.09-0.70] | 0.70x | 1.654 / 0.258 |
| 16 | Metal | silu_fwd | 0.20x [0.13-0.28] | 0.50x | 2.267 / 0.466 |
| 17 | Metal | mul_fwd | 0.21x [0.21-0.24] | 0.76x | 2.899 / 0.615 |
| 18 | Metal | silu_bwd | 0.23x [0.18-0.30] | 0.56x | 3.191 / 0.691 |
| 19 | Metal | block_fwd | 0.24x [0.24-0.27] | 0.37x | 67.25 / 16.92 |
| 20 | wgpu | muon vs torch bf16 NS5 (all three shapes) | 0.25x-0.27x | 0.31x-0.36x | see tables |
| 21 | wgpu | linear, all eight rows | 0.29x-0.38x | 0.34x-0.43x | see tables |
| 22 | Metal | gate_bwd / gate_fwd | 0.33x / 0.36x | 0.64x / 0.98x | 4.069 / 1.350, 1.773 / 0.681 |
| 23 | Metal | cross_entropy_bwd | 0.35x [0.33-0.38] | 0.56x | 96.68 / 34.30 |
| 24 | Metal | block_fwd_bwd | 0.37x [0.34-0.38] | 0.60x | 230.5 / 83.64 |
| 25 | wgpu | muon vs torch fp32 NS5 (all three shapes) | 0.41x-0.44x | 0.54x-0.68x | see tables |
| 26 | Metal | muon vs torch bf16 NS5 (all three shapes) | 0.44x-0.45x | 0.58x-0.76x | see tables |
| 27 | Metal | linear fwd (all four), qkv/down/lm_head bwd | 0.46x-0.70x | 0.55x-0.98x | see tables |
| 28 | Metal | sdpa_b4h8t2048d64_fwd | 0.50x [0.47-0.65] | 0.76x | 6.909 / 3.542 |
| 29 | wgpu | cross_entropy_bwd | 0.53x [0.50-0.54] | 0.87x | 63.71 / 34.30 |
| 30 | Metal | vres_bwd | 0.64x [0.32-0.79] | 0.94x | 1.943 / 1.192 |

**The composed block, the training-relevant number:**
- Metal: 0.37x forward + backward and 0.24x forward. That is about 2.7 and 4.2 times slower.
- wgpu: 0.18x forward + backward and 0.14x forward. That is about 5.6 and 7 times slower.
- Direction verified in 15/15 rounds for all four; magnitudes as ranged above.

### The worst five, with likely causes

`residual_add_bwd` (ranks 1 and 3) is an **API-shape artifact, not a kernel
problem**. ojas's `residual_add_backward` returns two fresh copies of the
incoming gradient. torch autograd returns the same tensor with no kernel;
its 0.016 ms is the timer floor. A composed backward passes the gradient
through: `bench/ojas_rows.rs::block_backward` never calls it. It costs
training nothing unless a caller (or `Tape`) calls it. This is verified from
the row semantics; whether `Tape` calls it was not checked.

1. **wgpu causal SDPA, 0.01–0.04x (verified direction; cause inferred from
   source).** `ojas-kernels/src/wgsl/attention.wgsl:50-100` gives each lane
   one query row of a 64-lane workgroup. Each lane holds private `q[D]` and
   `acc[D]` arrays (64 or 128 floats each, likely spilled), computes a scalar
   dot product per key, and does a branchy online-softmax rescale of `acc`
   per key. No matrix units are used. Metal's kernels for the same op are
   FlashAttention-2 on the TensorOps matrix units
   (`ojas-metal/kernels/ojas_backend.metal:701-719`). They run the same shapes
   25–45 times faster than wgpu (run A: 3.08 vs 77.05 ms forward, 8.11 vs
   277.8 ms backward at the nanolab shape). This one op is about 75% of wgpu's
   block forward + backward. Run A medians: SDPA forward + backward is
   355 ms of the block's 475 ms, inferred from row sums.

2. **Metal `adamw_full`, 0.09x (direction verified, cause verified from
   source).** `ojas-metal/src/device.rs:1404-1448`. Per tensor:
   - allocates three fresh buffers;
   - runs four `ojas_check_finite` passes over the inputs;
   - makes three copies out, runs tessl's AdamW step, and three more checks;
   - `finish` waits on the device and reads the status words back;
   - makes three copies back, then a second `synchronize`.

   That is two full device waits and about 13 full-tensor passes, times 170
   calls, about 3.5 ms per call. wgpu's AdamW (0.70x) decides
   finite-or-not on the device with no host wait per call (`ojas-wgpu/src/backend.rs` module docs).
   torch's is one `optimizer.step()`.

3. **Metal `rms_qk_norm` backward 0.10x and forward 0.11x (direction
   verified; cause verified from source).** `rms_qk_norm_backward` is two
   `rms_norm_backward` calls (`ojas-metal/src/backend.rs:716-729`). Each runs
   `ojas_rms_bwd_w` (`ojas_backend.metal:287-303`). That kernel uses one thread
   per column, so dim = **64 threads for the whole GPU**, and each loops over
   49,152 rows serially. The forward runs `ojas_rms_rstd` with one 256-lane
   threadgroup per 64-element row (`ojas_backend.metal:199-232`), so 3 of 4
   lanes are idle, plus a separate `ojas_rms_apply` pass. wgpu's QK-norm
   forward (0.17x) has the same one-256-lane-workgroup-per-row shape
   (`ojas-kernels/src/wgsl/norm.wgsl`). Its backward (0.95x) leaves the
   weight gradient out of the per-row kernel and reduces it with a two-stage
   column sum (`col_sum`, `ojas-wgpu/src/backend.rs:488`, called at 1269), not
   one thread per column. That is verified from source; that it explains
   wgpu's 9x lead over Metal on this row is inferred.

4. **Metal per-op fixed cost on small and elementwise ops, 0.18–0.36x
   (direction verified; cause verified from source, split inferred).** Rows
   affected: `rms_norm_fwd`, `silu`, `mul`, `residual_add_fwd`, `gate`,
   `vres_fwd`, `permute`. Every Metal op:
   - allocates and initialises a status buffer (`device.rs:328-334`);
   - runs a separate `ojas_check_finite` dispatch over **every input and
     every output** (`device.rs:337-345`, e.g. `linear` at 728-746);
   - ends with `finish`: a device wait plus a host read of the status words
     (`device.rs:370-393`);
   - crosses the channel to the device thread and back.

   The 1-element `floor_silu_1` row is 0.13–0.19 ms at best, but 1.5–1.8 ms
   median under this contention. wgpu's is 0.24–0.35 ms median, and torch's
   0.08–0.15. wgpu records into a shared encoder and checks finiteness inline
   (`report()` in the WGSL) at one sync per iteration. The Metal block runs 24
   ops forward and 53 forward + backward (counted from `bench/ojas_rows.rs`),
   each paying this. Run A's Metal block forward, 67 ms, is close to the
   49 ms sum of its kernel rows (inferred).

5. **wgpu GEMM, 0.29–0.38x on every linear row (direction verified; cause
   inferred).** `ojas-kernels/src/wgsl/gemm.wgsl:1-20` uses a 16x16 workgroup
   per 64x64 tile with scalar vec4 FMAs from workgroup memory and no
   simdgroup matrix. lm_head forward reaches 316 GFLOP / 194.7 ms = 1.6
   TFLOP/s, against torch's 5.2 TFLOP/s and Metal's 2.4 (tessl `ExactF32`).
   The same kernel family drives wgpu Muon (0.41–0.44x against torch's fp32
   NS5).

Next in line: Metal `cross_entropy_bwd` at 0.35x. `ojas_ce_rows`
(`ojas_backend.metal:605-659`) uses one 256-lane threadgroup per 50,304-wide
row, three passes over the row with `precise::exp` twice and
`precise::divide` twice per element. That comes on top of the finite checks
over the 824 MB input and the 824 MB output. Metal linear trails at
0.46–0.70x; part of that is the extra checks over the 824 MB lm_head output
and the inputs (inferred).

## Where ojas is faster (verified direction, 15/15 rounds)

| backend | row | run A median ratio [15-round range] | why (inferred) |
| :-- | :-- | :-- | :-- |
| Metal | clip_grad_norm_full | 4.58x [4.29-9.36] | one device reduction plus one scale. torch's `clip_grad_norm_` on MPS goes through `linalg.vector_norm` per tensor (pytorch-parity-plan.md section 3) |
| wgpu | clip_grad_norm_full | 6.36x [5.81-9.10] | same |
| Metal | sdpa backward, nanolab and T=2048 shapes | 1.24x and 1.38x [1.01-1.77] | FlashAttention-2 tiled backward; torch with grad runs `_scaled_dot_product_attention_math` |
| wgpu | rope fwd / bwd | 3.67x / 4.73x [2.78-6.78] | one kernel; torch's `apply_rope` is several kernels (negate, `cat`, two multiplies, an add) |
| wgpu | cross_entropy_fwd | 2.79x [1.93-3.21] | one fused pass; torch materializes `log_softmax` |
| wgpu | rms_norm_bwd | 3.42x [1.67-5.76] | one fused per-row kernel; torch's autograd backward of `F.rms_norm` is several kernels |
| wgpu | vres fwd / bwd | 1.32x / 1.67x [1.01-2.17] | fused; torch does sigmoid, two multiplies and an add |

Metal rope is mixed (its per-op fixed cost eats the fused-kernel advantage).
Metal SDPA backward at head dim 128 is mixed (14/15 rounds slower).

## Findings for the owners (no library code was changed here)

1. **`Tape` cannot drive the GPU block from these examples.** Neither
   `ojas-metal` nor `ojas-wgpu` depends on `ojas-autograd`, even as a
   dev-dependency, and this lane may not edit `Cargo.toml`. The block forward
   and backward are composed by hand from Backend ops. Parity against
   nanolab's real `Block` checks the composition, not `Tape`. A `Tape`-driven
   GPU row needs an `ojas-autograd` dev-dependency on one GPU crate, or an
   example in `ojas-autograd` with GPU dev-dependencies.
2. **`docs/bench-cpu-vs-torch.md:153` is stale.** It records Metal attention
   backward at T=2048 as 306 ms against torch's 60 ms. This harness measures
   `sdpa_b4h8t2048d64_bwd` at 16.5–19.4 ms (median across runs) against 25–26
   ms for torch, so ojas is faster in all 15 rounds. Its 3.86 ms vs 2.81 ms
   linear 2048³ line was not re-measured here. That doc is outside this
   lane's ownership.
3. **`docs/pytorch-parity-plan.md` section 3, target 2 ("a one-pass gradient
   norm, about 3 ms where torch takes 150") is partly met.** torch takes
   137–154 ms here. ojas takes 16–35 ms (Metal) and 16–25 ms (wgpu), which
   includes the scale pass, not 3 ms.
4. Fix candidates by measured cost:
   - wgpu attention on matrix units, or a tiled multi-row-per-lane design;
   - Metal AdamW without the copy-in/copy-out and double wait (or a batched
     multi-tensor step);
   - `ojas_rms_bwd_w` as a two-stage column reduction;
   - one finite check per op output instead of separate passes over every
     input and output on Metal;
   - a shared-encoder mode for Metal like wgpu's.

   None was attempted.

## Reproduce

```bash
bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 5                    # run A protocol
BENCH_ITERS=30 bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 5     # runs B and C
/opt/homebrew/opt/python@3.14/bin/python3.14 /Users/bharath/Code/research/ojas/bench/aggregate.py \
    bench/results/2026-10-01/runA bench/results/2026-10-01/runB bench/results/2026-10-01/runC   # direction table
```

Each run takes about 15 minutes for 5 rounds at 20 iterations and 18 minutes
at 30. Run it on an idle GPU: check `ioreg -r -d 1 -w 0 -c IOAccelerator` for
"Device Utilization %" near 0 first. Under contention the spread flags will
say so, as they did here.
