# SAM memory cross-attention: flash vs split-K timing (2026-11-02)

Rules: `rules.md` here, written before any timed row (its mtime, 20:44:57
local, is printed at the head of `job.log`, before the 20:46 build and the
20:48 first row). Runs Q2 of `../2026-11-01-sam-xattn-route/rules.md`, which
never ran. Harness: `ojas-metal/examples/attention_flash_bench.rs` at HEAD
`286a444` plus the `sam xattn` rows added for this run. The edit is
uncommitted; it was the only source change in the checkout at start and end
(`job.log`), and it is saved here as **`harness.patch`** (apply to `286a444`).
tessl `52bdd19`, clean. Binary sha256 `7170449a…c2c5`. The new rows are
appended after every existing row, so the Q2 row keeps its index and seeds.
**Verified:** run a's correctness line (`dispatcher-vs-splitK max abs out
1.54e-7 lse 6.68e-6 vs scalar out 1.33e-7 lse 1.34e-5 vs pad-q 2.79e-8`) is
identical to 10-27's (`../2026-10-27-attention-mpp/a-at3g/attempt-2.txt:40`).
The edit passes rustfmt (stdin check, no diff) and
`cargo clippy -p ojas-metal --release --example attention_flash_bench --no-deps -- -D warnings`
(`clippy-nodeps.log`). Without `--no-deps` (`clippy.log`), clippy stops first
on `ojas-cpu/src/quant_sparse.rs:468` (`chunks_exact_to_as_chunks`, toolchain
1.99). That lint was already there and has nothing to do with this edit, but it
will also fail CI's workspace clippy.

Files: `a-q2-m36320.txt` (Q2 as pre-registered), `b-sam-xattn.txt` (3–7
frames), `c-bf16.txt` (bf16 accumulation), `job.log` (lock, provenance,
load), `run.sh` (exact commands).

## Outcome in one line

**Flash is the right kernel and the dispatcher takes it**, 5–8× faster than
split-K at every SAM memory-frame count, but it runs at **~52 ms / 3.7 TFLOP/s
at 36,320 keys, not the predicted ≈42 ms / 4.5 TFLOP/s**. That is about 0.6×
the f32 GEMM control. bf16 accumulation does not help: it is **23× slower** and
its error is larger than the output's peak.

## Cleanliness: every run is `repeat`, so all milliseconds are CONTENDED

| run | load start → end (1-min) | fc1 control (bar 6.11–6.49) | noisy pairs | verdict |
|---|---|---|---|---|
| a, Q2 `m=36320` | 5.80 → 6.76 | 6.05 TFLOP/s | 0 | repeat |
| b, `sam xattn Tk=` | 5.75 → 14.94 | 5.69 TFLOP/s | 0 | repeat |
| c, `sam xattn bf16` | 10.21 → 16.08 | 1.23 TFLOP/s | 0 | repeat |

Run a missed the bar by a small margin: load ended 0.76 over 6.0, and the
control came in 1% under its floor. It is the closest to clean this harness has
come on this machine (10-27 had loads of 6.3–111). Run c started after a
20-minute wait for load ≤ 6 that never came, so its absolute times are heavily
contended. Its **ratios** come from interleaved A/A′ rounds and every pair is
`ok`. Dispatch counts and kernel identity do not depend on load, so they stand.

## Q2: the pre-registered row, `memory cross m=36320`

| arm | p10 | median | p90 | rate |
|---|---|---|---|---|
| dispatcher (flash) | 50.29 | **52.60 ms** | 54.85 | **3.67 TFLOP/s** |
| split-K | 286.07 | 294.02 | 304.88 | 0.66 |
| pad-q (bound lifted) | 284.48 | 294.84 | 303.81 | 0.65 |

**Prediction (≈42 ms / ≈4.5 TFLOP/s): not met.** The median is 25% above it,
in the quietest window this harness has had (control 4% under nominal). The
10-27 contended run had the same row at 50.3 ms / 3.83 TFLOP/s
(`../2026-10-27-attention-mpp/a-at3g/attempt-2.txt:37`), so ~50–53 ms is
reproducible. SpikeTrack's profiled ~49 ms per call agrees with the kernel
time. The 42.6 ms figure the prediction came from does not reproduce.

**Kernel identity at 36,320 (Q1/Q2c), confirmed by dispatch count:** the
dispatcher encoded `Ran { flash: 1, scalar: 0, part: 0, combine: 0 }` in
**6 dispatches**, the same as the scalar call (6) and the forced-flash call
(6/`flash:1`). Split-K took 7 (`part:1, combine:1`). Independently, the
dispatcher call ran 43.2× faster than the same call forced onto the scalar
kernel (48.8 vs 2106.6 ms).

## Q2b: faster kernel per Tk (run b, 25 runs × A/A′, 3 interleaved arms)

Noise bar = the larger p90−p10 of the two arms' A passes. Rate = 4·8·5184·Tk·32 / time.

| frames | Tk | flash (forced) median | split-K median | split-K − flash | bar | faster | dispatcher − flash (bar) |
|---|---|---|---|---|---|---|---|
| 3 | 15,600 | 25.05 ms (3.31) | 145.44 (0.57) | 120.4 | 60.2 | **flash** | +1.7 (27.2) |
| 4 | 20,800 | 35.63 (3.10) | 281.53 (0.39) | 245.9 | 198.2 | **flash** | +12.6 (36.7) |
| 5 | 26,000 | 36.41 (3.79) | 221.21 (0.62) | 184.8 | 50.0 | **flash** | +2.9 (20.3) |
| 6 | 31,200 | 48.79 (3.39) | 274.04 (0.60) | 225.2 | 54.9 | **flash** | +0.1 (17.9) |
| 7 | 36,400 | 58.25 (3.32) | 314.49 (0.61) | 256.2 | 74.6 | **flash** | −0.7 (28.1) |
| — | 36,464 | 64.07 (3.02) | 340.10 (0.57) | 276.0 | 76.5 | **flash** | −4.5 (28.5) |

- Flash wins every row, by far more than the noise bar: 5.8× (15,600) to 8×
  (20,800). Every A/A′ pair is `ok`.
- The **dispatcher is within noise of forced flash on every row**. In the
  timed rounds, each row's encoded kernels matched exactly:
  `Ran { flash: 120, part: 60, combine: 60 }` for 60 calls per arm. Forced flash
  and the dispatcher each encoded the flash pipeline once per call, and split-K
  encoded nothing else. Every row's dispatch counts were 6 / 7 / 6
  (dispatcher / split-K / scalar).
- Correctness held on every row: dispatcher vs split-K out ≤ 1.6e-7, lse
  ≤ 6.7e-6, and the first head plane agrees with the CPU (exact f32).
- Rows with load rising toward 15 show 3.0–3.8 TFLOP/s and no trend with Tk.
  Flash time grows about linearly in Tk, ~1.6 ms per 1,000 keys.

## The throughput ceiling, and what bf16 accumulation does

**Reference point, not headroom.** Per the 11-01 rules, the fc1 control is
tessl's exactness-constrained `ExactF32` GEMM, not the `matmul2d` path flash
uses, so it cannot bound flash. Same-run ratio of flash to the control:

| run | flash at 36k | control (same run) | ratio |
|---|---|---|---|
| a (quietest) | 3.67 TFLOP/s @ 36,320 | 6.05 | **0.61** |
| b | 3.32 @ 36,400 | 5.69 | 0.58 |

So flash at d=32 runs at about **0.6× the f32 GEMM control**, in the same
runs. Per both rules files, no headroom figure (in milliseconds or otherwise)
is drawn from this ratio.

**bf16 accumulation (`set_bf16_accumulation(true)`, forced flash, Tk 36,400,
run c):**

| | median | rate | vs f32 flash (interleaved) |
|---|---|---|---|
| f32 flash | 62.59 ms | 3.09 | — |
| bf16-accum flash | **1428.33 ms** | 0.14 | **22.8× slower** |

The ratio is contended too: a 1.4 s call absorbs more interference than a
60 ms one, so treat 22.8× as an order of magnitude (well over 10×), not a
precise figure. The error figures below do not depend on load.
`rel to peak` is `max|got − want| / max|want|` (`rel_to_peak` in the harness).

| error | bf16-accum vs f32 flash (all planes) | bf16-accum vs CPU (head plane 0) | f32 dispatcher vs CPU |
|---|---|---|---|
| out max abs | 1.61e-2 | 1.43e-2 | 1.41e-7 |
| out max abs / output peak | **1.35** | **1.53** | — |
| out mean abs | 8.93e-4 | — | — |
| lse max abs | 7.43e-1 | 7.03e-1 | 1.05e-5 |
| lse mean abs | 2.90e-1 | — | — |

As `rules.md` predicted from source, this mode is the kernel's scalar
`bf16_acc` branch (`ojas_backend.metal`, flash forward body), not a matrix-unit
bf16 path. It rounds to bf16 after every multiply and every add. It encodes the
same pipeline name (`flash: 1`), so the dispatch count cannot tell the
branches apart; the 23× time difference and the output can. At 36k keys its
running row sum absorbs most of each new block's contribution (lse off by
0.29 on average and 0.74 at worst), and the output error exceeds the output's
own peak. **It measures bf16 rounding error, not bf16 throughput. It is not a
speed lever at this shape, and its output is not usable here.** A faster
reduced-precision path would be bf16 K/V storage on the matrix units (ojas
feature 18, not landed), which this run does not measure. Note also that
`attention_forward` skips the pad route when bf16 accumulation is on
(`backend.rs:2730`). That is irrelevant at F ≥ 3.

## Is the routing right for these shapes?

- **Kernel choice: yes.** At every F = 3…7 shape the dispatcher encodes flash,
  and flash beats split-K by 5–8× beyond noise. Nothing in the router should
  move for these shapes.
- **`FLASH_MIN_GROUPS` (384): not exercised.** These calls launch
  `ceil(5184/32)·8 = 1296` threadgroups, 3.4× the floor, so no plausible value
  of the constant changes routing here. This run is no evidence for or against
  its value. That question belongs to the short-Q sweep rows, which were not
  run.
- **`ENCODER_ATTN_MAX_PAD_RATIO` (2.25): right from the far side.** Every shape
  here has Tk/Tq ≥ 3.0, so the pad route is refused. That is correct: pad-q with
  the bound lifted costs 294.8 ms at 36,320 against flash's 52.6 (5.6×, 50/50
  paired samples). The F ≤ 2 side (pad route taken) was not measured.
- SpikeTrack calls `attention_forward`; the harness times
  `attention_forward_saving`. For F ≥ 3 both reach the same flash ladder, and
  the pad pre-check is refused by the ratio.

## An unexplained change worth an owner look: split-K is ~3.2× slower than on 10-27

At `memory cross m=36320`, split-K took **81.4–83.9 ms** in
`../2026-10-27-attention-stage2/bench-output.txt:33,75` and **92.6 ms** in
`../2026-10-27-attention-mpp/a-at3g/attempt-2.txt:38` (that one at load ~26).
Today it takes **294.0 ms** at load ~6. Flash (50.3 in attempt-2 → 52.6) and
pad-q did not move like that. Pad-q's quiet anchor is 281.8 ms
(`QUIET_PADQ_MS`, stage 2) and it measured 294.8 today, ×1.05. About ten commits have touched `ojas-metal/src/device.rs`
since 10-27. One candidate is `d92a2e7` (2026-11-02, "opt-in bf16
accumulation, … in-kernel finiteness"), which added the bf16 bit to the
split-K `flags` and `splits` words and to the kernels' bodies. **Not
investigated or bisected; this is a lead, not a cause.**
It does not change routing (flash wins either way: 52 vs 82–93 ms on 10-27).
It does matter to any shape that still routes to split-K: bias below 56 groups,
key lengths, head dim 16, short Q under 384 groups.

## What this does and does not establish

- **Verified by running:** kernel identity at 36,320 and at 3–7 frames
  (dispatch counts and timed-round kernel counts); flash is faster than split-K
  at every Tk beyond noise; dispatcher ≈ forced flash; bf16-accum is ~23×
  slower than f32 flash with the errors above.
- **Contended:** every absolute millisecond and TFLOP/s. Run a is the best
  available: 0.76 over the load bar and 1% under the control floor.
- **Not claimed:** headroom (the control cannot bound `matmul2d`); any F ≤ 2
  number; the cause of the split-K slowdown.
- **For SpikeTrack:** the cross-attention kernel itself is the cost (~52 ms ×
  4 ≈ 210 ms per frame at 7 frames), and it is already the fastest kernel ojas
  has for the call. There is no routing lever. The remaining levers are inside
  the flash kernel (it runs at ~0.6× the f32 GEMM control at d = 32), or bf16
  K/V on the matrix units. bf16 accumulation is not one. Any of these is a
  kernel change for the owner to decide; none was made.
