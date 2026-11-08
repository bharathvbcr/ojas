# Rules — SAM memory cross-attention: flash vs split-K timing (2026-11-02)

**Written before the first timed run of this folder.** No commit is possible
from this session (shared checkout, eleven other sessions), so the ordering
evidence is the job log: `run.sh` prints this file's `stat` mtime, `git
rev-parse HEAD`, `git status --short` and `uptime` as its first lines, before
the build and before any timed row.

This run executes **Q2 of `bench/results/2026-11-01-sam-xattn-route/rules.md`**,
which was pre-registered and never ran (`mac_heavy` exit 76). That file is not
edited. Its Q2, verbatim in substance: the dispatcher arm's median ms on
`memory cross m=36320` and the rate over `4·B·H·Tq·Tk·D = 192.8 GFLOP`,
**predicted ≈ 42 ms / ≈ 4.5 TFLOP/s**. Its cleanliness bar is kept: verdict
`clean`, 1-minute load ≤ 6.0 at both ends, the `control fc1 … linear f32
(control)` row inside 6.11–6.49 TFLOP/s, every A-vs-A′ pair `ok`. A `repeat`
verdict makes the milliseconds **CONTENDED**; kernel identity (dispatch counts)
still stands.

## Deviations from the 11-01 rules, stated now

- **Commit.** 11-01 named `dac0b5b`. The harness and the router have changed
  since (`d92a2e7`: bias now takes flash at ≥ 56 groups, in-kernel finiteness).
  This run is at HEAD `286a444` plus the harness additions below. Q2's row
  (`memory cross m=36320`) is unchanged: same index, so the same seeds, same
  arms (dispatcher, split-K, pad-q).
- **Binary.** Built into the gitignored `target-xattn-q2/` at the repo root
  (the 11-01 probe target dir no longer exists), `CARGO_BUILD_JOBS=2`.

## What is added (rows appended after every existing row)

`sam xattn Tk=N`: `[1,8,5184,32] x [1,8,N,32]`, f32, no bias, no key lengths,
non-causal. N = `F·5184 + 16·F` (one 16-token object-pointer group per frame,
SpikeTrack's 16-object run) for F = 3…7: **15600, 20800, 26000, 31200, 36400**,
plus **36464** (the brief's upper end). 36400 is SpikeTrack's profiled count.
Arms, interleaved as the harness does (A, A′, odd rounds reversed):
`flash (forced)` = `attention_forward_saving_flash`, `split-K` =
`attention_forward_saving_splitk`, `dispatcher` = `attention_forward_saving`.
No pad-q arm (the ratio bound excludes it at F ≥ 3 and 11-01 already timed it
only as a lifted-bound comparison).

`sam xattn bf16 Tk=36400`: the same three arms plus `flash bf16-accum (forced)`
(`set_bf16_accumulation(true)` around the forced flash call, reset after).

## Questions, with predictions fixed now

- **Q2 (as 11-01).** ≈ 42 ms / ≈ 4.5 TFLOP/s on the dispatcher arm at 36320.
- **Q2b (faster kernel per Tk).** Flash beats split-K at every added Tk by
  more than the pair's noise bar (larger p90−p10 of the two A passes).
  Predicted from 11-01's sweep design: yes, at 1296 groups. If split-K wins any
  row, that is reported as an owner question; routing is not changed.
- **Q2c (kernel identity at 36320 and 36400).** The dispatcher encodes
  `Ran::FLASH`, with the scalar call's dispatch count (split-K's is one more).
- **Q2d (bf16 accumulation).** Reading `ojas_backend.metal` (`bf16_acc` branch
  of the flash forward body): it is a scalar per-(q, k) loop with bf16 rounding
  after every multiply and add, not the matrix-unit path. **Predicted: slower
  than f32 flash, by more than 2×**, with output error vs f32 well above the
  1e-5 f32 gate. It is reported as a numerics mode, not a speed lever; its
  output is kept out of the harness's correctness-fail gate.

## The "ceiling", and what will not be claimed

The brief asks for the winning kernel's TFLOP/s against the fc1 control
(≈ 6.3 TFLOP/s). That ratio is reported, **labelled a reference point, not
headroom**: per 11-01 the control is tessl's exactness-constrained
`GemmOperands::ExactF32`, not the `matmul2d` path the flash kernel uses, so it
cannot bound it. Where the brief's wording and the 11-01 rules disagree, the
rules win.

## Routing verdict, stated in advance

At Tq = 5184 the flash launch is `ceil(5184/32)·8 = 1296` threadgroups, far
above `FLASH_MIN_GROUPS` (384): that constant is **not exercised** by these
shapes, whatever the timing. `ENCODER_ATTN_MAX_PAD_RATIO` (2.25) is exercised
only from the far side (every added row is ≥ 3.0 · Tq). SpikeTrack calls
`attention_forward`; the harness times `attention_forward_saving`; for F ≥ 3
both reach the same flash ladder (the pad route is refused by the ratio).
