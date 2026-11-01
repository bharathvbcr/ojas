# Rules — SAM memory cross-attention: which kernel runs, and what it costs (2026-11-01)

**Committed before the first timed run.** Written because card ft-8d3aa337's
brief assumed this call falls through to the scalar kernel; reading the router
says it takes the TensorOps flash kernel. This run measures the call in ojas's
own harness instead of SpikeTrack's profiler.

## What is measured

`ojas-metal/examples/attention_flash_bench.rs` at commit `dac0b5b`, release,
rows selected with the example's `only` substring filter:

1. **`memory cross m=36320`** — `h=8, tq=5184, tk=36320, d=32`, f32, no bias,
   no key lengths, non-causal. This is SAM 3.1's memory cross-attention
   geometry (SpikeTrack's profiled call is the same with 36,400 keys;
   `SpikeTrack/crates/st-model/src/tracker/memory_attention.rs:20-22,150,163-173`).
   Arms: dispatcher (`Auto`), split-K, pad-q.
2. **`detector bias [1,8,201,32]x5184`** — the DETR boxRPB additive-bias row.
   Arms: dispatcher, split-K, scalar.

The harness's own rules are `bench/results/2026-10-27-attention-mpp/rules.md`;
this file does not restate them, it states what *this* run concludes.

## Questions, with predictions fixed now

- **Q1 (kernel identity, row 1).** Which kernel does the dispatcher encode?
  **Predicted: the flash kernel**, and the dispatcher's dispatch count equals
  the scalar call's (split-K's is one more). Read from the example's
  `# dispatches …` / `# kernels …` lines, not from timing.
- **Q2 (baseline, row 1).** The dispatcher arm's median ms and the effective
  rate over `4·B·H·Tq·Tk·D = 192.8 GFLOP`. **Predicted ≈ 42 ms / ≈ 4.5 TFLOP/s**,
  from SpikeTrack's profiled 42.6 ms and the contended 3.72–4.22 TFLOP/s that
  `attention_causal_rate_bench` saw at T=15584.
- **Q3 (row b, row 2).** Which kernel takes a biased call, and its cost.
  **Predicted: split-K, not scalar** — `xattn_splits`
  (`ojas-metal/src/backend.rs:1912`) never inspects the bias, and
  `AttnExtra::scalar_only()` is only radius/dead-rows.

## Cleanliness, and what I will not conclude

A row's numbers are **clean** only if the example says so: verdict `clean`,
1-minute load ≤ 6.0 at both ends, the `control fc1 … linear f32 (control)` row
inside 6.11–6.49 TFLOP/s, and every A-vs-A′ pair `ok`.

- If the verdict is `repeat`, the absolute milliseconds are reported as
  **CONTENDED** and no efficiency claim is drawn from them. Q1 and Q3's
  *kernel identity* still stand, because a dispatch count does not depend on
  load.
- This machine has already failed to produce a clean window for this harness:
  `2026-10-27-attention-mpp` logged six attempts and a three-hour wait at loads
  of 6.3 to 111, verdict `repeat` throughout. Load at the time of writing is
  21.9 / 27.5 / 33.2. **A run that yields no clean window is a legitimate
  outcome and will be reported as such, not substituted with a stand-in.**
- **No headroom figure will be claimed from this run.** The ceiling that would
  bound it — f32 GEMM throughput *on the matrix units* — is not measured
  anywhere in this repo. The f32 linear control is tessl's exactness-constrained
  `GemmOperands::ExactF32` (`ojas-metal/src/device.rs:2266-2271`), which is not
  the matrix-unit path the flash kernel uses, so it cannot bound a `matmul2d`
  kernel. This run establishes the **baseline**, not the headroom.

## Provenance

The binary is built from the **main checkout** at `dac0b5b`, the same commit as
this worktree, with `CARGO_TARGET_DIR=/Users/bharath/Code/research/ojas-xattn-probe-target`:
the worktree cannot build, because ojas's sibling path dependencies
(`tessl = ../../tessl`, gusset `../../../devtools/gusset`) do not resolve from
`.claude/worktrees/<lane>/` and cannot be fixed with symlinks (cargo
canonicalizes, and tessl depends back on `../ojas/ojas-core`). No source
differs between the two checkouts; both are clean at `dac0b5b`.

Run under the machine-wide heavy-job lock,
`/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh` (lock
`.../Lappi-decision/build/mac-heavy.lock.d`), which waits for a 1-minute load
of 12 or less before taking the lock.
