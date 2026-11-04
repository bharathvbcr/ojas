# Ojas convolutions on tessl's matrix-unit op (`Conv2dTensorOps`, 2026-11-04)

`conv2d_forward` on Metal can now send the convolutions tessl's `conv2d_nchw` covers to that op. The
op is Metal Performance Primitives' `convolution2d` (tessl `src/conv.rs`; its semantics and probe are
in `tessl/bench/results/conv2d_20261104/README.md`). It covers 3x3 kernels, stride 1, one group,
equal padding, and output channels a multiple of 16. Every other convolution keeps the im2col route.

**The route is opt-in:** `MetalBackend::set_conv2d_tensorops(Conv2dTensorOps::F16 | F32 | Off)`, or
`OJAS_METAL_CONV2D_TENSOROPS=f16|f32` when a backend opens. It defaults to Off. The f16 mode rounds
inputs and weights to f16 (relative L2 ≈ 3e-4), which fails `conv2d_gemm`'s 1e-5/1e-4 tolerance by
design. So adopting it in SpikeTrack needs SpikeTrack's accuracy gates. f32 mode passes every
existing conv test.

`rules.md` was written before the run. `run_fused.log` holds the raw rows of the shipped kernel;
`run.log` holds the first version's.

## Per call [V, `run_fused.log`: 15 interleaved rounds, wall time through `conv2d_forward`]

The shipped kernel: weights, pack, and the op storing its tile straight to NCHW. Load average 3.12
before, 3.37 after. Medians, with p10-p90 for the two arms claim 2 compares.

| 3x3 conv, pad 1 | im2col f32 (today's route) | tensorops f32 | tensorops f16 | f16 speed-up |
|---|---|---|---|---|
| 288², 256 → 256 | 53.15 ms (50.19-58.53) | 17.14 ms (16.91-17.91) | **5.40 ms** | **9.8x** |
| 144², 256 → 256 | 6.43 ms (6.02-6.61) | 4.91 ms (4.71-5.22) | 1.62 ms | 4.0x |
| 72², 256 → 256 | 1.71 ms (1.65-1.97) | 1.67 ms (1.63-1.93) | 0.67 ms | 2.6x |
| 1152², 32 → 16 | 87.27 ms (82.69-97.33) | 14.87 ms (14.02-15.36) | **3.07 ms** | **28.4x** |

**The pre-stated claims:**
- **Claim 1 holds.** f16 at 288² is more than 5x faster than im2col: 9.8x.
- **Claim 2 holds at three shapes and is a tie at 72².** f32 is faster with p10-p90 ranges apart
  at 288², 144² and 1152². At 72² its median is 2.7% lower but the ranges overlap (1.63-1.93
  against 1.65-1.97 ms). By `rules.md` that is not a refutation (it is not slower), but f32 is not
  shown faster there.

**The first version** (`run.log`, op into an NHWO scratch then an unpack pass, deleted) ran at a
load of 8.5 → 5.9, so its rows are not paired with these. They gave the same picture: f16 9.9x at
288², 22.8x at 1152², 2.6x at 72², and f32 at 72² 1.58 against 1.68 ms with p10-p90 apart.
tessl's paired A/B of the two stores is in `tessl/bench/results/conv2d_20261104/README.md` ("Fused
NCHW store"): fused 2-6% faster at 256 channels, 15-20% at 1152², 32 → 16, 2-5% slower at 72².

The bench's `COUNT` lines count `with_binder` calls (tessl `runtime.rs` `dispatch_count`), so the
whole tessl::conv call counts once, plus Ojas's finite scans: 5 for both versions.

**The strongest baseline is MPSGraph** (`tessl/bench/results/conv2d_20261104/mpsgraph.log`, GPU time
of the conv alone).
- f16: 3.13 ms at 288² and 0.29 ms at 72². The op alone is at parity (3.26 ms at 288²). The
  end-to-end gap (5.40 ms) is the weight and pack passes and Ojas's finite scans of the input,
  weights and output.
- f32: MPSGraph takes 7.98 ms at 288², 2.1x faster than tensorops f32 (17.14 ms).
- Reaching MPSGraph from Rust needs an MPSGraph binding crate, which is a new dependency.

## Whole SpikeTrack frame [V, `spiketrack_speed.log`, `spiketrack/`]

A speed probe, not a gate: `spiketrack_speed.sh`. SpikeTrack's default frame on held-out VISEM 36,
32 frames, 3 rounds with the arm order rotated, one binary (`spiketrack/versions.txt`). Each arm
ran at a load average below 7. Ratios are per round against that round's base.

| round | f16 / base (push_frame median, wall) | f32 / base (push_frame median, wall) |
|---|---|---|
| 1 | 1.000, 0.918 | 0.994, 0.917 |
| 2 | 0.972, 0.952 | 0.978, 0.963 |
| 3 | 0.961, 0.931 | 0.978, 0.950 |

push_frame median is over frames ≥ 8. Round 1's base ran first after the build with no warm-up
(wall 1,059 ms against 997 and 992 in rounds 2 and 3), so round 1's wall ratios overstate.

- **The frame gains 2-4% in push_frame median and 4-7% in wall time** (rounds 2 and 3). It is small
  because few of the frame's convs run on Ojas: the neck and the memory encoder run as Core ML
  packages in the default frame (the `neck on the trunk's worker: coreml` and `memory encoder`
  lines of every run log). Which Ojas convs took the route was not traced.
- **Tracks are unchanged:** in every round and both modes, all 32 frames had the same track ids as
  base, with mean box IoU 1.0000 over shared ids. The minimum box IoU was 0.9966 (f16) and 0.9921
  (f32). Predictions are not byte-identical, as expected from a different summation order.
- The per-call speed-ups above are where this route pays: an Ojas-run neck, or any model whose
  3x3 convs run on Ojas.

## Tests

- `ojas-metal/tests/conv2d_tensorops.rs`:
  - covered shapes take the op, with no `ojas_im2col` and no `ojas_conv_nchw`, and match the CPU
    backend: f32 within 1e-5 relative L2, f16 within 2e-3;
  - uncovered shapes (stride 2, 5x5, 24 output channels) keep im2col, bit for bit what Off gives.
  - the Budget is charged the op's own scratch (tessl's `conv2d_nchw_scratch_elems`), not the
    im2col route's. Before this, a one-column output was charged less than the op allocates (its
    width is padded to a 32-wide tile); the test failed against that code.
- `tessl/tests/conv2d_tensorops.rs` checks against an f64 reference, with poisoned scratch.
- All of these were re-run on the fused-only code and pass: tessl 5/5; Ojas `conv2d_tensorops` 3,
  `conv2d_gemm` 13 (3 ignored), `conv2d_pointwise` 13 and `conv2d_via_im2col` 2; and
  `conv2d_gemm` again with `OJAS_METAL_CONV2D_TENSOROPS=f32`.
- `conv2d_gemm`, `conv2d_pointwise` and `conv2d_via_im2col` pass with the route Off. With
  `OJAS_METAL_CONV2D_TENSOROPS=f32`, `conv2d_gemm` passes too. With `f16`, one tolerance test fails.
