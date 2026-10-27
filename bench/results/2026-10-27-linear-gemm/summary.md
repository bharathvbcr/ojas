# Linear and conv GEMM lanes at SAM 3.1 shapes on Metal (2026-10-27)

Question: why does `linear_bias_forward` take 26% of SpikeTrack's Metal frame (tessl-45's per-op profile at Ojas d5d33cc), which tessl GEMM lane each linear and conv runs, how close each runs to tessl's documented M5 Pro peak, and whether f32 has headroom without bf16. Measurement only: no routing changed.

Method: `ojas-metal/examples/linear_gemm_bench.rs`, release, under mac_heavy on the M5 Pro (f1bf555, parent local main cbd5565). Device-resident tensors; 5 warm-ups, then 25 rounds with the arms interleaved per round; min, p10, median, p90 per arm in `bench-output.txt`. TFLOP/s = 2mkn / median. The p10-p90 spread is about ±2% on the large shapes. Peaks are tessl's `docs/benchmarking.md:366-377` (M5 Pro): exact f32 6.3-6.4, tf32-relaxed 13.4-16.3, bf16 21.2-26.6 TFLOP/s.

## Lanes (read from code by Lane A; file:line in tessl and ojas-metal)

- f32 Fast, untagged operands: exact-f32 NT, `gemm_nt_f32` (`matmul2d_tensorops_nt_f32`, a 32x32 single-simdgroup tile).
- Autocast bf16, or both operands tagged bf16-exact: bf16 NT, `gemm_nt_bf16` (128x64 cooperative tile, four simdgroups).
- `linear_bias_forward` is `linear_forward` plus a separate `broadcast_add_forward`.
- `conv2d` is im2col plus the exact-f32 NT GEMM plus layout passes, always exact f32.
- tessl's relaxed f32 cooperative lane is behind a runtime-wide flag that Ojas never sets and `GemmOperands::ExactF32` refuses; not reachable, not measured.

## Linears, median ms (TFLOP/s)

| Shape | f32 | f32 + bias | bias add | bf16 lane | bf16 + bias | Autocast (casts included) |
|---|---|---|---|---|---|---|
| fc1 [5184,1024] x [4736,1024] | 8.00 (6.29) | 8.70 | 0.93 | 2.67 (18.9) | 3.38 | 3.70 |
| fc2 [5184,4736] x [1024,4736] | 7.44 (6.76) | 7.58 | 0.34 | 3.11 (16.2) | 3.24 | 4.20 |
| qkv [5184,1024] x [3072,1024] | 5.19 (6.29) | 5.69 | 0.67 | 1.88 (17.3) | 2.33 | 2.62 |
| proj [5184,1024] x [1024,1024] | 1.92 (5.65) | 2.06 | 0.34 | 0.89 (12.2) | 1.05 | 1.24 |
| windowed proj [9x576,1024]^2 | 1.92 (5.66) | 2.04 | 0.33 | 0.88 (12.3) | 1.05 | 1.23 |

The 256-wide linears are in `bench-output.txt` (2 to 4.7 TFLOP/s on f32; the [201,256] and [4,256] ones sit at a latency floor of about 0.14 ms).

- The large ViT linears run the exact-f32 lane at 98-106% of its documented peak: no f32 headroom inside that lane.
- The bf16 lane is 2.2-3.0x faster in isolation (46-71% of its peak, so it has headroom of its own); through Autocast, casts included, 1.7-2.4x faster than f32 + bias.
- The separate bias add is 5-16% of large f32 linears, 15-21% under bf16, and 28-52% of the 256-wide ones. tessl's fused, bit-exact `gemm_epilogue` is NN only; Ojas weights are NT `[n, k]`.
- fc1 measures 8.70 ms with bias here against 12.4 ms per call in the d5d33cc frame profile; the gap (about 40%) is the profile's contention and sync, not the kernel.

## Correction (2026-10-27, later the same day): the conv rows below were contended

The conv table in this section came from a run on a contended machine, and its im2col + layout share is wrong. Re-measured on a quieter machine (load average 4 to 5, recorded; three runs, 25 interleaved rounds each; `bench/results/2026-10-27-conv-im2col/old-path-repeats.txt`), the same conv2d path gives:

| Shape | conv2d | GEMM alone | im2col + layout |
|---|---|---|---|
| neck 3x3 p1 [1,256,288,288] | 25.28-25.29 | 16.64-16.93 | 8.4-8.6 ms (about 33%), not 36.7 ms (68%) |
| neck 3x3 p1 [1,256,144,144] | 6.13-6.16 | 4.27-4.33 | 1.8 ms, not 8.7 ms |

The GEMM, compute-bound, moved by under 2% between the runs; the memory-bound im2col and layout passes ran about 2.4x slower in the contended run [inferred: load was not recorded in that run]. The bias-add column above is memory-bound too and was taken in the same run, so read it as an upper bound until re-measured on a quiet machine. The f32 and bf16 GEMM columns are compute-bound and agree with the quiet runs.

## Conv, median ms (contended run; superseded by the correction above)

| Shape | conv2d (im2col + GEMM + layout) | GEMM alone | im2col + layout share |
|---|---|---|---|
| neck 3x3 p1 [1,256,288,288] | 53.73 (1.82 TFLOP/s) | 16.99 (5.76) | 68% |
| neck 3x3 p1 [1,256,144,144] | 13.09 | 4.35 | 67% |
| neck 3x3 p1 [1,256,72,72] | 1.69 | 1.29 | 24% |
| patch embed [1,3,1008,1008] k14 s14 | 1.65 | 1.28 | 22% |

At the two large neck convs the im2col and layout passes cost more than the GEMM.

## Reading

- f32 linear: at its lane's ceiling. Faster linear means another lane (bf16, measured; relaxed f32, documented only), both of which change numerics, or bias fusion (bit-exact, 5-16% on f32).
- Conv: on a quiet machine im2col and layout are about a third of the large neck convs (8.4 ms at 288^2), within 1.25x of im2col's write floor; the GEMM is the larger part. See the correction above.

## Limits

- One run on one machine; correctness of the bf16 and Autocast arms is not gated here (they are existing, separately tested paths).
- The relaxed f32 lane is documented, not measured.
- Per-frame weights of these calls need a fresh frame profile; this bench measures calls, not frames.
