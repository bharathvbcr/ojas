# Rules: tessl's matrix-unit convolution against the im2col route (2026-11-04)

Written before the timed run.

**Arms**, interleaved in rounds by `ojas-metal/examples/linear_gemm_bench.rs`: `conv2d f32` (today's
route: im2col, tessl's exact-f32 NT GEMM, layout pass), `conv2d tensorops f32` and `conv2d tensorops
f16` (`Conv2dTensorOps`, tessl::conv: pack, MPP convolution2d, unpack), and the probe arms the bench
already runs (GEMM alone f32 and bf16, fills, permute).

**Shapes**: every conv row whose name has `3x3` (the neck at 288², 144² and 72², 256 → 256; the
memory-encoder downsampler at 1152², 32 → 16).

**Run**: 3 warm-ups and 15 timed rounds per arm, in one mac_heavy hold. Load average is recorded
before and after.

**Claims and what refutes them**:
- At 288², `tensorops f16` median is at least 5x faster than `conv2d f32`. Refuted below 2x.
- `tensorops f32` is faster than `conv2d f32` at every 3x3 shape. Refuted if it is slower at any
  shape by more than the two arms' p10-p90 overlap.
- A shape where the op loses is reported as such, and it is a reason to keep the route opt-in.

Correctness is from the tests, not this bench: `tessl/tests/conv2d_tensorops.rs` (f64 reference,
partial tiles, 16-channel outputs, batch 2, padding 0/1/2) and `ojas-metal/tests/conv2d_tensorops.rs`
(CPU backend, and proof that im2col did not run).
