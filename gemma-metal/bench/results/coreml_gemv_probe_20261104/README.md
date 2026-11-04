# Core ML at E4B's dominant decode shape: fp16 and int4, GPU and Neural Engine (2026-11-04)

The question: would exporting Gemma's decode to Core ML, or running it with fp16 weights, beat
gemma-metal's tessl Q4 GEMV? The MLP gate/up and down GEMVs are 75% of an E4B token's weight bytes
(`../gemv_fraction_of_peak_20261104/README.md`), so their per-layer time decides it.

**Setup** (`probe.py`, log `run.log`, numbers `results.json` is in the scratch output; the rows are
below): a Core ML program of 24 one-token linears alternating 2560 -> 10240 -> 2560 with a tanh GELU
between, random weights, built with coremltools 9.0 (MIL builder, iOS18 opset, fp16 compute).
- `fp16`: fp16 weights.
- `int4`: `linear_quantize_weights`, int4 symmetric, per block of 64 (MLX Q4 uses group 64).
Each arm loaded on CPU_AND_GPU, CPU_AND_NE and ALL; 5 warm-up predicts, then wall time of 30
predicts, divided by 24 layers. That includes one predict's overhead per 24 layers. One run under
mac_heavy, load average 2.3 (quiet), M5 Pro. One run per arm: no repeat, so no measured noise floor.

| arm | units | us per layer, min | median | weight GB/s at min |
|---|---|---|---|---|
| fp16 | CPU_AND_GPU | 205.5 | 210.9 | 255 |
| fp16 | CPU_AND_NE | 332.2 | 334.9 | 158 |
| fp16 | ALL | 331.2 | 332.3 | 158 |
| int4 | CPU_AND_GPU | 67.0 | 69.9 | 208 |
| int4 | CPU_AND_NE | 776.7 | 792.3 | 18 |
| int4 | ALL | 66.4 | 69.2 | 210 |

tessl's `gemv_q4_mlx_simd` at the same shapes: 59.4-64.2 us GPU time and 68-78 us wall per call
(`../gemv_fraction_of_peak_20261104/README.md`, a contended run at load 9-13).

## Reading

- **[V] Core ML int4 on the GPU ties tessl's Q4 GEMV:** 66-67 us per layer against 68-78 us wall
  (59-64 us GPU). A Core ML export of the decode would not be faster per token. tessl also skips
  Core ML's per-predict overhead and keeps the KV cache and sampling on the device.
- **[V] fp16 weights are 3.1-3.4x slower** (205 us per layer) at the same bandwidth (255 GB/s): they
  stream four times the bytes of int4. An "fp16 path" for the weights is a slowdown. Activations
  and the KV cache are already bf16 in gemma-metal.
- **[V] The Neural Engine is slower at batch-1 decode:** 332 us (fp16) and 777 us (int4) per layer.
  ALL picks the GPU for int4 and the Neural Engine for fp16.
- **Not checked:** correctness. The outputs of the arms differ (rel L2 0.31 for fp16 GPU against
  Neural Engine, ~1.0 for int4 against fp16) after 24 random layers, where small per-layer errors
  compound. This probe times bandwidth; it is not a parity test.
