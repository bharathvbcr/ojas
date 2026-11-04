# Gemma E4B decode exported to Core ML: int4 and fp16 against gemma-metal (2026-11-04)

The question: does a Core ML export of the decode step, or fp16 weights, beat gemma-metal's Q4 decode
(12.0 ms per token at E4B shapes, `../dispatch_profile_20261104/README.md`)?

**What was exported** (`export_decode.py`, log `run.log`, numbers `results.json`): an E4B-shaped
decode step in PyTorch, converted with coremltools 9.0 to an ML Program (macOS 15, fp16 compute) with
the KV cache as Core ML state, random weights.
- Per layer: RMSNorm; q/k/v with 8 query heads over 2 KV heads, head dim 256, or 512 on the global
  layer; an in-place cache update at the step's position; masked softmax attention over the 256-slot
  cache; the o projection; RMSNorm; GELU-tanh gate/up 2560 -> 10240; down; and the per-layer-input
  gate and projection.
- Then the final norm, the tied 262,144-row output head and an argmax.
- Left out, all small in bytes, so the export has less work than gemma-metal: RoPE, q/k norms,
  logit softcap, the per-layer-input embedding table, and E4B's KV sharing in its last 18 layers.
- Arms: fp16 weights, and int4 (`linear_quantize_weights`, symmetric, per block of 64).
- Timing: CPU_AND_GPU, 8 warm-up steps, then 48 decode steps at positions 8-55, median wall per
  predict. One run per arm, load average ~3-5, M5 Pro. No repeat, so no measured noise floor.

A whole-model export exceeded the mac_heavy footprint cap during conversion (the first attempt, a
6-layer model with the head, was killed), so three programs were timed:

| program | arm | ms median (min) | weight GB | GB/s at median |
|---|---|---|---|---|
| noop (one add: predict overhead) | fp16 | 0.035 (0.033) | 0 | |
| 6 layers (5 sliding + 1 global) | fp16 | 6.14 (5.54) | 1.143 | 186 |
| 6 layers (5 sliding + 1 global) | int4 | 2.69 (2.44) | 0.304 | 113 |
| head, a quarter of the rows (65,536) + argmax | fp16 | 1.58 (1.47) | 0.336 | 212 |
| head, a quarter of the rows (65,536) + argmax | int4 | 0.63 (0.58) | 0.089 | 143 |

**Estimated 42-layer token** = 7 x 6 layers + 4 x quarter head - 10 x noop [I]:
- **Core ML int4: 21.0 ms** (1.75x gemma-metal's 12.0 ms).
- **Core ML fp16: 49.0 ms** (4.1x gemma-metal).

## Reading

- **[I] A Core ML export of Gemma's decode is slower than gemma-metal**, about 21 ms against 12 ms per
  token, even with less work per layer. The int4 GEMVs alone tie tessl's
  (`../coreml_gemv_probe_20261104`); the loss is in the rest of the layer: the 6-layer int4 block
  streams its weights at 113 GB/s, against ~256 GB/s for gemma-metal's whole token.
- **[I] fp16 weights make the decode about 2.3x slower than int4 in Core ML** (49 ms against 21 ms),
  because each token streams about four times the bytes.
- Every number is from random weights, with the composition estimated from three separately timed
  programs. Real E4B weights were not used (not on disk; the download was not approved).
