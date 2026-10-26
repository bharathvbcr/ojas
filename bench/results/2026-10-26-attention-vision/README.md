# SAM 3.1 ViT attention on Metal: ojas vs tessl `encoder_attn`, 2026-10-26

**Question.** GitPulse card ft-53d639de (general attention), acceptance 4:
how fast is `MetalBackend::attention_forward` at the SAM 3.1 ViT shapes, and
should the unmasked `Tq = Tk`, `D` in {64, 256} case be routed through
tessl's `embedgemma2::encoder_attn`?

**Run.**
- `cargo run --release -p ojas-metal --example attention_vision_bench -- 25`
  (`ojas-metal/examples/attention_vision_bench.rs`), base d2922fb, under the
  machine lock (`mac_heavy`), 15:36-15:40 local.
- Apple M5 Pro, TensorOps present; tessl picked the `Tiled` f32 kernel
  (`encoder_attn_tiled_h64_q32_k32_sg2`).
- The machine was not quiet: load 67-80, the GPU read 36% busy at the start
  and 98% at the end (other sessions' work; the lock admits one job of this
  lane's kind, not an idle machine). Both arms saw the same conditions,
  interleaved.
- Raw output: `attention-vision.md`.
- Data: q, k uniform in [-2, 2), v in [-1, 1), so the softmax is peaked and
  the output (max |o| 0.1-0.5) discriminates. f32, non-causal, no bias, no key
  lengths, scale 1/8.

## Method

- Ojas arm: `attention_forward` on device tensors uploaded once, then
  `Backend::sync`. Six dispatches per call (five finite checks and the
  kernel, `MetalBackend::dispatches`).
- tessl arm: three `[B,H,T,D]` to `[B,T,H,D]` permutes, `encoder_attn`, the
  output permuted back, `synchronize`. Nine dispatches per call (`infer_trace`):
  four of check + permute, one kernel. The lens buffer (all `T`) is built once
  outside the timer: it depends only on `(B, T)`, so a routed path would cache
  it. Intermediates are allocated fresh every run, as every `MetalBackend`
  op allocates its output.
- **Deviation, stated.** `MetalBackend` owns its tessl runtime on a private
  device thread, so its tensors cannot be passed to tessl, and the example
  cannot call `Backend::permute` on tessl's buffers. The tessl arm runs on a
  `GpuRuntime` of its own on the same GPU, with this crate's overlay metallib
  loaded, and records each permute as `fn permute` in `device.rs` does: the
  `ojas_check_finite` pass over the input, then the same `ojas_permute`
  kernel with the same bindings. So the permute cost is the routed path's.
  What differs: the ojas arm also pays the device-thread channel, and the
  tessl arm does not. One tiny op plus sync costs 0.108 ms at best and
  0.164 ms median through `MetalBackend` (`../2026-10-01-floor/floor.md`),
  against 4.7-22.8 ms for the tessl arm here: under 4% of it, and it works
  against ojas by at most that much. A route inside the device thread would
  not pay it per op.
- Correctness before timing. Pass iff `max|a-b| / max|ref| <= 1e-4` and every
  element satisfies the suite's `|a-b| <= 4e-5 sqrt(T) + 1e-4 |ref|`. Both
  arms against `CpuBackend` under `Numerics::Exact` (whole output at the
  windowed shape; planes 0 and 15 at the global shape, where the full CPU
  reference is too slow), and the arms against each other over the whole
  output. All pass by 2-3 orders of magnitude (below).
- Timing: 5 warm-ups per arm, then 25 rounds of A, B, A', B' (A', B' repeat
  the identical configuration: the noise floor). Wall clock around submit and
  wait; results dropped after the clock stops.
- Rule, fixed before the first run: route iff at the global shape the median
  of tessl (B and B' pooled) is below the median of ojas (A and A' pooled) by
  more than the A-vs-A spread, taken as p90 - p10 of the pooled A and A'.

## Correctness

| shape | comparison | max abs diff | max abs diff / max abs ref |
|---|---|---:|---:|
| windowed | ojas vs tessl, all | 7.2e-7 | 1.5e-6 |
| windowed | ojas vs CPU, all | 1.0e-6 | 2.1e-6 |
| windowed | tessl vs CPU, all | 8.6e-7 | 1.8e-6 |
| global | ojas vs tessl, all | 4.8e-7 | 2.5e-6 |
| global | ojas vs CPU, planes 0 / 15 | 5.5e-7 / 5.3e-7 | 4.7e-6 / 5.6e-6 |
| global | tessl vs CPU, planes 0 / 15 | 6.2e-7 / 4.5e-7 | 5.2e-6 / 4.8e-6 |

## Timings (ms, 25 rounds per arm; min / median / p10-p90)

| shape | ojas A | ojas A' | tessl B | tessl B' |
|---|---|---|---|---|
| windowed [9,16,576,64] | 73.5 / 76.7 / 74.5-81.9 | 73.8 / 77.0 / 74.3-86.3 | 3.70 / 4.74 / 3.76-8.18 | 3.67 / 4.75 / 3.87-8.80 |
| global [1,16,5184,64] | 587 / 606 / 592-615 | 591 / 607 / 591-634 | 20.5 / 22.8 / 20.9-29.3 | 20.0 / 22.8 / 20.9-28.2 |

Pooled medians: windowed ojas 77.0, tessl 4.75 (0.062x); global ojas 606.0,
tessl 22.8 (0.038x). Noise floor at the global shape: |median A - median A'|
1.2 ms, median per-round |A - A'| 7.1 ms, pooled p90 - p10 22.8 ms.

## Answer: route through tessl

At the global shape ojas - tessl = 583 ms against a spread of 22.8 ms, so the
rule says **route**. The gap is 25 times the spread and every one of the 25
tessl rounds (max 32 ms) is below every ojas round (min 587 ms). The windowed
shape agrees (72 ms against a spread of 7.5 ms). The permutes are part of the
tessl figure.

Rough rates from the FLOP count (4 T^2 D per head): global 110 GFLOP in 606 ms
is 0.18 TFLOP/s for the scalar `ojas_attn_gen_fwd`, 4.8 TFLOP/s for the tiled
TensorOps kernel with its permutes.

**Scope of the verdict.**
- Measured at D = 64 only. The D = 256 half of the rule is not measured: this
  example runs D = 64, and tessl has a D = 256 tiled kernel
  (`encoder_attn_tiled_h256_*`) that was not exercised here. Treat D = 256 as
  inferred, not verified.
- Unmasked, `Tq = Tk`, `H = Hkv`, no bias, no key lengths only: `encoder_attn`
  takes none of the others.
- f32 operands. The tiled kernel's f32 path agrees with the CPU Exact
  reference to 5e-6 of the output scale here, but that is two shapes of one
  data distribution.
- Noisy host (load 67-80). The effect is two orders larger than the
  observed spread, so it is not the load; the absolute milliseconds are.
