Decision rules, fixed before the first measurement (committed ahead of the results). No timing of `ojas_attn_fwd_d32` at these shapes, and no code for the cross-attention generalisation, existed when this file was written.

Background. Stage 2 (a simdgroup-matrix f32 partials kernel) was 3x slower than split-K at m = 15584 (109 ms against 35; `bench/results/2026-10-27-attention-stage2`). Fable's wave-5 ruling (scratchpad/fable-wave5.md) is (b), narrowly scoped, behind a day-0 check that can kill it before any code: generalise the TensorOps flash forward `attn_fwd_tiled_body` (`ojas_backend.metal`, instantiated for D32 as `ojas_attn_fwd_d32`) from Tq = Tk with a window to Tk != Tq with no window and an explicitly masked ragged last key block. If the day-0 check fails, (a): stage 2 is abandoned and split-K stays routed.

The 5.2 TFLOP/s that motivates (b) is not in any tessl document. It is derived from this repository's own pad-q medians (`bench/results/2026-10-27-attention-stage2/bench-output.txt`): the padded work 4 B H m^2 D over the pad-q median is 5.3 TFLOP/s at m = 15584, 4.8 at m = 36320 and 4.9 at m = 5200 (inferred; pad-q pads Q to m rows, so it does m^2 work to produce Tq x m of it).

Day-0 kill check (no new code). The existing `ojas_attn_fwd_d32` runs the body. It is bench'd as `Backend::causal_sdpa_forward(q, k, v, None)` (windowless; causal, so a query sees the keys up to itself and the kernel visits only the key blocks on or below the diagonal) on `[1, 8, T, 32]` self-attention at T = 5184, 15584 and 36320, interleaved, load recorded, with the fc1 linear as the control row.

- Rate. Effective TFLOP/s = useful work / median, with useful work 4 H D T (T + 1) / 2 floating-point operations (both matmuls over the visible (query, key) pairs only: the diagonal blocks' masked part counts as waste, which is what an effective rate means). The bidirectional `encoder_attn` (`attention_forward` at Tq = Tk, routed to tessl's TensorOps kernel) is timed in the same pairs at the same T with work 4 H D T^2, as a reference for the engine, not as a decision input.
- Kill rule, verbatim from Fable's wave-5 ruling: "Under 4.3 TFLOP/s effective (192.8 GFLOP in 45 ms), the body needs retuning, out of scope: (b) dies." Where it is read: the median of the causal arm at T = 15584 in a clean run (T = 36320 is reported beside it; T = 5184 is reported for scale). 4.3 TFLOP/s is what the m = 36320 gate needs (4 B H Tq m D = 192.8 GFLOP in 45 ms = 4.28 TFLOP/s) and the m = 15584 gate needs 4.14 (82.7 GFLOP in 20 ms).
- A clean run: both 1-minute load averages at most 6.0, every A-vs-A' pair within the pair's p90 - p10 spread (the memory-bound and stage-2 benches' rules), and the fc1 control within 6.3 TFLOP/s +- 3% (6.11 to 6.49). A run that is not clean is repeated, up to six attempts; every attempt stays in the committed output; a row of an unclean run counts for nothing.
- Interleaving: every arm of a row runs twice per round (A and A'), odd rounds in reverse order; 5 warm-ups, 25 rounds; each call is timed with a sync around it, which carries the harness floor of about 0.12 to 0.15 ms (negligible against 30 to 600 ms calls).
- No correctness gate: no code changed, and the kernel's own tests (`attention_forward`, `causal_sdpa`) already pin it. The example records `MetalBackend::dispatches` per call to show which route each arm took.
- If the rate is under 4.3, the report says so and (b) is closed; nothing else of this work is committed beyond rules, example and results.

If the check passes, (b) is built (this section is the plan Fable approved; its numbers are the gate and are copied verbatim):

- Scope: D32, no bias, no lens: the F >= 3 route. Split-K keeps bias, D16 and lens; a contract split, recorded in the dispatcher with named constants, not accumulation. The stage-2 simdgroup kernel (`wip/78-attn2-simdgroup`) does not merge in either outcome.
- Gate: <= 20 ms at m = 15584; <= 45 ms at m = 36320; no regression at m <= 10624 (pad-q keeps F <= 2 unless beaten), on detector bias, D16, lens. Baseline the better of split-K and pad-q, interleaved, dispatch count checked.
- Parity: full output against split-K and the scalar oracle at stage 1's tolerance; split-K itself matches the scalar kernel only to rounding (`ojas_backend.metal`, the split-K section header). The ragged last key block (15584, 36320 and 5200 are not multiples of the 64-key block) and every tile edge are tested.
- The 2.25 crossover row (`ENCODER_ATTN_MAX_PAD_RATIO`) is re-run against pad-q when either kernel changes; the constant is not changed without telling the coordinator.
- Kill: the gate unmet by the end of Oct 30.
- Other conditions: a demo steady state of F <= 2 means (a); a reprofile that shows cross-attention inflated in-frame means re-measure.

What would refute the check's pass: the causal arm's median at T = 15584 under 4.3 TFLOP/s in every clean attempt.
