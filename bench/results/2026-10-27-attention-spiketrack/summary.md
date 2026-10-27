# SpikeTrack attention shapes on Metal (Lane C2, 2026-10-27)

What decides what: each row below has its pre-stated rule quoted verbatim first, then the numbers, then a verdict against that rule. The rules were committed before the run (`rules.md` in commits 255837c and 485b987; the run started after 485b987). The bias threshold was changed from the provisional 275 ms to the measured 35 ms before any measurement, on the coordinator's message, and both are reported.

## Machine, commit, contention

- Device: Apple M5 Pro (Mac17,8), macOS 27.0.1, tensorops true. tessl kernel at D=32: `Tiled`.
- Source: branch `gp/78-bench2`, base 105e203. `ojas-metal/src` has no `routes_to_encoder_attn` (checked with `rg`), so every "ojas scalar" arm is `ojas_attn_gen_fwd`; the scalar dispatch count is 6 at D=32, the same as the D=64 scalar arm of Lane C (7 with a bias). The example is `ojas-metal/examples/attention_spiketrack_bench.rs`; the run is `c2-run` (heavy.sh, release, target dir `ojas/target-78c`), finished 23:51 local. Raw output: `rounds.csv` (every timed run, 1000 rows, arm field quoted), `rows.jsonl` (one summary per row), `bench-output.txt` (stdout, with the mac_heavy queue lines).
- Contention: the run held the mac_heavy lock, but the GPU is shared with other sessions' non-lock work. Load average at the start of rows was 5.7 to 16 (Lane C measured 60 to 80), and ioreg "Device Utilization" read 0% at the start of nearly every row. Those are instantaneous samples and a row's start sample can include this process's own tail (bx started at 96%, row ends read 83 to 99%). I cannot rule out short bursts from other sessions. Absolute milliseconds are therefore labelled contended-unknown; the p90 - p10 spread was 0.6% to 7% of the median for every row of 5 ms or more, and the verdicts rest on ratios and on the paired per-round sign.
- Correctness gate (before any timing, every arm vs the Ojas CPU backend `Numerics::Exact` on the same inputs, plus arm vs arm over the whole output): all comparisons passed. Worst max abs error over all rows: 5.96e-7 absolute, 1.1e-5 relative to the output scale (c3, tessl vs CPU, plane 0); the suite ratio was at most 0.002 (bound 1). CPU planes checked: all 8 for b, d, e, f and g; planes 0 and 7 for a, c1 and c2; plane 0 only for c3 (a plane is over 3e9 multiply-adds); planes 0 and 127 for bx. The whole-output ojas vs tessl comparison covers every plane.
- Intended path ran: ojas arms 6 dispatches (7 with a bias); tessl arms issued exactly the expected dispatches (9 for a and e, 10 for the fused pad variant, 12 for the explicit pad variant), checked against `infer_trace` before timing (the run aborts on a mismatch).
- CSV note: the per-round CSV arm field was written unquoted in the run and contained commas for the pad arms; the committed `rounds.csv` was re-quoted losslessly by a script (join of the middle fields), and the example now quotes the field. The example change (escaped quotes in one `format!`) is rustfmt-clean but was not compiled or clippy-checked after the run (job c2-check was refused by mac_heavy with exit 74, disk under 15 GB), and the bench was not re-run.

## Rules (verbatim from rules.md)

Decision rules, fixed before the first measurement (committed ahead of the results).

Common definitions. Each row runs 5 warm-ups of every arm, then 25 rounds; a round is two passes over the arms in order (arm 1, arm 2, ..., arm 1', arm 2', ...). Arm 1 is always the Ojas scalar kernel, so arm 1 vs arm 1' is the A-vs-A pair. "Median" is the median of the pooled samples of the two passes. "Spread" is the p90 - p10 of the pooled scalar samples (arm 1 and arm 1', 50 samples), the definition Lane C used; the median gap |median(arm 1) - median(arm 1')| is also reported, and a verdict that holds against the spread also holds against the gap. "Sign consistent" means the alternative arm is faster than the scalar arm in every paired round (pass 0 against pass 0, pass 1 against pass 1: 50 of 50). Absolute milliseconds are contended (the GPU is shared with other sessions); only arm-vs-arm ratios are quoted as findings.

Correctness gate. Before any timing, every arm must match the Ojas CPU backend (`Numerics::Exact`) on the same inputs with `max|a-b| / max|ref| <= 1e-4` and `|a-b| <= 4e-5 sqrt(Tk) + 1e-4 |ref|` per element, and each arm's dispatch count must equal what its intended path issues. A failure stops the row with no timings.

(a) D32 self-attention [1,8,5184,32] for q, k and v, no mask. Arm 1: Ojas `attention_forward` on Metal (the scalar kernel at 105e203). Arm 2: tessl `encoder_attn` at head dim 32, including the permutes. Per-layer ms x 10 layers per frame (6 detector encoder + 4 memory attention). Rule: routing D32 is recommended iff arm 2's median margin exceeds the A-vs-A spread and the sign is consistent across rounds.

(b) Detector cross-attention with additive bias: q [1,8,201,32], k and v [1,8,5184,32], bias [1,8,201,5184] f32, scalar kernel only. Per-frame cost = per-layer median x 6 layers. Rule: a tiled bias kernel is warranted iff the per-frame cost exceeds 10% of the CoreML frame time. The rule was first stated with a provisional denominator of 2.75 s per frame (the port's published 2.05 s backbone + 0.7 s tracker, SpikeTrack HANDOFF.md:16), a provisional threshold of 275 ms per frame. Before any measurement here, tessl-45 measured the CoreML frame on this M5 Pro: 0.272 s backbone + 0.0775 s track step = 0.3495 s (SpikeTrack bench/results/baselines/coreml.json, median of 5, contended), so the threshold is 10% = 35 ms per frame. The verdict is judged against 35 ms; 275 ms is also reported. If the Budget refuses the bias, report it and run the largest batch that fits. Extra row (not rule-bearing): the earlier default shape q [16,8,200,32], k and v [16,8,5184,32], bias [16,8,200,5184] at the largest batch the Budget lets through.

(c) Memory cross-attention: q [1,8,5184,32], k and v [1,8,m,32], no mask, m in {5200 (frame 1), 15616, 36544 (max)}; 4 layers per frame. Arm 1: scalar. Arm 2: tessl `encoder_attn` with q zero-padded to m rows, lens = m, and the output sliced back to 5184 rows, with the pad and slice copies counted in the arm's time. Two variants are timed: fused (the permute writes straight into the padded buffer and the closing permute reads the first 5184 rows) and explicit (separate pad copy and slice copy). The rule is judged on the explicit variant, the conservative one. Rule: the padding route is recommended iff it wins by more than the spread at the given m (same sign-consistency requirement as (a)). For m < 5184 the route is unavailable: padding k would be needed and `lens` zeroes the query rows. SpikeTrack's m is always above 5184, so the m < 5184 case does not arise.

(d) Decoder D16 sanity row: q [1,8,16,16], k and v [1,8,4096,16], scalar only. Expectation: under 1 ms. No rule; the number is reported.

(e) Detector decoder self-attention [1,8,201,32], Tq = Tk, no mask, 6 layers; scalar against tessl `encoder_attn` (D32). Rule: same as (a), judged at this shape.

(f) Encoder to text cross-attention: q [1,8,5184,32], k and v [1,8,33,32], key_lens [33] and [20], 6 layers, scalar only (Tq > Tk, no pad route). No rule; the cost is reported.

(g) Decoder to text cross-attention: q [1,8,201,32], k and v [1,8,33,32], no mask and key_lens [20], 6 layers, scalar only. No rule; the cost is reported.

Per-frame totals: median ms x layers summed over a (x10), b (x6), c at each m (x4), e (x6), f33 (x6), g (x6), once with the scalar kernel everywhere and once with each rule-bearing row on its tessl arm where its rule says route. d has no layer count and is excluded; f20, g20 and the extra batch row are variants and excluded.

## Results

All cells are ms over the 50 pooled samples of the two passes (25 rounds x 2), shown as min / median / p10-p90. Spread is the p90 - p10 of the pooled scalar samples. "Wins" is the number of the 50 paired runs in which the tessl arm was faster than the scalar arm of the same round and pass.

| row | shape | scalar | tessl | spread | A-vs-A median gap | median margin | wins | verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| a | [1,8,5184,32] self | 224.5 / 229.0 / 227.0-231.2 | 5.61 / 5.70 / 5.64-6.18 | 4.21 | 0.42 | 223.3 | 50/50 | ROUTE D32 |
| c1 | q 5184, m=5200 | 225.6 / 230.2 / 227.7-234.8 | explicit 5.71 / 5.85 / 5.76-6.32 (fused 5.67 / 5.77 / 5.69-6.24) | 7.11 | 0.14 | 224.3 | 50/50 | ROUTE pad |
| c2 | q 5184, m=15616 | 635.2 / 642.2 / 637.4-646.7 | explicit 47.14 / 48.18 / 47.59-49.54 (fused 47.14 / 47.96 / 47.39-48.83) | 9.31 | 0.07 | 594.0 | 50/50 | ROUTE pad |
| c3 | q 5184, m=36544 | 1471.2 / 1478.4 / 1474.2-1483.4 | explicit 266.2 / 276.0 / 270.0-280.1 (fused 268.1 / 274.4 / 270.3-279.8) | 9.20 | 0.17 | 1202.4 | 50/50 | ROUTE pad |
| e | [1,8,201,32] self | 0.540 / 0.553 / 0.544-0.577 | 0.178 / 0.189 / 0.182-0.214 | 0.032 | 0.003 | 0.364 | 50/50 | ROUTE D32 |

### (a) D32 self-attention

Rule (verbatim): "Rule: routing D32 is recommended iff arm 2's median margin exceeds the A-vs-A spread and the sign is consistent across rounds."

Result: margin 223.3 ms against spread 4.21 ms (and gap 0.42 ms); tessl faster in 50 of 50 paired runs. **Verdict: route D32 through `encoder_attn`.** The ratio is 0.025 (about 40x). Per layer 229.0 ms scalar, 5.70 ms tessl; x10 layers = 2290 ms against 57.0 ms per frame. (The earlier brief's x14 would be 3205 ms against 79.8 ms; the coordinator's count is 10.)

### (b) Detector cross-attention with bias (scalar only)

Rule (verbatim from rules.md): "a tiled bias kernel is warranted iff the per-frame cost exceeds 10% of the CoreML frame time ... the threshold is 10% = 35 ms per frame. The verdict is judged against 35 ms; 275 ms is also reported."

Thresholds and sources: 275 ms is the provisional figure, 10% of 2.75 s (the port's published 2.05 s backbone + 0.7 s tracker, SpikeTrack HANDOFF.md:16). 35 ms is the measured figure, 10% of the 0.3495 s CoreML frame on this M5 Pro (0.272 s backbone + 0.0775 s track step, SpikeTrack `bench/results/baselines/coreml.json`, median of 5, contended; supplied by tessl-45, not re-measured here).

Result: bias 1x8x201x5184 f32 = 33 MB, accepted by the Budget. Scalar per layer 11.10 / 12.06 / 11.56-12.39 ms (spread 0.83, A-vs-A gap 0.09); x6 layers = **72.4 ms per frame**. **Verdict: 72.4 ms > 35 ms, so a tiled bias kernel is warranted** (it is 20.7% of the CoreML frame). Against the provisional 275 ms it would not have been (72.4 <= 275); the measured threshold governs. Caveats: 35 ms is what the rule gives; the saving a tiled kernel can achieve is not measured here. The extra default row (q [16,8,200,32], bias [16,8,200,5184], 531 MB) was accepted by the Budget at batch 16: 154.2 / 159.9 / 157.5-162.8 ms per layer, x6 / 16 = 60.0 ms per frame at batch 16 (informational; it shows the per-frame cost falls only from 72 to 60 ms with batching, so the scalar bias kernel is not launch-bound at batch 1).

### (c) Memory cross-attention, pad-q route

Rule (verbatim): "Rule: the padding route is recommended iff it wins by more than the spread at the given m (same sign-consistency requirement as (a)). For m < 5184 the route is unavailable: padding k would be needed and `lens` zeroes the query rows. SpikeTrack's m is always above 5184, so the m < 5184 case does not arise." The rule is judged on the explicit pad + slice variant.

Result: at all three m the explicit variant wins by more than 31x the spread (margins 224, 594, 1202 ms against spreads 7.1, 9.3, 9.2 ms) and in 50 of 50 paired runs. **Verdict: the pad route is recommended at m = 5200, 15616 and 36544.** The fused variant is within 0.1 to 1.6 ms of the explicit one, so the copies cost about 0.08 ms at c1 and about 0.2 to 1.6 ms at c2 and c3 (within about 0.6% of the arm). For m < 5184 there is no pad route (stated by the rule; not applicable to SpikeTrack since m = F x 5184 + P x 16 with F >= 1).

Per frame (x4 layers): scalar 920.6 / 2568.7 / 5913.8 ms; pad route 23.4 / 192.7 / 1104.1 ms at m = 5200 / 15616 / 36544.

The pad route's cost grows with m squared, not linearly: 5.85 ms at m=5200, 48.2 ms at 15616 (x8.2), 276.0 ms at 36544 (x47) while m grew x3.0 and x7.0. Padding q to m rows makes the kernel do m x m work where 5184 x m is needed, and the scalar kernel's time is linear in m (230, 642, 1478 ms). This explains why the margin shrinks from 39x to 5.4x as m grows. Inferred (not measured): a tiled kernel with separate q and kv lengths would cost about 5.7 ms x m / 5184 = 5.7 / 17 / 40 ms per layer (about 23 / 69 / 161 ms per frame), which would turn the max-m frame from 1104 ms into about 161 ms in memory attention. That is the highest-value kernel in this report after D32 routing.

### (d) Decoder D16 sanity

Rule (verbatim): "Expectation: under 1 ms. No rule; the number is reported."

Result: q [1,8,16,16], k and v [1,8,4096,16]: 1.974 / 2.027 / 1.997-2.420 ms (spread 0.42, A-vs-A gap 0.000). **The expectation is not met: 2.03 ms per call.** The dispatch count is 6, the same as the other scalar rows; the f33 row (5184 query rows, 33 keys) costs 1.6 ms and g (201 x 33) costs 0.22 ms, so this is not a fixed launch cost. Inferred: only 8 heads x 16 query rows = 128 independent rows each walk 4096 keys serially in the scalar kernel, which leaves the GPU almost empty. There is no layer count for D16, so it is not in the per-frame totals; at 2 ms per call it matters if it runs several times per frame.

### (e) Detector decoder self-attention

Rule (verbatim): "Rule: same as (a), judged at this shape."

Result: margin 0.364 ms against spread 0.032 ms, 50 of 50 paired runs; scalar 0.553 ms, tessl 0.189 ms (ratio 0.34); x6 = 3.32 ms against 1.13 ms per frame. **Verdict: routable (D32 routing covers it), but the saving is 2.2 ms per frame**, so it is a by-product of (a), not a reason to route.

### (f) and (g) Scalar-only text cross-attention

Rule (verbatim): "No rule; the cost is reported." (for f33, f20, g and g20).

| row | shape | per layer, min / median / p10-p90 | x6 per frame (median) |
|---|---|---|---:|
| f33 | q [1,8,5184,32], kv [1,8,33,32], key_lens [33] | 1.586 / 1.623 / 1.593-1.652 | 9.74 ms |
| f20 | same, key_lens [20] | 1.068 / 1.093 / 1.075-1.113 | 6.56 ms |
| g | q [1,8,201,32], kv [1,8,33,32], no mask | 0.208 / 0.223 / 0.212-0.245 | 1.34 ms |
| g20 | same, key_lens [20] | 0.190 / 0.202 / 0.193-0.227 | 1.21 ms |

Tq > Tk in f, so there is no pad route. These total about 11 ms per frame; nothing here earns a kernel. The mask (key_lens 20) is applied correctly (CPU parity passed) and is cheaper than the unmasked case.

## Per-frame totals

Median ms x layers: a x10, b x6, c x4, e x6, f33 x6, g x6. d (no layer count), f20, g20 and bx are excluded. "Scalar everywhere" is every row on `ojas_attn_gen_fwd`. "Routed" puts a, e and c on their tessl arms (the pad route, explicit variant, for c) as their rules say, and leaves b, f33 and g scalar.

| memory length | scalar everywhere | routed | of which c (routed) | of which b (scalar) | CoreML frame (measured) |
|---|---:|---:|---:|---:|---:|
| m = 5200 (frame 1) | 3297 ms | 165 ms | 23 ms | 72 ms | 350 ms |
| m = 15616 | 4945 ms | 334 ms | 193 ms | 72 ms | 350 ms |
| m = 36544 (max) | 8290 ms | 1246 ms | 1104 ms | 72 ms | 350 ms |

Routed, frame-1 attention alone is 47% of the CoreML frame, mid is 96%, and at max m it is 3.6x the whole CoreML frame, with 89% of that in the memory cross-attention. With the inferred native Tq != Tk kernel (about 23 / 69 / 161 ms per frame for c) the totals would be about 165 / 210 / 302 ms.

## What this says for Lane A's queue (in order of measured value)

1. Route D32 self-attention through `encoder_attn` (a, e): rule passes; removes 2290 ms and 2 ms per frame.
2. Route the memory cross-attention through the pad-q route (c): rule passes at every m; saves 897 / 2376 / 4810 ms per frame. It is the interim route.
3. A tiled kernel with separate q and kv lengths: not covered by a rule; inferred to cut the max-m pad route from 1104 to about 161 ms per frame. The measurement above (quadratic growth of the pad route) is the evidence.
4. A tiled bias kernel (b): rule passes against the measured 35 ms threshold (72.4 ms per frame).
5. Nothing for f and g (11 ms per frame together). D16 (d) at 2.0 ms per call is the one surprise; whether it needs a kernel depends on calls per frame.

## Not measured

- bf16 variants; D64 (Lane C measured it at 3eba4d2); the CPU-side cost of building the length buffers (cached outside the timer, as a routed path would); a native Tq != Tk tiled kernel (item 3 is inferred from scaling); the decoder's call count per frame for (d).
- GitPulse tools were not loaded in this session, so `gitpulse_collision_risk` was not checked. Only files under `ojas-metal/examples/` and `bench/results/2026-10-27-attention-spiketrack/` were created or edited.
