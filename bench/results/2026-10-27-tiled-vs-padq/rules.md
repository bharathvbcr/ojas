Decision rules, fixed before the first measurement (committed ahead of the results). No timing of either route on this branch had been seen when this file was written.

Question. For memory cross-attention, q `[1, 8, 5184, 32]` and k and v `[1, 8, m, 32]`, f32, no bias, no key_lens, non-causal, on this M5 Pro: at which m does Ojas's split-K tiled kernel start beating the pad-q route? m in {5200, 10400, 10624, 15616}, ratio = m / 5184 = 1.003, 2.006, 2.049, 3.012.

Arms, both through `MetalBackend` on device tensors uploaded once per arm, then `Backend::sync` (wall clock around submit and wait):
- Tiled: `attention_forward_saving(q, k, v, AttentionOpts::default())`. It never routes to tessl and takes the split-K kernel at head dim 32 non-causal. Its log-sum-exp output is computed and dropped; the pad-q arm has no such output. That small extra work (8 x 5184 floats) is on the tiled arm.
- Pad-q: `attention_forward(q, k, v, AttentionOpts::default())`, which pads Q to Tk rows and runs tessl's TensorOps `encoder_attn` (`routes_to_encoder_attn`: head dim 32, 0 < Tq <= Tk <= 8 Tq, no bias or key_lens).

Decision rules (verbatim from the brief):
- R1: the routing bound is set between the largest m where pad-q's median beats tiled by more than the noise, and the smallest m where tiled's median beats pad-q by more than the noise. The ratio is m/5184.
- R2: if the crossover ratio is in [2.05, 2.5], Fable's chosen range 2.25-2.5 stands; report the exact gap.
- R3: if the crossover is below 1.5 or above 3, flag "reset the constant".
- R4: if tiled beats pad-q at m = 5200 beyond noise, flag "delete pad-q".
- Ties within noise favour the route with fewer dispatches and less scratch; say which that is.

Gates, before any timing, for every m. A failure stops the run with no timings.
- Dispatch gate: the dispatch count of each arm (`MetalBackend::dispatches` around the call plus `sync`), on its first call on fresh uploads and again on its second call, must equal that of a small yardstick known to take the same route: the causal scalar call on `[1, 8, 40, 32]` against 40 keys, and the tiled and pad-q calls on the same q against 80 keys. The yardstick itself must show tiled = scalar + 1 and pad-q > tiled on both counts. The counts are recorded.
- Correctness gate: tiled and pad-q outputs agree within 1e-5 max abs error over the whole output (and are finite), and both agree with `ojas_cpu::CpuBackend` under `Numerics::Exact` on the same inputs, at every m, over the first and last head planes (the example's existing bound: `max|a-b| / max|ref| <= 1e-4` and `|a-b| <= 4e-5 sqrt(Tk) + 1e-4 |ref|` per element). The brief required the CPU check at m = 5200 at least.

Timing. Inputs from the example's `Host::new` (q and k in [-2, 2), v in [-1, 1)). 5 warm-ups of each arm, then 25 rounds; a round is two passes over the two arms (A B A' B'), where A' and B' repeat the identical configuration. Odd rounds reverse the arm order inside each pass (B A B' A'), so neither arm is always second. Report min, median and p10-p90 per arm per m, per pass and pooled (50 samples per arm). "Median" is the pooled median. Absolute milliseconds are contended (the GPU and CPUs are shared with other sessions; the load at the start and end of each m is recorded); the verdict is about the arm-vs-arm comparison inside each m.

Noise and "beyond noise".
- margin(m) = pad-q median - tiled median (positive: tiled is faster).
- noise(m), the primary bar = the larger of the two arms' pooled p90 - p10 (the pooled-spread definition the brief allows; it is a spread of single runs, so conservative for a median).
- gap(m), the secondary bar = the larger of the two arms' A-vs-A' median gaps (|median(pass 0) - median(pass 1)| of the same arm).
- Tiled beats pad-q beyond noise iff margin > bar and tiled is faster in at least 45 of the 50 paired (pass, round) samples (pass 0 against pass 0, pass 1 against pass 1). Pad-q beats tiled beyond noise iff -margin > bar and pad-q is faster in at least 45 of 50. Otherwise the m is a tie.
- R1 to R4 are judged on the primary bar. The secondary bar is reported beside it; if the two disagree at any m, the summary says so and says which m.

Applying R1 to R4 mechanically (the example computes these; the summary quotes its output):
- R1 bracket: hi = the smallest m where tiled wins beyond noise; lo = the largest m below hi where pad-q wins beyond noise (the largest such m overall if tiled never wins). If pad-q wins beyond noise at an m above hi, the bracket is non-monotone and reported UNRESOLVED. Ties between lo and hi are listed. The routing bound is reported as the ratio of lo (pad-q is used only where it beat tiled beyond noise; ties favour the route named by the tie-break below), with hi as the upper end of the bracket. The grid has four m, so the bound is a bracket, not a located point.
- Crossover point estimate, for R2 and R3: the first zero crossing of the piecewise-linear margin(m) across the four raw medians, by ratio, from margin <= 0 to margin > 0. If margin > 0 at m = 5200 the estimate is "below" (under 1.003); if margin never turns positive it is "above" (over 3.012). The estimate is an interpolation, labelled inferred; the bracket is what the measurement guarantees.
- R2: the estimate in [2.05, 2.5] means Fable's range 2.25-2.5 stands; the gap reported is the signed distance of the estimate from 2.25 and from 2.5, and the bracket (lo, hi]. An estimate in [1.5, 2.05) or (2.5, 3] triggers neither R2 nor R3: the range moves to the measured bracket and the estimate.
- R3: "reset the constant" when the estimate is below 1.5 or above 3 ("below" and "above" count). The summary states whether the end-of-grid row confirms it beyond noise (tiled wins at 5200 for "below", pad-q wins at 15616 for "above") or the estimate rests on raw medians of a tie.
- R4: tiled wins at m = 5200 beyond noise on the primary bar: flag "delete pad-q".
- Tie-break: the route with fewer dispatches is the route whose measured first-call count is lower; scratch is compared from the code (tiled: split partials, splits x 5184 x 8 x (D + 2) floats, and a log-sum-exp; pad-q: four Tk-row copies and a length per batch row) and labelled inferred.

What would refute the measurement: a gate failing; either arm's dispatch count not matching its yardstick; the A-vs-A' gap exceeding the margins at every m (then every m is a tie and the summary says the grid cannot place the bound); non-monotone winners.
