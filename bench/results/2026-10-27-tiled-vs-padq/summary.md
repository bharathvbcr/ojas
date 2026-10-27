# Tiled kernel vs pad-q route: memory cross-attention crossover (2026-10-27)

Question: for SpikeTrack's memory cross-attention, q `[1, 8, 5184, 32]` over k and v `[1, 8, m, 32]` (f32, no bias or key_lens, non-causal) on the M5 Pro, at which m does Ojas's split-K tiled kernel (stage 1, f5de4c0) start beating the pad-q route (Q padded to Tk, tessl TensorOps `encoder_attn`)? It sets the routing bound that replaces `ENCODER_ATTN_MAX_PAD_RATIO = 8`. Rules R1 to R4 were committed before any timing (`rules.md`, bd5edcf; the bench started after it). Raw data: `rounds.csv`, `rows.jsonl`, `bench-output.txt`. Run by `ojas-metal/examples/attention_spiketrack_bench.rs` rows `x5200 x10400 x10624 x15616` (alias `xover`), 35717a9, on branch gp/78-bench3 (local main fdb030a plus a merge of f5de4c0).

## Gates (all passed before timing)

- Dispatches, first call and repeat: yardstick scalar (6, 6), tiled (7, 7), pad-q (9, 9); every m matched its yardstick, tiled (7, 7) and pad-q (9, 9).
- Tiled vs pad-q, whole output, max abs diff: 3.95e-7, 4.21e-7, 4.84e-7, 4.66e-7 at m = 5200, 10400, 10624, 15616 (bound 1e-5); no non-finite values.
- Both arms vs `CpuBackend` `Numerics::Exact`, head planes 0 and 7, at every m: worst max abs diff 8.2e-7, worst max abs diff / max abs ref 1.07e-5 (bound 1e-4); all pass.

## Verdict

5 warm-ups, then 25 rounds of A B A' B' with the arm order reversed on odd rounds; 50 samples per arm per m. Noise is the larger of the two arms' pooled p90 - p10; the secondary bar is the A-vs-A' median gap.

| m | ratio m/5184 | tiled median ms | pad-q median ms | noise ms | A-vs-A' gap ms | paired samples won | winner |
|---:|---:|---:|---:|---:|---:|---|---|
| 5200 | 1.003 | 11.886 | 5.700 | 0.635 | 0.034 | pad-q 50/50 | pad-q |
| 10400 | 2.006 | 23.583 | 21.314 | 0.878 | 0.248 | pad-q 50/50 | pad-q |
| 10624 | 2.049 | 24.032 | 22.144 | 0.924 | 0.156 | pad-q 50/50 | pad-q |
| 15616 | 3.012 | 35.261 | 47.715 | 0.961 | 0.001 | tiled 50/50 | tiled |

Primary and secondary bars agree at every m; no m is a tie.

- R1: pad-q wins beyond noise up to ratio 2.049 (m = 10624), tiled from ratio 3.012 (m = 15616). The measured bracket is (2.049, 3.012]; nothing was measured inside it.
- R2: the rules' point estimate (zero of the piecewise-linear margin over the raw medians) is ratio 2.176 [inferred], inside [2.05, 2.5], so Fable's range 2.25-2.5 stands (estimate 0.074 below 2.25, 0.324 below 2.5). The linear estimate is biased low: pad-q's time grows with m² (its median rises about 4.7x from m = 5200 to 15616, tiled's 3.0x), so the margin is convex; a two-parameter fit puts the zero near 2.23 [inferred].
- R3: not triggered.
- R4: not triggered; pad-q is 2.09x faster at m = 5200, so it stays.
- Tie-break (not needed): it would favour tiled, 7 dispatches against 9 [verified]; tiled holds about 11.5 MB of split partials at every m, pad-q four Tk-row copies, about 21 MB at m = 5200 and 64 MB at m = 15616 [inferred from code].

## What it means for the bound

SpikeTrack's ratio is F + P·16/5184 with F memory frames (1 to 7) and P object pointers (1 to 16): F = 2 spans 2.003 to 2.049 and F = 3 starts at 3.003. Any bound in (2.049, 3.003) therefore routes every SpikeTrack shape the same way: F <= 2 to pad-q, F >= 3 to the tiled kernel, which is the faster route at every measured point on both sides. Inside that gap the bound only matters for other callers; 2.25 sits at the fitted crossover (about 2.23), so it is the choice. The old bound of 8 sent F = 3 to 7 to pad-q: 47.7 ms against 35.3 ms per layer at F = 3 here, and 278.7 ms against 82.7 ms at F = 7 in Lane A's earlier min-of-N run (not re-measured in this interleaved run).

## Limits

- One run of 25 rounds per m on a shared machine (load average about 7); absolute milliseconds are contended, the verdict is the within-m comparison.
- The tiled arm also computes and drops a log-sum-exp (8 x 5184 floats), a small handicap against tiled.
- The grid places the bound in a bracket; it does not locate the crossover inside (2.049, 3.012]. Timing m of about 11500, 12000 and 12960 would.
- The example's header comment still says the crate has no `encoder_attn` routing, which is stale for rows `a` to `bx` on this branch.
