# Memory-bound Metal ops at SpikeTrack shapes, measured quiet (2026-10-27)

Why: the GEMM lane study's conv rows turned out to be contended (its im2col + layout share read 68%; quiet it is about 33%). Its bias-add column and the old SpikeTrack profile's permute row are memory-bound too, so both were re-measured with load recorded before any decision rests on them.

Method: `ojas-metal/examples/memory_bound_bench.rs`, release, under mac_heavy on the M5 Pro. Rules in `rules.md`, committed before the first run (and an addendum before run 2): a row counts only if its run's load is recorded and its A-vs-A' gap is within its spread. Two runs, 5 warm-ups and 25 interleaved rounds each; every arm has an identical A' arm. Load averages (1/5/15 min) at start and end: run 1 3.83/4.08/4.74 both; run 2 4.18/4.10/4.71 and 4.01/4.07/4.69. No row was flagged noisy. The f32 and bf16 linear arms are controls: fc1 6.30-6.34 TFLOP/s f32 and 18.9-19.2 bf16, fc2 6.71-6.72 f32, against 6.29, 6.76 and 18.87 in the GEMM study, so neither run was contended. GB/s counts bytes read plus written.

## Bias add (broadcast_add of the bias row after the linear), run 2 medians

| Shape | bias add ms | GB/s | f32 linear ms | add share of linear + add |
|---|---|---|---|---|
| fc1 [5184,4736] | 0.966 | 203 | 7.98 | 10.8% |
| fc2 [5184,1024] | 0.347 | 123 | 7.49 | 4.4% |
| qkv [5184,3072] | 0.672 | 190 | 5.25 | 11.3% |
| proj [5184,1024] | 0.332 | 128 | 1.90 | 14.9% |
| 256 [5184,256] | 0.189 | 56 | 0.31 | 38% |
| 256 [10400,256] | 0.237 | 90 | 0.50 | 32% |
| 256 up [5184,2048] | 0.493 | 172 | 1.38 | 26% |
| 256 down [5184,256] | 0.199 | 53 | 1.13 | 15% |

The GEMM study's bias-add column was not contended (fc1 0.928, qkv 0.669 ms there). Large adds stream at 172-203 GB/s, above a write-only `fill` of the same output (96-120 GB/s), likely because the product is still partly cached (inferred). Small adds sit near the about 0.14 ms dispatch-and-sync floor, so their GB/s are not bandwidth limits. Fusing the bias into the GEMM saves at most these milliseconds.

## Permute at SpikeTrack's call sites (st-model, read-only), run 2 medians

| Shape, dims | Call site | permute ms (GB/s) | streaming relu, same bytes, ms (GB/s) | Transpose? |
|---|---|---|---|---|
| [9,576,16,64], [0,2,1,3] | backbone.rs:613 | 0.472 (90) | 0.319 (133) | no, 256 B runs |
| [1,5184,16,64], [0,2,1,3] | backbone.rs:613 | 0.454 (94) | 0.311 (137) | no |
| [1,3,24,3,24,1024], [0,1,3,2,4,5] | ojas spatial.rs:635 (window_partition) | 0.550 (77) | 0.319 (133) | no, 4 KB rows |
| [1,3,3,24,24,1024], [0,1,3,2,4,5] | ojas spatial.rs:674 (window_merge) | 0.549 (77) | 0.311 (137) | no |
| [1,1024,72,72], [0,2,3,1] | backbone.rs:507 | 0.456 (93) | 0.310 (137) | yes |
| [256,82944], [1,0] | detector.rs:372 | 1.374 (124) | 0.789 (215) | yes |
| [1,5184,8,32], [0,2,1,3] | layers.rs:116 | 0.226 (47) | 0.187 (57) | no, 64 B runs |
| [1,8,201,32], [0,2,1,3] | layers.rs:122 | 0.149 | 0.145 | latency floor |

All 19 rows are in `bench-output.txt`. The large permutes run at 58-70% of a streaming pass over the same bytes, transposes and plain head or window moves alike, which points at the generic rank-loop kernel (per-element index arithmetic, strided stores) rather than at transposes [inferred; a specialised copy path was not built or tested]. The head swaps and window moves are the bulk of the calls; a tiled transpose would reach only the NCHW to NHWC and [256,82944] rows (roughly 4 ms of permute per frame, about 1.5-2 ms to gain). Against the old contended profile, [9,576,16,64] was 8-20% inflated there (0.50-0.57 ms) and [256,82944] matches (1.41 ms).

## Limits

- Per-frame totals need call counts per frame; those belong to the frame ledger, not this file.
- One machine; load 3.8-4.2 is not idle, but the controls repeat to within 1%.
