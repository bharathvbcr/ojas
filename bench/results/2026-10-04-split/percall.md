device: Apple M5 Pro, 20 timed runs
start: GPU 69%, load 7.40 7.41 6.97

`MetalBackend`, decode at H 12, D 64, 1024 cached positions; medians over 500 runs.

| scenario | runs | min ms | p50 ms | calls ms | per call µs | sync ms | event wait ms | per run: commits / residency flushes / cold allocs / dispatches / barriers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 request, 1 call | 500 | 0.123 | 0.181 | 0.011 | 11.1 | 0.171 | 0.148 | 1.00 / 2.00 / 2.00 / 2.00 / 2.00 |
| 16 requests, 1 batched call | 500 | 0.503 | 0.741 | 0.020 | 19.6 | 0.720 | 0.686 | 1.00 / 2.00 / 1.00 / 1.00 / 1.00 |
| 16 requests, 16 calls | 500 | 0.807 | 1.162 | 0.135 | 8.5 | 1.019 | 0.979 | 1.00 / 17.00 / 32.00 / 32.00 / 32.00 |

GPU span of one command buffer, tessl directly, 500 buffers.

| command buffer holds | min µs | median µs |
|---|---:|---:|
| 1 request, 1 dispatch | 34.2 | 41.7 |
| 16 requests, 1 batched dispatch | 398.3 | 584.1 |
| 16 dispatches, a barrier after each (as `MetalBackend` records them) | 721.4 | 910.9 |
| 16 dispatches, no barrier between them | 724.8 | 924.0 |

Split cache walk (`ojas_cached_attn` in parts, then `ojas_cached_attn_merge`), GPU span, 500 buffers.

| requests | splits | threadgroups | min µs | median µs | GB/s at min |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 12 | 34.7 | 48.2 | 181 |
| 1 | 2 | 24 | 28.7 | 39.2 | 219 |
| 1 | 4 | 48 | 16.8 | 38.0 | 374 |
| 1 | 8 | 96 | 28.0 | 37.7 | 225 |
| 1 | 16 | 192 | 34.4 | 44.8 | 183 |
| 1 | 32 | 384 | 47.0 | 59.4 | 134 |
| 2 | 1 | 24 | 46.7 | 66.0 | 269 |
| 2 | 2 | 48 | 47.8 | 76.5 | 263 |
| 2 | 4 | 96 | 43.2 | 65.8 | 292 |
| 2 | 8 | 192 | 47.0 | 70.4 | 268 |
| 2 | 16 | 384 | 57.9 | 77.9 | 217 |
| 2 | 32 | 768 | 76.1 | 100.5 | 165 |
| 4 | 1 | 48 | 104.9 | 172.1 | 240 |
| 4 | 2 | 96 | 93.3 | 165.2 | 270 |
| 4 | 4 | 192 | 95.2 | 159.9 | 264 |
| 4 | 8 | 384 | 97.0 | 154.8 | 259 |
| 4 | 16 | 768 | 110.3 | 163.9 | 228 |
| 4 | 32 | 1536 | 147.9 | 190.1 | 170 |
| 16 | 1 | 192 | 21.9 | 572.7 | 4594 |
| 16 | 2 | 384 | 362.8 | 573.9 | 277 |
| 16 | 4 | 768 | 393.6 | 566.3 | 256 |
| 16 | 8 | 1536 | 44.5 | 591.3 | 2263 |
| 16 | 16 | 3072 | 409.5 | 593.0 | 246 |
| 16 | 32 | 6144 | 507.8 | 694.6 | 198 |

Host time to record 16 dispatches (no wait inside), tessl directly; medians over 500 runs.

| recording | µs per dispatch | per run: residency flushes / cold allocs |
|---|---:|---|
| all into one reused output | 1.1 | 0.00 / 0.00 |
| each into a fresh output (as `MetalBackend`) | 1.7 | 16.00 / 16.00 |

end: GPU 94%, load 7.02 7.33 6.94
