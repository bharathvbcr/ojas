device: Apple M5 Pro, 20 timed runs
start: GPU 71%, load 9.28 7.03 6.41

`MetalBackend`, decode at H 12, D 64, 1024 cached positions; medians over 500 runs.

| scenario | runs | min ms | p50 ms | calls ms | per call µs | sync ms | event wait ms | per run: commits / residency flushes / cold allocs / dispatches / barriers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 request, 1 call | 500 | 0.134 | 0.197 | 0.010 | 10.2 | 0.186 | 0.163 | 1.00 / 2.00 / 1.00 / 1.00 / 1.00 |
| 16 requests, 1 batched call | 500 | 0.520 | 0.725 | 0.016 | 15.7 | 0.708 | 0.687 | 1.00 / 2.00 / 1.00 / 1.00 / 1.00 |
| 16 requests, 16 calls | 500 | 0.931 | 1.349 | 0.126 | 7.9 | 1.184 | 1.110 | 1.00 / 17.00 / 16.00 / 16.00 / 16.00 |

GPU span of one command buffer, tessl directly, 500 buffers.

| command buffer holds | min µs | median µs |
|---|---:|---:|
| 1 request, 1 dispatch | 34.5 | 40.2 |
| 16 requests, 1 batched dispatch | 393.7 | 562.3 |
| 16 dispatches, a barrier after each (as `MetalBackend` records them) | 718.4 | 1049.6 |
| 16 dispatches, no barrier between them | 718.7 | 1037.9 |

Host time to record 16 dispatches (no wait inside), tessl directly; medians over 500 runs.

| recording | µs per dispatch | per run: residency flushes / cold allocs |
|---|---:|---|
| all into one reused output | 1.2 | 0.00 / 0.00 |
| each into a fresh output (as `MetalBackend`) | 1.8 | 16.00 / 16.00 |

end: GPU 95%, load 9.49 7.11 6.44
