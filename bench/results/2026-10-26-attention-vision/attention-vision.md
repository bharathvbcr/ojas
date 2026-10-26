device: Apple M5 Pro, tensorops true
tessl kernel picked at D=64: Tiled
start: GPU 36%, load 67.59 70.83 58.07
25 reps of A, B, A', B' after 5 warm-ups of each

## windowed [B=9, H=16, T=576, D=64]

windowed: dispatches per call: ojas 6 (MetalBackend::dispatches), tessl 9 (infer_trace; 4 x [check + permute] + 1 kernel = 9 expected)

| windowed comparison | max abs diff | max abs ref | max abs diff / max abs ref | worst suite ratio | verdict |
|---|---:|---:|---:|---:|---|
| ojas vs tessl, whole output | 7.153e-7 | 0.4870 | 1.469e-6 | 0.001 | pass |
| ojas vs CPU Exact, whole output | 1.013e-6 | 0.4870 | 2.081e-6 | 0.001 | pass |
| tessl vs CPU Exact, whole output | 8.643e-7 | 0.4870 | 1.775e-6 | 0.001 | pass |

| arm | n | min ms | median | p10 | p90 | max | p90 - p10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ojas A | 25 | 73.518 | 76.734 | 74.527 | 81.900 | 84.877 | 7.373 |
| ojas A' (same config) | 25 | 73.754 | 76.965 | 74.348 | 86.260 | 88.495 | 11.912 |
| tessl B (with permutes) | 25 | 3.700 | 4.744 | 3.759 | 8.182 | 14.545 | 4.422 |
| tessl B' (same config) | 25 | 3.666 | 4.753 | 3.865 | 8.796 | 11.407 | 4.930 |
| ojas A + A' pooled | 50 | 73.518 | 76.965 | 74.527 | 81.996 | 88.495 | 7.469 |
| tessl B + B' pooled | 50 | 3.666 | 4.753 | 3.782 | 8.182 | 14.545 | 4.399 |

noise floor (A vs A'): |median A - median A'| = 0.231 ms, median per-round |A - A'| = 2.397 ms, pooled p90 - p10 = 7.469 ms
tessl median 4.753 ms vs ojas median 76.965 ms: ojas - tessl = 72.212 ms (tessl/ojas = 0.062), spread 7.469 ms

## global [B=1, H=16, T=5184, D=64]

global: dispatches per call: ojas 6 (MetalBackend::dispatches), tessl 9 (infer_trace; 4 x [check + permute] + 1 kernel = 9 expected)

| global comparison | max abs diff | max abs ref | max abs diff / max abs ref | worst suite ratio | verdict |
|---|---:|---:|---:|---:|---|
| ojas vs tessl, whole output | 4.768e-7 | 0.1925 | 2.477e-6 | 0.000 | pass |
| ojas vs CPU Exact, plane 0 | 5.513e-7 | 0.1185 | 4.652e-6 | 0.000 | pass |
| tessl vs CPU Exact, plane 0 | 6.184e-7 | 0.1185 | 5.217e-6 | 0.000 | pass |
| ojas vs CPU Exact, plane 15 | 5.290e-7 | 0.0949 | 5.574e-6 | 0.000 | pass |
| tessl vs CPU Exact, plane 15 | 4.508e-7 | 0.0949 | 4.750e-6 | 0.000 | pass |

| arm | n | min ms | median | p10 | p90 | max | p90 - p10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ojas A | 25 | 587.411 | 605.757 | 592.465 | 615.206 | 620.641 | 22.741 |
| ojas A' (same config) | 25 | 590.847 | 606.915 | 591.479 | 634.378 | 652.604 | 42.898 |
| tessl B (with permutes) | 25 | 20.458 | 22.816 | 20.855 | 29.259 | 32.240 | 8.404 |
| tessl B' (same config) | 25 | 20.006 | 22.781 | 20.942 | 28.153 | 29.588 | 7.210 |
| ojas A + A' pooled | 50 | 587.411 | 605.981 | 592.369 | 615.206 | 652.604 | 22.836 |
| tessl B + B' pooled | 50 | 20.006 | 22.816 | 20.942 | 28.359 | 32.240 | 7.416 |

noise floor (A vs A'): |median A - median A'| = 1.158 ms, median per-round |A - A'| = 7.069 ms, pooled p90 - p10 = 22.836 ms
tessl median 22.816 ms vs ojas median 605.981 ms: ojas - tessl = 583.165 ms (tessl/ojas = 0.038), spread 22.836 ms

end: GPU 98%, load 80.12 73.87 59.86

routing rule (global shape): tessl median 22.816 ms, ojas median 605.981 ms, A-vs-A spread 22.836 ms: ojas - tessl = 583.165 ms > spread -> ROUTE through tessl
