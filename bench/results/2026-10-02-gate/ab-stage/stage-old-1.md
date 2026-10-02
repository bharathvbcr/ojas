device: Apple M5 Pro, 40 timed runs
start: GPU 55%, load 4.20 5.75 9.25
GPU timestamp: 41.8829 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.3 | 83.8 | 317 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.6 | 55.5 | 258 |
| ojas_per_head_gate_bwd | 38.1 | 86.4 | 167.4 | 441 |
| ojas_per_head_gate_dbias | 0.2 | 121.0 | 125.1 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 112.6 | 114.0 | 114 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 44.3 | 45.9 | 289 |
| four standalone ojas_check_finite passes | 25.2 | 77.7 | 83.6 | 325 |
| whole backward, one command buffer | 88.1 | 620.4 | 636.1 | 142 |
end: GPU 95%, load 4.20 5.75 9.25
