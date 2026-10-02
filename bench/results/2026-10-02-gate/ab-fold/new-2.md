device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 5.42 5.32 4.82
GPU timestamp: 41.8957 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 81.7 | 83.9 | 308 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 50.9 | 56.2 | 252 |
| ojas_per_head_gate_bwd | 38.1 | 147.6 | 167.8 | 258 |
| ojas_per_head_gate_dbias | 0.2 | 120.7 | 125.1 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 108.8 | 114.1 | 118 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 224.6 | 230.6 | 57 |
| four standalone ojas_check_finite passes | 25.2 | 75.8 | 83.5 | 333 |
| whole backward, one command buffer | 88.1 | 764.4 | 797.7 | 115 |
end: GPU 95%, load 5.07 5.25 4.80
