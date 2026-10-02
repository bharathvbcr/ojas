device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 5.07 5.25 4.80
GPU timestamp: 41.8878 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.3 | 84.2 | 318 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 53.7 | 55.3 | 239 |
| ojas_per_head_gate_bwd | 38.1 | 138.3 | 165.8 | 276 |
| ojas_per_head_gate_dbias | 0.2 | 123.1 | 125.2 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 113.6 | 115.1 | 113 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 227.4 | 233.0 | 56 |
| four standalone ojas_check_finite passes | 25.2 | 73.5 | 78.0 | 343 |
| whole backward, one command buffer | 88.1 | 774.8 | 799.5 | 114 |
end: GPU 96%, load 5.07 5.25 4.80
