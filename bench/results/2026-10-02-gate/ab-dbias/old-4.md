device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 3.53 7.95 9.07
GPU timestamp: 41.9185 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.1 | 84.3 | 318 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.5 | 55.0 | 259 |
| ojas_per_head_gate_bwd | 38.1 | 136.7 | 145.4 | 279 |
| ojas_per_head_gate_dbias | 0.2 | 807.4 | 852.9 | 0 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.5 | 114.6 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 212.7 | 219.3 | 60 |
| ten ojas_check_finite passes | 63.2 | 227.9 | 235.7 | 277 |
| whole backward, one command buffer | 88.1 | 1585.8 | 1599.0 | 56 |
end: GPU 94%, load 3.53 7.95 9.07
