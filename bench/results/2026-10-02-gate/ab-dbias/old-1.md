device: Apple M5 Pro, 40 timed runs
start: GPU 58%, load 3.49 8.02 9.10
GPU timestamp: 41.9279 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 80.4 | 85.3 | 313 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.6 | 54.5 | 258 |
| ojas_per_head_gate_bwd | 38.1 | 138.4 | 145.6 | 276 |
| ojas_per_head_gate_dbias | 0.2 | 807.1 | 885.5 | 0 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 113.6 | 114.5 | 113 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 212.2 | 217.1 | 60 |
| ten ojas_check_finite passes | 63.2 | 232.3 | 236.6 | 272 |
| whole backward, one command buffer | 88.1 | 1611.0 | 1625.7 | 55 |
end: GPU 93%, load 3.49 8.02 9.10
