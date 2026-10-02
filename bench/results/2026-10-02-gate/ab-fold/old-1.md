device: Apple M5 Pro, 40 timed runs
start: GPU 49%, load 5.42 5.32 4.82
GPU timestamp: 41.8938 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.1 | 84.1 | 318 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 52.3 | 56.7 | 245 |
| ojas_per_head_gate_bwd | 38.1 | 143.8 | 173.5 | 265 |
| ojas_per_head_gate_dbias | 0.2 | 124.1 | 125.1 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 112.8 | 114.2 | 114 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 224.1 | 229.7 | 57 |
| ten ojas_check_finite passes | 63.2 | 232.8 | 243.5 | 271 |
| whole backward, one command buffer | 88.1 | 963.4 | 990.4 | 91 |
end: GPU 93%, load 5.42 5.32 4.82
