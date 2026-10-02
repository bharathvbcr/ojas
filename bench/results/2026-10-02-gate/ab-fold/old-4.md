device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 5.07 5.25 4.80
GPU timestamp: 41.8793 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.7 | 83.8 | 320 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.4 | 56.0 | 260 |
| ojas_per_head_gate_bwd | 38.1 | 142.1 | 168.6 | 269 |
| ojas_per_head_gate_dbias | 0.2 | 120.7 | 125.4 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 110.7 | 115.6 | 116 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 226.9 | 234.4 | 56 |
| ten ojas_check_finite passes | 63.2 | 241.6 | 254.3 | 262 |
| whole backward, one command buffer | 88.1 | 965.1 | 1003.6 | 91 |
end: GPU 95%, load 5.07 5.25 4.80
