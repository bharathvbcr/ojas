device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 5.07 5.25 4.80
GPU timestamp: 41.8961 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.7 | 83.8 | 320 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 54.3 | 55.9 | 236 |
| ojas_per_head_gate_bwd | 38.1 | 131.3 | 163.3 | 290 |
| ojas_per_head_gate_dbias | 0.2 | 120.3 | 125.3 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.3 | 114.4 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 226.5 | 230.4 | 57 |
| ten ojas_check_finite passes | 63.2 | 231.8 | 239.4 | 273 |
| whole backward, one command buffer | 88.1 | 957.8 | 973.6 | 92 |
end: GPU 95%, load 5.07 5.25 4.80
