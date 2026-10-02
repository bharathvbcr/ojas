device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 5.07 5.25 4.80
GPU timestamp: 41.9002 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.2 | 83.1 | 322 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.3 | 55.3 | 260 |
| ojas_per_head_gate_bwd | 38.1 | 150.6 | 165.6 | 253 |
| ojas_per_head_gate_dbias | 0.2 | 120.5 | 125.2 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.3 | 114.8 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 223.3 | 230.0 | 57 |
| four standalone ojas_check_finite passes | 25.2 | 67.5 | 73.6 | 374 |
| whole backward, one command buffer | 88.1 | 769.5 | 798.9 | 114 |
end: GPU 96%, load 5.07 5.25 4.80
