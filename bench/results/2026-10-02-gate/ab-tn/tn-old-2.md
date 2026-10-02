device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 19.11 13.32 11.55
GPU timestamp: 41.9260 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 83.9 | 84.7 | 300 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.0 | 54.5 | 262 |
| ojas_per_head_gate_bwd | 38.1 | 166.0 | 172.0 | 230 |
| ojas_per_head_gate_dbias | 0.2 | 120.6 | 125.0 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.3 | 114.1 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 227.4 | 232.2 | 56 |
| four standalone ojas_check_finite passes | 25.2 | 80.1 | 86.6 | 315 |
| whole backward, one command buffer | 88.1 | 790.4 | 803.4 | 111 |
end: GPU 96%, load 19.11 13.32 11.55
