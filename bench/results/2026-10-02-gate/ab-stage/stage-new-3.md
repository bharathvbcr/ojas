device: Apple M5 Pro, 40 timed runs
start: GPU 40%, load 4.20 5.75 9.25
GPU timestamp: 41.9047 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.7 | 83.4 | 320 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 52.6 | 54.5 | 244 |
| ojas_per_head_gate_bwd | 38.1 | 162.0 | 167.4 | 235 |
| ojas_per_head_gate_dbias | 0.2 | 56.4 | 58.0 | 3 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.4 | 114.2 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 45.7 | 46.4 | 281 |
| four standalone ojas_check_finite passes | 25.2 | 75.1 | 79.9 | 336 |
| whole backward, one command buffer | 88.1 | 558.0 | 571.6 | 158 |
end: GPU 92%, load 4.20 5.75 9.25
