device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 19.11 13.32 11.55
GPU timestamp: 41.9141 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 82.4 | 83.8 | 305 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 54.5 | 56.0 | 235 |
| ojas_per_head_gate_bwd | 38.1 | 163.6 | 170.9 | 233 |
| ojas_per_head_gate_dbias | 0.2 | 120.3 | 124.9 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 113.5 | 115.0 | 113 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 45.0 | 46.7 | 285 |
| four standalone ojas_check_finite passes | 25.2 | 82.5 | 86.8 | 306 |
| whole backward, one command buffer | 88.1 | 638.2 | 653.3 | 138 |
end: GPU 95%, load 19.11 13.32 11.55
