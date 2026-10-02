device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 5.42 5.32 4.82
GPU timestamp: 41.8602 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 82.5 | 84.1 | 305 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 54.3 | 56.0 | 236 |
| ojas_per_head_gate_bwd | 38.1 | 139.1 | 164.4 | 274 |
| ojas_per_head_gate_dbias | 0.2 | 120.1 | 125.0 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 112.3 | 114.2 | 114 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 221.9 | 225.4 | 58 |
| four standalone ojas_check_finite passes | 25.2 | 77.3 | 82.4 | 327 |
| whole backward, one command buffer | 88.1 | 772.6 | 807.7 | 114 |
end: GPU 95%, load 5.42 5.32 4.82
