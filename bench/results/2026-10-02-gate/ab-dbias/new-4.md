device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 3.53 7.95 9.07
GPU timestamp: 41.9039 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 80.0 | 85.3 | 315 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.2 | 54.4 | 260 |
| ojas_per_head_gate_bwd | 38.1 | 148.3 | 154.5 | 257 |
| ojas_per_head_gate_dbias | 0.2 | 120.3 | 125.2 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 108.8 | 114.1 | 118 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 212.5 | 217.6 | 60 |
| ten ojas_check_finite passes | 63.2 | 230.1 | 236.7 | 275 |
| whole backward, one command buffer | 88.1 | 938.1 | 948.8 | 94 |
end: GPU 95%, load 3.53 7.95 9.07
