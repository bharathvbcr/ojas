device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 3.53 7.95 9.07
GPU timestamp: 41.9380 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.3 | 84.4 | 317 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.5 | 54.7 | 259 |
| ojas_per_head_gate_bwd | 38.1 | 146.6 | 156.8 | 260 |
| ojas_per_head_gate_dbias | 0.2 | 120.4 | 125.3 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.1 | 114.4 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 211.6 | 217.0 | 61 |
| ten ojas_check_finite passes | 63.2 | 229.9 | 236.7 | 275 |
| whole backward, one command buffer | 88.1 | 941.7 | 949.0 | 94 |
end: GPU 94%, load 3.53 7.95 9.07
