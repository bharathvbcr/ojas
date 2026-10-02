device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 19.11 13.32 11.55
GPU timestamp: 41.8900 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.6 | 83.7 | 320 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 52.5 | 55.1 | 244 |
| ojas_per_head_gate_bwd | 38.1 | 166.3 | 174.9 | 229 |
| ojas_per_head_gate_dbias | 0.2 | 120.6 | 125.5 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 111.6 | 114.0 | 115 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 230.2 | 236.5 | 56 |
| four standalone ojas_check_finite passes | 25.2 | 82.3 | 87.5 | 307 |
| whole backward, one command buffer | 88.1 | 805.5 | 818.7 | 109 |
end: GPU 96%, load 19.11 13.32 11.55
