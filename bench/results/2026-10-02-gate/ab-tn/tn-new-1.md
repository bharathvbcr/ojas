device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 19.11 13.32 11.55
GPU timestamp: 41.8898 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 80.5 | 83.2 | 313 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 53.2 | 54.7 | 241 |
| ojas_per_head_gate_bwd | 38.1 | 165.2 | 172.5 | 231 |
| ojas_per_head_gate_dbias | 0.2 | 121.0 | 124.8 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 110.9 | 114.9 | 116 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 43.1 | 46.5 | 297 |
| four standalone ojas_check_finite passes | 25.2 | 81.6 | 89.1 | 309 |
| whole backward, one command buffer | 88.1 | 631.7 | 642.2 | 139 |
end: GPU 96%, load 19.11 13.32 11.55
