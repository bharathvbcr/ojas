device: Apple M5 Pro, 40 timed runs
start: GPU 49%, load 4.20 5.75 9.25
GPU timestamp: 41.8295 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.3 | 83.9 | 317 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 51.7 | 55.8 | 248 |
| ojas_per_head_gate_bwd | 38.1 | 160.0 | 170.7 | 238 |
| ojas_per_head_gate_dbias | 0.2 | 56.1 | 58.0 | 4 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 110.1 | 115.0 | 116 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 44.5 | 46.5 | 288 |
| four standalone ojas_check_finite passes | 25.2 | 76.3 | 87.5 | 331 |
| whole backward, one command buffer | 88.1 | 548.6 | 568.0 | 161 |
end: GPU 95%, load 4.20 5.75 9.25
