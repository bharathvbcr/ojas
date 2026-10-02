device: Apple M5 Pro, 40 timed runs
start: GPU 36%, load 4.20 5.75 9.25
GPU timestamp: 41.8810 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.9 | 83.7 | 319 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.2 | 55.6 | 261 |
| ojas_per_head_gate_bwd | 38.1 | 153.8 | 167.3 | 248 |
| ojas_per_head_gate_dbias | 0.2 | 53.4 | 57.9 | 4 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 111.9 | 113.9 | 114 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 44.0 | 45.7 | 291 |
| four standalone ojas_check_finite passes | 25.2 | 81.3 | 87.7 | 310 |
| whole backward, one command buffer | 88.1 | 554.3 | 568.9 | 159 |
end: GPU 94%, load 4.20 5.75 9.25
