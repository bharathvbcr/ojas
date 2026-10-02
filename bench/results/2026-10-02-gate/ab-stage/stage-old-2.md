device: Apple M5 Pro, 40 timed runs
start: GPU 33%, load 4.20 5.75 9.25
GPU timestamp: 41.9160 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.9 | 83.7 | 319 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 50.6 | 55.0 | 254 |
| ojas_per_head_gate_bwd | 38.1 | 156.8 | 168.1 | 243 |
| ojas_per_head_gate_dbias | 0.2 | 123.4 | 124.9 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.5 | 114.1 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 44.0 | 45.5 | 291 |
| four standalone ojas_check_finite passes | 25.2 | 76.0 | 85.3 | 332 |
| whole backward, one command buffer | 88.1 | 621.4 | 632.0 | 142 |
end: GPU 93%, load 4.20 5.75 9.25
