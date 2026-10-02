device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 19.11 13.32 11.55
GPU timestamp: 41.8918 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 83.2 | 84.5 | 302 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 55.7 | 57.1 | 230 |
| ojas_per_head_gate_bwd | 38.1 | 155.3 | 161.4 | 246 |
| ojas_per_head_gate_dbias | 0.2 | 123.8 | 125.2 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 113.1 | 114.7 | 113 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 46.0 | 46.9 | 278 |
| four standalone ojas_check_finite passes | 25.2 | 75.7 | 78.8 | 334 |
| whole backward, one command buffer | 88.1 | 601.7 | 613.7 | 146 |
end: GPU 94%, load 19.11 13.32 11.55
