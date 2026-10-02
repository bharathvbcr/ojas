device: Apple M5 Pro, 40 timed runs
start: GPU 45%, load 4.20 5.75 9.25
GPU timestamp: 41.8570 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 78.6 | 83.3 | 320 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 50.4 | 55.9 | 255 |
| ojas_per_head_gate_bwd | 38.1 | 147.1 | 171.3 | 259 |
| ojas_per_head_gate_dbias | 0.2 | 120.4 | 125.0 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 113.4 | 113.9 | 113 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 41.7 | 46.6 | 307 |
| four standalone ojas_check_finite passes | 25.2 | 78.1 | 86.8 | 323 |
| whole backward, one command buffer | 88.1 | 619.9 | 628.9 | 142 |
end: GPU 94%, load 4.20 5.75 9.25
