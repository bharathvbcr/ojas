device: Apple M5 Pro, 40 timed runs
start: GPU 45%, load 4.20 5.75 9.25
GPU timestamp: 41.8683 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.3 | 84.0 | 318 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 53.6 | 55.1 | 239 |
| ojas_per_head_gate_bwd | 38.1 | 129.8 | 170.2 | 294 |
| ojas_per_head_gate_dbias | 0.2 | 122.9 | 125.2 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 113.5 | 114.6 | 113 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 45.4 | 46.4 | 282 |
| four standalone ojas_check_finite passes | 25.2 | 77.7 | 84.8 | 325 |
| whole backward, one command buffer | 88.1 | 620.6 | 636.8 | 142 |
end: GPU 94%, load 4.20 5.75 9.25
