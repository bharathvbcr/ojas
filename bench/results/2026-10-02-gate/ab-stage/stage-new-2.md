device: Apple M5 Pro, 40 timed runs
start: GPU 33%, load 4.20 5.75 9.25
GPU timestamp: 41.8941 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 82.7 | 84.0 | 304 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 52.3 | 55.1 | 245 |
| ojas_per_head_gate_bwd | 38.1 | 162.5 | 171.4 | 235 |
| ojas_per_head_gate_dbias | 0.2 | 54.3 | 58.2 | 4 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 112.3 | 113.9 | 114 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 44.4 | 46.7 | 288 |
| four standalone ojas_check_finite passes | 25.2 | 73.9 | 83.5 | 341 |
| whole backward, one command buffer | 88.1 | 561.2 | 581.7 | 157 |
end: GPU 92%, load 4.20 5.75 9.25
