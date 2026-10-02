device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 19.11 13.32 11.55
GPU timestamp: 41.9437 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 82.1 | 83.8 | 306 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 54.6 | 56.5 | 235 |
| ojas_per_head_gate_bwd | 38.1 | 166.1 | 172.6 | 230 |
| ojas_per_head_gate_dbias | 0.2 | 123.6 | 125.0 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 112.6 | 114.3 | 114 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 234.1 | 239.8 | 55 |
| four standalone ojas_check_finite passes | 25.2 | 87.3 | 91.5 | 289 |
| whole backward, one command buffer | 88.1 | 806.4 | 815.9 | 109 |
end: GPU 96%, load 19.11 13.32 11.55
