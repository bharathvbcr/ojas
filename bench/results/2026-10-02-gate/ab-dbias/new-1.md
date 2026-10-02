device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 3.49 8.02 9.10
GPU timestamp: 41.8862 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 80.0 | 85.1 | 315 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 50.3 | 55.7 | 255 |
| ojas_per_head_gate_bwd | 38.1 | 146.6 | 154.5 | 260 |
| ojas_per_head_gate_dbias | 0.2 | 120.0 | 124.8 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.1 | 114.3 | 118 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 217.2 | 223.5 | 59 |
| ten ojas_check_finite passes | 63.2 | 232.1 | 235.6 | 272 |
| whole backward, one command buffer | 88.1 | 940.1 | 946.9 | 94 |
end: GPU 94%, load 3.49 8.02 9.10
