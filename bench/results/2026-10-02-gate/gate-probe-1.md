device: Apple M5 Pro, 40 timed runs
start: GPU 53%, load 3.11 11.17 17.82
GPU timestamp: 41.9194 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 80.4 | 84.7 | 313 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 50.3 | 55.2 | 255 |
| ojas_per_head_gate_bwd | 38.1 | 153.8 | 158.2 | 248 |
| ojas_per_head_gate_dbias | 0.2 | 807.1 | 840.7 | 0 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 113.9 | 114.9 | 113 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 224.4 | 226.2 | 57 |
| ten ojas_check_finite passes | 63.2 | 231.0 | 235.5 | 274 |
| whole backward, one command buffer | 88.1 | 1636.3 | 1650.4 | 54 |
end: GPU 93%, load 3.11 11.17 17.82
