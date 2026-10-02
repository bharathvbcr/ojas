device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 2.86 10.99 17.71
GPU timestamp: 41.9260 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 80.7 | 84.0 | 312 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 53.1 | 55.0 | 241 |
| ojas_per_head_gate_bwd | 38.1 | 150.9 | 158.3 | 253 |
| ojas_per_head_gate_dbias | 0.2 | 802.8 | 838.3 | 0 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.7 | 114.8 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 218.9 | 224.0 | 59 |
| ten ojas_check_finite passes | 63.2 | 230.2 | 235.2 | 274 |
| whole backward, one command buffer | 88.1 | 1617.7 | 1628.3 | 54 |
end: GPU 94%, load 2.86 10.99 17.71
