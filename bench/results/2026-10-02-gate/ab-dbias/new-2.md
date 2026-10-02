device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 3.49 8.02 9.10
GPU timestamp: 41.9394 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.3 | 84.2 | 317 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 54.1 | 55.6 | 237 |
| ojas_per_head_gate_bwd | 38.1 | 142.3 | 148.4 | 268 |
| ojas_per_head_gate_dbias | 0.2 | 120.4 | 125.0 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.5 | 114.7 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 221.7 | 227.4 | 58 |
| ten ojas_check_finite passes | 63.2 | 231.7 | 237.3 | 273 |
| whole backward, one command buffer | 88.1 | 941.6 | 952.3 | 94 |
end: GPU 95%, load 3.49 8.02 9.10
