device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 5.07 5.25 4.80
GPU timestamp: 41.9002 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 77.3 | 83.6 | 325 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 53.1 | 54.7 | 241 |
| ojas_per_head_gate_bwd | 38.1 | 140.2 | 164.9 | 272 |
| ojas_per_head_gate_dbias | 0.2 | 120.0 | 125.3 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.6 | 115.1 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 219.2 | 226.4 | 58 |
| ten ojas_check_finite passes | 63.2 | 230.5 | 240.5 | 274 |
| whole backward, one command buffer | 88.1 | 957.0 | 969.3 | 92 |
end: GPU 95%, load 5.07 5.25 4.80
