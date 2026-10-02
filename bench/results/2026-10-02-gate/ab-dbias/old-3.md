device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 3.49 8.02 9.10
GPU timestamp: 41.9121 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.9 | 84.8 | 315 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 48.8 | 54.7 | 262 |
| ojas_per_head_gate_bwd | 38.1 | 145.0 | 150.5 | 263 |
| ojas_per_head_gate_dbias | 0.2 | 804.5 | 848.3 | 0 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.6 | 114.5 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 215.9 | 221.3 | 59 |
| ten ojas_check_finite passes | 63.2 | 226.3 | 236.8 | 279 |
| whole backward, one command buffer | 88.1 | 1631.5 | 1649.9 | 54 |
end: GPU 92%, load 3.53 7.95 9.07
