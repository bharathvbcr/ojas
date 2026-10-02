device: Apple M5 Pro, 40 timed runs
start: GPU 0%, load 3.49 8.02 9.10
GPU timestamp: 41.9286 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 79.4 | 80.4 | 317 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 49.4 | 54.7 | 259 |
| ojas_per_head_gate_bwd | 38.1 | 150.9 | 154.8 | 253 |
| ojas_per_head_gate_dbias | 0.2 | 806.5 | 839.2 | 0 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.4 | 114.6 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 211.7 | 217.1 | 61 |
| ten ojas_check_finite passes | 63.2 | 232.0 | 236.1 | 272 |
| whole backward, one command buffer | 88.1 | 1632.4 | 1646.1 | 54 |
end: GPU 95%, load 3.49 8.02 9.10
