device: Apple M5 Pro, 40 timed runs
start: GPU 55%, load 19.11 13.32 11.55
GPU timestamp: 41.9457 ns per tick (upper bound, see docs)
| dispatch or sequence | f32 MB moved | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|
| tessl scale_f32_inplace over attn (reference read + write) | 25.2 | 83.3 | 84.1 | 302 |
| nt pre = x · w^T (4096, 12, 768) | 12.8 | 54.8 | 55.7 | 234 |
| ojas_per_head_gate_bwd | 38.1 | 166.4 | 172.6 | 229 |
| ojas_per_head_gate_dbias | 0.2 | 120.6 | 125.3 | 2 |
| nn gx = d_pre · w (4096, 768, 12) | 12.8 | 109.8 | 114.7 | 117 |
| tn gw = d_pre^T · x (12, 768, 4096) | 12.8 | 231.5 | 238.1 | 55 |
| four standalone ojas_check_finite passes | 25.2 | 82.7 | 86.1 | 305 |
| whole backward, one command buffer | 88.1 | 796.6 | 810.5 | 111 |
end: GPU 94%, load 19.11 13.32 11.55
