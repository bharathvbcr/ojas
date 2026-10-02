device: Apple M5 Pro, 40 timed runs
start: GPU 100%, load 19.23 25.59 39.06
GPU timestamp: 41.9090 ns per tick (upper bound, see docs)
| kernel or sequence | elements | f32 passes | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|---:|
| tessl scale_f32_inplace (reference read + write) | 8388608 | 2 | 240.1 | 261.1 | 280 |
| ojas_check_finite, 1 per thread | 8388608 | 1 | 217.3 | 224.1 | 154 |
| ojas_check_finite, 2 per thread | 8388608 | 1 | 118.7 | 140.1 | 283 |
| ojas_check_finite, 4 per thread | 8388608 | 1 | 125.9 | 149.4 | 267 |
| ojas_check_finite, 8 per thread | 8388608 | 1 | 118.1 | 132.2 | 284 |
| ojas_check_finite, 16 per thread | 8388608 | 1 | 121.6 | 136.2 | 276 |
| ojas_check_finite, 32 per thread | 8388608 | 1 | 124.2 | 144.7 | 270 |
| ojas_check_finite, 64 per thread | 8388608 | 1 | 119.0 | 144.0 | 282 |
| ojas_check_finite [4096, 768] | 3145728 | 1 | 28.8 | 37.1 | 436 |
| ojas_silu_fwd (checks inside) | 8388608 | 2 | 250.6 | 279.7 | 268 |
| ojas_silu_bwd (checks inside) | 8388608 | 3 | 409.5 | 435.6 | 246 |
| ojas_mul_fwd (checks inside) | 8388608 | 3 | 419.8 | 450.5 | 240 |
| ojas_mul_bwd (checks inside) | 8388608 | 5 | 673.4 | 730.5 | 249 |
| ojas_add_fwd (checks inside) | 8388608 | 3 | 402.8 | 434.8 | 250 |
| ojas_add_bwd (checks inside) | 8388608 | 5 | 651.9 | 718.2 | 257 |
end: GPU 100%, load 19.23 25.59 39.06
