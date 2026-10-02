device: Apple M5 Pro, 40 timed runs
start: GPU 100%, load 58.87 57.75 43.08
GPU timestamp: 42.7454 ns per tick (upper bound, see docs)
| kernel or sequence | elements | f32 passes | min µs | median µs | GB/s at min |
|---|---:|---:|---:|---:|---:|
| tessl scale_f32_inplace (reference read + write) | 8388608 | 2 | 237.1 | 259.8 | 283 |
| ojas_check_finite | 8388608 | 1 | 222.3 | 236.5 | 151 |
| ojas_check_finite [4096, 768] | 3145728 | 1 | 84.5 | 92.0 | 149 |
| ojas_silu_fwd | 8388608 | 2 | 262.1 | 271.6 | 256 |
| silu_forward sequence: check x, silu, check y | 8388608 | 4 | 686.4 | 705.6 | 196 |
| ojas_silu_bwd | 8388608 | 3 | 389.3 | 415.7 | 259 |
| silu_backward sequence: check x, check gy, kernel, check out | 8388608 | 6 | 1013.0 | 1049.9 | 199 |
| ojas_mul_fwd | 8388608 | 3 | 387.1 | 402.4 | 260 |
| mul_forward sequence: check a, check b, mul, check y | 8388608 | 6 | 1012.6 | 1048.7 | 199 |
| ojas_mul_bwd | 8388608 | 5 | 639.4 | 700.9 | 262 |
| mul_backward sequence: 3 input checks, kernel, 2 output checks | 8388608 | 10 | 1674.3 | 1738.7 | 200 |
end: GPU 100%, load 58.32 57.66 43.14
