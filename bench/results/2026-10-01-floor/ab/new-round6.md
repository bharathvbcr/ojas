device: Apple M5 Pro, 4 timed runs
start: GPU 100%, load 18.36 28.77 37.40; slow = op + sync > 1 ms
GPU timestamp: 41.9490 ns per tick (upper bound, see docs)
| scenario | runs | min ms | p10 | p50 | p90 | slow | fast: op / sync / wait / gpu ms | slow: op / sync / wait / gpu ms | per run: commits / residency flushes / cold allocs |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tessl: scale 1 + synchronize | 100 | 0.103 | 0.118 | 0.142 | 0.203 | 0/100 | 0.007 / 0.148 / 0.124 / 0.014 | - | 1.00 / 0.00 / 0.00 |
| tessl: same, 5 ms idle before | 100 | 0.147 | 0.200 | 0.233 | 0.332 | 1/100 | 0.045 / 0.214 / 0.163 / 0.021 | 0.052 / 1.278 / 0.124 / 0.014 | 1.00 / 0.00 / 0.00 |
| tessl: fresh 4 B buffer + scale + synchronize | 100 | 0.103 | 0.111 | 0.132 | 0.191 | 0/100 | 0.015 / 0.141 / 0.114 / 0.017 | - | 1.00 / 2.00 / 1.00 |
| backend: sync alone | 100 | 0.002 | 0.002 | 0.005 | 0.010 | 0/100 | 0.000 / 0.009 / 0.000 / - | - | 0.00 / 0.00 / 0.00 |
| backend: silu 1 + sync | 100 | 0.109 | 0.133 | 0.196 | 0.286 | 0/100 | 0.026 / 0.187 / 0.144 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 1 ms idle before | 100 | 0.149 | 0.208 | 0.269 | 0.423 | 2/100 | 0.069 / 0.218 / 0.164 / - | 0.431 / 1.673 / 1.541 / - | 1.00 / 2.00 / 1.00 |
| backend: same, 5 ms idle before | 100 | 0.192 | 0.222 | 0.278 | 0.373 | 0/100 | 0.079 / 0.218 / 0.161 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 20 ms idle before | 100 | 0.215 | 0.246 | 0.305 | 0.443 | 0/100 | 0.098 / 0.230 / 0.158 / - | - | 1.00 / 2.00 / 1.00 |
| backend: 8 x silu 1 + sync | 100 | 0.172 | 0.184 | 0.198 | 0.314 | 0/100 | 0.072 / 0.163 / 0.137 / - | - | 1.00 / 9.00 / 8.00 |
end: GPU 100%, load 18.25 28.57 37.28
