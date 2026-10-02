device: Apple M5 Pro, 4 timed runs
start: GPU 100%, load 19.50 29.88 38.03; slow = op + sync > 1 ms
GPU timestamp: 41.9216 ns per tick (upper bound, see docs)
| scenario | runs | min ms | p10 | p50 | p90 | slow | fast: op / sync / wait / gpu ms | slow: op / sync / wait / gpu ms | per run: commits / residency flushes / cold allocs |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tessl: scale 1 + synchronize | 100 | 0.096 | 0.107 | 0.135 | 0.196 | 0/100 | 0.007 / 0.146 / 0.126 / 0.010 | - | 1.00 / 0.00 / 0.00 |
| tessl: same, 5 ms idle before | 100 | 0.159 | 0.186 | 0.217 | 0.306 | 0/100 | 0.039 / 0.200 / 0.154 / 0.017 | - | 1.00 / 0.00 / 0.00 |
| tessl: fresh 4 B buffer + scale + synchronize | 100 | 0.107 | 0.149 | 0.176 | 0.264 | 0/100 | 0.011 / 0.189 / 0.164 / 0.019 | - | 1.00 / 2.00 / 1.00 |
| backend: sync alone | 100 | 0.002 | 0.002 | 0.003 | 0.004 | 0/100 | 0.000 / 0.004 / 0.000 / - | - | 0.00 / 0.00 / 0.00 |
| backend: silu 1 + sync | 100 | 0.101 | 0.120 | 0.180 | 0.245 | 0/100 | 0.018 / 0.166 / 0.136 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 1 ms idle before | 100 | 0.134 | 0.184 | 0.243 | 0.385 | 0/100 | 0.059 / 0.207 / 0.157 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 5 ms idle before | 100 | 0.182 | 0.226 | 0.285 | 0.475 | 1/100 | 0.081 / 0.238 / 0.179 / - | 0.070 / 8.717 / 3.895 / - | 1.00 / 2.00 / 1.00 |
| backend: same, 20 ms idle before | 100 | 0.207 | 0.245 | 0.321 | 0.482 | 0/100 | 0.101 / 0.250 / 0.178 / - | - | 1.00 / 2.00 / 1.00 |
| backend: 8 x silu 1 + sync | 100 | 0.171 | 0.193 | 0.249 | 0.514 | 1/100 | 0.087 / 0.211 / 0.176 / - | 1.007 / 0.371 / 0.258 / - | 1.00 / 9.00 / 8.00 |
end: GPU 100%, load 19.06 29.62 37.89
