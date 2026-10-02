device: Apple M5 Pro, 4 timed runs
start: GPU 100%, load 19.19 30.37 38.35; slow = op + sync > 1 ms
GPU timestamp: 41.8661 ns per tick (upper bound, see docs)
| scenario | runs | min ms | p10 | p50 | p90 | slow | fast: op / sync / wait / gpu ms | slow: op / sync / wait / gpu ms | per run: commits / residency flushes / cold allocs |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tessl: scale 1 + synchronize | 100 | 0.091 | 0.105 | 0.146 | 0.227 | 0/100 | 0.008 / 0.149 / 0.131 / 0.013 | - | 1.00 / 0.00 / 0.00 |
| tessl: same, 5 ms idle before | 100 | 0.166 | 0.190 | 0.229 | 0.322 | 1/100 | 0.038 / 0.210 / 0.167 / 0.021 | 0.040 / 1.113 / 1.064 / 0.022 | 1.00 / 0.00 / 0.00 |
| tessl: fresh 4 B buffer + scale + synchronize | 100 | 0.096 | 0.106 | 0.120 | 0.173 | 0/100 | 0.008 / 0.130 / 0.106 / 0.016 | - | 1.00 / 2.00 / 1.00 |
| backend: sync alone | 100 | 0.002 | 0.003 | 0.004 | 0.005 | 0/100 | 0.000 / 0.004 / 0.000 / - | - | 0.00 / 0.00 / 0.00 |
| backend: silu 1 + sync | 100 | 0.107 | 0.117 | 0.176 | 0.330 | 0/100 | 0.030 / 0.166 / 0.129 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 1 ms idle before | 100 | 0.164 | 0.195 | 0.241 | 0.405 | 0/100 | 0.060 / 0.210 / 0.158 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 5 ms idle before | 100 | 0.196 | 0.227 | 0.280 | 0.424 | 0/100 | 0.094 / 0.218 / 0.154 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 20 ms idle before | 100 | 0.215 | 0.242 | 0.299 | 0.473 | 0/100 | 0.090 / 0.239 / 0.162 / - | - | 1.00 / 2.00 / 1.00 |
| backend: 8 x silu 1 + sync | 100 | 0.171 | 0.183 | 0.252 | 0.418 | 3/100 | 0.089 / 0.196 / 0.163 / - | 1.076 / 0.295 / 0.228 / - | 1.00 / 9.00 / 8.00 |
end: GPU 100%, load 18.69 30.08 38.20
