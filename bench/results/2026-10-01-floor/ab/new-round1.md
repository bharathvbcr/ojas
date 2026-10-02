device: Apple M5 Pro, 4 timed runs
start: GPU 100%, load 19.64 30.65 38.49; slow = op + sync > 1 ms
GPU timestamp: 41.8757 ns per tick (upper bound, see docs)
| scenario | runs | min ms | p10 | p50 | p90 | slow | fast: op / sync / wait / gpu ms | slow: op / sync / wait / gpu ms | per run: commits / residency flushes / cold allocs |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tessl: scale 1 + synchronize | 100 | 0.098 | 0.115 | 0.153 | 0.234 | 0/100 | 0.011 / 0.169 / 0.138 / 0.014 | - | 1.00 / 0.00 / 0.00 |
| tessl: same, 5 ms idle before | 100 | 0.153 | 0.188 | 0.230 | 0.348 | 0/100 | 0.038 / 0.220 / 0.174 / 0.027 | - | 1.00 / 0.00 / 0.00 |
| tessl: fresh 4 B buffer + scale + synchronize | 100 | 0.111 | 0.126 | 0.173 | 0.199 | 0/100 | 0.008 / 0.169 / 0.148 / 0.018 | - | 1.00 / 2.00 / 1.00 |
| backend: sync alone | 100 | 0.002 | 0.002 | 0.014 | 0.021 | 0/100 | 0.000 / 0.015 / 0.000 / - | - | 0.00 / 0.00 / 0.00 |
| backend: silu 1 + sync | 100 | 0.118 | 0.124 | 0.137 | 0.202 | 0/100 | 0.017 / 0.145 / 0.111 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 1 ms idle before | 100 | 0.122 | 0.179 | 0.225 | 0.345 | 0/100 | 0.050 / 0.204 / 0.152 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 5 ms idle before | 100 | 0.193 | 0.224 | 0.258 | 0.414 | 1/100 | 0.087 / 0.203 / 0.143 / - | 0.080 / 1.469 / 1.393 / - | 1.00 / 2.00 / 1.00 |
| backend: same, 20 ms idle before | 100 | 0.227 | 0.249 | 0.306 | 0.471 | 1/100 | 0.104 / 0.236 / 0.166 / - | 0.540 / 0.636 / 0.550 / - | 1.00 / 2.00 / 1.00 |
| backend: 8 x silu 1 + sync | 100 | 0.176 | 0.191 | 0.245 | 0.400 | 0/100 | 0.073 / 0.203 / 0.174 / - | - | 1.00 / 9.00 / 8.00 |
end: GPU 100%, load 19.19 30.37 38.35
