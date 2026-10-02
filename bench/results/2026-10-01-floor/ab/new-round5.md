device: Apple M5 Pro, 4 timed runs
start: GPU 100%, load 18.56 28.98 37.52; slow = op + sync > 1 ms
GPU timestamp: 41.9336 ns per tick (upper bound, see docs)
| scenario | runs | min ms | p10 | p50 | p90 | slow | fast: op / sync / wait / gpu ms | slow: op / sync / wait / gpu ms | per run: commits / residency flushes / cold allocs |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tessl: scale 1 + synchronize | 100 | 0.086 | 0.101 | 0.150 | 0.192 | 0/100 | 0.010 / 0.142 / 0.125 / 0.010 | - | 1.00 / 0.00 / 0.00 |
| tessl: same, 5 ms idle before | 100 | 0.151 | 0.193 | 0.242 | 0.338 | 1/100 | 0.040 / 0.219 / 0.167 / 0.021 | 0.031 / 1.283 / 0.786 / 0.014 | 1.00 / 0.00 / 0.00 |
| tessl: fresh 4 B buffer + scale + synchronize | 100 | 0.100 | 0.109 | 0.121 | 0.159 | 0/100 | 0.008 / 0.132 / 0.110 / 0.016 | - | 1.00 / 2.00 / 1.00 |
| backend: sync alone | 100 | 0.002 | 0.002 | 0.003 | 0.004 | 0/100 | 0.000 / 0.003 / 0.000 / - | - | 0.00 / 0.00 / 0.00 |
| backend: silu 1 + sync | 100 | 0.110 | 0.124 | 0.155 | 0.251 | 0/100 | 0.022 / 0.162 / 0.118 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 1 ms idle before | 100 | 0.129 | 0.174 | 0.219 | 0.301 | 0/100 | 0.057 / 0.187 / 0.138 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 5 ms idle before | 100 | 0.189 | 0.224 | 0.270 | 0.533 | 2/100 | 0.083 / 0.229 / 0.164 / - | 0.775 / 1.585 / 1.489 / - | 1.00 / 2.00 / 1.00 |
| backend: same, 20 ms idle before | 100 | 0.206 | 0.246 | 0.307 | 0.538 | 1/100 | 0.105 / 0.243 / 0.169 / - | 1.181 / 0.216 / 0.164 / - | 1.00 / 2.00 / 1.00 |
| backend: 8 x silu 1 + sync | 100 | 0.165 | 0.193 | 0.252 | 0.357 | 0/100 | 0.080 / 0.197 / 0.168 / - | - | 1.00 / 9.00 / 8.00 |
end: GPU 100%, load 18.36 28.77 37.40
