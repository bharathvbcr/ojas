device: Apple M5 Pro, 4 timed runs
start: GPU 100%, load 19.06 29.62 37.89; slow = op + sync > 1 ms
GPU timestamp: 41.8935 ns per tick (upper bound, see docs)
| scenario | runs | min ms | p10 | p50 | p90 | slow | fast: op / sync / wait / gpu ms | slow: op / sync / wait / gpu ms | per run: commits / residency flushes / cold allocs |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| tessl: scale 1 + synchronize | 100 | 0.092 | 0.111 | 0.135 | 0.218 | 0/100 | 0.007 / 0.145 / 0.125 / 0.012 | - | 1.00 / 0.00 / 0.00 |
| tessl: same, 5 ms idle before | 100 | 0.157 | 0.188 | 0.242 | 0.393 | 3/100 | 0.040 / 0.222 / 0.181 / 0.018 | 0.044 / 1.735 / 1.300 / 0.013 | 1.00 / 0.00 / 0.00 |
| tessl: fresh 4 B buffer + scale + synchronize | 100 | 0.100 | 0.107 | 0.114 | 0.136 | 0/100 | 0.007 / 0.118 / 0.100 / 0.020 | - | 1.00 / 2.00 / 1.00 |
| backend: sync alone | 100 | 0.002 | 0.002 | 0.003 | 0.012 | 0/100 | 0.000 / 0.005 / 0.000 / - | - | 0.00 / 0.00 / 0.00 |
| backend: silu 1 + sync | 100 | 0.103 | 0.117 | 0.122 | 0.146 | 0/100 | 0.013 / 0.121 / 0.096 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 1 ms idle before | 100 | 0.133 | 0.195 | 0.244 | 0.387 | 0/100 | 0.058 / 0.220 / 0.163 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 5 ms idle before | 100 | 0.182 | 0.229 | 0.289 | 0.453 | 0/100 | 0.083 / 0.232 / 0.172 / - | - | 1.00 / 2.00 / 1.00 |
| backend: same, 20 ms idle before | 100 | 0.222 | 0.251 | 0.294 | 0.490 | 0/100 | 0.095 / 0.235 / 0.167 / - | - | 1.00 / 2.00 / 1.00 |
| backend: 8 x silu 1 + sync | 100 | 0.174 | 0.182 | 0.191 | 0.228 | 0/100 | 0.063 / 0.151 / 0.127 / - | - | 1.00 / 9.00 / 8.00 |
end: GPU 100%, load 18.73 29.37 37.76
