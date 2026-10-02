| scenario | side | slow runs | min ms | p50 median of rounds [range] | p90 median of rounds [range] |
|---|---|---:|---:|---|---|
| tessl: scale 1 + synchronize | old | 388/600 | 0.09 | 1.413 [1.361-1.429] | 1.523 [1.481-1.543] |
| tessl: scale 1 + synchronize | new | 0/600 | 0.086 | 0.144 [0.135-0.153] | 0.211 [0.192-0.234] |
| tessl: same, 5 ms idle before | old | 507/600 | 0.17 | 1.468 [1.462-1.47] | 1.558 [1.532-1.584] |
| tessl: same, 5 ms idle before | new | 6/600 | 0.147 | 0.232 [0.217-0.242] | 0.335 [0.306-0.393] |
| tessl: fresh 4 B buffer | old | 427/600 | 0.102 | 1.398 [0.248-1.434] | 1.517 [1.475-1.589] |
| tessl: fresh 4 B buffer | new | 0/600 | 0.096 | 0.127 [0.114-0.176] | 0.182 [0.136-0.264] |
| backend: silu 1 + sync | old | 404/600 | 0.097 | 1.427 [0.199-1.456] | 1.538 [1.476-1.632] |
| backend: silu 1 + sync | new | 0/600 | 0.101 | 0.165 [0.122-0.196] | 0.248 [0.146-0.33] |
| backend: same, 1 ms idle before | old | 421/600 | 0.126 | 1.460 [0.325-1.469] | 1.582 [1.522-1.637] |
| backend: same, 1 ms idle before | new | 2/600 | 0.122 | 0.242 [0.219-0.269] | 0.386 [0.301-0.423] |
| backend: same, 5 ms idle before | old | 396/600 | 0.208 | 1.488 [1.464-1.504] | 1.609 [1.585-1.636] |
| backend: same, 5 ms idle before | new | 4/600 | 0.182 | 0.279 [0.258-0.289] | 0.439 [0.373-0.533] |
| backend: same, 20 ms idle before | old | 390/600 | 0.228 | 1.510 [1.485-1.517] | 1.629 [1.6-1.723] |
| backend: same, 20 ms idle before | new | 2/600 | 0.206 | 0.305 [0.294-0.321] | 0.477 [0.443-0.538] |
| backend: 8 x silu 1 + sync | old | 480/600 | 0.207 | 1.500 [1.476-1.551] | 1.625 [1.547-1.766] |
| backend: 8 x silu 1 + sync | new | 4/600 | 0.165 | 0.247 [0.191-0.252] | 0.379 [0.228-0.514] |

load at each start:
new-round1.md: load 19.64 30.65 38.49
new-round2.md: load 19.19 30.37 38.35
new-round3.md: load 19.50 29.88 38.03
new-round4.md: load 19.06 29.62 37.89
new-round5.md: load 18.56 28.98 37.52
new-round6.md: load 18.36 28.77 37.40
old-round1.md: load 20.31 31.14 38.76
old-round2.md: load 18.69 30.08 38.20
old-round3.md: load 19.28 30.01 38.13
old-round4.md: load 18.73 29.37 37.76
old-round5.md: load 18.35 29.12 37.62
old-round6.md: load 18.25 28.57 37.28
summary.md: load 19.64 30.65 38.49
