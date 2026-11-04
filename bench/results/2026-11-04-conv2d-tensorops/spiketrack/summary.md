# Whole-frame ms, VISEM 36, steady frames >= 8: push_frame median / wall per frame

| round | base | f16 | f32 | f16 / base (push, wall) | f32 / base (push, wall) |
|---|---|---|---|---|---|
| 1 | 885 / 1059 | 885 / 972 | 880 / 971 | 1.000, 0.918 | 0.994, 0.917 |
| 2 | 895 / 997 | 870 / 949 | 875 / 960 | 0.972, 0.952 | 0.978, 0.963 |
| 3 | 895 / 992 | 860 / 924 | 875 / 943 | 0.961, 0.931 | 0.978, 0.950 |

Median (and worst) over rounds of the per-round ratio: f16 push 0.972 (max 1.000), wall 0.931 (max 0.952) over 3 round(s); f32 push 0.978 (max 0.994), wall 0.950 (max 0.963) over 3 round(s)

## Tracks against base (same round)

| round | arm | frames with equal id sets | mean box IoU over shared ids | min box IoU | byte-identical predictions |
|---|---|---|---|---|---|
| 1 | f16 | 32/32 | 1.0000 | 0.9966 | no |
| 1 | f32 | 32/32 | 1.0000 | 0.9921 | no |
| 2 | f16 | 32/32 | 1.0000 | 0.9966 | no |
| 2 | f32 | 32/32 | 1.0000 | 0.9921 | no |
| 3 | f16 | 32/32 | 1.0000 | 0.9966 | no |
| 3 | f32 | 32/32 | 1.0000 | 0.9921 | no |
