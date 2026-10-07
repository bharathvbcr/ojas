#!/bin/bash
# Per-round ratio of each variant's time to its case's reference in the same
# round (the public-API production time, or for raw accumulate cases the
# production accumulate kernel), then min / median / max over rounds. Rounds
# drift with GPU load, so a ratio taken within one round is steadier than a
# min-of-rounds per variant.
#   bash sweep_ratio.sh <log>...
set -u
awk '
  FNR == 1 { file++ }
  /^[a-z_0-9]+  M=/ {
    c = $1; ref = ""
    if (!(c in seen)) { seen[c] = 1; order[++n] = c; shape[c] = $2 " " $3 " " $4 }
    if ($0 ~ /production/) { ref = $6 + 0 }
    first = 1
    next
  }
  /^  [a-z]/ && $2 ~ /^[0-9.]+$/ {
    v = $1; t = $2 + 0
    if (ref == "" && first) { ref = t }
    first = 0
    if (!((c, v) in vseen)) { vseen[c, v] = 1; vord[c, ++nv[c]] = v }
    k = ++cnt[c, v]; r[c, v, k] = t / ref
  }
  END {
    for (i = 1; i <= n; i++) {
      c = order[i]
      printf "\n%s  %s\n", c, shape[c]
      for (j = 1; j <= nv[c]; j++) {
        v = vord[c, j]; m = cnt[c, v]
        # insertion sort of this variant ratios
        for (a = 1; a <= m; a++) s[a] = r[c, v, a]
        for (a = 2; a <= m; a++) { x = s[a]; b = a - 1; while (b >= 1 && s[b] > x) { s[b + 1] = s[b]; b-- } s[b + 1] = x }
        med = (m % 2) ? s[(m + 1) / 2] : (s[m / 2] + s[m / 2 + 1]) / 2
        printf "  %-40s n=%d  min %5.2f  med %5.2f  max %5.2f\n", v, m, s[1], med, s[m]
      }
    }
  }' "$@"
