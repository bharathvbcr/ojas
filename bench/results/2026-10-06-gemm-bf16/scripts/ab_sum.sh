#!/bin/bash
# Summarize an ab_prod.sh run: per case, the new/old time ratio of each
# back-to-back pair (same case, same round), then min / median / max over
# rounds, with each side's min time. A case's time is its public-API
# production line, or for the raw accumulate cases the production
# accumulate kernel (the only variant BENCH_PRODUCTION_ONLY keeps).
#   bash ab_sum.sh <tag>
set -u
DIR=/Users/bharath/Code/research/ojas/target-mathsrc/logs/${1:?tag}
for f in "$DIR"/*-r*.txt; do
  b=$(basename "$f" .txt)
  side=old
  case "$b" in *-new-r*) side=new ;; esac
  round=${b##*-r}
  label=${b%-*-r*}
  awk -v side="$side" -v round="$round" -v label="$label" '
    /^[a-z_0-9]+  M=/ {
      c = $1; t = ""
      if ($0 ~ /production/) { t = $6 + 0; print label, c, round, side, t; c = "" }
      next
    }
    c != "" && /^  matmul2d/ && $2 ~ /^[0-9.]+$/ { print label, c, round, side, $2 + 0; c = "" }
  ' "$f"
done | awk '
  { key = $1 SUBSEP $2 SUBSEP $3; t[key, $4] = $5; cases[$2] = 1; keys[key] = $2
    if (!(($2, $4) in best) || $5 < best[$2, $4]) best[$2, $4] = $5 }
  END {
    for (k in keys) {
      if (((k, "old") in t) && ((k, "new") in t)) {
        c = keys[k]; n[c]++; r[c, n[c]] = t[k, "new"] / t[k, "old"]
      }
    }
    printf "%-22s %4s %9s %9s %6s %6s %6s\n", "case", "n", "old min", "new min", "min", "med", "max"
    for (c in cases) {
      m = n[c]
      for (a = 1; a <= m; a++) s[a] = r[c, a]
      for (a = 2; a <= m; a++) { x = s[a]; b = a - 1; while (b >= 1 && s[b] > x) { s[b + 1] = s[b]; b-- } s[b + 1] = x }
      med = (m % 2) ? s[(m + 1) / 2] : (s[m / 2] + s[m / 2 + 1]) / 2
      printf "%-22s %4d %9.3f %9.3f %6.2f %6.2f %6.2f\n", c, m, best[c, "old"], best[c, "new"], s[1], med, s[m]
    }
  }' | sort
