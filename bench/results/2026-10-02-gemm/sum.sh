#!/bin/bash
# Min-of-rounds per row for old and new, and new/old time ratio (< 1 = faster).
set -euo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gemm-ab
awk -F'|' '
  FNR == 1 { side = (FILENAME ~ /\/old-/) ? "old" : "new" }
  /^\| (nt|nn|tn) / {
    row = $2; gsub(/^ +| +$/, "", row); ms = $6 + 0
    if (!(row in seen)) { seen[row] = 1; order[++n] = row }
    if (!((side, row) in best) || ms < best[side, row]) best[side, row] = ms
    cnt[side, row]++
  }
  END {
    printf "| row | old min ms | new min ms | new/old | runs old/new |\n|---|---:|---:|---:|---|\n"
    for (i = 1; i <= n; i++) {
      r = order[i]; o = best["old", r]; w = best["new", r]
      printf "| %s | %.3f | %.3f | %.2f | %d/%d |\n", r, o, w, w / o, cnt["old", r], cnt["new", r]
    }
  }' "$dir"/old-*.md "$dir"/new-*.md
grep -h "^start\|^end" "$dir"/old-*.md "$dir"/new-*.md | sort | uniq -c | head -20
