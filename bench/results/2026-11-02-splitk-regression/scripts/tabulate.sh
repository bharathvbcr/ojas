#!/bin/bash
# tabulate.sh <log>: per binary and arm, the per-process medians in run order,
# the median of those medians, their range, and the largest within-process
# p90 - p10. Input lines: r<N> probe <label> <arm> p10 x median y p90 z ...
grep '^r[0-9]' "$1" | awk -F'\t' '
{
  split($5, a, " "); split($6, b, " "); split($7, c, " ");
  k = $3 "\t" $4
  n[k]++; med[k, n[k]] = b[2]
  s = c[2] - a[2]; if (s > spr[k]) spr[k] = s
  if (!(k in seen)) { seen[k] = 1; order[++no] = k }
}
END {
  printf "label\tarm\tn\tmedian_of_medians\tmin\tmax\tmax_p90-p10\tper-process medians\n"
  for (i = 1; i <= no; i++) {
    k = order[i]; m = n[k]
    delete v
    for (j = 1; j <= m; j++) v[j] = med[k, j]
    list = ""
    for (j = 1; j <= m; j++) list = list sprintf("%.1f ", v[j])
    # insertion sort
    for (x = 2; x <= m; x++) { t = v[x]; y = x - 1; while (y > 0 && v[y] > t) { v[y+1] = v[y]; y-- } v[y+1] = t }
    mid = (m % 2) ? v[(m+1)/2] : (v[m/2] + v[m/2+1]) / 2
    printf "%s\t%d\t%.1f\t%.1f\t%.1f\t%.1f\t%s\n", k, m, mid, v[1], v[m], spr[k], list
  }
}' | sort -t$'\t' -k2,2 -k1,1
