#!/usr/bin/env bash
# Sourced by fusion_ab.sh. An exactness gate that compares two arms' greedy
# streams proves nothing when the baseline itself has collapsed into a repeat
# (31B free decode without hidden capture does: fusion_ab_31b_20260719T080223Z
# recorded 3 distinct ids in 16, id 537 ten times, and both arms "PASS"ed).

# stream_degenerate "<[a, b, c, ...]>": exit 0 and print the reason when the
# stream has fewer distinct ids than half its length, or one id fills more
# than half of it; exit 1 otherwise. An empty or unparsable stream counts as
# degenerate, so a missing stream can never read as a healthy one.
stream_degenerate() {
  local ids n distinct top
  ids="$(printf '%s' "$1" | tr -d '[] ' | tr ',' '\n' | grep -E '^[0-9]+$' || true)"
  n="$(printf '%s\n' "$ids" | grep -c . || true)"
  if [ "$n" -eq 0 ]; then
    echo "no token ids parsed from: $1"
    return 0
  fi
  distinct="$(printf '%s\n' "$ids" | sort -u | grep -c .)"
  top="$(printf '%s\n' "$ids" | sort | uniq -c | sort -rn | awk 'NR==1 {print $1}')"
  if [ $((distinct * 2)) -lt "$n" ] || [ $((top * 2)) -gt "$n" ]; then
    echo "$distinct distinct ids in $n, most frequent id $top times"
    return 0
  fi
  return 1
}
