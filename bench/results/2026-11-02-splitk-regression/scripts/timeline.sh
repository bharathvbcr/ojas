#!/bin/bash
# Prints the bisect timeline: every ojas first-parent commit after 2b14dca up to
# 286a444, oldest first, each with the last tessl first-parent commit committed
# at or before it (the co-dated tessl). Columns: index ojas tessl ojas_date subject
set -eu
OJ=/Users/bharath/Code/research/ojas
TS=/Users/bharath/Code/research/tessl
git -C "$TS" log --first-parent --format='%ct %h' main > "$OJ/target-splitk/tessl-fp.txt"
i=0
git -C "$OJ" log --first-parent --reverse --format='%ct %h %s' 2b14dca^..286a444 |
while read -r ct h subj; do
  t=$(awk -v c="$ct" '$1 <= c {print $2; exit}' "$OJ/target-splitk/tessl-fp.txt")
  printf '%d\t%s\t%s\t%s\t%s\n' "$i" "$h" "$t" "$(date -r "$ct" '+%m-%d %H:%M')" "$subj"
  i=$((i + 1))
done
