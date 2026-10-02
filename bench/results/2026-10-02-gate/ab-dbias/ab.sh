#!/bin/bash
# Interleaved A/B of the gate probe: old binary (serial dbias) vs new,
# alternating which side goes first, 4 rounds per side.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
old=$dir/metal_bench-old
new=/Users/bharath/Code/research/ojas/target-lane-bench/release/examples/metal_bench
for i in 1 2 3 4; do
  if [ $((i % 2)) -eq 1 ]; then
    "$old" 40 gate > "$dir/old-$i.md" 2>&1; "$new" 40 gate > "$dir/new-$i.md" 2>&1
  else
    "$new" 40 gate > "$dir/new-$i.md" 2>&1; "$old" 40 gate > "$dir/old-$i.md" 2>&1
  fi
done
for side in old new; do
  echo "== $side (min µs per round: dbias | whole backward)"
  for i in 1 2 3 4; do
    d=$(grep 'gate_dbias' "$dir/$side-$i.md" | awk -F'|' '{print $4}')
    w=$(grep 'whole backward' "$dir/$side-$i.md" | awk -F'|' '{print $4}')
    echo "round $i:$d |$w"
  done
done
