#!/bin/bash
# Interleaved A/B of the exact-f32 GEMM tile walk: old (row-major) and new
# (column panels) metal_bench binaries, alternating, ROUNDS each.
set -euo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gemm-ab
rounds=${ROUNDS:-4}
iters=${ITERS:-8}
for r in $(seq 1 "$rounds"); do
  for side in old new; do
    "$dir/metal_bench-$side" "$iters" gemm > "$dir/$side-$r.md" 2>&1
    echo "round $r $side done"
  done
done
