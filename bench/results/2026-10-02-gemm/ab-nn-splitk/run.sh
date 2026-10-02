#!/bin/bash
# Interleaved A/B of NN split-K: old (panel walk only) and new (panel walk +
# NN split-K) metal_bench binaries, `gemm` group, alternating, ROUNDS each.
set -euo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/splitk-ab
rounds=${ROUNDS:-4}
iters=${ITERS:-8}
for r in $(seq 1 "$rounds"); do
  if [ $((r % 2)) -eq 1 ]; then order="old new"; else order="new old"; fi
  for side in $order; do
    "$dir/metal_bench-$side" "$iters" gemm > "$dir/$side-$r.md" 2>&1
    echo "round $r $side done"
  done
done
