#!/bin/bash
# The split-K metal_vs_torch on the rows whose NN GEMM routes to K partitions,
# 3 rounds, compared unpaired with bench/results/2026-10-02-gemm/ab-rows*.
set -euo pipefail
out=/Users/bharath/Code/research/ojas/bench/results/2026-10-02-gemm/splitk-rows
bin=/Users/bharath/Code/research/ojas/target-lane-bench/release/examples/metal_vs_torch
mkdir -p "$out"
export BENCH_ROWS=floor_silu_1,linear_lmhead_bwd,linear_ce_c4096x50304
export BENCH_ITERS=20 BENCH_WARMUP=5
{ date -u +%Y-%m-%dT%H:%M:%SZ; stat -f '%Sm %z' "$bin"; uptime; } > "$out/env.txt"
for r in 1 2 3; do
  OJAS_BENCH_REF=/Users/bharath/Code/research/ojas/bench/out/r2/ref OJAS_BENCH_OUT="$out/round$r.jsonl" "$bin" 2>> "$out/round$r.log"
  echo "round $r done"
done
jq -rs '[.[] | select(.status == "ok")] | group_by(.row) | .[] | "\(.[0].row) mins=\([.[] | .min_ms] | map(tostring) | join(",")) medians=\([.[] | .median_ms] | map(tostring) | join(",")) parity=\([.[] | .parity_max_abs] | max)"' "$out"/round*.jsonl
