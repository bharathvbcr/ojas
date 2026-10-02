#!/bin/bash
# Quiet slot: the paired gate rows against torch only (the binaries are
# already built; run_paired.sh rebuilds incrementally at -j 4).
set -uo pipefail
uptime
CARGO_BUILD_JOBS=4 BENCH_ROWS=gate OUT_DIR=/Users/bharath/Code/research/ojas/bench/results/2026-10-02-gate/paired-gate-quiet \
  bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 5 > /Users/bharath/Code/research/ojas/target-baseline/gate-ab/paired-quiet.log 2>&1
echo "paired exit=$?"
uptime
sed -n '/### ojas-metal vs torch-mps/,/### Ranked/p' /Users/bharath/Code/research/ojas/bench/results/2026-10-02-gate/paired-gate-quiet/summary.md
