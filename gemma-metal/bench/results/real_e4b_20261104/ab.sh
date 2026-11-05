#!/bin/bash
# Paired A/B of two dispatch_profile binaries on the real E4B snapshot: pipelined decode tok/s,
# arms interleaved per round, each a fresh process with 3 x STEPS-step runs, under mac_heavy.
#   bash ab.sh LABEL BIN_A BIN_B [ROUNDS] [STEPS]   (writes ab_LABEL.log next to this script)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=/Users/bharath/Code/research/models/gemma-4-e4b-it-4bit
MH=/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh
LABEL=$1 A=$2 B=$3 ROUNDS=${4:-3} STEPS=${5:-64}
export GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0
{
  echo "A $(shasum -a 256 "$A" | cut -c1-16) $A"
  echo "B $(shasum -a 256 "$B" | cut -c1-16) $B"
  for r in $(seq 1 "$ROUNDS"); do
    for arm in A B; do
      bin=$A; [ "$arm" = B ] && bin=$B
      echo "== round $r arm $arm load $(sysctl -n vm.loadavg)"
      "$MH" "e4b-ab-$LABEL-$arm" "$bin" --model "$MODEL" --out "$HERE/ab_out" \
        --pipelined-only --pipelined-steps "$STEPS" | rg "pipelined run"
    done
  done
} 2>&1 | tee "$HERE/ab_$LABEL.log"
