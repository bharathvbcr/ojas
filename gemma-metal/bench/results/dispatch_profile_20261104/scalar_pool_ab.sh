#!/bin/bash
# Paired A/B of IcbScalarPool on the E4B-shaped synthetic model: the stable GPU pool
# (GEMMA_METAL_STABLE_SCALARS=1, every push maps the pool: one commit and wait per dispatch)
# against the constant-arena path (default). Pipelined decode tok/s, 3 rounds, arms interleaved,
# each a fresh process with 3 x 64-step runs. Timing only: the weights are random.
#   bash scalar_pool_ab.sh   (writes scalar_pool_ab.log next to this script)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CRATE=/Users/bharath/Code/research/ojas/gemma-metal
MH=/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh
BIN="$CRATE/target-stamps/release/dispatch_profile"
export GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0
[ -x "$BIN" ] || { echo "build dispatch_profile into target-stamps first" >&2; exit 3; }
{
  echo "binary $(shasum -a 256 "$BIN" | cut -c1-16)"
  for r in 1 2 3; do
    for arm in stable arena; do
      s=0; [ "$arm" = stable ] && s=1
      echo "== round $r arm $arm load $(sysctl -n vm.loadavg)"
      GEMMA_METAL_STABLE_SCALARS=$s "$MH" "scalar-pool-ab-$arm" "$BIN" --synthetic-e4b \
        --out "$HERE/synthetic_e4b/ab" --pipelined-only --pipelined-steps 64 | rg "pipelined run"
    done
  done
} 2>&1 | tee "$HERE/scalar_pool_ab.log"
