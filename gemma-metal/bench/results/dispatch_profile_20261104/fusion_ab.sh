#!/bin/bash
# Paired A/B of gemma-metal's opt-in fusions on the E4B-shaped synthetic model, after the scalar-pool
# and split-KV decode fixes: base (defaults), GEMMA_METAL_FUSE_LAYER=1 (QKV in one GEMV, rms_qkv_rope
# + KV store, PLE lookup + residual), GEMMA_METAL_FUSE_GATE_DOWN=1 (bounded-TG gate->down), and both.
# Pipelined decode tok/s, 3 rounds with the arm order rotated, a fresh process per arm, 3 x 64 steps.
# Timing only: the weights are random.   bash fusion_ab.sh   (writes fusion_ab.log here)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CRATE=/Users/bharath/Code/research/ojas/gemma-metal
MH=/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh
BIN="$CRATE/target-stamps/release/dispatch_profile"
export GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0
[ -x "$BIN" ] || { echo "build dispatch_profile into target-stamps first" >&2; exit 3; }
ARMS=(base layer gatedown both)
{
  echo "binary $(shasum -a 256 "$BIN" | cut -c1-16)"
  for r in 1 2 3; do
    n=${#ARMS[@]}
    for ((i = 0; i < n; i++)); do
      arm=${ARMS[$(( (i + r - 1) % n ))]}
      l=0; g=0
      case $arm in layer) l=1 ;; gatedown) g=1 ;; both) l=1; g=1 ;; esac
      echo "== round $r arm $arm load $(sysctl -n vm.loadavg)"
      GEMMA_METAL_FUSE_LAYER=$l GEMMA_METAL_FUSE_GATE_DOWN=$g "$MH" "fusion-ab-$arm" "$BIN" \
        --synthetic-e4b --out "$HERE/synthetic_e4b/fusion" --pipelined-only --pipelined-steps 64 \
        | rg "pipelined run"
    done
  done
} 2>&1 | tee "$HERE/fusion_ab.log"
