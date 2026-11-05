#!/bin/bash
# gemma-metal against mlx_lm on the real E4B snapshot, interleaved per round, fresh process each,
# under mac_heavy. gemma-metal: dispatch_profile pipelined decode (STEPS tokens in one command
# buffer, 3 runs). mlx_lm: oracle/mlx_speed.py (stream_generate, STEPS tokens, 3 runs after one
# warm-up, mlx_lm's generation tok/s). Not like-for-like: mlx_lm hands each token to Python as it
# goes; the gemma-metal arm reads back once at the end.
#   bash ab_mlx.sh GEMMA_BIN [ROUNDS] [STEPS]   (writes ab_mlx_STEPS.log next to this script)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=/Users/bharath/Code/research/models/gemma-4-e4b-it-4bit
MH=/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh
PY=/Users/bharath/.venvs/ml/bin/python
BIN=$1 ROUNDS=${2:-3} STEPS=${3:-64}
export GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0
{
  echo "gemma-metal $(shasum -a 256 "$BIN" | cut -c1-16) $BIN"
  for r in $(seq 1 "$ROUNDS"); do
    echo "== round $r arm gemma-metal load $(sysctl -n vm.loadavg)"
    "$MH" "e4b-vs-mlx-gm" "$BIN" --model "$MODEL" --out "$HERE/ab_out" \
      --pipelined-only --pipelined-steps "$STEPS" | rg "pipelined run"
    echo "== round $r arm mlx_lm load $(sysctl -n vm.loadavg)"
    "$MH" "e4b-vs-mlx-mlx" "$PY" -I "$HERE/oracle/mlx_speed.py" "$MODEL" "$STEPS" 3 | rg "^run"
  done
} 2>&1 | tee "$HERE/ab_mlx_$STEPS.log"
rm -rf "$HERE/ab_out"
