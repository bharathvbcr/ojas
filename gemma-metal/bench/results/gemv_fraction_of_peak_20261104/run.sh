#!/bin/bash
# Interleaved: tessl probe, MLX probe, three times, inside one mac_heavy hold.
set -uo pipefail
S=$(dirname "$0")
for r in 1 2 3; do
  echo "== round $r tessl $(date -u +%T) load $(sysctl -n vm.loadavg)"
  "$S/gemv" "$S/gemv.metallib" 5 || echo "== tessl probe exit $?"
  echo "== round $r mlx $(date -u +%T) load $(sysctl -n vm.loadavg)"
  python3 -I "$S/mlx_qmv.py" 5 || echo "== mlx probe exit $?"
done
