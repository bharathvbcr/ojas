#!/bin/bash
# Speed probe, not a gate: SpikeTrack's default frame (E32 placements) with Ojas's 3x3 convolutions
# on tessl::conv (OJAS_METAL_CONV2D_TENSOROPS=f16 or f32) against the im2col route (unset).
# Held-out VISEM 36, first 32 frames, 3 rounds, arm order rotated, as SpikeTrack's E36 speed job.
# One binary, built from the SpikeTrack main tree into its own target dir (target-convtops), so the
# binary other sessions run is not touched. An arm whose start or end load is 8 or more is "not read".
# usage: Lappi-decision/tools/mac_heavy.sh conv-tops-frame bash <this file>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=/Users/bharath/Code/research/SpikeTrack
OUT=$HERE/spiketrack
mkdir -p "$OUT"
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VID=36
FRAMES=32
(cd "$M" && CARGO_TARGET_DIR=$M/target-convtops cargo build --release -j 2 -p spiketrack 2>&1 | tail -1)
[ -x "$M/target-convtops/release/spiketrack" ] || { echo "build failed: no binary" >&2; exit 3; }
BIN=$M/target-convtops/release/spiketrack
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SpikeTrack $(git -C "$M" log --oneline -1), binary $(shasum -a 256 "$BIN" | cut -c1-16)"
  echo "ojas $(git -C "$M/../ojas" log --oneline -1) + the uncommitted Conv2dTensorOps route"
  echo "tessl $(git -C "$M/../tessl" log --oneline -1) + the uncommitted tessl::conv"
} > "$OUT/versions.txt"
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
arm() { # name round
  local w=0
  until awk -v l="$(load1)" 'BEGIN { exit !(l < 7) }' || [ "$w" -ge 600 ]; do sleep 20; w=$((w + 20)); done
  local start; start=$(load1)
  local d="$OUT/$1-r$2"
  [ -e "$d" ] && { echo "$d exists" >&2; exit 2; }
  local mode=""
  case $1 in
    base) mode="" ;;
    f16) mode=f16 ;;
    f32) mode=f32 ;;
  esac
  OJAS_METAL_CONV2D_TENSOROPS=$mode "$BIN" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/$VID/images" \
    --prompt sperm --backend metal --max-frames "$FRAMES" --boxes detector --max-objects 64 \
    --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
    --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" \
    --out "$d" > "$OUT/$1-r$2.log" 2>&1
  local s=$? end; end=$(load1)
  find "$d/masks" -type f -delete 2>/dev/null
  rmdir "$d/masks" 2>/dev/null
  local read=read
  awk -v a="$start" -v b="$end" 'BEGIN { exit !(a < 8 && b < 8) }' || read="NOT READ"
  echo "== $1 r$2 exit $s load start $start end $end: $read"
}
ARMS=(base f16 f32)
for r in 1 2 3; do
  n=${#ARMS[@]}; for ((i = 0; i < n; i++)); do arm "${ARMS[$(( (i + r - 1) % n ))]}" $r; done
done
python3 -I "$M/tools/analysis/e25_frame.py" "$OUT" "$VID" "${ARMS[@]}"
