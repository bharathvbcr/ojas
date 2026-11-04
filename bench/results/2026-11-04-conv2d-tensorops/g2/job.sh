#!/bin/bash
# G2 (and, with N=300 OUT=<dir>, G3) of RULES.md: held-out HOTA, first N frames (default 32), for the
# Ojas conv route in SpikeTrack.
# usage: Lappi-decision/tools/mac_heavy.sh conv-g2 bash <this file> [CONFIG ...]   (base conv32 conv16)
#        N=300 OUT=../g3 Lappi-decision/tools/mac_heavy.sh conv-g3 bash <this file>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=/Users/bharath/Code/research/SpikeTrack
BIN=$M/target-convtops/release/spiketrack
N=${N:-32}
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
VISEM=$M/data/visem/VISEM_Tracking_Train_v4/Train
SPLIT=$M/tools/reference/visem_split.json
(cd "$M" && CARGO_TARGET_DIR=$M/target-convtops cargo build --release -j 2 -p spiketrack 2>&1 | tail -1)
[ -x "$BIN" ] || { echo "build failed: no binary" >&2; exit 3; }
ROOT=${OUT:-$HERE}
mkdir -p "$ROOT"
echo "binary $(shasum -a 256 "$BIN" | cut -c1-16), N=$N" > "$ROOT/provenance.txt"
for cfg in "${@:-base conv32 conv16}"; do
  for c in $cfg; do
    case $c in
      base) mode="" ;;
      conv32) mode=f32 ;;
      conv16) mode=f16 ;;
      *) echo "unknown config $c" >&2; exit 2 ;;
    esac
    OUT=$ROOT/$c
    rm -rf "$OUT/run"
    mkdir -p "$OUT/logs"
    for id in 21 22 36 47 82; do
      OJAS_METAL_CONV2D_TENSOROPS=$mode "$BIN" track "$VISEM/$id/images" --prompt sperm --backend metal \
        --max-frames $N --boxes detector --max-objects 64 --out "$OUT/run" \
        --lora "$M/fixtures/lora_runs/visem-lora-v3/adapter.safetensors" \
        --weight-edit "$M/bench/results/weight-space/lorav3_exact_it_3p00.json" > "$OUT/logs/$id.log" 2>&1
      echo "== $c video $id exit $? load $(sysctl -n vm.loadavg)"
    done
    "$BIN" eval-visem --pred "$OUT/run" --split "$SPLIT" --set held_out --max-frames $N --visem "$VISEM" \
      --out "$OUT/eval.json" > "$OUT/eval.txt" 2>&1
    echo "== $c eval exit $?"
  done
done
