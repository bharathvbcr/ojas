#!/bin/bash
# clip_grad_norm_full on ojas-metal, interleaved across binaries.
#   clip_ab.sh OUT ROUNDS BIN [BIN...]
# Odd rounds run the binaries in the given order, even rounds reversed.
# Every row records uptime load and GPU utilisation before and after.
set -euo pipefail
OUT=$1; ROUNDS=$2; shift 2
BINS=("$@")
REPO=/Users/bharath/Code/research/ojas
PY=/Users/bharath/.venvs/ml-2026-10/bin/python
export BENCH_ROWS=clip_grad_norm_full BENCH_ITERS=${BENCH_ITERS:-40} BENCH_WARMUP=${BENCH_WARMUP:-10}
export PYTHONDONTWRITEBYTECODE=1 RIPGREP_CONFIG_PATH=
mkdir -p "$OUT"
if [ ! -d "$OUT/ref" ]; then
    "$PY" -I "$REPO/bench/torch_rows.py" ref --ref "$OUT/ref" 2>"$OUT/ref.log"
fi
snap() {
    local util
    util=$(ioreg -r -d 1 -w 0 -c IOAccelerator | rg -o '"Device Utilization %"=[0-9]+' | head -1 | rg -o '[0-9]+$' || true)
    printf '{"round":%s,"bin":"%s","when":"%s","load":"%s","gpu_util_pct":%s}\n' \
        "$1" "$2" "$3" "$(uptime | sed 's/.*load averages*: //')" "${util:-null}" >> "$OUT/load.jsonl"
}
for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then order=("${BINS[@]}"); else order=(); for ((i=${#BINS[@]}-1; i>=0; i--)); do order+=("${BINS[$i]}"); done; fi
    for b in "${order[@]}"; do
        name=$(basename "$b")
        snap "$r" "$name" before
        OJAS_BENCH_REF="$OUT/ref" OJAS_BENCH_OUT="$OUT/$name.r$r.jsonl" "$b" 2>>"$OUT/$name.log" \
            || echo "{\"round\":$r,\"bin\":\"$name\",\"status\":\"crash\"}" >> "$OUT/$name.r$r.jsonl"
        snap "$r" "$name" after
    done
done
echo "done: $OUT"
