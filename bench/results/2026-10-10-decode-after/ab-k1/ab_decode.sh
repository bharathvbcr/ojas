#!/bin/bash
# Interleaved A/B of two decode_vs_torch binaries on the GPU lanes.
#   ab_decode.sh BIN_BEFORE BIN_AFTER REFDIR ROUNDS OUT
set -euo pipefail
A=$1; B=$2; REF=$3; ROUNDS=$4; OUT=$5
mkdir -p "$OUT"
export BENCH_ROWS=${BENCH_ROWS:-gen_prefill_p32,gen_greedy_p32,gen_topk50} OJAS_BENCH_REF="$REF"
for r in $(seq 1 "$ROUNDS"); do
    mkdir -p "$OUT/round$r"
    if [ $((r % 2)) -eq 1 ]; then order="before after"; else order="after before"; fi
    for lane in metal wgpu; do
        for side in $order; do
            bin=$A; [ "$side" = after ] && bin=$B
            printf '{"round":%s,"lane":"%s","side":"%s","time":"%s","load":"%s"}\n' "$r" "$lane" "$side" \
                "$(date +%H:%M:%S)" "$(uptime | sed 's/.*load averages*: //')" >> "$OUT/load.jsonl"
            OJAS_BENCH_OUT="$OUT/round$r/$lane-$side.jsonl" "$bin" "$lane" 2>>"$OUT/round$r/$lane-$side.log"
        done
    done
    echo "round $r/$ROUNDS done ($order)" >&2
done
