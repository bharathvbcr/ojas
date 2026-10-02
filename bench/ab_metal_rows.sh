#!/bin/bash
# Interleaved A/B of two builds of the Metal bench (`metal_vs_torch`) on chosen rows.
# Used for the round 3 regression check in docs/bench-gpu-vs-torch.md.
#
#   OLD_SRC=<exported ojas tree> REF=<torch ref dir> bench/ab_metal_rows.sh [rounds]   (default 10)
#
# OLD_SRC is a checkout or `git archive` export of the older revision, with the sibling
# `../tessl` its manifests expect. It is built with its own harness into OLD_TARGET.
# OLD_BIN instead names an already-built old binary and skips that build (OLD_SRC is
# then not needed): the way to A/B a change in a path dependency such as tessl, which
# a rebuild would pick up on both sides.
# The new side is the binary run_paired.sh already built (NEW_BIN); this script does not
# rebuild it, so build it first. REF is a torch reference directory written by
# `torch_rows.py ref` (run_paired.sh leaves one at <OUT_DIR>/ref).
#
# Environment (optional): BENCH_ROWS, comma-separated row-name prefixes (default:
# floor_silu_1 control plus the four round 2 rows), BENCH_ITERS (50), BENCH_WARMUP (10),
# OUT_DIR, OLD_TARGET, NEW_BIN. SUMMARY_ONLY=1 skips the build and the runs and summarises
# the rounds already in OUT_DIR.
set -euo pipefail

REPO=$(dirname "$(dirname "$(realpath "$0")")")
ROUNDS=${1:-10}
[ -n "${OLD_BIN:-}" ] || : "${OLD_SRC:?OLD_SRC must name the older ojas tree (or OLD_BIN an old binary)}"
: "${REF:?REF must name a torch reference directory}"
OUT=${OUT_DIR:-$REPO/bench/out/ab-$(date +%Y%m%d-%H%M%S)}
OLD_TARGET=${OLD_TARGET:-$REPO/target-baseline/ab-target}
NEW_BIN=${NEW_BIN:-$REPO/target-lane-bench/release/examples/metal_vs_torch}
export RIPGREP_CONFIG_PATH=
export BENCH_ROWS=${BENCH_ROWS:-floor_silu_1,rope_bwd,permute_bthd_bhtd,residual_add_fwd,residual_add_bwd}
export BENCH_ITERS=${BENCH_ITERS:-50}
export BENCH_WARMUP=${BENCH_WARMUP:-10}

if [ -z "${SUMMARY_ONLY:-}" ]; then
if [ -z "${OLD_BIN:-}" ]; then
    CARGO_TARGET_DIR=$OLD_TARGET cargo build --release \
        --manifest-path "$OLD_SRC/Cargo.toml" -p ojas-metal --example metal_vs_torch 2>&1 | tail -2
    OLD_BIN=$OLD_TARGET/release/examples/metal_vs_torch
fi
[ -x "$OLD_BIN" ] || { echo "OLD_BIN $OLD_BIN is missing" >&2; exit 1; }
[ -x "$NEW_BIN" ] || { echo "NEW_BIN $NEW_BIN is missing: run bench/run_paired.sh first" >&2; exit 1; }
mkdir -p "$OUT/old" "$OUT/new"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "old: ${OLD_SRC:-prebuilt} $OLD_BIN $(stat -f '%Sm %z' "$OLD_BIN")"
    echo "new: $(git -C "$REPO" rev-parse --short HEAD) (dirty files: $(git -C "$REPO" status --porcelain | wc -l | tr -d ' ')) $(stat -f '%Sm %z' "$NEW_BIN")"
    echo "rows: $BENCH_ROWS  iters: $BENCH_ITERS  warmup: $BENCH_WARMUP  rounds: $ROUNDS"
} > "$OUT/env.txt"

lane() { # side round
    local bin
    if [ "$1" = old ]; then bin=$OLD_BIN; else bin=$NEW_BIN; fi
    printf '{"side":"%s","round":%s,"when":"before","load":"%s","gpu_util_pct":%s}\n' "$1" "$2" \
        "$(uptime | sed 's/.*load averages*: //')" \
        "$(ioreg -r -d 1 -w 0 -c IOAccelerator | rg -o '"Device Utilization %"=[0-9]+' | head -1 | rg -o '[0-9]+$' || echo null)" \
        >> "$OUT/load.jsonl"
    OJAS_BENCH_REF=$REF OJAS_BENCH_OUT=$OUT/$1/round$2.jsonl "$bin" 2>> "$OUT/$1/round$2.log"
}

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then lane old "$r"; lane new "$r"; else lane new "$r"; lane old "$r"; fi
    echo "round $r/$ROUNDS done" >&2
done
fi

# Per row and side: the smallest round minimum, the median of round minimums, the median
# of round medians, and the worst parity seen. The rows are the ones the runs produced, in
# order (BENCH_ROWS holds prefixes).
ROWS=$(jq -rs '[.[] | select(.status=="ok") | .row] | reduce .[] as $r ([]; if index([$r]) then . else . + [$r] end) | .[]' "$OUT"/old/round*.jsonl)
[ -n "$ROWS" ] || { echo "no ok rows in $OUT/old" >&2; exit 1; }
for row in $ROWS; do
    for side in old new; do
        jq -rs --arg row "$row" --arg side "$side" '
            [ .[] | select(.status=="ok" and .row==$row) ] as $x
            | ($x|map(.min_ms)|sort) as $mins | ($x|map(.median_ms)|sort) as $meds
            | "\($row) \($side) rounds=\($x|length) min=\($mins[0]*1000|round/1000) med_of_mins=\($mins[($mins|length)/2|floor]*1000|round/1000) med_of_medians=\($meds[($meds|length)/2|floor]*1000|round/1000) parity_max_abs=\($x|map(.parity_max_abs)|max)"' \
            "$OUT/$side"/round*.jsonl
    done
done | tee "$OUT/summary.txt"

# Rows a lane recorded but could not time (no torch reference, a refused op, a crash) are
# listed, so a missing row is never read as an absent result.
SKIPPED=$(jq -rs '[.[] | select(.status != "ok" and .status != "info") | "\(.row): \(.status)"] | unique | .[]' "$OUT"/old/round*.jsonl "$OUT"/new/round*.jsonl)
if [ -n "$SKIPPED" ]; then
    { echo "not timed:"; echo "$SKIPPED"; } | tee -a "$OUT/summary.txt"
fi
