#!/bin/bash
# Paired A/B: ojas MetalBackend, ojas WgpuBackend and torch MPS, alternating
# the lane order every round. See bench/README.md.
#
#   bench/run_paired.sh [rounds]        (default 5)
#
# Environment (all optional):
#   BENCH_ROWS    comma-separated row-name prefixes (default: every row)
#   BENCH_ITERS   timed iterations per row (default 20, minimum 20)
#   BENCH_WARMUP  warm-up iterations per row (default 5, minimum 5)
#   OUT_DIR       output directory (default bench/out/<timestamp>)
#   PYTHON        python with torch (default /opt/homebrew/opt/python@3.14/bin/python3.14)
#   CARGO_TARGET_DIR  (default <repo>/target-lane-bench)
#   BENCH_LANES   space-separated lanes in odd-round order (default
#                 "ojas-metal ojas-wgpu torch-mps"; also ojas-cpu and
#                 ojas-cpu-host, which run only the decode rows). Even
#                 rounds run them in reverse.
#
# The metal and wgpu lanes run their kernel rows, then the decode rows
# (ojas-infer's decode_vs_torch), into the same lane file.
set -euo pipefail

REPO=$(dirname "$(dirname "$(realpath "$0")")")
ROUNDS=${1:-5}
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT_DIR:-$REPO/bench/out/$STAMP}
PY=${PYTHON:-/opt/homebrew/opt/python@3.14/bin/python3.14}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$REPO/target-lane-bench}
export PYTHONDONTWRITEBYTECODE=1
export RIPGREP_CONFIG_PATH=
EX=$CARGO_TARGET_DIR/release/examples

mkdir -p "$OUT"
echo "output: $OUT" >&2

# 1. Build first: a stale binary produces numbers that look valid.
cargo build --release --manifest-path "$REPO/Cargo.toml" -p ojas-metal -p ojas-wgpu \
    --example metal_vs_torch --example wgpu_vs_torch 2>&1 | tail -3 >&2
cargo build --release --manifest-path "$REPO/Cargo.toml" -p ojas-infer \
    --example decode_vs_torch 2>&1 | tail -3 >&2
LANES=${BENCH_LANES:-ojas-metal ojas-wgpu torch-mps}

# 2. Toolchain and machine.
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git: $(git -C "$REPO" rev-parse HEAD) (dirty files: $(git -C "$REPO" status --porcelain | wc -l | tr -d ' '))"
    echo "uncommitted diff of the benchmarked crates (sha1): $(git -C "$REPO" diff -- ojas-core ojas-metal ojas-wgpu ojas-kernels ojas-device ojas-cpu ojas-model ojas-infer | shasum | cut -c1-12)"
    echo "tessl: $(git -C "$REPO/../tessl" rev-parse HEAD 2>&1) (dirty files: $(git -C "$REPO/../tessl" status --porcelain 2>&1 | wc -l | tr -d ' '))"
    echo "rustc: $(rustc -V)"
    echo "cargo: $(cargo -V)"
    echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
    echo "memory_bytes: $(sysctl -n hw.memsize)"
    echo "metal_bin: $(stat -f '%Sm %z' "$EX/metal_vs_torch")"
    echo "wgpu_bin: $(stat -f '%Sm %z' "$EX/wgpu_vs_torch")"
    echo "decode_bin: $(stat -f '%Sm %z' "$EX/decode_vs_torch")"
    echo "rounds: $ROUNDS  iters: ${BENCH_ITERS:-20}  warmup: ${BENCH_WARMUP:-5}  rows: ${BENCH_ROWS:-all}  lanes: $LANES"
} > "$OUT/env.txt"
"$PY" "$REPO/bench/torch_rows.py" env --out "$OUT/env_torch.json" >&2

# 3. Torch reference outputs (the parity gate the ojas binaries apply).
"$PY" "$REPO/bench/torch_rows.py" ref --ref "$OUT/ref" 2>"$OUT/ref.log"

snapshot() { # round lane when
    local up util
    up=$(uptime | sed 's/.*load averages*: //')
    util=$(ioreg -r -d 1 -w 0 -c IOAccelerator | rg -o '"Device Utilization %"=[0-9]+' | head -1 | rg -o '[0-9]+$' || true)
    printf '{"round":%s,"lane":"%s","when":"%s","time":"%s","load":"%s","gpu_util_pct":%s}\n' \
        "$1" "$2" "$3" "$(date +%H:%M:%S)" "$up" "${util:-null}" >> "$OUT/load.jsonl"
}

run_lane() { # round lane
    local f="$OUT/round$1/$2.jsonl" rc=0
    snapshot "$1" "$2" before
    local log="$OUT/round$1/$2.log"
    export OJAS_BENCH_REF="$OUT/ref" OJAS_BENCH_OUT="$f"
    case "$2" in
        ojas-metal) "$EX/metal_vs_torch" 2>>"$log" && "$EX/decode_vs_torch" metal 2>>"$log" || rc=$? ;;
        ojas-wgpu) "$EX/wgpu_vs_torch" 2>>"$log" && "$EX/decode_vs_torch" wgpu 2>>"$log" || rc=$? ;;
        ojas-cpu) "$EX/decode_vs_torch" cpu 2>>"$log" || rc=$? ;;
        ojas-cpu-host) "$EX/decode_vs_torch" cpu-host 2>>"$log" || rc=$? ;;
        torch-mps) "$PY" "$REPO/bench/torch_rows.py" time --out "$f" 2>>"$log" || rc=$? ;;
        *) echo "unknown lane $2" >&2; rc=64 ;;
    esac
    snapshot "$1" "$2" after
    if [ "$rc" -ne 0 ]; then
        printf '{"runtime":"%s","row":"_process","status":"crash","exit":%s}\n' "$2" "$rc" >> "$f"
        echo "round $1 $2 exited $rc (see $OUT/round$1/$2.log)" >&2
    fi
}

# 4. Rounds, alternating which runtime goes first.
for r in $(seq 1 "$ROUNDS"); do
    mkdir -p "$OUT/round$r"
    if [ $((r % 2)) -eq 1 ]; then
        order="$LANES"
    else
        order=$(printf '%s\n' $LANES | tail -r | tr '\n' ' ')
    fi
    for lane in $order; do
        run_lane "$r" "$lane"
    done
    echo "round $r/$ROUNDS done ($order)" >&2
done

# 5. Per-round ratios, spread flags, ranking.
"$PY" "$REPO/bench/aggregate.py" "$OUT" > "$OUT/summary.md"
echo "summary: $OUT/summary.md" >&2
