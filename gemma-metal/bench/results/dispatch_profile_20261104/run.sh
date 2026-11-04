#!/bin/bash
# Per-dispatch GPU timestamp profile of one Gemma decode token.
#
#   bash run.sh --mini                 # synthetic mini graph, no weights (smoke / method check)
#   bash run.sh --e4b                  # real E4B Q4 token (needs the HF snapshot, see README)
#   bash run.sh --synthetic-e4b        # E4B shapes and tensor set, random Q4 weights (timing only)
#   bash run.sh --model /path/to/dir   # real model from an explicit snapshot directory
#   PROBES=1 bash run.sh --e4b         # also: pipelined tok/s under barrier / mid-commit env flags
#
# Heavy GPU work goes through the machine's one-heavy-job lock (mac_heavy.sh). The build uses
# its own CARGO_TARGET_DIR and -j 2. Output goes next to this script, in a directory named after
# the mode: <mode>/{run.log,dispatch_stamps.tsv,kernel_summary.tsv,family_summary.tsv,overhead.tsv}.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CRATE=/Users/bharath/Code/research/ojas/gemma-metal
MH=/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh
export CARGO_TARGET_DIR="$CRATE/target-stamps"

MODE="${1:---e4b}"
case "$MODE" in
  --mini) TAG=mini_smoke ;;
  --e4b) TAG=e4b ;;
  --synthetic-e4b) TAG=synthetic_e4b ;;
  --model) TAG=model ;;
  *) echo "usage: $0 --mini | --e4b | --synthetic-e4b | --model DIR" >&2; exit 2 ;;
esac
OUT="$HERE/$TAG"
mkdir -p "$OUT"

# Quiet the per-op diagnostic logging: it is host work inside the encode we are timing.
export GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0

run() { # label, env-assignments..., -- args...
  local label="$1"; shift
  echo "== $label: $(date -u +%FT%TZ) load: $(sysctl -n vm.loadavg)"
  "$MH" "dispatch-profile-$label" cargo run --manifest-path "$CRATE/Cargo.toml" --release -j 2 \
    --bin dispatch_profile -- "$@"
  echo "== $label done: $(date -u +%FT%TZ) load: $(sysctl -n vm.loadavg)"
}

{
  run profile "$@" --tokens 8 --rounds 8 --out "$OUT"
  if [ "${PROBES:-0}" = 1 ]; then
    # No "no barrier" flag exists in tessl (forgo_pending_barrier is crate-private), so the
    # probes vary barrier mode and commit granularity and read wall tok/s only.
    P=(--pipelined-only --out "$OUT/probe")
    run probe-default "$@" "${P[@]}"
    METAL_RUNTIME_HAZARD_BARRIERS=0 run probe-hazard0-alwayson "$@" "${P[@]}"
    TESSL_COARSE_BARRIERS=0 run probe-coarse0 "$@" "${P[@]}"
    TESSL_MID_COMMIT=100 run probe-midcommit100 "$@" "${P[@]}"
    run probe-default-again "$@" "${P[@]}"
  fi
} 2>&1 | tee "$OUT/run.log"
