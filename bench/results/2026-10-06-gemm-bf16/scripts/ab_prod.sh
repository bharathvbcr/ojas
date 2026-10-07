#!/bin/bash
# Interleaved A/B of tessl's production TN/NT GEMM kernels: the tune rig
# built before the tile-walk change (old, saved by `save_old`) against one
# built after it (new). The metallib is embedded in the binary
# (include_bytes!), so a copied binary keeps the shader it was built with.
# GPU load from other sessions drifts by tens of percent over minutes, so
# each case runs old and new back to back (alternating which goes first)
# rather than one whole suite after the other.
#   bash ab_prod.sh save_old            # copy the current tune build aside
#   bash ab_prod.sh run <tag> [rounds]  # rebuild new, then pair per case
# Env: OLD_BIN (default ab/old_bench) is the old side; PREFIXES (default
# the TN/NT and f32 lanes) picks the case labels, as an rg alternation,
# unless LABELS lists them outright. The new binary is kept as
# ab/<tag>-new_bench; NEW_BIN names a prebuilt one instead and skips the
# build (ab_tree.sh builds both sides from one tree that way).
set -u
ROOT=/Users/bharath/Code/research/ojas
TESSL=/Users/bharath/Code/research/tessl
LOGS=$ROOT/target-mathsrc/logs
AB=$ROOT/target-mathsrc/ab
mkdir -p "$LOGS" "$AB"
export CARGO_TARGET_DIR=$ROOT/target-mathsrc/tessl-tune
export CARGO_BUILD_JOBS=2
SRC=$TESSL/src/bin/bench_gemm_tnnt_tune.rs
BIN=$CARGO_TARGET_DIR/release/bench_gemm_tnnt_tune
OLD_BIN=${OLD_BIN:-$AB/old_bench}
PREFIXES=${PREFIXES:-ntp|tnp|ntap|tnap|f32nt|f32tn}
bin_for() {
  if [ "$1" = old ]; then echo "$OLD_BIN"; else echo "$NEW_BIN"; fi
}
MODE=${1:?save_old|run}
if [ "$MODE" = save_old ]; then
  cp "$BIN" "$OLD_BIN" || exit 1
  shasum "$BIN" "$OLD_BIN"
  exit 0
fi
TAG=${2:?tag}
ROUNDS=${3:-4}
if [ -z "${NEW_BIN:-}" ]; then
  echo "== build new bench_gemm_tnnt_tune (TESSL_GEMM_TUNE=1, release)"
  if ! TESSL_GEMM_TUNE=1 cargo build --release --manifest-path "$TESSL/Cargo.toml" \
    --bin bench_gemm_tnnt_tune > "$LOGS/$TAG-build.log" 2>&1; then
    rg -n '^(error|warning)|-->' "$LOGS/$TAG-build.log" | head -n 40
    tail -n 5 "$LOGS/$TAG-build.log"
    exit 1
  fi
  NEW_BIN=$AB/$TAG-new_bench
  cp "$BIN" "$NEW_BIN" || exit 1
fi
shasum "$OLD_BIN" "$NEW_BIN"
if [ -z "${LABELS:-}" ]; then
  LABELS=$(RIPGREP_CONFIG_PATH= rg -o "\"(($PREFIXES)_[a-z0-9_]+)\"" -r '$1' "$SRC")
fi
for label in $LABELS; do
  RIPGREP_CONFIG_PATH= rg -q -F "\"$label\"" "$SRC" || { echo "no case labelled $label in $SRC"; exit 1; }
done
echo "== $(echo "$LABELS" | wc -w | tr -d ' ') cases, $ROUNDS rounds"
mkdir -p "$LOGS/$TAG"
for r in $(seq 1 "$ROUNDS"); do
  i=0
  for label in $LABELS; do
    i=$((i + 1))
    # Alternate which side runs first, by case and by round.
    order="old new"
    [ $(((r + i) % 2)) -eq 0 ] && order="new old"
    for side in $order; do
      BENCH_PRODUCTION_ONLY=1 BENCH_ONLY=$label BENCH_WARMUP=4 BENCH_ITERS=12 \
        "$(bin_for "$side")" > "$LOGS/$TAG/$label-$side-r$r.txt" 2> "$LOGS/$TAG/$label-$side-r$r.err" \
        || { tail -n 20 "$LOGS/$TAG/$label-$side-r$r.err"; exit 1; }
    done
  done
  echo "== round $r done"
done
echo "== done; logs $LOGS/$TAG/"
