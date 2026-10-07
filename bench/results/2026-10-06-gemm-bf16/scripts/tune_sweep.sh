#!/bin/bash
# Build tessl's GEMM tune rig (TESSL_GEMM_TUNE=1, release, own target dir)
# and sweep the TN/NT coop column-panel variants:
#   bash tune_sweep.sh <tag> [rounds]
# Each round runs the panel cases once; logs go to target-mathsrc/logs.
set -u
ROOT=/Users/bharath/Code/research/ojas
TESSL=/Users/bharath/Code/research/tessl
LOGS=$ROOT/target-mathsrc/logs
mkdir -p "$LOGS"
TAG=${1:?tag}
ROUNDS=${2:-3}
export CARGO_TARGET_DIR=$ROOT/target-mathsrc/tessl-tune
export CARGO_BUILD_JOBS=2
echo "== build bench_gemm_tnnt_tune (TESSL_GEMM_TUNE=1, release)"
if ! TESSL_GEMM_TUNE=1 cargo build --release --manifest-path "$TESSL/Cargo.toml" \
  --bin bench_gemm_tnnt_tune > "$LOGS/$TAG-build.log" 2>&1; then
  rg -n '^(error|warning)|-->' "$LOGS/$TAG-build.log" | head -n 40
  tail -n 5 "$LOGS/$TAG-build.log"
  exit 1
fi
tail -n 1 "$LOGS/$TAG-build.log"
BIN=$CARGO_TARGET_DIR/release/bench_gemm_tnnt_tune
for r in $(seq 1 "$ROUNDS"); do
  echo "== round $r"
  BENCH_ONLY=ntp_,tnp_,ntap_,tnap_ BENCH_WARMUP=4 BENCH_ITERS=12 /usr/bin/time -l "$BIN" \
    > "$LOGS/$TAG-r$r.txt" 2> "$LOGS/$TAG-r$r.time" || { tail -n 20 "$LOGS/$TAG-r$r.time"; exit 1; }
  rg -n 'maximum resident' "$LOGS/$TAG-r$r.time"
done
cat "$LOGS/$TAG-r1.txt"
