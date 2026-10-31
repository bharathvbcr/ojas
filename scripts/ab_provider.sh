#!/bin/bash
# The paired, interleaved A/B of TapeStepProvider against the tessl provider
# (ojas-qwen35/tests/gpu_ab_provider.rs): every arm one process under
# mac_heavy.sh, the order rotated each rep (odd reps tape first, even reps
# tessl first), rep 1 also dumping both arms' gradients, then the parity
# comparison. One ROW line per arm goes to $OUT/rows.jsonl.
#
#   bash scripts/ab_provider.sh
#
# Settings (environment): REPS (5), TOKENS (2048), OUT (a new directory under
# the system temp dir; it is created and must not hold earlier rows),
# TARGET (the cargo target dir), LOAD (MAC_HEAVY_MAX_LOAD; the owner-approved
# number in target-ug/loadcap), HEAVY (the mac_heavy.sh path), LABEL (the
# lock label), MANIFEST (this checkout's Cargo.toml).
#
# The lock is taken per arm and released between arms, so other sessions'
# jobs run between arms, never during one; the order is recorded per row
# (`order`) and the load is in env.txt.

set -u
set -o pipefail

REPS="${REPS:-5}"
TOKENS="${TOKENS:-2048}"
LOAD="${LOAD:-30}"
LABEL="${LABEL:-ug-ab-harness}"
HEAVY="${HEAVY:-/Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh}"
MANIFEST="${MANIFEST:-/Users/bharath/Code/research/ojas-ug-ab-harness/Cargo.toml}"
TARGET="${TARGET:-/Users/bharath/Code/research/ojas-ug-ab-harness-build}"
OUT="${OUT:-${TMPDIR:-/tmp}/ojas-ab-provider-$$}"

if [ -e "$OUT/rows.jsonl" ]; then
  echo "refusing: $OUT/rows.jsonl exists" >&2
  exit 2
fi
mkdir -p "$OUT" || exit 2

run_test() {
  # run_test <test name> <env assignments...>: one cargo test process under the lock.
  local name="$1"
  shift
  env MAC_HEAVY_MAX_LOAD="$LOAD" bash "$HEAVY" "$LABEL" \
    env CARGO_TARGET_DIR="$TARGET" "$@" \
    cargo test --manifest-path "$MANIFEST" -p ojas-qwen35 --release --test gpu_ab_provider \
    -- --ignored --test-threads=2 --nocapture "$name"
}

run_arm() {
  # run_arm <arm> <rep> <order> <dump 0|1>
  run_test gpu_ab_arm AB_ARM="$1" AB_TOKENS="$TOKENS" AB_REP="$2" AB_ORDER="$3" \
    AB_DUMP="$4" AB_OUT="$OUT" | grep '^ROW ' | sed 's/^ROW //' >>"$OUT/rows.jsonl"
}

status=0
for rep in $(seq 1 "$REPS"); do
  dump=0
  if [ "$rep" -eq 1 ]; then
    dump=1
  fi
  if [ $((rep % 2)) -eq 1 ]; then
    run_arm tape "$rep" tape-first "$dump" || status=1
    run_arm tessl "$rep" tape-first "$dump" || status=1
  else
    run_arm tessl "$rep" tessl-first "$dump" || status=1
    run_arm tape "$rep" tessl-first "$dump" || status=1
  fi
done

run_test ab_compare_dumps AB_OUT="$OUT" | grep '^ROW ' | sed 's/^ROW //' >>"$OUT/rows.jsonl" || status=1
echo "rows: $OUT/rows.jsonl (status $status)"
exit "$status"
