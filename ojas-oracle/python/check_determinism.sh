#!/usr/bin/env bash
# Regenerate the tiny fixtures N times (default 2) and require every file to
# be byte-identical across runs. Leaves the last run's files in place.
#
#   bash ojas-oracle/python/check_determinism.sh [N]
set -euo pipefail

ORACLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${1:-2}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ojas-oracle-determinism.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "runs must be a positive integer, got $RUNS" >&2
    exit 2
fi
# C-style loops: BSD `seq 2 1` counts down instead of printing nothing.
for ((run = 1; run <= RUNS; run++)); do
    bash "$ORACLE/python/regen_tiny.sh" >"$WORK/run$run.log" 2>&1 || {
        echo "run $run failed:" >&2
        tail -20 "$WORK/run$run.log" >&2
        exit 1
    }
    (cd "$ORACLE/fixtures/tiny" && shasum -a 256 -- *) >"$WORK/run$run.sha256"
done
for ((run = 2; run <= RUNS; run++)); do
    if ! diff "$WORK/run1.sha256" "$WORK/run$run.sha256"; then
        echo "run $run differs from run 1" >&2
        exit 1
    fi
done
cat "$WORK/run1.sha256"
echo "byte-identical across $RUNS runs ($(wc -l <"$WORK/run1.sha256" | tr -d ' ') files)"
