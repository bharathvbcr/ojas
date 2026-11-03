#!/bin/bash
# Job 3 of rules.md: D1 exp diagnostic, small-op baseline, test and lint.
#   /Users/bharath/Code/research/SpikeTrack/tools/locked.sh flash-fwd-3 \
#     bash /Users/bharath/Code/kern1103/ojas/bench/results/2026-11-03-tracker-ops-baseline/job3.sh
set -uo pipefail
W=/Users/bharath/Code/kern1103
O=$W/ojas
R=$O/bench/results/2026-11-03-tracker-ops-baseline
BINS=$W/bins
export CARGO_TARGET_DIR=$W/target
export CARGO_BUILD_JOBS=2
BRANCH=claude/flash-fwd-specialize-1103
status=0

echo "== rules.md mtime $(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$R/rules.md")"
echo "== ojas HEAD $(git -C $O rev-parse HEAD); tessl $(git -C $W/tessl rev-parse HEAD)"
git -C $O status --short
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)"
[ -x "$BINS/A" ] || { echo "== no A binary"; exit 1; }
echo "A $(shasum -a 256 "$BINS/A" | cut -c1-16) (job 1)" | tee "$R/bins.tsv"

# D1: A + the fast-exp patch, same tree and target dir.
git -C $O checkout -q -- Cargo.lock 2>/dev/null
git -C $O checkout -q --detach 3406d65 || exit 1
git -C $O apply "$R/diag-fast-exp.patch" || { echo "== patch failed"; exit 1; }
echo "== build D1 $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
cargo build --release -j 2 --manifest-path $O/Cargo.toml -p ojas-metal --example flash_fwd_probe \
  > "$R/build-D1.log" 2>&1 || { echo "== build D1 FAILED"; tail -30 "$R/build-D1.log"; status=1; }
cp "$CARGO_TARGET_DIR/release/examples/flash_fwd_probe" "$BINS/D1"
echo "D1 $(shasum -a 256 "$BINS/D1" | cut -c1-16) (3406d65 + diag-fast-exp.patch)" | tee -a "$R/bins.tsv"
git -C $O checkout -q -- ojas-metal/kernels/ojas_backend.metal Cargo.lock
git -C $O checkout -q "$BRANCH"
git -C $O status --short

echo "== build ops probe + test at tip $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
cargo build --release -j 2 --manifest-path $O/Cargo.toml -p ojas-metal --example tracker_ops_probe \
  > "$R/build-ops.log" 2>&1 || { echo "== build ops FAILED"; tail -30 "$R/build-ops.log"; status=1; }
cp "$CARGO_TARGET_DIR/release/examples/tracker_ops_probe" "$BINS/ops"
echo "ops $(shasum -a 256 "$BINS/ops" | cut -c1-16) ($(git -C $O rev-parse --short HEAD))" | tee -a "$R/bins.tsv"

echo "== hash D1 $(date -u +%H:%M:%SZ)"
"$BINS/D1" hash > "$R/hash-D1.txt" 2>&1
diff <(cut -f2,3 "$O/bench/results/2026-11-03-flash-fwd-specialize/hash-A.txt") <(cut -f2,3 "$R/hash-D1.txt") > "$R/hash-diff-D1.txt"
echo "== D1 hash lines differing from A: $(grep -c '^>' "$R/hash-diff-D1.txt")"

cp "$BINS/A" "$BINS/A2"
: > "$R/time-d1.tsv"
for r in 0 1 2 3 4 5; do
  if [ $((r % 2)) -eq 0 ]; then order=(A D1 A2); else order=(A2 D1 A); fi
  echo "== d1 round $r order ${order[*]} $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  for l in "${order[@]}"; do
    "$BINS/$l" time 3 15 "$l" | sed "s/^/r$r\t/" | tee -a "$R/time-d1.tsv"
  done
done

: > "$R/ops.tsv"
for r in 0 1 2 3 4 5; do
  echo "== ops round $r $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  "$BINS/ops" 3 15 "r$r" | tee -a "$R/ops.tsv"
done

echo "== test attention_flash_bits $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
cargo test --release -j 2 --manifest-path $O/Cargo.toml -p ojas-metal --test attention_flash_bits \
  -- --test-threads=1 > "$R/test-flash-bits.log" 2>&1 && echo "== test passed" \
  || { echo "== test FAILED"; tail -40 "$R/test-flash-bits.log"; status=1; }

echo "== clippy $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
cargo clippy -j 2 --release --manifest-path $O/Cargo.toml -p ojas-metal --all-targets --no-deps \
  -- -D warnings > "$R/clippy.log" 2>&1 && echo "== clippy clean" \
  || { echo "== clippy FAILED"; grep -E "^(error|warning)" -A6 "$R/clippy.log" | head -60; status=1; }
echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime) status $status"
exit $status
