#!/bin/bash
# Job 5: the whole ojas-metal and ojas-model test suites and workspace clippy at the branch tip.
#   /Users/bharath/Code/research/SpikeTrack/tools/locked.sh flash-fwd-5 bash /Users/bharath/Code/kern1103/bins/job5.sh
set -uo pipefail
W=/Users/bharath/Code/kern1103
O=$W/ojas
OUT=$W/bins/suite
export CARGO_TARGET_DIR=$W/target
export CARGO_BUILD_JOBS=2
mkdir -p "$OUT"
status=0
echo "== ojas HEAD $(git -C $O rev-parse HEAD); tessl $(git -C $W/tessl rev-parse HEAD)"
git -C $O status --short
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)"
for p in ojas-metal ojas-model; do
  echo "== test $p $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  if cargo test --release -j 2 --manifest-path $O/Cargo.toml -p "$p" -- --test-threads=1 > "$OUT/test-$p.log" 2>&1; then
    echo "== test $p passed"
  else
    echo "== test $p FAILED"; status=1
  fi
  grep -E "^test result|FAILED|panicked" "$OUT/test-$p.log" | sort | uniq -c | sort -rn | head -20
done
echo "== clippy workspace $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
cargo clippy -j 2 --manifest-path $O/Cargo.toml --workspace --all-targets -- -D warnings > "$OUT/clippy-workspace.log" 2>&1 \
  && echo "== clippy workspace clean" \
  || { echo "== clippy workspace FAILED"; grep -E "^(error|warning)" -A4 "$OUT/clippy-workspace.log" | head -40; status=1; }
echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime) status $status"
exit $status
