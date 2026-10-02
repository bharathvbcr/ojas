#!/bin/bash
# Short slot at -j 4: tessl's library on its own (does the new TN split-K
# compile?), then metal_bench, then the `tn` sweep.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
uptime
cargo build --release -j 4 --lib --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml \
  --target-dir /Users/bharath/Code/research/ojas/target-lane-bench/tessl > "$dir/tn-tessl-build.log" 2>&1
code=$?
echo "tessl lib build exit=$code"
grep -E '^(error|warning)' -A6 "$dir/tn-tessl-build.log" | head -40
[ $code -eq 0 ] || exit 1
cargo build --release -j 4 --features metal -p ojas-metal --example metal_bench \
  --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
  --target-dir /Users/bharath/Code/research/ojas/target-lane-bench > "$dir/tn-bench-build.log" 2>&1
code=$?
echo "metal_bench build exit=$code"
grep -E '^(error|warning)' -A6 "$dir/tn-bench-build.log" | head -40
[ $code -eq 0 ] || exit 1
uptime
/Users/bharath/Code/research/ojas/target-lane-bench/release/examples/metal_bench 20 tn > "$dir/tn-sweep-1.md" 2>&1
echo "sweep exit=$?"
cat "$dir/tn-sweep-1.md"
