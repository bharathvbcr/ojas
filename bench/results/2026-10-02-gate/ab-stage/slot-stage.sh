#!/bin/bash
# One slot, serial at -j 2: the ojas-metal suite with the staged bias sum,
# clippy, the metal_bench build, then the gate probe A/B against
# the binary built before it (metal_bench-tn).
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
m=/Users/bharath/Code/research/ojas/Cargo.toml
t=/Users/bharath/Code/research/ojas/target-lane-bench
sum() {
  grep '^test result:' "$1" | awk '{p+=$4; f+=$6; i+=$8} END {print "passed " p ", failed " f ", ignored " i}'
  grep -E 'FAILED|panicked|^error' "$1" | head -20
}
uptime
cargo test --release -j 2 --no-fail-fast --features metal -p ojas-metal --manifest-path $m --target-dir $t > "$dir/stage-suite.log" 2>&1
code=$?; echo "ojas-metal suite exit=$code"; sum "$dir/stage-suite.log"
[ $code -eq 0 ] || exit 1
echo "mutants skipped: no fork-heavy steps on this Mac (resume, gentler)"
cargo clippy --release -j 2 --all-targets --features metal -p ojas-metal --manifest-path $m --target-dir $t -- -D warnings > "$dir/stage-clippy.log" 2>&1
echo "clippy exit=$?"; grep -E '^(error|warning)|-->' "$dir/stage-clippy.log" | head -10
cargo build --release -j 2 --features metal -p ojas-metal --example metal_bench --manifest-path $m --target-dir $t > "$dir/stage-build.log" 2>&1
echo "build exit=$?"
uptime
old=$dir/metal_bench-tn
new=$t/release/examples/metal_bench
for i in 1 2 3 4; do
  if [ $((i % 2)) -eq 1 ]; then
    "$old" 40 gate > "$dir/stage-old-$i.md" 2>&1; "$new" 40 gate > "$dir/stage-new-$i.md" 2>&1
  else
    "$new" 40 gate > "$dir/stage-new-$i.md" 2>&1; "$old" 40 gate > "$dir/stage-old-$i.md" 2>&1
  fi
done
for side in old new; do
  echo "== $side (min µs per round: dbias | whole backward | GPU at start)"
  for i in 1 2 3 4; do
    b=$(grep 'ojas_per_head_gate_dbias' "$dir/stage-$side-$i.md" | awk -F'|' '{print $4}')
    w=$(grep 'whole backward' "$dir/stage-$side-$i.md" | awk -F'|' '{print $4}')
    s=$(grep '^start:' "$dir/stage-$side-$i.md" | awk '{print $3}')
    echo "round $i: dbias$b | whole$w | $s"
  done
done
uptime
