#!/bin/bash
# One slot, serial at -j 4: tessl suite and clippy, the parallel TN split-K
# mutants, the ojas-metal suite, the gate probe A/B against the fold binary,
# and the paired gate rows against torch.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
tm=/Users/bharath/Code/research/tessl/Cargo.toml
tt=/Users/bharath/Code/research/ojas/target-lane-bench/tessl
m=/Users/bharath/Code/research/ojas/Cargo.toml
t=/Users/bharath/Code/research/ojas/target-lane-bench
sum() {
  grep '^test result:' "$1" | awk '{p+=$4; f+=$6; i+=$8} END {print "passed " p ", failed " f ", ignored " i}'
  grep -E 'FAILED|panicked|^error' "$1" | head -20
}
uptime
cargo test --release -j 4 --no-fail-fast --manifest-path $tm --target-dir $tt > "$dir/tn-tessl-suite.log" 2>&1
code=$?; echo "tessl suite exit=$code"; sum "$dir/tn-tessl-suite.log"
[ $code -eq 0 ] || exit 1
cargo clippy --release -j 4 --all-targets --manifest-path $tm --target-dir $tt -- -D warnings > "$dir/tn-tessl-clippy.log" 2>&1
echo "tessl clippy exit=$?"; grep -E '^(error|warning)|-->' "$dir/tn-tessl-clippy.log" | head -10
cargo clippy --release -j 4 --all-targets --manifest-path $tm --target-dir $tt -- -D warnings -A clippy::type_complexity > "$dir/tn-tessl-clippy2.log" 2>&1
echo "tessl clippy (type_complexity allowed) exit=$?"; grep -E '^(error|warning)|-->' "$dir/tn-tessl-clippy2.log" | head -10
bash "$dir/mutate-tn.sh"
cargo test --release -j 4 --no-fail-fast --features metal -p ojas-metal --manifest-path $m --target-dir $t > "$dir/tn-metal-suite.log" 2>&1
code=$?; echo "ojas-metal suite exit=$code"; sum "$dir/tn-metal-suite.log"
[ $code -eq 0 ] || exit 1
cargo build --release -j 4 --features metal -p ojas-metal --example metal_bench --manifest-path $m --target-dir $t > "$dir/tn-build.log" 2>&1
echo "build exit=$?"
uptime
old=$dir/metal_bench-fold
new=$t/release/examples/metal_bench
for i in 1 2 3 4; do
  if [ $((i % 2)) -eq 1 ]; then
    "$old" 40 gate > "$dir/tn-old-$i.md" 2>&1; "$new" 40 gate > "$dir/tn-new-$i.md" 2>&1
  else
    "$new" 40 gate > "$dir/tn-new-$i.md" 2>&1; "$old" 40 gate > "$dir/tn-old-$i.md" 2>&1
  fi
done
for side in old new; do
  echo "== $side (min µs per round: tn gw | whole backward | GPU at start)"
  for i in 1 2 3 4; do
    g=$(grep 'tn gw' "$dir/tn-$side-$i.md" | awk -F'|' '{print $4}')
    w=$(grep 'whole backward' "$dir/tn-$side-$i.md" | awk -F'|' '{print $4}')
    s=$(grep '^start:' "$dir/tn-$side-$i.md" | awk '{print $3}')
    echo "round $i: gw$g | whole$w | $s"
  done
done
uptime
CARGO_BUILD_JOBS=4 BENCH_ROWS=gate OUT_DIR=/Users/bharath/Code/research/ojas/bench/results/2026-10-02-gate/paired-gate \
  bash /Users/bharath/Code/research/ojas/bench/run_paired.sh 5 > "$dir/tn-paired.log" 2>&1
echo "paired exit=$?"
tail -30 "$dir/tn-paired.log"
uptime
