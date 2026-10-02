#!/bin/bash
# One heavy-job slot, serial at -j 4: the ojas-metal suite with the folded
# gate checks, their mutants, clippy, the metal_bench build, then an
# interleaved probe A/B against the bias-sum binary.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
m=/Users/bharath/Code/research/ojas/Cargo.toml
t=/Users/bharath/Code/research/ojas/target-lane-bench
uptime
cp "$t/release/examples/metal_bench" "$dir/metal_bench-dbias"
cargo test --release -j 4 --no-fail-fast --features metal -p ojas-metal --manifest-path $m --target-dir $t > "$dir/fold-suite.log" 2>&1
code=$?
echo "suite exit=$code"
grep '^test result:' "$dir/fold-suite.log" | awk '{p+=$4; f+=$6; i+=$8} END {print "passed " p ", failed " f ", ignored " i}'
grep -E 'FAILED|panicked|^error' "$dir/fold-suite.log" | head -20
[ $code -eq 0 ] || exit 1
bash "$dir/mutate-fold.sh"
cargo clippy --release -j 4 --features metal -p ojas-metal --all-targets --manifest-path $m --target-dir $t -- -D warnings > "$dir/fold-clippy.log" 2>&1
echo "clippy exit=$?"
grep -E '^(error|warning)|-->' "$dir/fold-clippy.log" | head -20
cargo build --release -j 4 --features metal -p ojas-metal --example metal_bench --manifest-path $m --target-dir $t > "$dir/fold-build.log" 2>&1
echo "build exit=$?"
uptime
old=$dir/metal_bench-dbias
new=$t/release/examples/metal_bench
for i in 1 2 3 4; do
  if [ $((i % 2)) -eq 1 ]; then
    "$old" 40 gate > "$dir/fold-old-$i.md" 2>&1; "$new" 40 gate > "$dir/fold-new-$i.md" 2>&1
  else
    "$new" 40 gate > "$dir/fold-new-$i.md" 2>&1; "$old" 40 gate > "$dir/fold-old-$i.md" 2>&1
  fi
done
for side in old new; do
  echo "== $side (min µs per round: checks | whole backward | GPU at start)"
  for i in 1 2 3 4; do
    c=$(grep 'ojas_check_finite passes' "$dir/fold-$side-$i.md" | awk -F'|' '{print $4}')
    b=$(grep 'ojas_per_head_gate_bwd' "$dir/fold-$side-$i.md" | awk -F'|' '{print $4}')
    w=$(grep 'whole backward' "$dir/fold-$side-$i.md" | awk -F'|' '{print $4}')
    s=$(grep '^start:' "$dir/fold-$side-$i.md" | awk '{print $3}')
    echo "round $i: checks$c | gate_bwd$b | whole$w | $s"
  done
done
