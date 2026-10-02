#!/bin/bash
# Each mutation of ojas_per_head_gate_dbias must fail the dbias unit tests.
# The live kernel is backed up first and restored after every mutation.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
src=/Users/bharath/Code/research/ojas/ojas-metal/kernels/per_head_gate.metal
cp "$src" "$dir/per_head_gate.metal.good"
run() {
  name=$1
  cargo test --release -j 4 --features metal -p ojas-metal --lib gate_dbias \
    --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
    --target-dir /Users/bharath/Code/research/ojas/target-lane-bench > "$dir/mutant-$name.log" 2>&1
  code=$?
  cp "$dir/per_head_gate.metal.good" "$src"
  if [ $code -eq 0 ]; then echo "mutant $name SURVIVED"; else echo "mutant $name killed (exit $code)"; fi
  grep -E '^test .*(ok|FAILED)$' "$dir/mutant-$name.log"
}
# 1: each chunk's lanes added in reverse.
sed -i '' 's/s += simd_shuffle(v\[q\], (ushort)j);/s += simd_shuffle(v[q], (ushort)(live - 1u - j));/' "$src"
grep -q 'live - 1u - j' "$src" || { echo "mutation 1 did not apply"; cp "$dir/per_head_gate.metal.good" "$src"; exit 1; }
run reversed
# 2: a final partial chunk is dropped.
sed -i '' 's/const uint live = (uint)min((ulong)sw, n - start);/const uint live = n - start < sw ? 0u : sw;/' "$src"
grep -q 'n - start < sw ? 0u : sw' "$src" || { echo "mutation 2 did not apply"; cp "$dir/per_head_gate.metal.good" "$src"; exit 1; }
run partial
# 3: each chunk reduced as a tree (simd_sum) instead of lane by lane.
sed -i '' 's/s += simd_shuffle(v\[q\], (ushort)j);/s += j == 0u ? simd_sum(v[q]) : 0.0f;/' "$src"
grep -q 'simd_sum(v\[q\])' "$src" || { echo "mutation 3 did not apply"; cp "$dir/per_head_gate.metal.good" "$src"; exit 1; }
run tree
cmp -s "$dir/per_head_gate.metal.good" "$src" && echo "kernel restored"
