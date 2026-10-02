#!/bin/bash
# Mutants of the staged bias sum; each must fail the gate_dbias_* tests in
# ojas-metal/src/gpu.rs. The live kernel is backed up first and restored
# after every mutant.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
k=/Users/bharath/Code/research/ojas/ojas-metal/kernels/per_head_gate.metal
cp "$k" "$dir/stage-k.good"
restore() { cp "$dir/stage-k.good" "$k"; }
run() {
  name=$1
  cargo test --release -j 4 --features metal -p ojas-metal \
    --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
    --target-dir /Users/bharath/Code/research/ojas/target-lane-bench \
    --lib gate_dbias > "$dir/stage-mutant-$name.log" 2>&1
  code=$?
  restore
  if [ $code -eq 0 ]; then echo "mutant $name SURVIVED"; else echo "mutant $name killed (exit $code)"; fi
  grep -E '^test .*FAILED$' "$dir/stage-mutant-$name.log"
}
apply() {
  from=$1; to=$2; marker=$3
  sed -i '' "s/$from/$to/" "$k"
  grep -q "$marker" "$k" || { echo "mutation $marker did not apply"; restore; exit 1; }
}
# 1: the unrolled adds out of row order (a1 before a0).
apply 's += a0; s += a1;' 's += a1; s += a0;' 's += a1; s += a0;'
run swap-pair
# 2: the tail loop dropped (rows past the last multiple of 8 in a block).
apply 'for (; j < live; j++) {' 'for (; j < 0u; j++) {' 'j < 0u'
run no-tail
# 3: every block reads from row 0 (base dropped from the load).
apply 'slot\[i\] = d_pre\[(base + i) \* n_head + h\];' 'slot[i] = d_pre[i * n_head + h];' 'd_pre\[i \* n_head + h\]'
run no-base
cmp -s "$dir/stage-k.good" "$k" && echo "kernel restored"
