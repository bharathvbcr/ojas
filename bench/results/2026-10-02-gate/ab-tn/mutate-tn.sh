#!/bin/bash
# Mutants of the parallel TN split-K; each must fail the `parallel_tn` tests
# in tessl/tests/gemm_ragged_shapes.rs. Live files are backed up first and
# restored after every mutant.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
k=/Users/bharath/Code/research/tessl/kernels/matmul_tensorops.metal
u=/Users/bharath/Code/research/tessl/kernels/utils.metal
g=/Users/bharath/Code/research/tessl/src/gemm.rs
cp "$k" "$dir/tn-k.good"; cp "$u" "$dir/tn-u.good"; cp "$g" "$dir/tn-g.good"
restore() { cp "$dir/tn-k.good" "$k"; cp "$dir/tn-u.good" "$u"; cp "$dir/tn-g.good" "$g"; }
run() {
  name=$1
  cargo test --release -j 4 --manifest-path /Users/bharath/Code/research/tessl/Cargo.toml \
    --target-dir /Users/bharath/Code/research/ojas/target-lane-bench/tessl \
    --test gemm_ragged_shapes parallel_tn > "$dir/tn-mutant-$name.log" 2>&1
  code=$?
  restore
  if [ $code -eq 0 ]; then echo "mutant $name SURVIVED"; else echo "mutant $name killed (exit $code)"; fi
  grep -E '^test .*FAILED$' "$dir/tn-mutant-$name.log"
}
apply() {
  file=$1; from=$2; to=$3; marker=$4
  sed -i '' "s/$from/$to/" "$file"
  grep -q "$marker" "$file" || { echo "mutation $marker did not apply"; restore; exit 1; }
}
# 1: every partition reads A from k = 0 (the parallel kernel only; the
# sequential split-K kernel has the same line).
before=$(grep -c 'auto mA = tensor(A, dextents' "$k")
sed -i '' '/kernel void matmul2d_tensorops_tn_splitk_par_f32(/,/op.run(tA, tB, tS);/s/auto mA = tensor(A + k0 \* M,/auto mA = tensor(A,/' "$k"
after=$(grep -c 'auto mA = tensor(A, dextents' "$k")
[ "$after" -eq $((before + 1)) ] || { echo "mutation a-offset did not apply once ($before -> $after)"; restore; exit 1; }
run a-offset
# 2: the reduction drops the last partition.
apply "$u" 'for (uint p = 1; p < partitions; p++) {' 'for (uint p = 1; p + 1 < partitions; p++) {' 'p + 1 < partitions'
run last-partition
# 3: slices packed at M·N, unpadded (misaligned when M·N is odd).
apply "$g" 'let slice = numel.div_ceil(4) \* 4;' 'let slice = numel;' 'let slice = numel;'
run unpadded-slice
cmp -s "$dir/tn-k.good" "$k" && cmp -s "$dir/tn-u.good" "$u" && cmp -s "$dir/tn-g.good" "$g" && echo "files restored"
