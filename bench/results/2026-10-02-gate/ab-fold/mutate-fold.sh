#!/bin/bash
# Each mutant of the folded gate checks must fail the gate tests (gpu.rs unit
# tests and tests/fused_checks.rs). Live files are backed up first and
# restored after every mutant.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
k=/Users/bharath/Code/research/ojas/ojas-metal/kernels/per_head_gate.metal
g=/Users/bharath/Code/research/ojas/ojas-metal/src/gpu.rs
cp "$k" "$dir/fold-kernel.good"
cp "$g" "$dir/fold-gpu.good"
restore() { cp "$dir/fold-kernel.good" "$k"; cp "$dir/fold-gpu.good" "$g"; }
run() {
  name=$1
  cargo test --release -j 4 --features metal -p ojas-metal --lib --test fused_checks gate \
    --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
    --target-dir /Users/bharath/Code/research/ojas/target-lane-bench > "$dir/fold-mutant-$name.log" 2>&1
  code=$?
  restore
  if [ $code -eq 0 ]; then echo "mutant $name SURVIVED"; else echo "mutant $name killed (exit $code)"; fi
  grep -E '^test .*FAILED$' "$dir/fold-mutant-$name.log"
}
apply() {
  file=$1; from=$2; to=$3; marker=$4
  sed -i '' "s/$from/$to/" "$file"
  grep -q "$marker" "$file" || { echo "mutation $marker did not apply"; restore; exit 1; }
}
apply "$k" 'if (bad_in) ojas_gate_flag' 'if (false \&\& bad_in) ojas_gate_flag' 'false && bad_in'
run kernel-in
apply "$k" 'if (bad_out) ojas_gate_flag' 'if (false \&\& bad_out) ojas_gate_flag' 'false && bad_out'
run kernel-out
apply "$k" 'if (!isfinite(s)) ojas_gate_flag' 'if (false \&\& !isfinite(s)) ojas_gate_flag' 'false && !isfinite(s)'
run dbias-out
apply "$g" '\.any(|&w| w != 0)' '.any(|\&w| w == 7)' 'w == 7'
run gpu-status-ignored
apply "$g" 'finite_f32("per_head_gate", &grad.d_input)?;' 'let _ = \&grad.d_input;' 'let _ = &grad.d_input;'
run gpu-d-input-unchecked
cmp -s "$dir/fold-kernel.good" "$k" && cmp -s "$dir/fold-gpu.good" "$g" && echo "files restored"
