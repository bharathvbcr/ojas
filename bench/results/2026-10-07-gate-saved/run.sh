#!/bin/bash
# Saved-sigmoid gate A/B on Metal, then wgpu (never concurrently).
set -u
W=/Users/bharath/Code/research/ojas/.gitpulse/worktrees/harden-gpu-runtime-64-bit-fault-word-typ-85133af8
R=/Users/bharath/Code/research/ojas
M="$R/Cargo.toml"
export CARGO_TARGET_DIR="$R/target-gpuhard"
echo "main HEAD: $(git -C $R rev-parse HEAD)"
git -C $R status --short --untracked-files=no
export CARGO_BUILD_JOBS=2
OUT="$W/target/gate-saved"
mkdir -p "$OUT"
cargo test -j 2 --release --manifest-path "$M" -p ojas-metal --test gate_saved --no-run || exit 1
cargo test -j 2 --release --manifest-path "$M" -p ojas-wgpu --test gate_saved --no-run || exit 1
for r in 1 2; do
  for b in metal wgpu; do
    echo "=== $b round $r: start $(date '+%H:%M:%S'), load $(sysctl -n vm.loadavg) ==="
    cargo test -j 2 --release --manifest-path "$M" -p "ojas-$b" --test gate_saved -- --ignored --nocapture bench_ > "$OUT/$b-$r.txt" 2>&1
    echo "=== $b round $r: exit $?, end load $(sysctl -n vm.loadavg) ==="
  done
done
echo "main HEAD after: $(git -C $R rev-parse HEAD)"
git -C $R status --short --untracked-files=no
