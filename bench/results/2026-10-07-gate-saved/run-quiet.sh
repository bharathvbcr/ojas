#!/bin/bash
# Quiet round of the saved-sigmoid gate A/B: binaries already built, so no
# compile runs alongside. Waits until the 1-minute load is <= 8 first.
set -u
W=/Users/bharath/Code/research/ojas/.gitpulse/worktrees/harden-gpu-runtime-64-bit-fault-word-typ-85133af8
R=/Users/bharath/Code/research/ojas
M="$R/Cargo.toml"
export CARGO_TARGET_DIR="$R/target-gpuhard"
OUT="$W/target/gate-saved"
for i in $(seq 1 60); do
  l=$(sysctl -n vm.loadavg | awk '{print int($2)}')
  [ "$l" -le 8 ] && break
  sleep 10
done
echo "main HEAD: $(git -C $R rev-parse HEAD)"
for r in 3 4; do
  for b in metal wgpu; do
    echo "=== $b round $r: start $(date '+%H:%M:%S'), load $(sysctl -n vm.loadavg) ==="
    cargo test -j 2 --release --manifest-path "$M" -p "ojas-$b" --test gate_saved -- --ignored --nocapture bench_ > "$OUT/$b-$r.txt" 2>&1
    echo "=== $b round $r: exit $?, end load $(sysctl -n vm.loadavg) ==="
  done
done
