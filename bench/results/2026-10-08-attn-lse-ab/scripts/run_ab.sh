#!/bin/bash
# Attention backward A/B, c74f3ba (old) against HEAD (new), both built
# against the same tessl and gusset. One CARGO_TARGET_DIR per tree. Then
# ROUNDS rounds; each round runs old and new back to back per backend,
# alternating which goes first.
set -u
D=/Users/bharath/Code/research/ojas/target-attn-ab
ROUNDS=${ROUNDS:-6}
OUT=$D/out
mkdir -p "$OUT"
export CARGO_BUILD_JOBS=2
for t in old new; do
  for be in metal wgpu; do
    echo "build $t $be"
    CARGO_TARGET_DIR=$D/target-$t cargo build --release \
      --manifest-path "$D/ab/$t/Cargo.toml" -p ojas-$be --example attn_ab_$be \
      >"$OUT/build-$t-$be.log" 2>&1 || { echo "build $t $be failed"; tail -30 "$OUT/build-$t-$be.log"; exit 1; }
  done
done
git -C /Users/bharath/Code/research/tessl log --oneline -1 >"$OUT/env.txt"
git -C /Users/bharath/Code/research/tessl status --short >>"$OUT/env.txt"
rustc --version >>"$OUT/env.txt"
: >"$OUT/ab.txt"
for r in $(seq 1 "$ROUNDS"); do
  for be in metal wgpu; do
    if [ $((r % 2)) -eq 1 ]; then order="old new"; else order="new old"; fi
    for t in $order; do
      echo "round $r $be $t load: $(uptime | sed 's/.*load averages*: //')" | tee -a "$OUT/ab.txt"
      "$D/target-$t/release/examples/attn_ab_$be" 2>&1 | sed "s/^AB /AB round=$r /" | tee -a "$OUT/ab.txt"
    done
  done
done
