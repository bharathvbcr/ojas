#!/bin/bash
# build_diag.sh <label> <ojas_sha> <tessl_sha>
# Diagnostic only (never a proposed fix): builds the probe at the given commits
# with ONE line of the bisect worktree's ojas_backend.metal changed: inside
# ojas_xattn_part_d##DD, `bf16_acc` becomes the constant `false`, so the
# compiler can drop every bf16 branch from the split-K partials kernel. The
# worktree is restored before and after; the shared checkout is untouched.
set -eu
LABEL=$1
OJ_SHA=$2
TS_SHA=$3
LANE=/Users/bharath/Code/research/ojas/target-splitk
WT=/Users/bharath/Code/splitk-bisect
MSL=ojas-metal/kernels/ojas_backend.metal
rm -f "$WT/ojas/ojas-metal/examples/splitk_probe.rs"
git -C "$WT/ojas" checkout --quiet -- .
git -C "$WT/ojas" checkout --quiet --detach "$OJ_SHA"
git -C "$WT/tessl" checkout --quiet --detach "$TS_SHA"
test -z "$(git -C "$WT/ojas" status --porcelain)" || { echo "ojas worktree dirty"; exit 1; }
test -z "$(git -C "$WT/tessl" status --porcelain)" || { echo "tessl worktree dirty"; exit 1; }
# The part kernel's line is the only one reading `flags & 32u` that sits next
# to `const bool live = row < Tq;` with a macro continuation.
n=$(grep -c 'const bool bf16_acc = (flags & 32u) != 0u; *\\$' "$WT/ojas/$MSL" || true)
echo "# candidate lines: $n"
lineno=$(grep -n -B1 'const bool bf16_acc = (flags & 32u) != 0u; *\\$' "$WT/ojas/$MSL" |
  grep -A1 'const bool live = row < Tq;' | grep 'bf16_acc' | cut -d: -f1)
test -n "$lineno" || { echo "part-kernel line not found"; exit 1; }
echo "# patching line $lineno"
sed -i '' "${lineno}s/const bool bf16_acc = (flags \& 32u) != 0u;/const bool bf16_acc = false;               /" "$WT/ojas/$MSL"
git -C "$WT/ojas" diff --stat
git -C "$WT/ojas" diff | tee "$LANE/bins/$LABEL.patch"
cp "$LANE/splitk_probe.rs" "$WT/ojas/ojas-metal/examples/splitk_probe.rs"
echo "# build $LABEL (diag) ojas $(git -C "$WT/ojas" rev-parse --short HEAD) tessl $(git -C "$WT/tessl" rev-parse --short HEAD) $(date -u +%H:%M:%SZ)"
CARGO_TARGET_DIR="$LANE/build" CARGO_BUILD_JOBS=2 \
  cargo build --manifest-path "$WT/ojas/Cargo.toml" -p ojas-metal --release \
  --example splitk_probe
cp "$LANE/build/release/examples/splitk_probe" "$LANE/bins/$LABEL"
printf '%s\t%s\t%s\t%s\tdiag:bf16_acc=false in xattn_part\n' "$LABEL" "$OJ_SHA" "$TS_SHA" \
  "$(shasum -a 256 "$LANE/bins/$LABEL" | cut -c1-16)" >> "$LANE/bins/manifest.tsv"
rm -f "$WT/ojas/ojas-metal/examples/splitk_probe.rs"
git -C "$WT/ojas" checkout --quiet -- .
echo "# built $LABEL $(date -u +%H:%M:%SZ)"
