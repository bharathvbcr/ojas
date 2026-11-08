#!/bin/bash
# build.sh <label> <ojas_sha> <tessl_sha>
# Moves the bisect worktrees (outside the shared checkout, detached) to the
# given commits, drops in the fixed probe harness, builds it at -j 2 into one
# target dir, and copies the binary to bins/<label>. Never touches the main
# ojas or tessl working trees.
set -eu
LABEL=$1
OJ_SHA=$2
TS_SHA=$3
LANE=/Users/bharath/Code/research/ojas/target-splitk
WT=/Users/bharath/Code/splitk-bisect
mkdir -p "$LANE/bins"
if [ ! -d "$WT/ojas" ]; then
  mkdir -p "$WT"
  git -C /Users/bharath/Code/research/ojas worktree add --detach "$WT/ojas" "$OJ_SHA"
fi
if [ ! -d "$WT/tessl" ]; then
  git -C /Users/bharath/Code/research/tessl worktree add --detach "$WT/tessl" "$TS_SHA"
fi
# The probe is the only untracked file in the ojas worktree; remove it before
# moving HEAD so checkout never sees a conflict.
rm -f "$WT/ojas/ojas-metal/examples/splitk_probe.rs"
# cargo rewrites Cargo.lock's gusset entry (live path dep at ../../../devtools,
# not used by ojas-metal); restore tracked files before moving HEAD.
git -C "$WT/ojas" checkout --quiet -- .
git -C "$WT/tessl" checkout --quiet -- .
git -C "$WT/ojas" checkout --quiet --detach "$OJ_SHA"
git -C "$WT/tessl" checkout --quiet --detach "$TS_SHA"
test -z "$(git -C "$WT/ojas" status --porcelain)" || { echo "ojas worktree dirty"; exit 1; }
test -z "$(git -C "$WT/tessl" status --porcelain)" || { echo "tessl worktree dirty"; exit 1; }
cp "$LANE/splitk_probe.rs" "$WT/ojas/ojas-metal/examples/splitk_probe.rs"
echo "# build $LABEL ojas $(git -C "$WT/ojas" rev-parse --short HEAD) tessl $(git -C "$WT/tessl" rev-parse --short HEAD) $(date -u +%H:%M:%SZ)"
CARGO_TARGET_DIR="$LANE/build" CARGO_BUILD_JOBS=2 \
  cargo build --manifest-path "$WT/ojas/Cargo.toml" -p ojas-metal --release \
  --example splitk_probe
cp "$LANE/build/release/examples/splitk_probe" "$LANE/bins/$LABEL"
printf '%s\t%s\t%s\t%s\n' "$LABEL" "$OJ_SHA" "$TS_SHA" \
  "$(shasum -a 256 "$LANE/bins/$LABEL" | cut -c1-16)" >> "$LANE/bins/manifest.tsv"
echo "# built $LABEL $(date -u +%H:%M:%SZ)"
