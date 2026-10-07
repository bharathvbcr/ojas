#!/bin/bash
# Both sides of a production A/B built from one copy of tessl, differing only
# in kernels/matmul_tensorops.metal: old is tessl HEAD's shader (refused
# unless kernels/matmul_tensorops.orig.metal still matches it), new is the
# working tree's. One target dir, so the second build is the metallib and a
# relink. ab_prod.sh then pairs the two binaries per case, back to back.
#   LABELS="case ..." bash ab_tree.sh <tag> [rounds]
set -u
D=/Users/bharath/Code/research/ojas/target-mathsrc
TESSL=/Users/bharath/Code/research/tessl
# Outside ojas: a tree under ojas/ is taken for a member of ojas's workspace.
TREE=/private/tmp/claude-501/-Users-bharath-Code-research-ojas/9cfc887e-5d3d-4a70-97ed-018336773f0f/scratchpad/ab-tree
TAG=${1:?tag}
ROUNDS=${2:-6}
: "${LABELS:?set LABELS to the cases to pair}"
ORIG=$D/kernels/matmul_tensorops.orig.metal
export CARGO_TARGET_DIR=$D/tessl-abtree
export CARGO_BUILD_JOBS=2
BIN=$CARGO_TARGET_DIR/release/bench_gemm_tnnt_tune
mkdir -p "$TREE" "$D/ab" "$D/logs"
head_sha=$(git --git-dir="$TESSL/.git" show HEAD:kernels/matmul_tensorops.metal | shasum | cut -d' ' -f1)
orig_sha=$(shasum "$ORIG" | cut -d' ' -f1)
[ "$head_sha" = "$orig_sha" ] || { echo "orig shader $orig_sha is not tessl HEAD's $head_sha"; exit 1; }
rsync -a --delete --exclude target --exclude .git "$TESSL/" "$TREE/" || exit 1
NEW_SHADER=$D/ab/$TAG-new.metal
cp "$TREE/kernels/matmul_tensorops.metal" "$NEW_SHADER" || exit 1
build() {
  local side=$1 src=$2
  cp "$src" "$TREE/kernels/matmul_tensorops.metal" || exit 1
  echo "== build $side (shader $(shasum "$src" | cut -c1-8))"
  if ! TESSL_GEMM_TUNE=1 cargo build --release --manifest-path "$TREE/Cargo.toml" \
    --bin bench_gemm_tnnt_tune > "$D/logs/$TAG-build-$side.log" 2>&1; then
    rg -n '^(error|warning)|-->' "$D/logs/$TAG-build-$side.log" | head -n 40
    exit 1
  fi
  cp "$BIN" "$D/ab/$TAG-${side}_bench" || exit 1
}
build old "$ORIG"
build new "$NEW_SHADER"
o=$(shasum "$D/ab/$TAG-old_bench" | cut -d' ' -f1)
n=$(shasum "$D/ab/$TAG-new_bench" | cut -d' ' -f1)
[ "$o" != "$n" ] || { echo "old and new binaries are identical; the shader swap did not rebuild"; exit 1; }
OLD_BIN=$D/ab/$TAG-old_bench NEW_BIN=$D/ab/$TAG-new_bench bash "$D/ab_prod.sh" run "$TAG" "$ROUNDS"
