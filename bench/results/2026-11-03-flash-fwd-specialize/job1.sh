#!/bin/bash
# Job 1 of rules.md: build A, B, C from one worktree, hash, then time 8 rounds.
# Run under the lock:
#   /Users/bharath/Code/research/SpikeTrack/tools/locked.sh flash-fwd-1 \
#     bash /Users/bharath/Code/kern1103/ojas/bench/results/2026-11-03-flash-fwd-specialize/job1.sh
set -uo pipefail
W=/Users/bharath/Code/kern1103
O=$W/ojas
R=$O/bench/results/2026-11-03-flash-fwd-specialize
BINS=$W/bins
export CARGO_TARGET_DIR=$W/target
export CARGO_BUILD_JOBS=2
BRANCH=claude/flash-fwd-specialize-1103
mkdir -p "$BINS"

echo "== rules.md mtime $(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$R/rules.md")"
echo "== ojas HEAD $(git -C $O rev-parse HEAD) ($BRANCH); tessl $(git -C $W/tessl rev-parse HEAD)"
echo "== ojas status:"; git -C $O status --short
echo "== tessl status:"; git -C $W/tessl status --short
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)"

build() { # label sha
  local label=$1 sha=$2
  git -C $O checkout -q -- Cargo.lock 2>/dev/null
  git -C $O checkout -q --detach "$sha" || { echo "== checkout $sha failed"; return 1; }
  echo "== build $label ($sha) $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  if ! cargo build --release -j 2 --manifest-path $O/Cargo.toml -p ojas-metal \
      --example flash_fwd_probe > "$R/build-$label.log" 2>&1; then
    echo "== build $label FAILED"; tail -30 "$R/build-$label.log"; return 1
  fi
  cp "$CARGO_TARGET_DIR/release/examples/flash_fwd_probe" "$BINS/$label"
  echo "$label $sha $(shasum -a 256 "$BINS/$label" | cut -c1-16)" | tee -a "$R/bins.tsv"
}

: > "$R/bins.tsv"
build A 3406d65 || exit 1
build B c80bda3 || exit 1
build C 33cddcd || exit 1
git -C $O checkout -q -- Cargo.lock 2>/dev/null
git -C $O checkout -q "$BRANCH"
cp "$BINS/A" "$BINS/A2"
echo "== builds done $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"

for l in A B C; do
  echo "== hash $l $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  "$BINS/$l" hash > "$R/hash-$l.txt" 2>&1 || { echo "== hash $l FAILED"; cat "$R/hash-$l.txt"; }
done
for l in B C; do
  if diff <(cut -f2,3 "$R/hash-A.txt") <(cut -f2,3 "$R/hash-$l.txt") > "$R/hash-diff-$l.txt"; then
    echo "== hashes $l: identical to A on every line"
  else
    echo "== hashes $l: DIFFER from A"; cat "$R/hash-diff-$l.txt"
  fi
done

labels=(A B C A2)
: > "$R/time.tsv"
for r in 0 1 2 3 4 5 6 7; do
  order=()
  for s in 0 1 2 3; do order+=("${labels[$(((r + s) % 4))]}"); done
  if [ $((r % 2)) -eq 1 ]; then order=("${order[3]}" "${order[2]}" "${order[1]}" "${order[0]}"); fi
  echo "== round $r order ${order[*]} $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  for l in "${order[@]}"; do
    "$BINS/$l" time 3 15 "$l" | sed "s/^/r$r\t/" | tee -a "$R/time.tsv"
  done
done
echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)"
