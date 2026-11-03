#!/bin/bash
# Job 2 of rules.md: SpikeTrack 38e0a35 (export at /Users/bharath/Code/kern1103/SpikeTrack, whose
# ../ojas is this worktree) built against A and against the winner, then the tracker A/B.
#   /Users/bharath/Code/research/SpikeTrack/tools/locked.sh flash-fwd-2 \
#     bash /Users/bharath/Code/kern1103/ojas/bench/results/2026-11-03-flash-fwd-specialize/job2.sh <win-label> <win-sha>
set -uo pipefail
WIN=$1
WIN_SHA=$2
W=/Users/bharath/Code/kern1103
O=$W/ojas
ST=$W/SpikeTrack
M=/Users/bharath/Code/research/SpikeTrack
R=$O/bench/results/2026-11-03-flash-fwd-specialize
E=$R/e2e
BINS=$W/bins
export CARGO_TARGET_DIR=$W/target
export CARGO_BUILD_JOBS=2
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
BRANCH=claude/flash-fwd-specialize-1103
mkdir -p "$E"
status=0

echo "== rules.md mtime $(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$R/rules.md")"
echo "== ojas HEAD $(git -C $O rev-parse HEAD); tessl $(git -C $W/tessl rev-parse HEAD); SpikeTrack export of 38e0a35"
git -C $O status --short
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)"

build() { # label sha
  local label=$1 sha=$2
  git -C $O checkout -q -- Cargo.lock 2>/dev/null
  git -C $O checkout -q --detach "$sha" || { echo "== checkout $sha failed"; return 1; }
  echo "== build st-$label ($sha) $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  if ! cargo build --release -j 2 --manifest-path $ST/Cargo.toml -p spiketrack \
      > "$E/build-st-$label.log" 2>&1; then
    echo "== build st-$label FAILED"; tail -30 "$E/build-st-$label.log"; return 1
  fi
  cp "$CARGO_TARGET_DIR/release/spiketrack" "$BINS/st-$label"
  echo "st-$label $sha $(shasum -a 256 "$BINS/st-$label" | cut -c1-16)" | tee -a "$E/bins.tsv"
}

: > "$E/bins.tsv"
build A 3406d65 || exit 1
build "$WIN" "$WIN_SHA" || exit 1
git -C $O checkout -q -- Cargo.lock 2>/dev/null
git -C $O checkout -q "$BRANCH"
git -C $O status --short

track() { # name binary [extra...]
  local name=$1 bin=$2; shift 2
  local d=$E/$name
  rm -rf "$d"; mkdir -p "$d"
  echo "== $name $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  "$bin" track "$M/data/visem/VISEM_Tracking_Train_v4/Train/82/images" --prompt sperm --backend metal \
    --max-frames 34 --boxes detector --lora "$M/fixtures/lora_runs/visem-lora-r16/adapter.safetensors" \
    --max-objects 16 --out "$d" "$@" > "$d/track.log" 2>&1 || { echo "== $name FAILED"; tail -20 "$d/track.log"; status=1; }
  rm -rf "$d/masks"
}
track ctl-$WIN "$BINS/st-$WIN"
track r1-A "$BINS/st-A" --profile stages; track r1-$WIN "$BINS/st-$WIN" --profile stages
track r2-$WIN "$BINS/st-$WIN" --profile stages; track r2-A "$BINS/st-A" --profile stages
track r3-A "$BINS/st-A" --profile stages; track r3-$WIN "$BINS/st-$WIN" --profile stages
track ctl-A "$BINS/st-A"

for r in ctl-A r1-A r1-$WIN r2-$WIN r2-A r3-A r3-$WIN; do
  [ -s "$E/$r/82.json" ] || { echo "== $r: no predictions"; status=1; continue; }
  if cmp -s "$E/ctl-$WIN/82.json" "$E/$r/82.json"; then echo "$r: byte-identical to ctl-$WIN"
  else echo "$r: DIFFERS from ctl-$WIN"; status=1; fi
done
python3 -I "$R/e2e_summary.py" "$E" "$WIN" | tee "$E/summary.txt" || status=1
echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime) status $status"
exit $status
