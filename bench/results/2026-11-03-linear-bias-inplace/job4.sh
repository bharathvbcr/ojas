#!/bin/bash
# Job 4 of rules.md: F's op-level A/B with hashes, tests and clippy at F, then SpikeTrack end to end.
#   /Users/bharath/Code/research/SpikeTrack/tools/locked.sh flash-fwd-4 \
#     bash /Users/bharath/Code/kern1103/ojas/bench/results/2026-11-03-linear-bias-inplace/job4.sh
set -uo pipefail
W=/Users/bharath/Code/kern1103
O=$W/ojas
ST=$W/SpikeTrack
M=/Users/bharath/Code/research/SpikeTrack
R=$O/bench/results/2026-11-03-linear-bias-inplace
E=$R/e2e
BINS=$W/bins
P_SHA=c1676d6
F_SHA=aade8fa
BRANCH=claude/flash-fwd-specialize-1103
export CARGO_TARGET_DIR=$W/target
export CARGO_BUILD_JOBS=2
export SPIKETRACK_WEIGHTS="$HOME/Code/research/models/sam3.1/sam3.1_multiplex.safetensors"
export SPIKETRACK_MODEL_DIR="$HOME/Code/research/models/sam3.1"
mkdir -p "$E"
status=0

echo "== rules.md mtime $(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$R/rules.md")"
echo "== ojas HEAD $(git -C $O rev-parse HEAD); tessl $(git -C $W/tessl rev-parse HEAD)"
git -C $O status --short
echo "== start $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime)"

at() { # sha: detach the worktree there
  git -C $O checkout -q -- Cargo.lock 2>/dev/null
  git -C $O checkout -q --detach "$1" || { echo "== checkout $1 failed"; exit 1; }
}
build_ops() { # label sha
  at "$2"
  echo "== build ops-$1 ($2) $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  cargo build --release -j 2 --manifest-path $O/Cargo.toml -p ojas-metal --example tracker_ops_probe \
    > "$R/build-ops-$1.log" 2>&1 || { echo "== build ops-$1 FAILED"; tail -30 "$R/build-ops-$1.log"; exit 1; }
  cp "$CARGO_TARGET_DIR/release/examples/tracker_ops_probe" "$BINS/ops-$1"
  echo "ops-$1 $2 $(shasum -a 256 "$BINS/ops-$1" | cut -c1-16)" | tee -a "$R/bins.tsv"
}
build_st() { # label sha
  at "$2"
  echo "== build st-$1 ($2) $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  cargo build --release -j 2 --manifest-path $ST/Cargo.toml -p spiketrack \
    > "$E/build-st-$1.log" 2>&1 || { echo "== build st-$1 FAILED"; tail -30 "$E/build-st-$1.log"; exit 1; }
  cp "$CARGO_TARGET_DIR/release/spiketrack" "$BINS/st-$1"
  echo "st-$1 $2 $(shasum -a 256 "$BINS/st-$1" | cut -c1-16)" | tee -a "$R/bins.tsv"
}

: > "$R/bins.tsv"
build_ops P $P_SHA
build_ops F $F_SHA
build_st A $P_SHA
build_st F $F_SHA
git -C $O checkout -q -- Cargo.lock 2>/dev/null
git -C $O checkout -q "$BRANCH"
git -C $O status --short
cp "$BINS/ops-P" "$BINS/ops-P2"

for l in P F; do "$BINS/ops-$l" hash > "$R/hash-$l.txt" 2>&1; done
if diff "$R/hash-P.txt" "$R/hash-F.txt" > "$R/hash-diff.txt"; then echo "== hashes F: identical to P"
else echo "== hashes F: DIFFER from P"; cat "$R/hash-diff.txt"; status=1; fi

labels=(P F P2)
: > "$R/ops.tsv"
for r in 0 1 2 3 4 5 6 7; do
  order=()
  for s in 0 1 2; do order+=("${labels[$(((r + s) % 3))]}"); done
  if [ $((r % 2)) -eq 1 ]; then order=("${order[2]}" "${order[1]}" "${order[0]}"); fi
  echo "== ops round $r order ${order[*]} $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
  for l in "${order[@]}"; do
    "$BINS/ops-$l" 3 15 "$l" | sed "s/^/r$r\t/" >> "$R/ops.tsv"
  done
done

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
track ctl-F "$BINS/st-F"
track r1-A "$BINS/st-A" --profile stages; track r1-F "$BINS/st-F" --profile stages
track r2-F "$BINS/st-F" --profile stages; track r2-A "$BINS/st-A" --profile stages
track r3-A "$BINS/st-A" --profile stages; track r3-F "$BINS/st-F" --profile stages
track ctl-A "$BINS/st-A"
for r in ctl-A r1-A r1-F r2-F r2-A r3-A r3-F; do
  [ -s "$E/$r/82.json" ] || { echo "== $r: no predictions"; status=1; continue; }
  if cmp -s "$E/ctl-F/82.json" "$E/$r/82.json"; then echo "$r: byte-identical to ctl-F"
  else echo "$r: DIFFERS from ctl-F"; status=1; fi
done
python3 -I "$R/e2e_summary.py" "$E" F | tee "$E/summary.txt" || status=1

echo "== tests at F $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
for t in linear_bias_fused attention_flash_bits encoder_ops encoder_ops_edges bf16_accumulation \
    autocast_precast_weight attention_flash attention_routing; do
  if cargo test --release -j 2 --manifest-path $O/Cargo.toml -p ojas-metal --test "$t" \
      -- --test-threads=1 > "$R/test-$t.log" 2>&1; then
    echo "== test $t passed: $(grep -E '^test result' "$R/test-$t.log" | tail -1)"
  else
    echo "== test $t FAILED"; grep -E "panicked|FAILED|error" "$R/test-$t.log" | head -20; status=1
  fi
done
echo "== clippy $(date -u +%H:%M:%SZ) load $(sysctl -n vm.loadavg)"
cargo clippy -j 2 --release --manifest-path $O/Cargo.toml -p ojas-metal --all-targets --no-deps \
  -- -D warnings > "$R/clippy.log" 2>&1 && echo "== clippy clean" \
  || { echo "== clippy FAILED"; grep -E "^(error|warning)" -A6 "$R/clippy.log" | head -60; status=1; }
echo "== end $(date -u +%Y-%m-%dT%H:%M:%SZ) $(uptime) status $status"
exit $status
