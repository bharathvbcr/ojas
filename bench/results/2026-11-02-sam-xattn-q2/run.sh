#!/bin/bash
# Runs under the machine-wide heavy-job lock, from the ojas root:
#   MAC_HEAVY_MAX_WAIT_S=3600 bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh \
#     ojas-sam-xattn-q2 bash bench/results/2026-11-02-sam-xattn-q2/run.sh
# Provenance first (rules.md mtime before any build or timed row), then the
# build at -j 2, then the three invocations of rules.md, each started only once
# the 1-minute load is at or below the harness's 6.0 bar (up to 20 minutes;
# the harness's own verdict still decides whether a run is clean).
set -u
ROOT=/Users/bharath/Code/research/ojas
OUT=$ROOT/bench/results/2026-11-02-sam-xattn-q2
BIN=$ROOT/target-xattn-q2/release/examples/attention_flash_bench

echo "# provenance $(date -u +%Y-%m-%dT%H:%M:%SZ)"
stat -f '# rules.md mtime %Sm' -t '%Y-%m-%dT%H:%M:%S%z' "$OUT/rules.md"
echo "# HEAD $(git -C "$ROOT" rev-parse HEAD)"
echo "# git status --short (start):"
git -C "$ROOT" status --short
uptime

echo "# build $(date -u +%Y-%m-%dT%H:%M:%SZ)"
CARGO_TARGET_DIR=$ROOT/target-xattn-q2 CARGO_BUILD_JOBS=2 \
  cargo build --manifest-path "$ROOT/Cargo.toml" -p ojas-metal --release \
  --example attention_flash_bench || exit 1
echo "# built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
shasum -a 256 "$BIN"

quiet() {
  for _ in $(seq 1 120); do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v l="$l" 'BEGIN{exit !(l <= 6.0)}'; then
      echo "# load $l at $(date -u +%H:%M:%SZ): starting $1"
      return 0
    fi
    sleep 10
  done
  echo "# load stayed above 6.0 for 20 minutes before $1; running anyway (verdict will say)"
}

quiet "a: Q2 as pre-registered (m=36320)"
"$BIN" 5 25 m=36320 > "$OUT/a-q2-m36320.txt" 2>&1
echo "# a exit $?"

quiet "b: sam xattn rows"
"$BIN" 5 25 "sam xattn Tk=" > "$OUT/b-sam-xattn.txt" 2>&1
echo "# b exit $?"

quiet "c: bf16 probe"
"$BIN" 2 8 "sam xattn bf16" > "$OUT/c-bf16.txt" 2>&1
echo "# c exit $?"

echo "# git status --short (end):"
git -C "$ROOT" status --short
uptime
echo "# done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
