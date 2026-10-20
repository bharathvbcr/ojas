#!/bin/bash
# Compile and test the phase-A branch (worktree ../ojas-phase-a) in its own target dir, through
# mac_heavy with alarms, and a keep-alive that only touches an existing directory. Launched detached;
# writes logs/phase-a/SUMMARY.txt and the logs/phase-a/DONE sentinel.
set -u
R=/Users/bharath/Code/research
WT="$R/ojas-phase-a"
TD="$WT/target"
LOGS="$R/ojas/backups/a6-tools/logs/phase-a"
mkdir -p "$LOGS" "$TD"
: > "$LOGS/SUMMARY.txt"
( while true; do [ -d "$TD" ] && touch "$TD"; sleep 300; done ) &
KEEP=$!
trap 'kill "$KEEP" 2>/dev/null' EXIT
step() { # step <name> <secs> <cargo subcommand> <args...>
  local name=$1 secs=$2 sub=$3; shift 3
  echo "=== $name" | tee -a "$LOGS/SUMMARY.txt"
  if [ -e "$TD" ] && [ ! -d "$TD" ]; then echo "    REFUSED: $TD is not a directory" | tee -a "$LOGS/SUMMARY.txt"; return; fi
  mkdir -p "$TD"
  env CARGO_TARGET_DIR="$TD" perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$secs" \
    bash "$R/Lappi-decision/tools/mac_heavy.sh" "a6-phasea-$name" \
    cargo "$sub" --manifest-path "$WT/Cargo.toml" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "    exit=$rc $(grep -E '^test result:' "$LOGS/$name.log" | awk '{p+=$4; f+=$6} END {if (NR) printf "passed=%d failed=%d (%d binaries)", p, f, NR}')" | tee -a "$LOGS/SUMMARY.txt"
  grep -E '^(error|warning)|panicked at|^test .* FAILED' "$LOGS/$name.log" | head -40 >> "$LOGS/SUMMARY.txt"
}
step check  14400 check -p ojas-train -p ojas-io --all-targets
step test   14400 test -p ojas-train -p ojas-io --no-fail-fast
step clippy 14400 clippy --keep-going -p ojas-train -p ojas-io --all-targets -- -D warnings
cat "$LOGS/SUMMARY.txt"
