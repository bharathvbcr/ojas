#!/bin/bash
# Verify branch gp/a6-phase-b (worktree ../ojas-phase-b), which contains phase A (import + rewire) and
# phase B (TapeStepProvider). One run covers both. Generates the torch accumulate fixture first, then
# check, tests, clippy. Each cargo step runs through mac_heavy with an alarm; the keep-alive only touches
# an existing directory. Launched detached; writes logs/phase-b/SUMMARY.txt and the DONE sentinel.
set -u
R=/Users/bharath/Code/research
WT="$R/ojas-phase-b"
TD="$WT/target"
LOGS="$R/ojas/backups/a6-tools/logs/phase-b"
BREW=/opt/homebrew/opt/python@3.14/bin/python3.14
mkdir -p "$LOGS" "$TD"
: > "$LOGS/SUMMARY.txt"
( while true; do [ -d "$TD" ] && touch "$TD"; sleep 300; done ) &
KEEP=$!
trap 'kill "$KEEP" 2>/dev/null' EXIT

note() { echo "$*" | tee -a "$LOGS/SUMMARY.txt"; }
step() { # step <name> <secs> <cargo subcommand> <args...>
  local name=$1 secs=$2 sub=$3; shift 3
  note "=== $name"
  if [ -e "$TD" ] && [ ! -d "$TD" ]; then note "    REFUSED: $TD is not a directory"; return; fi
  mkdir -p "$TD"
  env CARGO_TARGET_DIR="$TD" perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$secs" \
    bash "$R/Lappi-decision/tools/mac_heavy.sh" "a6-phaseb-$name" \
    cargo "$sub" --manifest-path "$WT/Cargo.toml" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  note "    exit=$rc $(grep -E '^test result:' "$LOGS/$name.log" | awk '{p+=$4; f+=$6} END {if (NR) printf "passed=%d failed=%d (%d binaries)", p, f, NR}')"
  grep -E '^(error|warning)|panicked at|^test .* FAILED' "$LOGS/$name.log" | head -60 >> "$LOGS/SUMMARY.txt"
}

# 0. The torch fixture for the 3-row accumulate test (CPU torch, small).
note "=== accum-fixture"
( cd "$WT" && perl -e 'alarm shift; exec @ARGV' 1800 "$BREW" -P ojas-oracle/python/qwen35_accum_fixture.py ) > "$LOGS/accum-fixture.log" 2>&1
note "    exit=$?"; tail -3 "$LOGS/accum-fixture.log" >> "$LOGS/SUMMARY.txt"

# 1. Compile everything touched (fast feedback on the uncompiled phase A + B code).
step check   14400 check -p ojas-io -p ojas-train -p ojas-model -p ojas-qwen35 --all-targets
# 2. CPU tests: ojas-io and ojas-train (moved oracles, frozen entries), ojas-model provider, bank, adapters.
step train   14400 test -p ojas-io -p ojas-train --no-fail-fast
step model   14400 test -p ojas-model --test qwen35_step_provider --test qwen35_bank --test qwen35_adapters \
  --test qwen35_adapters_adversarial --test qwen35_optim --test preflight --no-fail-fast
# 3. Metal provider tests (ignored by default; the 2B kill-9 one needs the snapshot and is skipped if absent).
step metal   14400 test -p ojas-qwen35 --release --test gpu_tape_provider --no-fail-fast -- --ignored --test-threads=1
# 4. Lints.
step clippy  14400 clippy --keep-going -p ojas-io -p ojas-train -p ojas-model -p ojas-qwen35 --all-targets -- -D warnings
cat "$LOGS/SUMMARY.txt"
