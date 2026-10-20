#!/bin/bash
# Re-run of the verify steps that failed or never ran, after the fix commits 43d5e82, 0a17930, aaed8b3
# and 111097d. Launched detached by launch-verify.sh (second argument). Same mac_heavy and alarm rules as
# owner-verify-main.sh, plus a keep-alive that touches target-a6-main every 5 min, so housekeeping
# cannot purge test binaries mid-run (three binaries vanished in the first run).
set -u
R=/Users/bharath/Code/research
M="$R/ojas"
LOGS="$M/backups/a6-tools/logs/rerun"
mkdir -p "$LOGS" "$M/target-a6-main"
: > "$LOGS/SUMMARY.txt"

# Keep-alive: touch only an existing directory. A bare `touch` on a deleted target dir creates a
# 0-byte FILE at that path and cargo then fails with "Not a directory" (happened 2026-10-18 07:32
# after the target dir was cleared during the disk-full cleanup).
( while true; do
    if [ -d "$M/target-a6-main" ]; then
      touch "$M/target-a6-main"
      find "$M/target-a6-main" -maxdepth 3 -name deps -type d -exec touch {} + 2>/dev/null
    fi
    sleep 300
  done ) &
KEEP=$!
trap 'kill "$KEEP" 2>/dev/null' EXIT

step() { # step <name> <secs> <cargo subcommand> <args...>
  local name=$1 secs=$2 sub=$3; shift 3
  echo "=== $name" | tee -a "$LOGS/SUMMARY.txt"
  if [ -e "$M/target-a6-main" ] && [ ! -d "$M/target-a6-main" ]; then
    echo "    REFUSED: $M/target-a6-main exists and is not a directory" | tee -a "$LOGS/SUMMARY.txt"; return
  fi
  mkdir -p "$M/target-a6-main"
  env CARGO_TARGET_DIR="$M/target-a6-main" perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$secs" \
    bash "$R/Lappi-decision/tools/mac_heavy.sh" "a6-rerun-$name" \
    cargo "$sub" --manifest-path "$M/Cargo.toml" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "    exit=$rc $(grep -E '^test result:' "$LOGS/$name.log" | awk '{p+=$4; f+=$6} END {if (NR) printf "passed=%d failed=%d (%d binaries)", p, f, NR}')" | tee -a "$LOGS/SUMMARY.txt"
  grep -E '^(error|warning)|panicked at|^test .* FAILED|could not execute' "$LOGS/$name.log" | head -20 >> "$LOGS/SUMMARY.txt"
}

# Re-run 2 (05:2x on 2026-10-18): model 137/137 and Metal 106/106 already passed in re-run 1
# (logs/rerun1/), and sha-bench passed in the first run (331 MiB/s). Only the two steps that timed out
# waiting for the heavy lock / cargo slots run here, with alarms matching mac_heavy's own 4 h wait.
# Re-run 3 (2026-10-18 ~12:53): Metal 124/124 (clip_multi included) and linux OK; clippy found 3 lints in
# ojas-io/src/sha256.rs, fixed in d0fbe97 (fix cycle 1 of 3; logs/rerun3/).
# Re-run 4 (~13:05): io 134/134 (sha256 unchanged); clippy then reached ojas-model and found 21 errors
# (result_large_err from AdapterError::QuantizerMismatch, is_multiple_of, as_chunks), fixed in the next
# commit (fix cycle 2 of 3; logs/rerun4/).
# Re-run 5 (~13:20): adapters 49/49; ojas-model lib clippy-clean; one lint left in tests/qwen35_optim.rs,
# fixed in 81a81ca (fix cycle 3 of 3, the last; logs/rerun5/).
# Re-run 6 (~13:35): clippy --keep-going over every target found exactly 2 errors (qwen35_bank
# single-element loop, qwen35_lora_merge_export useless vec!), fixed in 94aa3af with the owner awake
# (logs/rerun6/).
# Re-run 7 (~13:55): touched 17/17, clippy --keep-going clean; ojas main 94aa3af verified (logs/rerun7/).
# Re-run 8: 54fa635 (parts_words) against tessl bf091a2: the Metal clip and frozen suites, the
# qwen35 provider tests (tessl's qwen35_train changed), then clippy over the same crates.
# Re-run 8 passed (13:31; logs/rerun8/). Re-run 9: ojas main 4123dfb (phase A+B merged) against tessl
# 41d8866 (precise rsqrt in RMSNorm/qk-norm, wave 16 C/D, wave 17 E). Launched only when the disk has
# more than 50 GiB free (a cold build of ~10-15 GB; ENOSPC on 2026-10-18 blocked every session).
step qwen35cpu 7200 test -p ojas-qwen35 --no-fail-fast
step parity    7200 test -p ojas-qwen35 --release --test gpu_parity --no-fail-fast -- --ignored --test-threads=1
step provider  7200 test -p ojas-model --test qwen35_step_provider --no-fail-fast
step provgpu   7200 test -p ojas-qwen35 --release --test gpu_tape_provider --no-fail-fast -- --ignored --test-threads=1
step clippy  14400 clippy --keep-going -p ojas-io -p ojas-train -p ojas-autograd -p ojas-model -p ojas-metal -p ojas-qwen35 --all-targets -- -D warnings
cat "$LOGS/SUMMARY.txt"
