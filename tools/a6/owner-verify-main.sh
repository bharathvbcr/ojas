#!/bin/bash
# Verify the a6 commits now on local main (ffc6f2e..c695803), which were committed unverified on the owner's
# instruction. Run by the owner, from any directory:
#   ! MAC_HEAVY_MAX_LOAD=100 bash /Users/bharath/Code/research/ojas/backups/a6-tools/owner-verify-main.sh
# Every cargo step goes through mac_heavy.sh (lock per step, released between steps) with its own target dir
# (target-a6-main; no peer shares it) and a perl alarm. Logs and SUMMARY.txt go to backups/a6-tools/logs/.
set -u
R=/Users/bharath/Code/research
M="$R/ojas"
LOGS="$M/backups/a6-tools/logs"
mkdir -p "$LOGS"
: > "$LOGS/SUMMARY.txt"

step() { # step <name> <secs> <cargo subcommand> <args...>
  local name=$1 secs=$2 sub=$3; shift 3
  echo "=== $name" | tee -a "$LOGS/SUMMARY.txt"
  env CARGO_TARGET_DIR="$M/target-a6-main" perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$secs" \
    bash "$R/Lappi-decision/tools/mac_heavy.sh" "a6-main-$name" \
    cargo "$sub" --manifest-path "$M/Cargo.toml" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "    exit=$rc $(grep -E '^test result:' "$LOGS/$name.log" | awk '{p+=$4; f+=$6} END {if (NR) printf "passed=%d failed=%d (%d binaries)", p, f, NR}')" | tee -a "$LOGS/SUMMARY.txt"
  grep -E '^(error|warning: unused)|panicked at|^test .* FAILED' "$LOGS/$name.log" | head -20 >> "$LOGS/SUMMARY.txt"
}

# 1. Compile everything the a6 commits touch, with their tests (fast fail on compile errors).
step build    3600 test -p ojas-io -p ojas-train -p ojas-autograd -p ojas-model -p ojas-metal -p ojas-qwen35 --no-run
# 2. The new and touched suites.
step train    1200 test -p ojas-io -p ojas-train --no-fail-fast
step autograd 2400 test -p ojas-autograd --test linear_ce_value_only --test frozen_adversarial --test frozen --no-fail-fast
step model    3600 test -p ojas-model --test qwen35_adapters --test qwen35_adapters_adversarial --test qwen35_bank \
  --test qwen35_optim --test checkpoint --test save_adversarial --test qwen35_lora_fq8 --test qwen35_lora_fq8_metal \
  --test qwen35_lora --test qwen35_lora_scale1 --test qwen35_lora_adversarial --test qwen35_lora_metal \
  --test qwen35_readout --test qwen35_lora_merge_export --test lora_merge_adversarial --test adamw_torch_100 \
  --test preflight --test qwen35_load_pinned --test qwen35_load_adversarial --test qwen35_fixture --no-fail-fast
step metal    3600 test -p ojas-metal --release --test frozen_checked --test frozen_adversarial --test linear_ce \
  --test bf16_operands --no-fail-fast -- --test-threads=1
step metallib 2400 test -p ojas-metal --release --lib --no-fail-fast -- --test-threads=1
step qwen35   2400 test -p ojas-qwen35 --test cpu_groups --no-fail-fast
step lorarow  2400 test -p ojas-qwen35 --release --test gpu_memory_rows --no-fail-fast -- --ignored --test-threads=1 gpu_lora_step_trains_only_the_adapters
# 3. Whole touched crates (catches regressions outside the new tests).
step full     5400 test -p ojas-io -p ojas-train -p ojas-autograd -p ojas-model --no-fail-fast
# 4. Lints, the Linux check of ojas-train, and the sha256 bench.
step clippy   3600 clippy -p ojas-io -p ojas-train -p ojas-autograd -p ojas-model -p ojas-metal -p ojas-qwen35 --all-targets -- -D warnings
if rustup target list --installed | grep -qx aarch64-unknown-linux-gnu; then
  step linux 1200 check -p ojas-train --target aarch64-unknown-linux-gnu
else
  echo "=== linux: SKIPPED (aarch64-unknown-linux-gnu not installed)" | tee -a "$LOGS/SUMMARY.txt"
fi
step sha-bench 1200 run -p ojas-train --release --example sha256_throughput -- 7
tail -6 "$LOGS/sha-bench.log" >> "$LOGS/SUMMARY.txt"

echo; echo "Summary: $LOGS/SUMMARY.txt"; cat "$LOGS/SUMMARY.txt"
