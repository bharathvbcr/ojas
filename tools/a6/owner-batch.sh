#!/bin/bash
# ojas-a6 verification batch, wave Q1 (quantised Lappi LoRA path). Run by the owner:
#   ! bash backups/a6-tools/owner-batch.sh
# Every cargo step goes through mac_heavy.sh (one lock per step, released between steps),
# each lane has its own CARGO_TARGET_DIR, and every step is bounded by a perl alarm.
# Logs: backups/a6-tools/logs/<step>.log; a summary prints at the end.
set -u
R=/Users/bharath/Code/research
HEAVY="$R/Lappi-decision/tools/mac_heavy.sh"
LOGS="$R/ojas/backups/a6-tools/logs"
VENV=/Users/bharath/.venvs/ml/bin/python3          # mlx 0.31.2 (fake-quant only)
BREW=/opt/homebrew/opt/python@3.14/bin/python3.14  # torch 2.13: made the checked-in fixtures
# torch is in the user site (~/Library/Python/3.14), which -I drops: run BREW with -P only
# (no script-dir/cwd on sys.path), keep -I for the venv (its own site-packages hold mlx).
mkdir -p "$LOGS"
: > "$LOGS/SUMMARY.txt"

bounded() { # bounded <secs> <cmd...>
  perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$@"
}
step() { # step <name> <secs> <cmd...>
  local name=$1 secs=$2; shift 2
  echo "=== $name" | tee -a "$LOGS/SUMMARY.txt"
  bounded "$secs" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "    exit=$rc  $(grep -E '^test result:' "$LOGS/$name.log" | awk '{p+=$4; f+=$6} END {if (NR) printf "passed=%d failed=%d (%d binaries)", p, f, NR}')" | tee -a "$LOGS/SUMMARY.txt"
}
cargo_step() { # cargo_step <name> <worktree> <cargo subcommand> <args...>
  local name=$1 wt=$2 sub=$3; shift 3
  step "$name" 2400 env CARGO_TARGET_DIR="$R/$wt/target" bash "$HEAVY" "a6-$name" \
    cargo "$sub" --manifest-path "$R/$wt/Cargo.toml" "$@"
}

# 0. Format the touched Rust files (no compile).
rustfmt --edition 2024 "$R/ojas-a6-train/ojas-train/src/"*.rs "$R/ojas-a6-train/ojas-train/examples/sha256_throughput.rs"
rustfmt --edition 2021 "$R/ojas-a6-train/bench/sha256_throughput.rs" \
  "$R/ojas-a6-valuece/ojas-autograd/src/tape.rs" "$R/ojas-a6-valuece/ojas-autograd/tests/linear_ce_value_only.rs" \
  "$R/ojas-a6-adapters/ojas-model/src/qwen35/"{adapters,bank,optim,mod}.rs "$R/ojas-a6-adapters/ojas-model/src/checkpoint.rs" \
  "$R/ojas-a6-adapters/ojas-model/tests/"qwen35_{adapters,adapters_adversarial,bank,optim}.rs \
  "$R/ojas-a6-fq8/ojas-model/tests/qwen35_common/"{mod,lora}.rs "$R/ojas-a6-fq8/ojas-model/tests/"qwen35_lora_fq8{,_metal}.rs

# 1. Fake-quant fixture (lane 4). Generator tests, then the fq8 base (mlx), then torch refs over it.
FQ="$R/ojas-a6-fq8"
step fq8-gen-tests 600 bash -c "cd '$FQ' && '$VENV' -I ojas-oracle/python/test_qwen35_fakequant_base.py"
step fq8-base 600 bash -c "cd '$FQ' && '$VENV' -I ojas-oracle/python/qwen35_fakequant_base.py"
step fq8-refs 1200 bash -c "cd '$FQ' && '$BREW' -P ojas-oracle/python/qwen35_lora_fixture.py --base ojas-model/tests/fixtures/qwen35_tiny_fq8 --out ojas-model/tests/fixtures/qwen35_tiny_lora_fq8"
# Default arguments must reproduce the checked-in fixture byte for byte.
SCR="$LOGS/default-fixture"; rm -rf "$SCR"; mkdir -p "$SCR"
step fq8-default-regen 1200 bash -c "cd '$FQ' && '$BREW' -P ojas-oracle/python/qwen35_lora_fixture.py --out '$SCR'"
step fq8-default-cmp 60 bash -c "for f in '$SCR'/*; do cmp \"\$f\" '$FQ/ojas-model/tests/fixtures/qwen35_tiny_lora/'\"\$(basename \"\$f\")\" && echo \"same \$(basename \"\$f\")\"; done"

# 2. Tests, one lane at a time.
cargo_step train-test     ojas-a6-train    test -p ojas-train --no-fail-fast
cargo_step train-qwen35   ojas-a6-train    test -p ojas-qwen35 --test cpu_groups --no-fail-fast
cargo_step valuece-test   ojas-a6-valuece  test -p ojas-autograd --test linear_ce_value_only --test frozen_adversarial --no-fail-fast
cargo_step valuece-model  ojas-a6-valuece  test -p ojas-model --test qwen35_readout --no-fail-fast
cargo_step adapters-test  ojas-a6-adapters test -p ojas-model --test qwen35_adapters --test qwen35_adapters_adversarial --test qwen35_bank --test qwen35_optim --test checkpoint --test save_adversarial --no-fail-fast
cargo_step fq8-test       ojas-a6-fq8      test -p ojas-model --test qwen35_lora_fq8 --test qwen35_lora_fq8_metal --test qwen35_lora --test qwen35_lora_scale1 --test qwen35_lora_adversarial --test qwen35_lora_metal --test qwen35_readout --no-fail-fast

# 2b. Wave 2: Metal freeze_checked (lane C) and the doc sync (lane B, checked by sitegen after the merge).
rustfmt --edition 2021 "$R/ojas-a6-stepcost/ojas-metal/src/"{backend,device,link}.rs \
  "$R/ojas-a6-stepcost/ojas-metal/tests/frozen_checked.rs" "$R/ojas-a6-stepcost/ojas-qwen35/tests/gpu_memory_rows.rs"
cargo_step stepcost-test   ojas-a6-stepcost test -p ojas-metal --release --test frozen_checked --test frozen_adversarial --test linear_ce --test bf16_operands --no-fail-fast -- --test-threads=1
cargo_step stepcost-lib    ojas-a6-stepcost test -p ojas-metal --release --lib --no-fail-fast -- --test-threads=1
cargo_step stepcost-lora   ojas-a6-stepcost test -p ojas-qwen35 --release --test gpu_memory_rows --no-fail-fast -- --ignored --test-threads=1 gpu_lora_step_trains_only_the_adapters
cargo_step stepcost-clippy ojas-a6-stepcost clippy -p ojas-metal -p ojas-qwen35 --all-targets -- -D warnings

# 2c. Wave 3: memory pre-flight, AdamW vs torch over 100 steps, LoRA merge for export.
rustfmt --edition 2021 "$R/ojas-a6-preflight/ojas-model/src/"{preflight,lib}.rs "$R/ojas-a6-preflight/ojas-model/tests/preflight.rs" \
  "$R/ojas-a6-adamw100/ojas-model/tests/adamw_torch_100.rs" \
  "$R/ojas-a6-merge/ojas-model/src/lora.rs" "$R/ojas-a6-merge/ojas-model/src/qwen35/params.rs" \
  "$R/ojas-a6-merge/ojas-model/tests/"*merge*.rs
step adamw-gen 900 bash -c "cd '$R/ojas-a6-adamw100' && '$BREW' -P ojas-oracle/python/adamw_oracle.py"
cargo_step preflight-test  ojas-a6-preflight test -p ojas-model --test preflight --no-fail-fast
cargo_step adamw-test      ojas-a6-adamw100  test -p ojas-model --test adamw_torch_100 --no-fail-fast
cargo_step merge-test      ojas-a6-merge     test -p ojas-model --test qwen35_lora_merge_export --test lora_merge_adversarial --test qwen35_lora --no-fail-fast
cargo_step preflight-clippy ojas-a6-preflight clippy -p ojas-model --all-targets -- -D warnings
cargo_step adamw-clippy     ojas-a6-adamw100  clippy -p ojas-model --tests -- -D warnings
cargo_step merge-clippy     ojas-a6-merge     clippy -p ojas-model --all-targets -- -D warnings

# 3. Clippy.
cargo_step train-clippy    ojas-a6-train    clippy -p ojas-train -p ojas-qwen35 --all-targets -- -D warnings
cargo_step valuece-clippy  ojas-a6-valuece  clippy -p ojas-autograd --all-targets -- -D warnings
cargo_step adapters-clippy ojas-a6-adapters clippy -p ojas-model --all-targets -- -D warnings
cargo_step fq8-clippy      ojas-a6-fq8      clippy -p ojas-model --tests -- -D warnings

# 4. ojas-train builds off macOS (no tessl, no cfg(target_os)).
if rustup target list --installed | grep -qx aarch64-unknown-linux-gnu; then
  cargo_step train-linux ojas-a6-train check -p ojas-train --target aarch64-unknown-linux-gnu
else
  echo "=== train-linux: SKIPPED (aarch64-unknown-linux-gnu not installed)" | tee -a "$LOGS/SUMMARY.txt"
fi

# 5. sha256 throughput (min/median of 7 rounds over 64 MiB).
cargo_step train-bench ojas-a6-train run -p ojas-train --release --example sha256_throughput -- 7
tail -6 "$LOGS/train-bench.log" >> "$LOGS/SUMMARY.txt"

echo; echo "Summary: $LOGS/SUMMARY.txt"; cat "$LOGS/SUMMARY.txt"
