#!/bin/bash
# ft-2bbced18 criteria 1 and 4: interleaved A/B, min-of-5, at Qwen3.5-2B-Base, T = 2,048 and 9,638,
# micro-batch 1, 28 GiB MetalBackend budget. GPU, one arm per process, ~3-5 h total. Run by the owner
# when the Mac is free (never alongside another GPU job):
#   ! bash backups/a6-tools/owner-bench-stepcost.sh
# Criterion 1 (freeze_checked): tape-lora vs tape-lora-checked. Criterion 4 (dW skip): tape-lora vs tape-lora-fulldw
# (fulldw may refuse at T = 9,638 under the budget; that refusal is itself the row).
set -u
R=/Users/bharath/Code/research
WT="$R/ojas-a6-stepcost"     # before the merge; after it, set WT=$R/ojas and a fresh target dir
OUT="$R/ojas/backups/a6-tools/logs/bench-stepcost"
mkdir -p "$OUT"
for tok in 2048 9638; do
  for i in 1 2 3 4 5; do
    for arm in tape-lora tape-lora-checked tape-lora-fulldw; do
      perl -e 'alarm shift; exec @ARGV or die' 3600 env CARGO_TARGET_DIR="$WT/target" \
        QWEN35_MEM_ARM=$arm QWEN35_MEM_TOKENS=$tok QWEN35_MEM_PHASES=on \
        bash "$R/Lappi-decision/tools/mac_heavy.sh" "a6-$arm-$tok-$i" \
        cargo test --manifest-path "$WT/Cargo.toml" -p ojas-qwen35 --release --test gpu_memory_rows \
        -- --ignored --test-threads=1 --nocapture gpu_memory_row > "$OUT/$arm-$tok-$i.log" 2>&1
      echo "$arm T=$tok rep $i exit=$? $(grep -m1 '^ROW' "$OUT/$arm-$tok-$i.log" | cut -c1-200)"
    done
  done
done | tee "$OUT/rows.txt"
echo "Rows: $OUT/rows.txt (min min_step_ms per arm and T is the reading)"
