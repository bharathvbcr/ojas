#!/usr/bin/env bash
# Regenerate every tiny golden fixture in ojas-oracle/fixtures/tiny, in order.
#
#   bash ojas-oracle/python/regen_tiny.sh
#
# PYTHON overrides the interpreter (default: the torch 2.13 / Python 3.14 one
# bench/ uses). CARGO_TARGET_DIR defaults to the oracle lane's target dir.
# Everything runs on CPU; nothing here touches a GPU.
set -euo pipefail

ORACLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/opt/homebrew/opt/python@3.14/bin/python3.14}"
export PYTHONDONTWRITEBYTECODE=1
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ORACLE/../target-oracle}"
F="$ORACLE/fixtures/tiny"
mkdir -p "$F"

# 1. synthetic token bin (16 KiB)
"$PY" "$ORACLE/python/golden.py" token-bin --out "$F/tokens.bin"
# 2. ojas-data BatchSampler starts for 40 steps of B=2, K=2, T=32, seed 1337
cargo run -q --release --manifest-path "$ORACLE/../Cargo.toml" -p ojas-oracle \
    --example dump_batch_starts -- \
    --bin "$F/tokens.bin" --seed 1337 --steps 40 --batch 2 --accum 2 --seq 32 \
    --out "$F/batch_starts.json" --rows
# 3. the torch replay reads the same tokens as ojas TokenBin, byte for byte
"$PY" "$ORACLE/python/batches.py" verify --starts "$F/batch_starts.json" --bin "$F/tokens.bin"
# 4. nanolab init, tiny spec, seed 1337
"$PY" "$ORACLE/python/export_init.py" --tiny --seed 1337 --out "$F/init.safetensors"
# 5. forward, grads, traces (cross-checked against nanolab train()), LR schedules
"$PY" "$ORACLE/python/golden.py" fixtures --dir "$F"
# 6. one stock nanolab Muon step (bf16 NS5) per case
"$PY" "$ORACLE/python/muon_step.py" --out "$F/muon_step_bf16.safetensors"
