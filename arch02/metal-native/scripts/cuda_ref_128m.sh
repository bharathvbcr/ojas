#!/usr/bin/env bash
# Same-shape 128M CUDA reference — RUN THIS ON THE 3070 Ti BOX, not the Mac.
#
# Why this exists: the "~650-840 ms" and "BPB 1.9944" 3070 Ti figures in this
# repo are for `--preset sota` (4 layers x 128 dim x mlp 384). arch02-128m is
# 24 x 768 x 2304 — roughly 100x the parameters. Comparing them is invalid.
# This produces a real same-shape number.
#
# On the CUDA machine (from anywhere in this monorepo, or after copying this
# script into parameter-golf/):
#
#   ./cuda_ref_128m.sh probe      # ~2 min  can 8 GB hold it? finds the accum setting
#   ./cuda_ref_128m.sh bench      # ~5 min  throughput at the chosen accum
#   ./cuda_ref_128m.sh quality    # ~long   2000-step BPB, matches the Mac champion
#
# Then on the Mac, ingest:
#   ./scripts/ingest_cuda128m.sh /path/to/parameter-golf/logs/cuda128m
#
# Report alongside any Mac number: GPU, VRAM, accum steps, torch version.
set -euo pipefail

# Resolve real path so parameter-golf/scripts symlink still finds metal-native/.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
if command -v readlink >/dev/null 2>&1; then
  _real=$(readlink "$0" 2>/dev/null || true)
  if [ -n "${_real:-}" ]; then
    case "$_real" in
      /*) SCRIPT_DIR=$(cd "$(dirname "$_real")" && pwd) ;;
      *)  SCRIPT_DIR=$(cd "$(dirname "$0")/$(dirname "$_real")" && pwd) ;;
    esac
  fi
fi
# Resolve parameter-golf/ whether this script lives in metal-native/scripts
# or was copied into parameter-golf/.
find_pg() {
  local d
  for d in \
    "$PWD" \
    "$SCRIPT_DIR" \
    "$SCRIPT_DIR/.." \
    "$SCRIPT_DIR/../../.." \
    "$SCRIPT_DIR/../../../parameter-golf" \
    "$SCRIPT_DIR/../../../../parameter-golf"
  do
    if [ -f "$d/train_gpt.py" ]; then
      (cd "$d" && pwd)
      return 0
    fi
  done
  return 1
}

PG=$(find_pg) || {
  echo "train_gpt.py not found. Run from parameter-golf/ or keep this script"
  echo "inside the parameter_golf monorepo (metal-native/scripts/)."
  exit 1
}
cd "$PG"

# Canonical in-repo ingest home on Mac (metal-native). When running on the
# CUDA box, default writes under parameter-golf/logs/cuda128m; Mac ingest
# copies into metal-native/logs/cuda128m.
METAL_LOGS=
if [ -d "$SCRIPT_DIR/../logs" ] || [ -d "$SCRIPT_DIR/.." ]; then
  METAL_LOGS=$(cd "$SCRIPT_DIR/.." && pwd)/logs/cuda128m
fi
OUT=${OUT_ROOT:-"$PG/logs/cuda128m"}
mkdir -p "$OUT"

# Shapes matched to arch02-128m exactly:
#   head_dim = MODEL_DIM/NUM_HEADS = 768/24 = 32
#   mlp      = MODEL_DIM*MLP_MULT  = 768*3  = 2304
#   kv_dim   = NUM_KV_HEADS*head_dim = 12*32 = 384
shape_env() {
  echo "VOCAB_SIZE=1024 NUM_LAYERS=24 MODEL_DIM=768 NUM_HEADS=24 NUM_KV_HEADS=12"
  echo "MLP_MULT=3 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 ROPE_BASE=10000.0"
  echo "TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=4096"
}

run_cuda() { # accum iters log_every val_every tag
  local accum=$1 iters=$2 logev=$3 valev=$4 tag=$5
  if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "BLOCKER: no CUDA GPU visible (torch.cuda.is_available() is False)."
    echo "This host cannot run probe/bench/quality. Use the 3070 Ti box."
    echo "Mac prep path: $METAL_LOGS (ingest with scripts/ingest_cuda128m.sh)."
    exit 2
  fi
  echo "=== accum=$accum iters=$iters -> $OUT/$tag ==="
  echo "DISCLOSE: GRAD_ACCUM_STEPS=$accum"
  VOCAB_SIZE=1024 NUM_LAYERS=24 MODEL_DIM=768 NUM_HEADS=24 NUM_KV_HEADS=12 \
  MLP_MULT=3 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 ROPE_BASE=10000.0 \
  TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS="$accum" \
  ITERATIONS="$iters" WARMUP_STEPS=10 WARMDOWN_ITERS=0 MAX_WALLCLOCK_SECONDS=0 \
  VAL_LOSS_EVERY="$valev" TRAIN_LOG_EVERY="$logev" SEED=1337 \
  DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}" OUT_DIR="$OUT/$tag" \
  python3 train_gpt.py 2>&1 | tee "$OUT/$tag.log" | tail -20
}

probe() {
  echo "=== VRAM probe: smallest GRAD_ACCUM_STEPS that fits ==="
  echo "The Mac run holds this model at ~13.5 GB unified. On 8 GB this may need"
  echo "accumulation — which costs CUDA throughput and MUST be disclosed."
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true
  python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)" 2>/dev/null || true
  if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "BLOCKER: no CUDA GPU on this machine."
    echo "Recorded for Mac-side STATUS; probe/bench/quality must run on 3070 Ti."
    mkdir -p "$OUT"
    cat > "$OUT/BLOCKER.txt" <<EOF
no_cuda_on_host
host=$(uname -s)-$(uname -m)
probed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
action=run_on_3070_ti
script=cuda_ref_128m.sh probe && bench && quality
mac_ingest=arch02/metal-native/scripts/ingest_cuda128m.sh
EOF
    if [ -n "$METAL_LOGS" ]; then
      mkdir -p "$METAL_LOGS"
      cp "$OUT/BLOCKER.txt" "$METAL_LOGS/BLOCKER.txt"
    fi
    exit 2
  fi
  for accum in 1 2 4 8; do
    echo "--- trying GRAD_ACCUM_STEPS=$accum"
    if run_cuda "$accum" 12 5 0 "probe_accum$accum" >/dev/null 2>&1; then
      echo "FITS at GRAD_ACCUM_STEPS=$accum"
      echo "$accum" > "$OUT/.accum"
      echo "DISCLOSE: GRAD_ACCUM_STEPS=$accum (write this next to every CUDA ms/step and BPB)"
      return 0
    fi
    echo "    OOM or failure at accum=$accum"
  done
  echo
  echo "Did not fit at any tested accumulation."
  echo "That is a REPORTABLE RESULT, not a failure: state that the 128M config"
  echo "does not fit an 8 GB card, while 64 GB unified memory holds it outright."
  echo "none" > "$OUT/.accum"
  return 0
}

bench() {
  local accum; accum=$(cat "$OUT/.accum" 2>/dev/null || echo 1)
  echo "using GRAD_ACCUM_STEPS=$accum (from probe; override with ACCUM=n)"
  accum=${ACCUM:-$accum}
  if [ "$accum" = "none" ]; then
    echo "probe recorded no fitting accum — refuse bench"; exit 1
  fi
  run_cuda "$accum" 200 20 0 "bench_accum$accum"
  echo
  echo "Take the steady-state ms/step and tok/s from the log's later steps."
  echo "Disclose accum=$accum next to the number."
}

quality() {
  local accum; accum=$(cat "$OUT/.accum" 2>/dev/null || echo 1)
  accum=${ACCUM:-$accum}
  if [ "$accum" = "none" ]; then
    echo "probe recorded no fitting accum — refuse quality"; exit 1
  fi
  echo "=== 2000 steps for a same-shape BPB, comparable to the Mac champion ==="
  echo "DISCLOSE: GRAD_ACCUM_STEPS=$accum"
  run_cuda "$accum" 2000 100 500 "quality_accum$accum"
  echo
  echo "Compare val_bpb against the Mac champion's final_ema_sliding_bpb."
  echo "Note the evaluation paths differ (this script's val vs the native"
  echo "EMA-sliding eval) — say so, or re-evaluate both with one scorer:"
  echo "  ./scripts/score_cuda128m.py $OUT"
}

case "${1:-probe}" in
  probe) probe ;;
  bench) bench ;;
  quality) quality ;;
  env) shape_env ;;
  *) echo "usage: $0 [probe|bench|quality|env]"; exit 1 ;;
esac
