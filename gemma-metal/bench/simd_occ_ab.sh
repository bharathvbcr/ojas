#!/usr/bin/env bash
# Simd occupancy A/B (no layout swap): shipping rows=4×sg=2 vs r2 / sg4.
# One Metal process at a time. Fusion / encode-once / GEMV_BLOCKED stay OFF.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
STAMP="${SIMD_OCC_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="bench/results/simd_occ_ab_e4b_${STAMP}"
mkdir -p bench/results

busy="$(pgrep -lf 'target/release/(bench|diag_tok)|fusion_ab\.sh' 2>/dev/null || true)"
if [ -n "$busy" ] && [ "${FUSION_AB_ALLOW_BUSY:-0}" != "1" ]; then
  echo "FATAL: GPU busy — $busy"
  exit 1
fi

QUIET=(CARGO_TARGET_DIR=target GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0)
PIN=(
  GEMMA_METAL_FUSE_KV=1
  GEMMA_METAL_GEMV_INTERLEAVE=0
  GEMMA_METAL_GEMV_SIMD=1
  GEMMA_METAL_GEMV_BLOCKED=0
  GEMMA_METAL_FUSE_QKV=0
  GEMMA_METAL_FUSE_PLE=0
  GEMMA_METAL_FUSE_ROPE_KV=0
  GEMMA_METAL_FUSE_LAYER=0
  GEMMA_METAL_FUSE_GATE_DOWN=0
  GEMMA_METAL_ENCODE_ONCE=0
)
EXACT=(METAL_RUNTIME_HAZARD_BARRIERS=0)

apply_occ() {
  local arm="$1" rows sg tptg
  case "$arm" in
    ship) rows=4; sg=2; tptg=64 ;;
    r2)   rows=2; sg=2; tptg=64 ;;
    sg4)  rows=4; sg=4; tptg=128 ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  # Metal constants
  perl -i -pe "s/constant uint SIMD_ROWS = \\d+u;/constant uint SIMD_ROWS = ${rows}u;/" \
    kernels/gemv_q4_mlx.metal
  perl -i -pe "s/constant uint SIMD_SG_PER_TG = \\d+u;/constant uint SIMD_SG_PER_TG = ${sg}u;/" \
    kernels/gemv_q4_mlx.metal
  # Host dispatch must match
  perl -i -pe "s/const GEMV_SIMD_ROWS: u32 = \\d+;/const GEMV_SIMD_ROWS: u32 = ${rows};/" \
    src/kernels.rs
  perl -i -pe "s/const GEMV_SIMD_SG: u32 = \\d+;/const GEMV_SIMD_SG: u32 = ${sg};/" \
    src/kernels.rs
  perl -i -pe "s/const GEMV_SIMD_TPTG: usize = \\d+;/const GEMV_SIMD_TPTG: usize = ${tptg};/" \
    src/kernels.rs
  local rpt=$((rows * sg))
  perl -i -pe "s/let rpt = \\d+u32; \\/\\/ GEMV_SIMD_SG \\* GEMV_SIMD_ROWS/let rpt = ${rpt}u32; \\/\\/ GEMV_SIMD_SG * GEMV_SIMD_ROWS/" \
    src/gpu_model.rs
  echo "applied arm=$arm rows=$rows sg=$sg tptg=$tptg rpt=$rpt"
}

restore_ship() {
  apply_occ ship
}

trap 'echo "restoring ship constants…"; restore_ship' EXIT

run_arm() {
  local arm="$1"
  apply_occ "$arm"
  echo "=== build $arm ==="
  if ! env "${QUIET[@]}" cargo build --release --bin bench --bin diag_tok 2>&1 | tail -8; then
    echo "FATAL: build failed for $arm"
    exit 1
  fi
  echo "=== diag_tok $arm (HAZARD=0) ==="
  env "${QUIET[@]}" "${PIN[@]}" "${EXACT[@]}" \
    cargo run --release --bin diag_tok -- e4b \
    >"${OUT}_${arm}_diag.txt" 2>&1 || true
  echo "=== bench $arm (quiet shipping hazard) ==="
  env "${QUIET[@]}" "${PIN[@]}" \
    cargo run --release --bin bench -- --e4b \
    >"${OUT}_${arm}_bench.txt" 2>&1 || true
  # Extract tok/s + token list if present
  python3 - <<PY
import re, pathlib
arm="$arm"
out=pathlib.Path("${OUT}")
bench=(out.parent / f"{out.name}_{arm}_bench.txt").read_text(errors="replace")
diag=(out.parent / f"{out.name}_{arm}_diag.txt").read_text(errors="replace")
# Prefer real E4B decode line (last match); ignore mini-graph.
ms=re.findall(r"decode:\\s*([0-9.]+)\\s*tok/s", bench)
tok=ms[-1] if ms else "NA"
tm=re.search(r"new tokens:\\s*(\\[[0-9,\\s]+\\])", diag)
tokens=tm.group(1).replace(" ","") if tm else "NA"
print(f"ARM {arm}: tok/s={tok} tokens={tokens}")
PY
}

echo "simd_occ_ab stamp=$STAMP out=$OUT"
run_arm ship
run_arm r2
run_arm sg4
# trap restores ship + rebuild not forced; leave tree at ship constants
echo "done — summarize next"
