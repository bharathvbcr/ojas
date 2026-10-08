#!/usr/bin/env bash
# Quiet E4B product A/B: GEMMA_METAL_GEMV_INTERLEAVE=0 (ship) vs =1 (Interleaved4 Hot).
# Pin GEMV_SIMD=1; fusion / encode-once / GEMV_BLOCKED / FUSE_GATE_DOWN OFF.
# Exactness under HAZARD=0 outranks tok/s. One Metal process; E4B only.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
STAMP="${GEMV_I4_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="bench/results/gemv_interleave_ab_e4b_${STAMP}"
mkdir -p bench/results

busy="$(pgrep -lf 'target/release/(bench|diag_tok)|fusion_ab\.sh|simd_occ_ab' 2>/dev/null || true)"
# Ignore this harness / tee so self-match does not false-positive.
busy="$(echo "$busy" | grep -v 'gemv_interleave_ab' || true)"
if [ -n "$busy" ] && [ "${FUSION_AB_ALLOW_BUSY:-0}" != "1" ]; then
  echo "FATAL: GPU busy — $busy"
  exit 1
fi

QUIET=(CARGO_TARGET_DIR=target GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0)
BASE_PIN=(
  GEMMA_METAL_FUSE_KV=1
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

run_arm() {
  local arm="$1" interleave="$2"
  echo "=== arm=$arm GEMV_INTERLEAVE=$interleave ==="
  echo "=== diag_tok $arm (HAZARD=0) ==="
  env "${QUIET[@]}" "${BASE_PIN[@]}" "${EXACT[@]}" \
    GEMMA_METAL_GEMV_INTERLEAVE="$interleave" \
    cargo run --release --bin diag_tok -- e4b \
    >"${OUT}_${arm}_diag.txt" 2>&1 || true
  echo "=== bench $arm (quiet shipping hazard) ==="
  env "${QUIET[@]}" "${BASE_PIN[@]}" \
    GEMMA_METAL_GEMV_INTERLEAVE="$interleave" \
    cargo run --release --bin bench -- --e4b \
    >"${OUT}_${arm}_bench.txt" 2>&1 || true
  python3 - <<PY
import re, pathlib
arm="$arm"
out=pathlib.Path("${OUT}")
bench=(out.parent / f"{out.name}_{arm}_bench.txt").read_text(errors="replace")
diag=(out.parent / f"{out.name}_{arm}_diag.txt").read_text(errors="replace")
ms=re.findall(r"decode:\\s*([0-9.]+)\\s*tok/s", bench)
tok=ms[-1] if ms else "NA"
tm=re.search(r"new tokens:\\s*(\\[[0-9,\\s]+\\])", diag)
tokens=tm.group(1).replace(" ","") if tm else "NA"
print(f"ARM {arm}: tok/s={tok} tokens={tokens}")
PY
}

echo "gemv_interleave_ab stamp=$STAMP out=$OUT"
run_arm ship 0
run_arm i4 1
echo "done — summarize next"
