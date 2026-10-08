#!/usr/bin/env bash
# Audit 7 — exact-128M backward optimization A/B harness (M5 Pro).
#
# Flags (see src/ab_flags.rs). NOTE: as of M16 the Audit 7 KEEPs are
# **default ON** — CAST_ONCE, FA_FAST, FA_BF16. Baseline arms below set them to 0.
#   METAL_NATIVE_BWD_PROFILE=1    per-section backward timing (diagnostic)
#   METAL_NATIVE_BWD_CAST_ONCE=1  single bf16 cast for shared grad operands
#   METAL_NATIVE_GEMM_ACCUM_DX=1  multiply_accumulate for dX only (dW stays temp+add)
#   METAL_NATIVE_GEMM_ACCUM=1     Audit 6 full accum (known Soft-regress; speed ref)
#   METAL_NATIVE_FA_TILED=1       tiled FA bwd (16M Soft quality winner; untested at 128M)
#   METAL_NATIVE_FA_FAST=1        head-dim-32 row FA bwd, identical numerics (Audit 7)
#   METAL_NATIVE_FA_BF16=1        as FA_FAST + bf16 Q/K/V (matches the bf16 fwd's LSE)
#
# Usage:
#   ./scripts/bench_128m_ab.sh build        # lib gate + release build (run first)
#   ./scripts/bench_128m_ab.sh profile      # where do the ~2.2s backward go
#   ./scripts/bench_128m_ab.sh speed        # bench ladder over the speed flags
#   ./scripts/bench_128m_ab.sh cast-parity  # CAST_ONCE must be bit-identical (3-step loss diff)
#   ./scripts/bench_128m_ab.sh quality      # 2×500-step probes: default vs FA_TILED (slow, ~1 h)
#   ./scripts/bench_128m_ab.sh fa           # Audit 7 FA-bwd ladder (the 70% kernel)
#   ./scripts/bench_128m_ab.sh fa-quality   # 500-step BPB probe for the FA winner
#   ./scripts/bench_128m_ab.sh all          # build + profile + speed + cast-parity
#
# Gates (docs/optimization_map.md conventions):
#   - Speed flip KEEPs only on a clear ms/step win at equal loss trajectory.
#   - CAST_ONCE requires EXACT loss match in cast-parity before any KEEP.
#   - ACCUM_DX requires the quality probe (or a 1k Soft ladder) before default-on.
#   - Record every result in docs/optimization_map.md “Live baselines”.

set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}

ROOT=${PARAMETER_GOLF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
DATA=(--data-dir "$ROOT/parameter-golf/data/datasets/fineweb10B_sp1024"
      --token-bytes "$ROOT/Rust_MLKit/arch_02_value_resid/burn-port/token_bytes.json")
PRESET=(--preset arch02-128m --batch 16 --seq-len 256)
CHAMP=(--optimizer muon_polar_adamw --matrix-lr 0.05 --ema-decay 0.997)
BIN=target/release/train
OUT=out/audit7
mkdir -p "$OUT"

build() {
  echo "== lib gate (must stay 57/57 green) =="
  cargo test --release --lib
  cargo build --release --bin train
}

# Never trust an existing binary: kernels/ or src/ may have changed since it was
# built (the funnel has the same rule). cargo is a no-op when nothing changed.
ensure_build() {
  cargo build --release --bin train >/dev/null 2>&1 || {
    echo "build failed; re-running verbosely:" >&2
    cargo build --release --bin train
    exit 1
  }
}

bench_one() { # name env...
  local name=$1; shift
  echo
  echo "== bench: $name =="
  env "$@" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --bench --bench-steps 8 --seed 1337 --log-every 1 --out "$OUT/bench_$name" \
    2>&1 | tail -6
}

profile() {
  ensure_build
  # Default to the SHIPPED stack — profiling bare defaults just re-measures the
  # pre-Audit-7 path, which is no longer what the champion runs.
  local mode="${1:-shipped}"
  local envs=(METAL_NATIVE_BWD_PROFILE=1 METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_FA_BF16=1)
  local tag=shipped
  if [ "$mode" = baseline ]; then
    envs=(METAL_NATIVE_BWD_PROFILE=1 METAL_NATIVE_BWD_CAST_ONCE=0 METAL_NATIVE_FA_FAST=0 METAL_NATIVE_FA_BF16=0)
    tag=baseline
  fi
  echo "== BWD_PROFILE ($tag): per-section backward shares (synced; not a speed gate) =="
  env "${envs[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --bench --bench-steps 4 --seed 1337 --log-every 1 --out "$OUT/profile_$tag" 2>&1 \
    | tee "$OUT/profile_$tag.log" | { grep -A 32 "bwd_profile" || echo "(no bwd_profile output — is METAL_NATIVE_BWD_PROFILE wired in this binary?)"; } | tail -40
  echo
  echo "full log: $OUT/profile_$tag.log"
  echo "compare:  $0 profile baseline"
  echo "For NAX utilization: run the same command under Instruments"
  echo "→ Metal System Trace + Neural Accelerators counter (see src/log.rs docs)."
}

speed() {
  ensure_build
  bench_one baseline           METAL_NATIVE_BWD_CAST_ONCE=0 METAL_NATIVE_FA_FAST=0 METAL_NATIVE_FA_BF16=0
  bench_one cast_once          METAL_NATIVE_BWD_CAST_ONCE=1
  bench_one accum_dx           METAL_NATIVE_GEMM_ACCUM_DX=1
  bench_one cast_once_accum_dx METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_GEMM_ACCUM_DX=1
  bench_one accum_full_ref     METAL_NATIVE_GEMM_ACCUM=1   # speed ref only; Soft-REJECT default
  bench_one fa_tiled           METAL_NATIVE_FA_TILED=1
  echo
  echo "Compare ms/step + dispatches vs the 2816 ms / 1701 disp M13 smoke gate."
}

cast_parity() {
  ensure_build
  echo "== CAST_ONCE parity: 3 steps, losses must match EXACTLY =="
  for mode in off on; do
    envs=(METAL_NATIVE_BWD_CAST_ONCE=0)
    if [ "$mode" = on ]; then envs=(METAL_NATIVE_BWD_CAST_ONCE=1); fi
    env "${envs[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
      --total-steps 3 --seed 1337 --log-every 1 --eval-every 9999 \
      --no-final-weight-save --out "$OUT/castparity_$mode" >/dev/null 2>&1
  done
  python3 - "$OUT" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
def losses(d):
    return [json.loads(l)["loss"] for l in open(d/"metrics.jsonl") if '"loss"' in l and "final_ema" not in l]
a, b = losses(out/"castparity_off"), losses(out/"castparity_on")
print("off:", a); print("on: ", b)
# Fresh Metal queues are not bit-deterministic (DECISIONS M13: replay
# equivalence at loss atol 1e-5). Measured off-vs-off run noise ~4e-6;
# CAST_ONCE off-vs-on Δ ~3e-6 (2026-07-19) — inside run-to-run noise.
worst = max(abs(x - y) for x, y in zip(a, b))
print(f"max |Δloss| = {worst:.3e}")
print("VERDICT:", "PASS — within replay-equivalence atol 1e-5" if worst < 1e-5 else
      "FAIL — exceeds documented replay atol; investigate before enabling")
EOF
}

fa() {
  ensure_build
  echo "== Audit 7 FA-bwd ladder (fa_dqdkv was ~70% of backward) =="
  echo "-- kernel-level parity gate first --"
  cargo test --release --lib fa_bwd_row_d32_matches_generic -- --nocapture 2>&1 | tail -12
  bench_one fa_baseline  METAL_NATIVE_BWD_CAST_ONCE=0 METAL_NATIVE_FA_FAST=0 METAL_NATIVE_FA_BF16=0
  bench_one fa_fast      METAL_NATIVE_FA_FAST=1
  bench_one fa_bf16      METAL_NATIVE_FA_BF16=1
  # Stack the FA winner on the Audit 7 GEMM flips.
  bench_one fa_bf16_full METAL_NATIVE_FA_BF16=1 METAL_NATIVE_BWD_CAST_ONCE=1 \
                         METAL_NATIVE_GEMM_ACCUM_DX=1
  # Tiled + bf16: the Phase 4 twins, unreferenced before Audit 7.
  bench_one fa_tiled_bf16 METAL_NATIVE_FA_TILED=1 METAL_NATIVE_FA_BF16=1
  echo
  echo "FA_FAST must be a pure speed win (identical numerics)."
  echo "FA_BF16 trades bf16 Q/K/V for bandwidth — gate it with 'fa-quality'."
  echo "Re-run 'profile' after: fa_dqdkv should drop well below its 70% share."
}

fa_quality() {
  ensure_build
  # NOTE: these probes stack CAST_ONCE+ACCUM_DX, so they are NOT comparable to
  # the clean 2.4209/2.4163 baselines from `quality`. They compare FA variants
  # against *each other*. Use `seed-repeat` to isolate a single flag.
  echo "== 500-step BPB probes for the FA variants (~25 min each) =="
  for name in fa_fast fa_bf16 fa_tiled_bf16; do
    case $name in
      fa_fast) envs=(METAL_NATIVE_FA_FAST=1) ;;
      fa_bf16) envs=(METAL_NATIVE_FA_BF16=1) ;;
      fa_tiled_bf16) envs=(METAL_NATIVE_FA_TILED=1 METAL_NATIVE_FA_BF16=1) ;;
    esac
    echo "-- $name --"
    env "${envs[@]}" METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_GEMM_ACCUM_DX=1 \
      "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
      --total-steps 500 --seed 1337 --log-every 50 --eval-every 250 \
      --no-final-weight-save --out "$OUT/quality_$name" 2>&1 | tail -3
  done
  echo
  echo "Baselines at 500 steps: row default 2.4209 · FA_TILED 2.4163."
  echo "FA_FAST must land on 2.4209 (identical math). FA_BF16 is the real"
  echo "question: bf16 bwd matches the bf16 fwd's taped LSE, so it may improve."
}

quality() {
  ensure_build
  echo "== 500-step champion-config probes (~25 min each at ~3 s/step) =="
  for name in default fa_tiled; do
    envs=(NOOP=1)
    if [ "$name" = fa_tiled ]; then envs=(METAL_NATIVE_FA_TILED=1); fi
    echo "-- $name --"
    env "${envs[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
      --total-steps 500 --seed 1337 --log-every 50 --eval-every 250 \
      --no-final-weight-save --out "$OUT/quality_$name" 2>&1 | tail -3
  done
  echo
  echo "Compare final_ema_sliding_bpb in $OUT/quality_*/metrics.jsonl."
  echo "Champion reference: 2.0158 @2000 steps (no same-shape CUDA 128M ref yet;"
  echo "do not quote sota-ladder 1.9944 against 128M). FA_TILED won"
  echo "-0.013 BPB at 16M Soft; a win here motivates a FA_TILED champion rerun."
}

fwd() {
  ensure_build
  local SHIP=(METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_FA_BF16=1)
  echo "== Audit 8 forward: parity gate =="
  cargo test --release --lib fa_fwd_d32_matches_generic -- --nocapture 2>&1 | tail -10
  echo
  echo "== forward section profile (shipped bwd stack, generic fwd) =="
  env "${SHIP[@]}" METAL_NATIVE_FWD_PROFILE=1 "$BIN" "${PRESET[@]}" "${DATA[@]}" \
    "${CHAMP[@]}" --bench --bench-steps 4 --seed 1337 --log-every 1 \
    --out "$OUT/fwdprofile_generic" 2>&1 \
    | tee "$OUT/fwdprofile_generic.log" | { grep -A 16 "fwd_profile" || echo "(no fwd_profile output — is METAL_NATIVE_FWD_PROFILE wired in this binary?)"; } | tail -18
  echo
  echo "== forward section profile (FA_FWD_FAST) =="
  env "${SHIP[@]}" METAL_NATIVE_FWD_PROFILE=1 METAL_NATIVE_FA_FWD_FAST=1 \
    "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" --bench --bench-steps 4 \
    --seed 1337 --log-every 1 --out "$OUT/fwdprofile_fast" 2>&1 \
    | tee "$OUT/fwdprofile_fast.log" | { grep -A 16 "fwd_profile" || echo "(no fwd_profile output — is METAL_NATIVE_FWD_PROFILE wired in this binary?)"; } | tail -18
  echo
  echo "== step-time A/B =="
  bench_one fwd_ship      "${SHIP[@]}"
  bench_one fwd_fast      "${SHIP[@]}" METAL_NATIVE_FA_FWD_FAST=1
  echo
  echo "FA_FWD_FAST f32 is same-math; under Bf16 it also enables bf16 fwd flash"
  echo "(first path that reaches it — use_bf16_flash is hard-coded false)."
}

glue() {
  ensure_build
  echo "== Audit 8 resid_glue: atomics -> row-block reduction =="
  echo "-- parity gate (elementwise must be byte-identical; reductions <=1e-5) --"
  cargo test --release --lib glue_bwd_rowblock_matches_inline_atomics -- --nocapture 2>&1 | tail -10
  bench_one glue_inline    METAL_NATIVE_GLUE_ROWBLOCK=0
  bench_one glue_rb16      METAL_NATIVE_GLUE_ROWBLOCK=1 METAL_NATIVE_GLUE_ROWBLOCKS=16
  bench_one glue_rb32      METAL_NATIVE_GLUE_ROWBLOCK=1 METAL_NATIVE_GLUE_ROWBLOCKS=32
  bench_one glue_rb64      METAL_NATIVE_GLUE_ROWBLOCK=1 METAL_NATIVE_GLUE_ROWBLOCKS=64
  echo
  echo "Inline path issues 3.1M device atomics/call (6.2M for the dmix twin);"
  echo "row-blocks issue C*rb (768*32 = 24576). Sweep rb for the parallelism/"
  echo "atomic-count tradeoff, then re-run 'profile' — resid_glue was 14% of bwd."
}

seed_repeat() {
  ensure_build
  # The 500-step single-seed deltas (~0.004-0.005 BPB) are the same magnitude
  # the funnel already called "statistically tied" at 1000 steps, so they cannot
  # decide anything alone. This mirrors the funnel's two-seed methodology on the
  # two open questions, holding CAST_ONCE fixed (parity-clean, free speed):
  #   ctl      = CAST_ONCE                  (control)
  #   accumdx  = CAST_ONCE + ACCUM_DX       (does dX accumulate cost quality?)
  #   fabf16   = CAST_ONCE + FA_BF16        (does bf16 FA bwd help?)
  # 6 runs x ~23 min ~= 2.3 h.
  for seed in 42 2026; do
    for name in ctl accumdx fabf16; do
      case $name in
        ctl)     envs=(METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_FA_BF16=0 METAL_NATIVE_FA_FAST=1) ;;
        accumdx) envs=(METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_FA_BF16=0 METAL_NATIVE_FA_FAST=1 METAL_NATIVE_GEMM_ACCUM_DX=1) ;;
        fabf16)  envs=(METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_FA_BF16=1 METAL_NATIVE_FA_FAST=1) ;;
      esac
      echo "-- $name seed $seed --"
      env "${envs[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
        --total-steps 500 --seed "$seed" --log-every 100 --eval-every 500 \
        --no-final-weight-save --out "$OUT/seed${seed}_$name" 2>&1 | tail -2
    done
  done
  echo
  echo "Read with: $0 summary   (compare per-seed means, not single runs)"
}

summary() {
  python3 - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
if not out.exists():
    sys.exit(f"no results yet at {out}")
bench, qual = [], []
for d in sorted(out.iterdir()):
    f = d / "metrics.jsonl"
    if not f.is_file():
        continue
    steps, final = [], None
    with f.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "final_ema_sliding_bpb" in r:
                final = r["final_ema_sliding_bpb"]
            elif "step_ms" in r:
                steps.append(r)
    if final is not None:
        qual.append((d.name, final, len(steps)))
    if steps:
        tail = steps[-4:]
        ms = sum(s["step_ms"] for s in tail) / len(tail)
        bwd = [dict(s.get("phase_ms", [])).get("backward") for s in tail]
        bwd = [x for x in bwd if x]
        bench.append((d.name, ms, sum(bwd) / len(bwd) if bwd else 0.0,
                      tail[-1].get("dispatches", 0)))

if bench:
    print("\n== step time (mean of last 4 logged steps) ==")
    print(f"{'run':<26}{'ms/step':>10}{'bwd ms':>10}{'disp':>8}{'vs base':>10}")
    base = next((b[1] for b in bench if "baseline" in b[0]), None)
    for name, ms, bwd, disp in sorted(bench, key=lambda x: x[1]):
        rel = f"{(ms/base-1)*100:+.1f}%" if base else "-"
        print(f"{name:<26}{ms:>10.1f}{bwd:>10.1f}{disp:>8}{rel:>10}")
if qual:
    print("\n== final EMA sliding BPB (same-shape CUDA 128M ref: null until logs/cuda128m) ==")
    print(f"{'run':<26}{'BPB':>10}{'steps':>8}")
    for name, bpb, n in sorted(qual, key=lambda x: x[1]):
        print(f"{name:<26}{bpb:>10.4f}{n:>8}")
    print("\n500-step references: row default 2.4209 · FA_TILED 2.4163")
    print("2000-step champion:  2.0158  (sota-ladder 1.9944 is NOT a 128M twin)")
print()
EOF
}

everything() {
  build
  fa
  profile
  speed
  cast_parity
  fa_quality
  summary
  echo "Remaining: champion rerun with the winning flag stack (see README / M15)."
}

case "${1:-all}" in
  build) build ;;
  profile) profile "${2:-shipped}" ;;
  speed) speed ;;
  cast-parity) cast_parity ;;
  quality) quality ;;
  fa) fa ;;
  fa-quality) fa_quality ;;
  summary) summary ;;
  fwd) fwd ;;
  glue) glue ;;
  seed-repeat) seed_repeat ;;
  everything) everything ;;
  all) build; profile; speed; cast_parity ;;
  *)
    echo "usage: $0 [build|profile|speed|cast-parity|quality|fa|fa-quality|fwd|glue|seed-repeat|summary|everything|all]"
    exit 1
    ;;
esac
