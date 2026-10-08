#!/usr/bin/env bash
# Audit 7/8 — full result set for write-up. Runs every remaining measurement
# and emits one summary table at the end.
#
# Stages (each can be run alone):
#   ./scripts/blog_results.sh env               capture chip/OS/toolchain versions
#   ./scripts/blog_results.sh gates     ~10 min  lib tests + all parity gates
#   ./scripts/blog_results.sh variance  ~5 min   run-to-run + thermal spread (quote the RANGE)
#   ./scripts/blog_results.sh power             GPU watts protocol (callback to post #1)
#   ./scripts/blog_results.sh cuda-ref          how to get a REAL 128M CUDA number
#   ./scripts/blog_results.sh glue      ~10 min  resid_glue atomics A/B (row_blocks sweep)
#   ./scripts/blog_results.sh profiles  ~5 min   bwd + fwd section tables, before/after
#   ./scripts/blog_results.sh baseline  ~5 min   pre-Audit-7 vs winning stack, same session
#   ./scripts/blog_results.sh champion  ~50 min  2000-step champion with every Audit 7+8 win
#   ./scripts/blog_results.sh champion-seed2 ~50 min  second seed (needed for a quality claim)
#   ./scripts/blog_results.sh mlx       ~5 min   MLX vs metal-native at MATCHED 128M shapes
#   ./scripts/blog_results.sh table              collate everything measured so far
#   ./scripts/blog_results.sh all       ~75 min  gates + glue + profiles + champion + mlx + table
#   ./scripts/blog_results.sh long      ~8 h     20k-step run (the quality proof; run overnight)
#
# Honest-reporting rules baked in (see docs/optimization_map.md Audit 7e):
#   - 500-step BPB deltas below ~0.03 are NOISE at this scale. Do not claim a
#     quality win from one seed. The champion gate is "no regression", not "better".
#   - Speed numbers come from steady-state logged steps, not warmup.
#   - The MLX comparison runs parameter-golf/train_gpt_mlx.py at MATCHED 128M
#     shapes (L24/C768/H24/KV12/MLP2304/head_dim 32/vocab 1024, 4096 tok/step).
#     Shapes and token budget are identical; the MLX script is still a plain
#     GQA+RoPE+SiLU transformer without arch_02's bigram / value-embedding /
#     value-residual modules. Report it as "same shapes, framework vs
#     hand-written kernels", not as a bit-for-bit port.

set -euo pipefail
cd "$(dirname "$0")/.."
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}

ROOT=${PARAMETER_GOLF_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
DATA=(--data-dir "$ROOT/parameter-golf/data/datasets/fineweb10B_sp1024"
      --token-bytes "$ROOT/Rust_MLKit/arch_02_value_resid/burn-port/token_bytes.json")
PRESET=(--preset arch02-128m --batch 16 --seq-len 256)
CHAMP=(--optimizer muon_polar_adamw --matrix-lr 0.05 --ema-decay 0.997)
BIN=target/release/train
OUT=out/blog
MLX=../mlx-baseline  # toy-shape port; the 128M comparison uses parameter-golf/train_gpt_mlx.py
mkdir -p "$OUT"

# Winning stack. CAST_ONCE / FA_* / FA_FWD_FAST are default-ON as of Audit 9A;
# listed explicitly so the write-up records exactly what produced each number.
# GLUE_ROWBLOCK deliberately omitted — Audit 8 A/B ≈ noise (REJECT as WIN).
WIN=(METAL_NATIVE_BWD_CAST_ONCE=1 METAL_NATIVE_FA_FAST=1 METAL_NATIVE_FA_BF16=1
     METAL_NATIVE_FA_FWD_FAST=1)
OLD=(METAL_NATIVE_BWD_CAST_ONCE=0 METAL_NATIVE_FA_FAST=0 METAL_NATIVE_FA_BF16=0
     METAL_NATIVE_FA_FWD_FAST=0 METAL_NATIVE_GLUE_ROWBLOCK=0)

ensure_build() {
  cargo build --release --bin train >/dev/null 2>&1 || {
    echo "build failed; verbose:" >&2; cargo build --release --bin train; exit 1; }
}

bench() { # name env...
  local name=$1; shift
  echo "-- bench: $name"
  env "$@" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --bench --bench-steps 10 --seed 1337 --log-every 1 --out "$OUT/bench_$name" 2>&1 | tail -3
}

gates() {
  ensure_build
  echo "=== correctness gates (every claim in the write-up depends on these) ==="
  cargo test --release --lib 2>&1 | tail -20
  echo
  echo "--- targeted kernel parity ---"
  for t in fa_bwd_row_d32_matches_generic fa_fwd_d32_matches_generic \
           glue_bwd_rowblock_matches_inline_atomics; do
    echo "== $t"
    cargo test --release --lib "$t" -- --nocapture 2>&1 | grep -E "max\||test result|Δ|drift" | head -6
  done
}

glue() {
  ensure_build
  echo "=== resid_glue: 3.1M inline atomics vs row-block reduction ==="
  bench glue_off "${WIN[@]}" METAL_NATIVE_GLUE_ROWBLOCK=0
  for rb in 8 16 32 64 128; do
    bench "glue_rb$rb" "${WIN[@]}" METAL_NATIVE_GLUE_ROWBLOCK=1 METAL_NATIVE_GLUE_ROWBLOCKS=$rb
  done
}

profiles() {
  ensure_build
  echo "=== section profiles: before (pre-Audit-7) vs after (winning stack) ==="
  for tag in old win; do
    if [ "$tag" = old ]; then envs=("${OLD[@]}"); else envs=("${WIN[@]}"); fi
    for which in BWD FWD; do
      echo "--- $which profile ($tag)"
      env "${envs[@]}" "METAL_NATIVE_${which}_PROFILE=1" "$BIN" "${PRESET[@]}" \
        "${DATA[@]}" "${CHAMP[@]}" --bench --bench-steps 4 --seed 1337 --log-every 1 \
        --out "$OUT/prof_${which}_${tag}" 2>&1 \
        | tee "$OUT/prof_${which}_${tag}.log" \
        | { grep -A 20 "$(echo "$which" | tr 'A-Z' 'a-z')_profile" || echo "(no profile output)"; } \
        | tail -22
    done
  done
  echo
  echo "Also worth capturing for the write-up: optimizer split"
  env "${WIN[@]}" METAL_NATIVE_OPTIM_PROFILE=1 "$BIN" "${PRESET[@]}" "${DATA[@]}" \
    "${CHAMP[@]}" --bench --bench-steps 4 --seed 1337 --log-every 1 \
    --out "$OUT/prof_OPTIM_win" 2>&1 | tee "$OUT/prof_OPTIM_win.log" \
    | { grep -A 6 "optim_profile" || echo "(no optim_profile output)"; } | tail -8
}

champion() {
  ensure_build
  echo "=== 2000-step champion with every Audit 7+8 win (~50 min) ==="
  echo "GATE: must land near 2.0158 (no regression). Do NOT claim a BPB win —"
  echo "      the 500-step seed spread is 0.032, ~180x this delta."
  env "${WIN[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --total-steps 2000 --seed 1337 --final-warmdown 350 \
    --eval-every 250 --checkpoint-every 250 --log-every 50 \
    --research-manifest "$ROOT/research/optimizer-study.json" \
    --out "$ROOT/out/champion_128m_seed1337_audit8" 2>&1 | tail -6
}

# One seed supports "no regression". Two supports a quality claim. Given the
# measured 500-step seed spread of 0.032, a single run cannot distinguish
# better from worse — so if the write-up says anything stronger than "held",
# it needs this.
champion_seed2() {
  ensure_build
  echo "=== champion, second seed (~50 min) — needed for any quality claim ==="
  env "${WIN[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --total-steps 2000 --seed 2026 --final-warmdown 350 \
    --eval-every 250 --checkpoint-every 250 --log-every 50 \
    --research-manifest "$ROOT/research/optimizer-study.json" \
    --out "$ROOT/out/champion_128m_seed2026_audit8" 2>&1 | tail -6
}

# The pre-Audit-7 baseline, re-run on TODAY's machine and thermal state. The
# 2895 ms figure was measured days ago; quoting it against a fresh number
# silently attributes any drift to the kernel work.
baseline_rerun() {
  ensure_build
  echo "=== pre-Audit-7 baseline, re-measured now (same session, same thermals) ==="
  env "${OLD[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --bench --bench-steps 20 --seed 1337 --log-every 1 \
    --out "$OUT/bench_baseline_today" 2>&1 | tail -3
  echo "--- winning stack, immediately after, same thermal state ---"
  env "${WIN[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --bench --bench-steps 20 --seed 1337 --log-every 1 \
    --out "$OUT/bench_win_today" 2>&1 | tail -3
  echo
  echo "Quote THIS pair as the headline speedup — same binary, same session."
}

mlx() {
  ensure_build
  if ! python3 -c "import mlx" 2>/dev/null; then
    echo "mlx not installed: pip install mlx"; return 0
  fi
  echo "=== MLX vs metal-native at MATCHED 128M shapes ==="
  echo "parameter-golf/train_gpt_mlx.py is env-configurable and lines up exactly:"
  echo "  L24 / C768 / H24 / KV12 / MLP_MULT=3 -> 2304 / head_dim 768/24 = 32"
  echo "  vocab 1024, tied embeddings, logit softcap 30, rope base 10000"
  echo "Same tokens/step as the native bench: B16 x T256 = 4096."
  echo
  echo "Remaining parity gaps (state them in any write-up): the MLX script is a"
  echo "GQA+RoPE+SiLU transformer without arch_02's bigram hash embedding, value"
  echo "embedding, or value residual, and it uses its own Muon/AdamW group split."
  echo "It is a same-shape framework reference, not a bit-for-bit port."
  echo
  echo "--- MLX (30 steps, wallclock cap off) ---"
  ( cd "$ROOT/parameter-golf" && \
    VOCAB_SIZE=1024 NUM_LAYERS=24 MODEL_DIM=768 NUM_HEADS=24 NUM_KV_HEADS=12 \
    MLP_MULT=3 TIE_EMBEDDINGS=1 LOGIT_SOFTCAP=30.0 ROPE_BASE=10000.0 \
    TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=4096 GRAD_ACCUM_STEPS=1 \
    MLX_MAX_MICROBATCH_TOKENS=4096 MLX_EAGER_EVAL=0 \
    ITERATIONS=30 WARMUP_STEPS=3 WARMDOWN_ITERS=0 MAX_WALLCLOCK_SECONDS=0 \
    VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=5 SEED=1337 \
    DATA_PATH="$ROOT/parameter-golf/data/datasets/fineweb10B_sp1024" \
    OUT_DIR="$ROOT/out/blog_mlx128m" \
    python3 train_gpt_mlx.py ) 2>&1 | tail -12
  echo
  echo "--- metal-native, same 128M shapes, same 4096 tok/step ---"
  env "${WIN[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --bench --bench-steps 30 --seed 1337 --log-every 5 \
    --out "$OUT/bench_128m_native" 2>&1 | tail -3
  echo
  echo "Compare the MLX 'tok/s' line against the native 'BENCH done' tok/s."
}

env_capture() {
  echo "=== environment (paste verbatim into the write-up) ==="
  {
    echo "date: $(date -u +%FT%TZ)"
    echo "chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    echo "cores: $(sysctl -n hw.ncpu) cpu / GPU cores via system_profiler below"
    echo "memory_GB: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
    echo "macos: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "xcode: $(xcodebuild -version 2>/dev/null | tr '\n' ' ')"
    echo "metal_toolchain: $(xcrun metal --version 2>&1 | head -1)"
    echo "rustc: $(rustc --version)"
    echo "python: $(python3 --version)"
    echo "mlx: $(python3 -c 'import mlx.core as m; print(m.__version__)' 2>/dev/null || echo absent)"
    system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Total Number of Cores|Metal" | head -4
    echo "data_shards: $(ls "$ROOT"/parameter-golf/data/datasets/fineweb10B_sp1024/*.bin 2>/dev/null | wc -l | tr -d ' ')"
  } | tee "$OUT/environment.txt"
}

# Laptops throttle. A 10-step bench on a cold machine is not the number a 20k
# run will see, and reporting the cold number is the single easiest way to get
# a benchmark torn apart. This runs the same config N times back-to-back and
# reports spread, so the write-up can quote a sustained range.
variance() {
  ensure_build
  local reps=${1:-5}
  echo "=== run-to-run + thermal variance: $reps repeats, back-to-back ==="
  echo "(no cooldown between reps — this is deliberately the throttled case)"
  for i in $(seq 1 "$reps"); do
    env "${WIN[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
      --bench --bench-steps 20 --seed 1337 --log-every 1 \
      --out "$OUT/var_rep$i" 2>&1 | grep "BENCH done"
  done
  python3 - "$OUT" "$reps" <<'EOF'
import json, pathlib, statistics, sys
out, reps = pathlib.Path(sys.argv[1]), int(sys.argv[2])
ms = []
for i in range(1, reps+1):
    f = out / f"var_rep{i}" / "metrics.jsonl"
    if not f.is_file(): continue
    s = [json.loads(l)["step_ms"] for l in f.open()
         if '"step_ms"' in l]
    if len(s) > 3: ms.append(statistics.mean(s[3:]))
if len(ms) > 1:
    print(f"\nsustained ms/step across {len(ms)} reps: "
          f"min {min(ms):.1f}  max {max(ms):.1f}  mean {statistics.mean(ms):.1f}  "
          f"stdev {statistics.stdev(ms):.1f}  spread {(max(ms)-min(ms))/min(ms)*100:.1f}%")
    print("Quote the RANGE in the write-up, not the best single run.")
EOF
}

# Callback to the first post: utilization lies, power draw doesn't.
power() {
  ensure_build
  echo "=== GPU power draw during a sustained run (needs sudo) ==="
  echo "First post's thesis was that nvidia-smi utilization lies and watts don't."
  echo "Same instrument, Apple side. Ctrl-C the powermetrics window when done."
  echo
  echo "Run in terminal A:"
  echo "  sudo powermetrics --samplers gpu_power -i 1000 -n 60 | grep -E 'GPU Power|GPU HW active'"
  echo "Run in terminal B:"
  echo "  cd $(pwd) && env ${WIN[*]} $BIN ${PRESET[*]} \\"
  echo "    --data-dir <data> --token-bytes <tb> ${CHAMP[*]} \\"
  echo "    --bench --bench-steps 200 --seed 1337 --out $OUT/power"
  echo
  echo "Report: mean GPU W during steady state, before vs after the kernel work."
  echo "A fixed-work speedup at ~equal watts = real efficiency gain, not just"
  echo "a higher power ceiling."
}

cuda_ref() {
  cat <<EOF
=== 128M CUDA reference — RUN THIS ON THE 3070 Ti, NOT THE MAC ===

GAP FOUND: the "~650-840 ms" and "BPB 1.9944" references throughout this repo
are for '--preset sota' = 4 layers x 128 dim x mlp 384. arch02-128m is
24 x 768 x 2304 — roughly 100x the parameters. Any 128M-vs-those-numbers
comparison is invalid. train.rs no longer prints them for non-sota shapes.

Same-shape harness (probe → bench → quality, seed 1337, L24/C768/H24/Hkv12/MLP×3/T256/4096 tok):

  # on the 3070 Ti box
  ./scripts/cuda_ref_128m.sh probe      # finds GRAD_ACCUM_STEPS that fits 8 GB
  ./scripts/cuda_ref_128m.sh bench      # ~200 steps throughput; DISCLOSE accum
  ./scripts/cuda_ref_128m.sh quality    # 2000-step val_bpb

  # on the Mac (when logs arrive)
  ./scripts/ingest_cuda128m.sh /path/to/parameter-golf/logs/cuda128m
  ./scripts/score_cuda128m.py logs/cuda128m

Canonical Mac ingest dir: logs/cuda128m (STATUS.md + summary.json).
Always disclose GRAD_ACCUM_STEPS next to every CUDA ms/step and BPB.
Do NOT claim Mac vs CUDA speed until same-shape numbers exist there.
Nanolab run128m_* is out of scope.

8 GB VRAM is the risk: the Mac run sits at ~13.5 GB unified. Probe tries
accum 1/2/4/8 with TRAIN_BATCH_TOKENS=4096. If nothing fits, that itself is
the reportable result — not a license to quote sota 1.9944 against 128M.
EOF
  # Local probe records the Mac blocker without requiring NVIDIA drivers.
  if [ -x ./scripts/cuda_ref_128m.sh ]; then
    echo
    echo "--- local CUDA availability probe (expected BLOCKER on Mac) ---"
    ./scripts/cuda_ref_128m.sh probe || true
  fi
}

long_run() {
  ensure_build
  echo "=== 20k-step run — the quality proof (~8 h). Run overnight. ==="
  echo "16M recipe reference: FA_TILED + --warmdown 3500 gave FINAL EMA 1.8969,"
  echo "beating the 3070 Ti sota-ladder 1.9944 (sota shape only). This is the"
  echo "128M analogue — do not treat 1.9944 as a 128M CUDA twin."
  env "${WIN[@]}" "$BIN" "${PRESET[@]}" "${DATA[@]}" "${CHAMP[@]}" \
    --total-steps 20000 --warmdown 3500 --seed 1337 \
    --eval-every 1000 --checkpoint-every 1000 --log-every 100 \
    --research-manifest "$ROOT/research/optimizer-study.json" \
    --out "$ROOT/out/long20k_128m_audit8" 2>&1 | tail -8
}

table() {
  python3 - "$OUT" "$ROOT/out" <<'EOF'
import json, pathlib, sys
def load(f):
    steps, final = [], None
    if not f.is_file(): return steps, final
    with f.open() as fh:
        for line in fh:
            try: r = json.loads(line)
            except Exception: continue
            if "final_ema_sliding_bpb" in r: final = r["final_ema_sliding_bpb"]
            elif "step_ms" in r: steps.append(r)
    return steps, final

rows = []
for root in (pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])):
    if not root.exists(): continue
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        steps, final = load(d / "metrics.jsonl")
        if not steps and final is None: continue
        tail = steps[-8:] if steps else []
        ms  = sum(s["step_ms"] for s in tail)/len(tail) if tail else 0
        tps = sum(s.get("tokens_per_s",0) for s in tail)/len(tail) if tail else 0
        ph  = [dict(s.get("phase_ms", [])) for s in tail]
        gp  = lambda k: (sum(p.get(k,0) for p in ph)/len(ph)) if ph else 0
        rows.append((d.name, ms, tps, gp("forward"), gp("backward"), gp("optim"),
                     tail[-1].get("dispatches",0) if tail else 0, final, len(steps)))

print("\n" + "="*118)
print("AUDIT 7/8 RESULTS")
print("="*118)
print(f"{'run':<34}{'ms/step':>9}{'tok/s':>8}{'fwd':>7}{'bwd':>8}{'optim':>7}{'disp':>7}{'BPB':>10}{'steps':>7}")
print("-"*118)
for n, ms, tps, f, b, o, dsp, bpb, ns in sorted(rows, key=lambda r: r[1] or 9e9):
    print(f"{n:<34}{ms:>9.1f}{tps:>8.0f}{f:>7.0f}{b:>8.0f}{o:>7.0f}{dsp:>7}"
          f"{(f'{bpb:.4f}' if bpb else '-'):>10}{ns:>7}")
print("-"*118)
print("""
Reference points for the write-up:
  pre-Audit-7 champion   2895 ms | 1431 tok/s | BPB 2.015756
  Audit 7 champion       2005 ms | 2071 tok/s | BPB 2.015576
  Audit 8 WIN champion   ~1683 ms | ~2400 tok/s | BPB 2.0107 (seed1337)
  Audit 8 seed2026       ~1592 ms |           | BPB 2.0404
  long20k WIN            ~1582 ms |           | BPB 1.8155
  burn-port (Rust/Burn)  ~2900 ms   (sota shape, not 128M)

Claim discipline:
  * BPB differences below ~0.03 at 500 steps are noise (measured seed spread 0.032).
  * Report speed as a range across logged steady-state steps, not a single best.
  * The MLX number is toy-shape only and the port omits modules - label it.
  * 3070 Ti 1.9944 / 650-840 ms are sota-ladder ONLY. 128M CUDA = null until
    logs/cuda128m has probe+bench(+quality) with disclosed GRAD_ACCUM_STEPS.
  * GLUE_ROWBLOCK / ACCUM_DX / muon binder-fuse are NOT speed KEEPs (Audit 9).
""")
EOF
}

case "${1:-all}" in
  env) env_capture ;;
  variance) variance "${2:-5}" ;;
  power) power ;;
  cuda-ref) cuda_ref ;;
  gates) gates ;;
  glue) glue ;;
  profiles) profiles ;;
  champion) champion ;;
  champion-seed2) champion_seed2 ;;
  baseline) baseline_rerun ;;
  mlx) mlx ;;
  long) long_run ;;
  table) table ;;
  all) env_capture; gates; baseline_rerun; glue; profiles; variance 5; champion; mlx; table ;;
  *)
    echo "usage: $0 [env|gates|baseline|glue|profiles|variance|power|champion|champion-seed2|mlx|cuda-ref|long|table|all]"
    exit 1
    ;;
esac
