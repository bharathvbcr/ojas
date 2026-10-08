#!/usr/bin/env bash
# Layer-fusion v1 A/B gate — run on the M5 host (needs Metal).
#
#   bench/fusion_ab.sh [e4b|31b]        # default: e4b (Lane B dispatch)
#
# E4B: baseline / qkv / ple / rope / both
# 31B: baseline + qkv required. Optional smoke:
#   FUSION_AB_EXTRA_ARMS=1  → also run ple + rope + both
#
# Light / focused arms (comma-separated; overrides default arm list):
#   FUSION_AB_ARMS=baseline,gate_down   # D18 Hot FUSE_GATE_DOWN (expect Δ≈−42)
#   FUSION_AB_ARMS=baseline,qkv FUSION_AB_FUSE_KV=0  # QKV exactness vs solo K/V
#
# Exactness baseline pin (shipping decode still defaults FUSE_KV=1):
#   FUSION_AB_FUSE_KV=0|1   → pin GEMMA_METAL_FUSE_KV for A/B (default 1).
#                             Use 0 to gate fused QKV against solo K/V gemv.
#                             (Bank-split gemv_kv matches solo; FUSE_KV=1 is
#                             again a valid product-path exactness baseline.)
#
# TRACE / OOM mitigations (Hot+TRACE can jetsam a 64 GB M5; serialize always):
#   FUSION_AB_TRACE=1       → run TRACE dispatch scrape (default OFF for e4b
#                             and 31b — opt in when measuring dispatch drops)
#   FUSION_AB_ALLOW_BUSY=1  → skip GPU-busy preflight abort
#
# Gates, in order of authority:
#   1. EXACTNESS — fused token stream must equal unfused, token for token.
#      ALWAYS under METAL_RUNTIME_HAZARD_BARRIERS=0. Shipping hazard skip-auto
#      makes E4B streams non-deterministic even with fusion OFF, so exactness
#      under hazard=1 is not a valid fusion gate.
#   2. DISPATCHES — required arms must drop vs baseline, else not engaging.
#   3. TOK/S — shipping hazard path (GemmaGpu default skip-auto); informational.
#
# Compatible with macOS /bin/bash 3.2 (no associative arrays).
#
# Each arm runs in a fresh process with ambient FUSE_* cleared so overrides cannot
# poison the both/master flag path.
set -euo pipefail

MODEL="${1:-e4b}"
cd "$(dirname "$0")/.." || exit 1
OUT="bench/results/fusion_ab_${MODEL}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p bench/results

# Pin shipping decode path (names from src/kernels.rs).
# FUSION_AB_FUSE_KV overrides the KV fusion pin for exactness A/B only
# (product default remains FUSE_KV=1; do not default-on FUSE_QKV).
FUSE_KV_PIN="${FUSION_AB_FUSE_KV:-1}"
case "$FUSE_KV_PIN" in
  0|1) ;;
  *)
    echo "FATAL: FUSION_AB_FUSE_KV must be 0 or 1 (got: $FUSE_KV_PIN)"
    exit 2
    ;;
esac
PIN_ENV=(
  "GEMMA_METAL_FUSE_KV=${FUSE_KV_PIN}"
  GEMMA_METAL_GEMV_INTERLEAVE=0
  GEMMA_METAL_GEMV_SIMD=1
)
# Exactness must use always-on Device barriers (deterministic).
EXACT_ENV=(METAL_RUNTIME_HAZARD_BARRIERS=0)
QUIET_ENV=(CARGO_TARGET_DIR=target GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0)

case "$MODEL" in
  e4b)
    BENCH_ARGS=(--e4b)
    DIAG_ARGS=(e4b)
    ARM_NAMES=(baseline qkv ple rope both)
    REQUIRE_DISPATCH_DROP_ARMS=(qkv ple rope both)
    # TRACE off by default on E4B (opt in for dispatch-drop hard-fail).
    DO_TRACE="${FUSION_AB_TRACE:-0}"
    ;;
  31b)
    BENCH_ARGS=(--model "$(python3 - <<'PY'
import glob, os
c = sorted(glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/*')))
print(c[-1] if c else '')
PY
)")
    if [ -z "${BENCH_ARGS[1]:-}" ]; then
      echo "FATAL: no 31B HF snapshot under ~/.cache/huggingface/hub/...gemma-4-31b-it-4bit"
      exit 1
    fi
    DIAG_ARGS=(31b)
    ARM_NAMES=(baseline qkv)
    REQUIRE_DISPATCH_DROP_ARMS=(qkv)
    if [ "${FUSION_AB_EXTRA_ARMS:-0}" = "1" ]; then
      ARM_NAMES+=(ple rope both)
      # rope saves 1 store/producer; both includes qkv+rope (ple is N/A).
      REQUIRE_DISPATCH_DROP_ARMS+=(rope both)
    fi
    # Hot upload + TRACE peaked ~36–55 GiB — skip TRACE unless opted in.
    DO_TRACE="${FUSION_AB_TRACE:-0}"
    ;;
  *)
    echo "usage: $0 [e4b|31b]"
    exit 2
    ;;
esac

# Optional light arm list (e.g. D18 gate→down only). Always require baseline first.
if [ -n "${FUSION_AB_ARMS:-}" ]; then
  IFS=',' read -r -a ARM_NAMES <<< "${FUSION_AB_ARMS}"
  REQUIRE_DISPATCH_DROP_ARMS=()
  for arm in "${ARM_NAMES[@]}"; do
    case "$arm" in
      baseline) ;;
      qkv|ple|rope|both|gate_down) REQUIRE_DISPATCH_DROP_ARMS+=("$arm") ;;
      *)
        echo "FATAL: unknown FUSION_AB_ARMS entry: $arm"
        exit 2
        ;;
    esac
  done
  if [ "${ARM_NAMES[0]}" != "baseline" ]; then
    echo "FATAL: FUSION_AB_ARMS must start with baseline"
    exit 2
  fi
fi

# Abort if another heavy Metal job is already alive (jetsam / scheduler noise).
if [ "${FUSION_AB_ALLOW_BUSY:-0}" != "1" ]; then
  busy="$(pgrep -lf 'target/release/(bench|diag_tok)|fusion_ab\.sh' 2>/dev/null \
    | grep -v "$$" | grep -v "pgrep" || true)"
  if [ -n "$busy" ]; then
    echo "FATAL: another Metal job appears busy — wait or set FUSION_AB_ALLOW_BUSY=1"
    echo "$busy"
    exit 1
  fi
fi

echo "=== layer-fusion v1 A/B · model=$MODEL · arms=${ARM_NAMES[*]} · TRACE=${DO_TRACE} · $(date -u) ==="
echo "building (release)…"
if ! env "${QUIET_ENV[@]}" cargo build --release --bin bench --bin diag_tok 2>&1 | tail -5; then
  echo "BUILD FAILED — fix before A/B"
  exit 1
fi

# --- arm definitions: name -> flag assignments (per arm) -------------
arm_flags() {
  case "$1" in
    # Explicit FUSE_GATE_DOWN=0 on layer-fusion arms (D18 is separate / default OFF).
    baseline) echo "GEMMA_METAL_FUSE_QKV=0 GEMMA_METAL_FUSE_PLE=0 GEMMA_METAL_FUSE_ROPE_KV=0 GEMMA_METAL_FUSE_LAYER=0 GEMMA_METAL_FUSE_GATE_DOWN=0" ;;
    qkv)      echo "GEMMA_METAL_FUSE_QKV=1 GEMMA_METAL_FUSE_PLE=0 GEMMA_METAL_FUSE_ROPE_KV=0 GEMMA_METAL_FUSE_LAYER=0 GEMMA_METAL_FUSE_GATE_DOWN=0" ;;
    ple)      echo "GEMMA_METAL_FUSE_QKV=0 GEMMA_METAL_FUSE_PLE=1 GEMMA_METAL_FUSE_ROPE_KV=0 GEMMA_METAL_FUSE_LAYER=0 GEMMA_METAL_FUSE_GATE_DOWN=0" ;;
    rope)     echo "GEMMA_METAL_FUSE_QKV=0 GEMMA_METAL_FUSE_PLE=0 GEMMA_METAL_FUSE_ROPE_KV=1 GEMMA_METAL_FUSE_LAYER=0 GEMMA_METAL_FUSE_GATE_DOWN=0" ;;
    # Explicit all fusions — do not rely on FUSE_LAYER alone (ambient 0 would stick).
    both)     echo "GEMMA_METAL_FUSE_LAYER=1 GEMMA_METAL_FUSE_QKV=1 GEMMA_METAL_FUSE_PLE=1 GEMMA_METAL_FUSE_ROPE_KV=1 GEMMA_METAL_FUSE_GATE_DOWN=0" ;;
    # D18 Hot bounded-TG gate→down (not under FUSE_LAYER master). Expect Δ≈−42 on E4B.
    gate_down) echo "GEMMA_METAL_FUSE_QKV=0 GEMMA_METAL_FUSE_PLE=0 GEMMA_METAL_FUSE_ROPE_KV=0 GEMMA_METAL_FUSE_LAYER=0 GEMMA_METAL_FUSE_GATE_DOWN=1" ;;
    *)        echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

# bash 3.2-safe key/value store (no declare -A).
kv_set() { eval "$1__$2=\"\$3\""; }
kv_get() { eval "printf '%s' \"\${$1__$2-}\""; }

# Real-model decode tok/s: last "decode:" after === Real model banner (not mini).
scrape_real_decode_toks() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
idx = text.rfind("=== Real model")
if idx < 0:
    idx = text.rfind("=== Real E4B")
if idx < 0:
    sys.exit(0)
chunk = text[idx:]
vals = re.findall(r"^\s+decode:\s+([0-9]+(?:\.[0-9]+)?)", chunk, re.M)
if vals:
    print(vals[-1])
PY
}

# Mean TRACE disp= after real-model banner, decode tokens only (bytes≈ > 0 skips
# mini/warmup/prefill lines that report bytes≈0.00GiB).
scrape_real_decode_disp() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
idx = text.rfind("=== Real model")
if idx < 0:
    idx = text.rfind("=== Real E4B")
if idx < 0:
    sys.exit(0)
chunk = text[idx:]
pat = re.compile(
    r"\[trace\].*?\bdisp=(\d+)\b.*?bytes≈([0-9]+(?:\.[0-9]+)?)GiB"
)
disps = []
for m in pat.finditer(chunk):
    disp = int(m.group(1))
    gib = float(m.group(2))
    if gib > 0.01:
        disps.append(disp)
if not disps:
    all_d = [int(x) for x in re.findall(r"\[trace\].*?\bdisp=(\d+)\b", chunk)]
    disps = all_d[-16:] if len(all_d) >= 16 else all_d
if not disps:
    sys.exit(0)
print(f"{sum(disps) / len(disps):.1f}")
PY
}

require_nonempty() {
  local label="$1" val="$2" log="$3"
  if [ -z "$val" ]; then
    echo "FATAL: empty $label — see $log"
    exit 1
  fi
}

# Run a command with pinned env. Uses a subshell + unset (portable) instead of
# `env -u` — some macOS PATH setups still trip over leading `env -u`.
# Usage: run_pinned <unset_hazard:0|1> <log> <env assignments...> -- <cmd...>
run_pinned() {
  local unset_hazard="$1" log="$2"
  shift 2
  local assigns=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    assigns+=("$1")
    shift
  done
  [ "${1:-}" = "--" ] && shift
  (
    # Clear ambient fusion overrides so per-arm assigns win.
    unset GEMMA_METAL_FUSE_QKV GEMMA_METAL_FUSE_PLE GEMMA_METAL_FUSE_ROPE_KV \
      GEMMA_METAL_FUSE_LAYER GEMMA_METAL_FUSE_GATE_DOWN GEMMA_METAL_PERSISTENT_INTERP
    if [ "$unset_hazard" = 1 ]; then
      unset METAL_RUNTIME_HAZARD_BARRIERS
    fi
    local kv
    for kv in "${assigns[@]}"; do
      export "$kv"
    done
    exec "$@"
  ) >"$log" 2>&1
}

for arm in "${ARM_NAMES[@]}"; do
  # shellcheck disable=SC2206
  flags=( $(arm_flags "$arm") )
  echo
  echo "--- arm=$arm  [${flags[*]}  + pin FUSE_KV=${FUSE_KV_PIN} GEMV_INTERLEAVE=0 GEMV_SIMD=1]"

  # (1) exactness: fixed prompt, 16 greedy tokens — ALWAYS hazard=0
  d_log="${OUT}_${arm}_diag.txt"
  run_pinned 0 "$d_log" "${QUIET_ENV[@]}" "${PIN_ENV[@]}" "${EXACT_ENV[@]}" "${flags[@]}" \
    -- cargo run --release --bin diag_tok -- "${DIAG_ARGS[@]}"
  tok="$(grep -m1 '^new tokens:' "$d_log" | sed 's/^new tokens: //' || true)"
  require_nonempty "tokens ($arm)" "$tok" "$d_log"
  kv_set TOKENS "$arm" "$tok"
  echo "    tokens: $tok  (exactness @ HAZARD=0)"

  # (2) tok/s — shipping hazard (env unset → GemmaGpu skip-auto); TRACE off
  b_log="${OUT}_${arm}_bench.txt"
  run_pinned 1 "$b_log" "${QUIET_ENV[@]}" "${PIN_ENV[@]}" "${flags[@]}" \
    -- cargo run --release --bin bench -- "${BENCH_ARGS[@]}"
  ts="$(scrape_real_decode_toks "$b_log")"
  require_nonempty "decode tok/s ($arm)" "$ts" "$b_log"
  kv_set TOKS "$arm" "$ts"
  echo "    decode: $ts tok/s  (real model, shipping hazard)"

  # (3) dispatch count — TRACE optional (31b defaults off to avoid jetsam OOM)
  if [ "$DO_TRACE" = "1" ]; then
    t_log="${OUT}_${arm}_trace.txt"
    run_pinned 1 "$t_log" "${QUIET_ENV[@]}" "${PIN_ENV[@]}" "${flags[@]}" GEMMA_METAL_TRACE=1 \
      -- cargo run --release --bin bench -- "${BENCH_ARGS[@]}"
    disp="$(scrape_real_decode_disp "$t_log")"
    require_nonempty "dispatches ($arm)" "$disp" "$t_log"
    kv_set DISP "$arm" "$disp"
    echo "    dispatches/token (mean decode): $disp"
  else
    echo "    dispatches/token: SKIPPED (set FUSION_AB_TRACE=1 to enable; default off)"
    kv_set DISP "$arm" ""
  fi
done

echo
echo "=== VERDICT ==="
base_tok="$(kv_get TOKENS baseline)"
fail=0

if [ -z "$base_tok" ]; then
  echo "  EXACTNESS: INDETERMINATE — baseline produced no token line"
  fail=1
else
  for arm in "${ARM_NAMES[@]}"; do
    [ "$arm" = baseline ] && continue
    tok="$(kv_get TOKENS "$arm")"
    if [ "$tok" = "$base_tok" ]; then
      echo "  EXACTNESS $arm: PASS (stream identical to baseline)"
    else
      echo "  EXACTNESS $arm: *** FAIL *** — do not ship this arm"
      echo "      baseline: $base_tok"
      echo "      $arm:     $tok"
      fail=1
    fi
  done
fi

# Hard-fail if required fusion arms did not drop dispatches vs baseline.
base_disp="$(kv_get DISP baseline)"
if [ "$DO_TRACE" != "1" ]; then
  echo "  DISPATCHES: SKIPPED (FUSION_AB_TRACE off — exactness still gated)"
elif [ -z "$base_disp" ]; then
  echo "  DISPATCHES: INDETERMINATE — no baseline disp"
  fail=1
else
  for arm in "${REQUIRE_DISPATCH_DROP_ARMS[@]}"; do
    d="$(kv_get DISP "$arm")"
    if [ -z "$d" ]; then
      echo "  DISPATCHES $arm: *** FAIL *** — missing disp scrape"
      fail=1
      continue
    fi
    drop="$(python3 -c "print(float('${base_disp}') - float('${d}'))")"
    if python3 -c "import sys; sys.exit(0 if float('${d}') < float('${base_disp}') else 1)"; then
      echo "  DISPATCHES $arm: PASS (Δ=${drop} vs baseline ${base_disp})"
      # D18: Hot gate→down should save ~1 dispatch/layer → ≈42 on E4B.
      if [ "$arm" = gate_down ]; then
        if python3 -c "import sys; sys.exit(0 if abs(float('${drop}') - 42.0) <= 8.0 else 1)"; then
          echo "  DISPATCHES gate_down: engagement OK (Δ≈−42 expected; got Δ=${drop})"
        else
          echo "  DISPATCHES gate_down: *** FAIL *** — Δ=${drop} not ≈−42 (tolerance ±8; fusion under-engaged?)"
          fail=1
        fi
      fi
    else
      echo "  DISPATCHES $arm: *** FAIL *** — ${d} ≥ baseline ${base_disp} (fusion not engaging)"
      fail=1
    fi
  done
  for arm in "${ARM_NAMES[@]}"; do
    [ "$arm" = baseline ] && continue
    skip=
    for req in "${REQUIRE_DISPATCH_DROP_ARMS[@]}"; do
      [ "$arm" = "$req" ] && skip=1 && break
    done
    [ -n "$skip" ] && continue
    d="$(kv_get DISP "$arm")"
    if [ -n "$d" ]; then
      drop="$(python3 -c "print(float('${base_disp}') - float('${d}'))")"
      echo "  DISPATCHES $arm: smoke only (Δ=${drop}; no drop required)"
    fi
  done
fi

printf '\n  %-9s %10s %10s %12s\n' arm tok/s dispatch speedup
base_ts="$(kv_get TOKS baseline)"
for arm in "${ARM_NAMES[@]}"; do
  ts="$(kv_get TOKS "$arm")"
  d="$(kv_get DISP "$arm")"
  sp="—"
  if [ -n "$ts" ] && [ -n "$base_ts" ]; then
    sp="$(python3 -c "print(f'{float(\"${ts}\")/float(\"${base_ts}\"):.3f}x')" 2>/dev/null || echo '—')"
  fi
  printf '  %-9s %10s %10s %12s\n' "$arm" "${ts:-?}" "${d:-?}" "$sp"
done

# JSON artifact
{
  echo '{'
  echo "  \"lane\": \"layer-fusion-v1-ab\","
  echo "  \"model\": \"$MODEL\","
  echo "  \"date_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"arms_run\": [$(printf '"%s",' "${ARM_NAMES[@]}" | sed 's/,$//')],"
  echo "  \"exactness_hazard\": 0,"
  echo "  \"tok_s_hazard\": \"shipping_skip_auto\","
  echo "  \"exactness_all_pass\": $([ "$fail" -eq 0 ] && echo true || echo false),"
  echo "  \"pin\": {\"FUSE_KV\": ${FUSE_KV_PIN}, \"GEMV_INTERLEAVE\": 0, \"GEMV_SIMD\": 1},"
  echo '  "arms": {'
  n=${#ARM_NAMES[@]}
  i=0
  for arm in "${ARM_NAMES[@]}"; do
    i=$((i + 1))
    sep=','
    [ "$i" -eq "$n" ] && sep=''
    tok="$(kv_get TOKENS "$arm")"
    ts="$(kv_get TOKS "$arm")"
    d="$(kv_get DISP "$arm")"
    exact=false
    [ "$tok" = "$base_tok" ] && exact=true
    echo "    \"$arm\": {\"tok_s\": ${ts:-null}, \"dispatches\": ${d:-null}, \"tokens\": \"$tok\", \"exact\": $exact}$sep"
  done
  echo '  }'
  echo '}'
} > "${OUT}.json"
echo
echo "artifact: ${OUT}.json"
echo "logs:     ${OUT}_*.txt"
[ "$fail" -eq 0 ] || echo "NOTE: exactness and/or required dispatch-drop failed — fusion must not ship."
exit "$fail"
