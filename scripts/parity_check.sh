#!/usr/bin/env bash
# PyTorch Parity Verification Loop runner for ojas.
# Executes isolated op numerical parity checks against PyTorch reference fixtures
# (from SpikeTrack/fixtures/backbone/visem_22_f0.safetensors) on CPU and Metal backends.
#
# Usage:
#   ./scripts/parity_check.sh [OPTIONS]
#
# Examples:
#   ./scripts/parity_check.sh                     # Runs CPU parity checks (default)
#   ./scripts/parity_check.sh --backend metal     # Runs Metal GPU parity checks
#   ./scripts/parity_check.sh --backend all       # Runs CPU and Metal parity checks
#   ./scripts/parity_check.sh --op gelu_erf       # Checks single op
#   ./scripts/parity_check.sh --release           # Runs in release profile (fast)
#   ./scripts/parity_check.sh --op attention --backend metal

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_TIME=$(date +%s)

TMP_LOG_FILES=()
cleanup_logs() {
  if [[ ${#TMP_LOG_FILES[@]} -gt 0 ]]; then
    for f in "${TMP_LOG_FILES[@]}"; do
      [[ -f "$f" ]] && rm -f "$f"
    done
  fi
  return 0
}
trap cleanup_logs EXIT INT TERM

show_help() {
  cat <<'EOF'
parity_check.sh: PyTorch Oracle Fixture Parity Verification Loop

Usage:
  ./scripts/parity_check.sh [OPTIONS]

Options:
  -b, --backend BACKEND    Target backend: cpu, metal, or all (default: cpu)
  -o, --op OP_FILTER       Filter to specific op/test name (e.g. gelu_erf, attention, rope)
  -r, --release            Run with cargo --release (recommended for gigabyte fixtures)
  -d, --debug              Run with debug profile (unoptimized)
  --fixtures PATH          Explicit path to SAM 3.1 activation fixtures safetensors
  --weights PATH           Explicit path to SAM 3.1 weights safetensors
  --groupnorm PATH         Explicit path to GroupNorm pixel decoder safetensors
  --no-gpu-lock            Do not require or attempt mac_heavy GPU lock for Metal
  -h, --help               Show this help message

Environment Variables:
  OJAS_SAM_FIXTURES           Path to visem_22_f0.safetensors
  OJAS_SAM_GROUPNORM_FIXTURES Path to pixel_decoder.safetensors
  OJAS_SAM_WEIGHTS            Path to sam3.1_multiplex.safetensors
  OJAS_ALLOW_NO_GPU           1 allows clean skip on hosts lacking Metal GPU
  MAC_HEAVY_LABEL             Indicates active exclusive GPU lock under mac_heavy.sh
EOF
}

BACKEND="cpu"
OP_FILTER=""
PROFILE="release"
CUSTOM_FIXTURES=""
CUSTOM_WEIGHTS=""
CUSTOM_GROUPNORM=""
NO_GPU_LOCK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -b|--backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    -o|--op)
      OP_FILTER="${2:-}"
      shift 2
      ;;
    -r|--release)
      PROFILE="release"
      shift
      ;;
    -d|--debug)
      PROFILE="debug"
      shift
      ;;
    --fixtures)
      CUSTOM_FIXTURES="${2:-}"
      shift 2
      ;;
    --weights)
      CUSTOM_WEIGHTS="${2:-}"
      shift 2
      ;;
    --groupnorm)
      CUSTOM_GROUPNORM="${2:-}"
      shift 2
      ;;
    --no-gpu-lock)
      NO_GPU_LOCK=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

# Validate backend
case "$BACKEND" in
  cpu|metal|all) ;;
  *)
    echo "ERROR: Invalid backend '$BACKEND'. Must be 'cpu', 'metal', or 'all'." >&2
    exit 1
    ;;
esac

# Resolve Fixtures path
if [[ -n "$CUSTOM_FIXTURES" ]]; then
  export OJAS_SAM_FIXTURES="$CUSTOM_FIXTURES"
elif [[ -z "${OJAS_SAM_FIXTURES:-}" ]]; then
  CANDIDATES=(
    "$ROOT/../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors"
    "${HOME}/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors"
    "/Users/bharath/Code/research/SpikeTrack/fixtures/backbone/visem_22_f0.safetensors"
    "$ROOT/fixtures/backbone/visem_22_f0.safetensors"
  )
  for cand in "${CANDIDATES[@]}"; do
    if [[ -f "$cand" ]]; then
      export OJAS_SAM_FIXTURES="$cand"
      break
    fi
  done
fi

if [[ -z "${OJAS_SAM_FIXTURES:-}" || ! -f "$OJAS_SAM_FIXTURES" ]]; then
  echo "======================================================================" >&2
  echo "ERROR: SAM 3.1 fixture file not found!" >&2
  echo "Looked for: visem_22_f0.safetensors in SpikeTrack/fixtures/backbone/" >&2
  echo "Set OJAS_SAM_FIXTURES or pass --fixtures <path>." >&2
  echo "To generate fixtures, run: python3 SpikeTrack/tools/reference/dump_fixtures.py" >&2
  echo "======================================================================" >&2
  exit 1
fi

# Resolve GroupNorm fixtures (optional)
if [[ -n "$CUSTOM_GROUPNORM" ]]; then
  export OJAS_SAM_GROUPNORM_FIXTURES="$CUSTOM_GROUPNORM"
elif [[ -z "${OJAS_SAM_GROUPNORM_FIXTURES:-}" ]]; then
  GN_CANDIDATES=(
    "$ROOT/../SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors"
    "${HOME}/Code/research/SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors"
    "/Users/bharath/Code/research/SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors"
  )
  for cand in "${GN_CANDIDATES[@]}"; do
    if [[ -f "$cand" ]]; then
      export OJAS_SAM_GROUPNORM_FIXTURES="$cand"
      break
    fi
  done
fi

# Resolve Weights (optional)
if [[ -n "$CUSTOM_WEIGHTS" ]]; then
  export OJAS_SAM_WEIGHTS="$CUSTOM_WEIGHTS"
elif [[ -z "${OJAS_SAM_WEIGHTS:-}" ]]; then
  W_CANDIDATES=(
    "$ROOT/../SpikeTrack/models/sam3.1/sam3.1_multiplex.safetensors"
    "${HOME}/Code/research/SpikeTrack/models/sam3.1/sam3.1_multiplex.safetensors"
    "/Users/bharath/Code/research/SpikeTrack/models/sam3.1/sam3.1_multiplex.safetensors"
  )
  for cand in "${W_CANDIDATES[@]}"; do
    if [[ -f "$cand" ]]; then
      export OJAS_SAM_WEIGHTS="$cand"
      break
    fi
  done
fi

PROFILE_FLAG=""
if [[ "$PROFILE" == "release" ]]; then
  PROFILE_FLAG="--release"
fi

echo "======================================================================"
echo "==> [Parity Check] PyTorch Oracle Parity Runner"
echo "    Backend:           $BACKEND"
echo "    Op Filter:         ${OP_FILTER:-<all>}"
echo "    Profile:           $PROFILE"
echo "    Fixtures:          $OJAS_SAM_FIXTURES ($(du -h "$OJAS_SAM_FIXTURES" | cut -f1))"
if [[ -n "${OJAS_SAM_GROUPNORM_FIXTURES:-}" ]]; then
  echo "    GroupNorm:         $OJAS_SAM_GROUPNORM_FIXTURES"
fi
if [[ -n "${OJAS_SAM_WEIGHTS:-}" ]]; then
  echo "    Weights:           $OJAS_SAM_WEIGHTS"
fi
echo "======================================================================"

run_test_with_verification() {
  local suite_label="$1"
  local test_file="$2"
  shift 2
  local -a cmd=("$@")

  local log_file
  log_file=$(mktemp /tmp/ojas_parity_XXXXXX)
  TMP_LOG_FILES+=("$log_file")

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    rm -f "$log_file"
    exit "$rc"
  fi

  if grep -qE "(running 0 tests|test result: .* 0 passed; 0 failed)" "$log_file"; then
    rm -f "$log_file"
    echo "" >&2
    echo "======================================================================" >&2
    if [[ -n "$OP_FILTER" ]]; then
      echo "ERROR: No parity tests matched op filter '$OP_FILTER' in $suite_label suite!" >&2
    else
      echo "ERROR: 0 parity tests were executed in $suite_label suite!" >&2
    fi
    echo "Available parity tests in $test_file:" >&2
    echo "  conv2d, conv_transpose2d, gelu_erf, global_attention, group_norm," >&2
    echo "  layer_norm, linear_bias, rope_interleaved, rope_tables, windowed_attention" >&2
    echo "======================================================================" >&2
    exit 1
  fi

  rm -f "$log_file"
  return 0
}

run_cpu_parity() {
  echo ""
  echo "==> [CPU Parity] Running ojas-oracle sam_fixtures_cpu..."
  local cmd=(cargo test $PROFILE_FLAG -p ojas-oracle --test sam_fixtures_cpu)
  if [[ -n "$OP_FILTER" ]]; then
    cmd+=("$OP_FILTER")
  fi
  cmd+=(-- --ignored --nocapture)
  echo "    Command: ${cmd[*]}"
  run_test_with_verification "CPU" "ojas-oracle/tests/sam_fixtures_cpu.rs" "${cmd[@]}"
}

run_metal_parity() {
  echo ""
  echo "==> [Metal Parity] Running ojas-metal sam_fixtures_metal..."
  if [[ "$(uname -s)" != "Darwin" ]]; then
    if [[ "${OJAS_ALLOW_NO_GPU:-0}" == "1" ]]; then
      echo "    [SKIP] Not running on Darwin and OJAS_ALLOW_NO_GPU=1."
      return 0
    else
      echo "    ERROR: Metal backend tests require macOS Darwin (Apple Silicon)." >&2
      exit 1
    fi
  fi

  # Check GPU lock
  local lock_wrapper=()
  if [[ -n "${MAC_HEAVY_LABEL:-}" ]]; then
    echo "    GPU Lock: Active under MAC_HEAVY_LABEL='$MAC_HEAVY_LABEL'"
  elif [[ "$NO_GPU_LOCK" -eq 1 ]]; then
    echo "    GPU Lock: Bypassed (--no-gpu-lock)"
  else
    # Check for mac_heavy.sh in candidate paths
    local LOCK_SCRIPTS=(
      "$ROOT/../Lappi-decision/tools/mac_heavy.sh"
      "$ROOT/../tessl/tools/mac_heavy.sh"
      "${HOME}/Code/research/Lappi-decision/tools/mac_heavy.sh"
    )
    for s in "${LOCK_SCRIPTS[@]}"; do
      if [[ -x "$s" ]]; then
        lock_wrapper=("$s" "ojas-parity-check")
        echo "    GPU Lock: Wrapping with $s"
        break
      fi
    done
    if [[ "${#lock_wrapper[@]}" -eq 0 ]]; then
      echo "    Notice: mac_heavy.sh not found. Running with --test-threads=1 serialization."
    fi
  fi

  local base_cmd=(cargo test $PROFILE_FLAG -p ojas-metal --test sam_fixtures_metal)
  if [[ -n "$OP_FILTER" ]]; then
    base_cmd+=("$OP_FILTER")
  fi
  base_cmd+=(-- --ignored --nocapture --test-threads=1)

  local full_cmd=()
  if [[ "${#lock_wrapper[@]}" -gt 0 ]]; then
    full_cmd=("${lock_wrapper[@]}" "${base_cmd[@]}")
  else
    full_cmd=("${base_cmd[@]}")
  fi
  echo "    Command: ${full_cmd[*]}"
  run_test_with_verification "Metal" "ojas-metal/tests/sam_fixtures_metal.rs" "${full_cmd[@]}"
}

case "$BACKEND" in
  cpu)
    run_cpu_parity
    ;;
  metal)
    run_metal_parity
    ;;
  all)
    run_cpu_parity
    run_metal_parity
    ;;
esac

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo ""
echo "======================================================================"
echo "==> [Parity Check] SUCCESS in ${TOTAL_TIME}s"
echo "======================================================================"
