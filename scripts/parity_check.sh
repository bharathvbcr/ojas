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
#
# Lock: on macOS the Metal suite runs under mac_heavy (OJAS_MAC_HEAVY, else
# ../Lappi-decision/tools/mac_heavy.sh, else
# ~/Code/research/Lappi-decision/tools/mac_heavy.sh), the same lookup ci_local.sh
# uses. It is resolved BEFORE any suite runs when --backend is metal or all, and the
# script fails closed when none exists: exit 1 (exit 64 when OJAS_MAC_HEAVY is not an
# absolute path to an executable file). There is no unlock flag. mac_heavy's own exit
# codes (75, 76, ...) pass through with no banner. When MAC_HEAVY_LABEL is already
# set the run is already under mac_heavy and goes in place, so the lock is never
# taken twice. Known trade-off: a stale MAC_HEAVY_LABEL inherited from an unrelated
# shell also skips the lock (ci_local.sh and tessl do the same).

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
# EXIT only cleans up. INT and TERM clean up and then END the script with the
# conventional 128+signal status: a handler that merely returns lets the script
# resume (with its log deleted) and run on to a banner.
trap cleanup_logs EXIT
trap 'cleanup_logs; exit 130' INT
trap 'cleanup_logs; exit 143' TERM

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
  -h, --help               Show this help message

The Metal suite on macOS runs under mac_heavy and fails closed (exit 1) without it.
There is no unlock flag.

Environment Variables:
  OJAS_MAC_HEAVY              Absolute path to mac_heavy.sh (else ../Lappi-decision/tools/,
                              else ~/Code/research/Lappi-decision/tools/); not
                              absolute or not an executable file: exit 64
  OJAS_SAM_FIXTURES           Path to visem_22_f0.safetensors
  OJAS_SAM_GROUPNORM_FIXTURES Path to pixel_decoder.safetensors
  OJAS_SAM_WEIGHTS            Path to sam3.1_multiplex.safetensors
  OJAS_ALLOW_NO_GPU           1 allows clean skip on hosts lacking Metal GPU
  MAC_HEAVY_LABEL             Set: already under mac_heavy.sh, run in place (no second lock)
EOF
}

BACKEND="cpu"
OP_FILTER=""
PROFILE="release"
CUSTOM_FIXTURES=""
CUSTOM_WEIGHTS=""
CUSTOM_GROUPNORM=""

# Which backend suites actually executed and which were skipped. The closing
# banner is derived from these, so a skipped suite can never print the line a
# passed one does. A suite that FAILS exits the script before either is touched.
RAN_BACKENDS=()
SKIPPED_BACKENDS=()

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

# Resolve the machine lock before any suite runs, so `--backend all` cannot run the
# CPU suite and only then refuse. macOS only: off Darwin the Metal suite is an error
# or a documented skip in run_metal_parity and no lock is looked up.
LOCK_WRAPPER=""
if [[ "$BACKEND" != "cpu" && "$(uname -s)" == "Darwin" ]]; then
  if [[ -n "${MAC_HEAVY_LABEL:-}" ]]; then
    :
  elif [[ -n "${OJAS_MAC_HEAVY:-}" ]]; then
    case "$OJAS_MAC_HEAVY" in
      /*) ;;
      *)
        echo "ERROR: OJAS_MAC_HEAVY=$OJAS_MAC_HEAVY must be an absolute path (a relative one would resolve against the repo root, not your cwd)" >&2
        exit 64
        ;;
    esac
    if [[ ! -f "$OJAS_MAC_HEAVY" || ! -x "$OJAS_MAC_HEAVY" ]]; then
      echo "ERROR: OJAS_MAC_HEAVY=$OJAS_MAC_HEAVY is not an executable file" >&2
      exit 64
    fi
    LOCK_WRAPPER="$OJAS_MAC_HEAVY"
  else
    for candidate in \
      "$ROOT/../Lappi-decision/tools/mac_heavy.sh" \
      "${HOME:-/nonexistent}/Code/research/Lappi-decision/tools/mac_heavy.sh"; do
      if [[ -f "$candidate" && -x "$candidate" ]]; then
        LOCK_WRAPPER="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$LOCK_WRAPPER" && -z "${MAC_HEAVY_LABEL:-}" ]]; then
    echo "ERROR: mac_heavy.sh not found; the Metal suite takes the machine lock on macOS and will not run without it. Set OJAS_MAC_HEAVY to its path." >&2
    exit 1
  fi
fi

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

  # Exit 0 is not evidence. A suite counts as passed only on what its own log
  # says. grep exits 2 on an error (e.g. an unreadable log); that is a failed
  # check, never "no match".
  log_matches() {
    local rc_grep=0
    grep -qE "$1" "$log_file" || rc_grep=$?
    if [[ $rc_grep -gt 1 ]]; then
      echo "ERROR: could not read the $suite_label suite log (grep exit $rc_grep); refusing to report it as passed." >&2
      exit 1
    fi
    return "$rc_grep"
  }

  if log_matches "test result: FAILED"; then
    rm -f "$log_file"
    echo "ERROR: $suite_label suite printed 'test result: FAILED' although cargo exited 0." >&2
    exit 1
  fi

  if log_matches "(running 0 tests|test result: .* 0 passed; 0 failed)"; then
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

  if ! log_matches "test result: ok\\. [1-9][0-9]* passed"; then
    rm -f "$log_file"
    echo "ERROR: $suite_label suite exited 0 but its log has no 'test result: ok. N passed' line with N > 0; it did not demonstrably run." >&2
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
  RAN_BACKENDS+=("cpu")
}

run_metal_parity() {
  echo ""
  echo "==> [Metal Parity] Running ojas-metal sam_fixtures_metal..."
  if [[ "$(uname -s)" != "Darwin" ]]; then
    if [[ "${OJAS_ALLOW_NO_GPU:-0}" == "1" ]]; then
      echo "    [SKIP] Not running on Darwin and OJAS_ALLOW_NO_GPU=1."
      SKIPPED_BACKENDS+=("metal")
      return 0
    else
      echo "    ERROR: Metal backend tests require macOS Darwin (Apple Silicon)." >&2
      exit 1
    fi
  fi

  # The lock was resolved (or refused) before any suite ran; see LOCK_WRAPPER.
  local lock_wrapper=()
  if [[ -n "$LOCK_WRAPPER" ]]; then
    lock_wrapper=(bash "$LOCK_WRAPPER" ojas-parity-check)
    echo "    GPU Lock: Wrapping with $LOCK_WRAPPER"
  else
    echo "    GPU Lock: Active under MAC_HEAVY_LABEL='${MAC_HEAVY_LABEL:-}'"
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
  RAN_BACKENDS+=("metal")
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

# Closing banner. Exit status is 0 in every case that reaches here (a skip under
# OJAS_ALLOW_NO_GPU=1 is a documented clean skip), but only a run in which every
# requested suite executed says SUCCESS. A caller grepping for SUCCESS therefore
# cannot mistake a skipped Metal suite, or a run where nothing executed, for a pass.
print_summary() {
  local elapsed="$1"
  local ran="none" skipped="none"
  [[ ${#RAN_BACKENDS[@]} -gt 0 ]] && ran="${RAN_BACKENDS[*]}"
  [[ ${#SKIPPED_BACKENDS[@]} -gt 0 ]] && skipped="${SKIPPED_BACKENDS[*]}"
  echo ""
  echo "======================================================================"
  if [[ ${#RAN_BACKENDS[@]} -eq 0 ]]; then
    echo "==> [Parity Check] NOTHING RAN in ${elapsed}s (skipped: $skipped)"
  elif [[ ${#SKIPPED_BACKENDS[@]} -gt 0 ]]; then
    echo "==> [Parity Check] INCOMPLETE, NOT A FULL PASS in ${elapsed}s (ran: $ran; skipped: $skipped)"
  else
    echo "==> [Parity Check] SUCCESS in ${elapsed}s (ran: $ran)"
  fi
  echo "======================================================================"
}

print_summary "$(($(date +%s) - START_TIME))"
