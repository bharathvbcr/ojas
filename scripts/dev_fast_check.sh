#!/usr/bin/env bash
# Fast inner development loop for ojas crates (< 15s).
# Checks syntax (cargo check), runs surgical lints (cargo clippy -D warnings),
# and executes in-crate parallel unit tests (cargo test --lib).
#
# Usage:
#   ./scripts/dev_fast_check.sh [OPTIONS] [CRATE_NAME...]
#
# Examples:
#   ./scripts/dev_fast_check.sh                     # Defaults to ojas-core
#   ./scripts/dev_fast_check.sh ojas-cpu            # Checks ojas-cpu
#   ./scripts/dev_fast_check.sh ojas-core ojas-cpu  # Checks both
#   ./scripts/dev_fast_check.sh --all               # Fast check across workspace
#   ./scripts/dev_fast_check.sh --skip-clippy ojas-core

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

START_TIME=$(date +%s)

show_help() {
  cat <<'EOF'
dev_fast_check.sh: Fast inner dev loop syntax & unit check (< 15s)

Usage:
  ./scripts/dev_fast_check.sh [OPTIONS] [CRATE_NAME...]

Options:
  -h, --help       Show this help message
  --all            Check all workspace packages
  --check-only     Run only syntax/type checking (cargo check)
  --skip-clippy    Skip clippy linting stage
  --skip-tests     Skip in-crate unit tests stage

Default crate if none specified: ojas-core
EOF
}

CRATES=()
CHECK_ONLY=0
SKIP_CLIPPY=0
SKIP_TESTS=0
ALL_WORKSPACE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --all)
      ALL_WORKSPACE=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --skip-clippy)
      SKIP_CLIPPY=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 1
      ;;
    *)
      CRATES+=("$1")
      shift
      ;;
  esac
done

if [[ "$ALL_WORKSPACE" -eq 1 ]]; then
  PKG_ARGS=(--workspace)
  TARGET_DESC="workspace"
elif [[ "${#CRATES[@]}" -eq 0 ]]; then
  PKG_ARGS=(-p ojas-core)
  TARGET_DESC="ojas-core"
else
  PKG_ARGS=()
  for c in "${CRATES[@]}"; do
    PKG_ARGS+=(-p "$c")
  done
  TARGET_DESC="${CRATES[*]}"
fi

echo "======================================================================"
echo "==> [Fast Check] Target: $TARGET_DESC"
echo "======================================================================"

# Stage 1: Syntax & Typecheck
echo "==> [Stage 1/3] Syntax & Typecheck (cargo check)"
STAGE1_START=$(date +%s)
cargo check "${PKG_ARGS[@]}"
STAGE1_END=$(date +%s)
echo "    -> Completed in $((STAGE1_END - STAGE1_START))s"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  TOTAL_TIME=$(($(date +%s) - START_TIME))
  echo "==> Fast check completed in ${TOTAL_TIME}s (check-only mode)."
  exit 0
fi

# Stage 2: Surgical Clippy
if [[ "$SKIP_CLIPPY" -eq 0 ]]; then
  echo "==> [Stage 2/3] Surgical Clippy (cargo clippy -D warnings)"
  STAGE2_START=$(date +%s)
  cargo clippy "${PKG_ARGS[@]}" -- -D warnings
  STAGE2_END=$(date +%s)
  echo "    -> Completed in $((STAGE2_END - STAGE2_START))s"
else
  echo "==> [Stage 2/3] Clippy skipped (--skip-clippy)"
fi

# Stage 3: In-crate Parallel Unit Tests
if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "==> [Stage 3/3] In-crate Unit Tests (cargo test --lib)"
  STAGE3_START=$(date +%s)
  cargo test "${PKG_ARGS[@]}" --lib -- --nocapture
  STAGE3_END=$(date +%s)
  echo "    -> Completed in $((STAGE3_END - STAGE3_START))s"
else
  echo "==> [Stage 3/3] Unit tests skipped (--skip-tests)"
fi

TOTAL_TIME=$(($(date +%s) - START_TIME))
echo "======================================================================"
echo "==> [Fast Check] SUCCESS in ${TOTAL_TIME}s (target: < 15s)"
echo "======================================================================"
