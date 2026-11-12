#!/usr/bin/env bash
# Dated-nightly gate for ojas-simd, the crate that holds the workspace's arch
# intrinsics and inline asm. (It replaces scripts/simd_bench.rs, which only
# exercised `std::simd` and no ojas code.)
#
# Exit codes (the contract ci_local.sh reads):
#   0   ran and passed
#   3   NOT RUN: rustup or the dated nightly is absent
#   1   anything else that failed (an internal rc of 3 is remapped to 1)
#   >=128  a signal; passed through
# The run path needs the dated nightly, which this host does not have: it is
# unverified here and only a stub covers it.
set -euo pipefail

# not_run REASON: print the NOT RUN line, record REASON for ci_local.sh (which
# sets OJAS_NOT_RUN_REASON_FILE) and exit 3.
not_run() {
  echo "NOT RUN: $1" >&2
  if [ -n "${OJAS_NOT_RUN_REASON_FILE:-}" ]; then printf '%s\n' "$1" >"$OJAS_NOT_RUN_REASON_FILE"; fi
  exit 3
}

# Dated, bumped together with gusset's RUST_NIGHTLY
# (gusset/.github/workflows/matrix.yml and nightly.yml).
NIGHTLY=nightly-2026-10-05

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v rustup >/dev/null 2>&1; then
  not_run "rustup not found; the dated nightly gate needs it"
fi

# Capture first, never pipe into `grep -q`: under pipefail the early exit sends
# SIGPIPE to rustup, which would read as "not installed".
if ! list="$(rustup toolchain list)"; then
  echo "ERROR: could not list toolchains ('rustup toolchain list' failed)" >&2
  exit 1
fi
case $'\n'"$list" in
  *$'\n'"$NIGHTLY"-*) ;;
  *)
    not_run "$NIGHTLY is not installed (rustup toolchain install $NIGHTLY --profile minimal)"
    ;;
esac

cd "$ROOT"
rc=0
cargo "+$NIGHTLY" test -p ojas-simd --release --no-fail-fast || rc=$?
if [ "$rc" -eq 3 ]; then
  echo "ERROR: cargo exited 3, which this script reserves for NOT RUN; reporting 1" >&2
  rc=1
fi
exit "$rc"
