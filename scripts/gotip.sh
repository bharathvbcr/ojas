#!/usr/bin/env bash
# Mirror of gusset .github/workflows/tip.yml for this tree: build the
# staticlib with stable Rust, then run the Go package on gotip.
#
# Exit codes (the contract ci_local.sh reads):
#   0   ran and passed
#   3   NOT RUN: gotip is missing, or installed but never downloaded
#   1   anything else that failed (an internal rc of 3 is remapped to 1, so
#       3 only ever means "not run")
#   >=128  a signal; passed through
# A run that did not happen never exits 0. The tip toolchain has not been
# executed on this host: the run path below is unverified.
set -euo pipefail

# not_run REASON: print the NOT RUN line, record REASON for ci_local.sh (which
# sets OJAS_NOT_RUN_REASON_FILE) and exit 3.
not_run() {
  echo "NOT RUN: $1" >&2
  if [ -n "${OJAS_NOT_RUN_REASON_FILE:-}" ]; then printf '%s\n' "$1" >"$OJAS_NOT_RUN_REASON_FILE"; fi
  exit 3
}

if ! command -v gotip >/dev/null 2>&1; then
  not_run "gotip is not on PATH (go install golang.org/dl/gotip@latest && gotip download); the tip gate did not execute"
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! version="$(gotip version 2>&1)"; then
  not_run "gotip is installed but 'gotip version' failed (run 'gotip download'): $version"
fi

# go/ holds the Darwin .pc; go/linux holds the Linux one (test.yml does the same).
case "$(uname -s)" in
  Darwin) pc_dir="$ROOT/go" ;;
  Linux) pc_dir="$ROOT/go/linux" ;;
  *)
    echo "ERROR: gotip.sh supports Darwin and Linux, not $(uname -s)" >&2
    exit 1
    ;;
esac

# Called under `||`, where `set -e` does not apply: every step is checked by hand.
run() {
  echo "==> gotip $version"
  cd "$ROOT" || return 1
  cargo build --offline -p ojas-gusset-engine || return $?
  cd "$ROOT/go" || return 1
  # -a: the go/README.md calls rebuilding every package strictly required.
  PKG_CONFIG_PATH="$pc_dir" gotip test -a -tags gusset_pkgconfig -count=1 -timeout 20m ./... || return $?
}

rc=0
run || rc=$?
if [ "$rc" -eq 3 ]; then
  echo "ERROR: a gotip step exited 3, which this script reserves for NOT RUN; reporting 1" >&2
  rc=1
fi
exit "$rc"
