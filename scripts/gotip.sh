#!/usr/bin/env bash
# Mirror of gusset .github/workflows/tip.yml for this tree: build the
# staticlib with stable Rust, then run the Go package on gotip.
#
# If gotip is not on PATH this script skips. That skip is unverified:
# this host has not executed the tip toolchain.
set -euo pipefail

if ! command -v gotip >/dev/null 2>&1; then
  echo "gotip is not installed; tip gate skipped (unverified)" >&2
  exit 0
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "==> gotip $(gotip version)"
cargo build --offline -p ojas-gusset-engine
cd "$ROOT/go"
PKG_CONFIG_PATH="$PWD" gotip test -tags gusset_pkgconfig -count=1 -timeout 20m ./...
