#!/usr/bin/env bash
# Every Cargo target source (lib, bin, test, example, bench) of every workspace
# package must be tracked by git. A manifest naming an untracked file builds
# locally and fails on a fresh checkout; this check fails first, in seconds.
# Invoke: ./scripts/check_tracked_targets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

srcs=$(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].targets[].src_path')
if [ -z "$srcs" ]; then
  echo "check_tracked_targets: cargo metadata listed no targets" >&2
  exit 1
fi

missing=0
count=0
while IFS= read -r src; do
  count=$((count + 1))
  rel=${src#"$ROOT"/}
  if [ "$rel" = "$src" ]; then
    echo "check_tracked_targets: $src is outside $ROOT" >&2
    missing=$((missing + 1))
  elif ! git ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
    echo "check_tracked_targets: $rel is a Cargo target source but is not tracked" >&2
    missing=$((missing + 1))
  fi
done <<<"$srcs"

if [ "$missing" -ne 0 ]; then
  echo "check_tracked_targets: $missing of $count target sources are not tracked" >&2
  exit 1
fi
echo "check_tracked_targets: all $count target sources are tracked"
