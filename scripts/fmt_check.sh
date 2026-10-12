#!/usr/bin/env bash
# rustfmt check over the ojas workspace members only. `cargo fmt --all` also
# formats local path dependencies (tessl, gusset), so a formatting change in a
# sibling repository would fail ojas CI. The package list comes from cargo
# metadata, so a new member is covered without editing this file.
# Invoke: ./scripts/fmt_check.sh [--fix]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pkgs=()
while IFS= read -r name; do
  pkgs+=(-p "$name")
done < <(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name')
if [ "${#pkgs[@]}" -eq 0 ]; then
  echo "fmt_check: cargo metadata listed no workspace packages" >&2
  exit 1
fi

if [ "${1:-}" = "--fix" ]; then
  cargo fmt "${pkgs[@]}"
else
  cargo fmt "${pkgs[@]}" --check
fi
