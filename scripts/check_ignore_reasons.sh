#!/usr/bin/env bash
# Every `#[ignore]` in tracked Rust source must carry a reason:
# `#[ignore = "why, and how to run it"]`. A bare `#[ignore]` leaves a reader of
# `cargo test` output with "ignored" and nothing else (no way to tell a slow
# sweep from a test that needs a device or a fixture). Doc and line comments
# that merely mention `#[ignore]` are not attributes and are not flagged. A
# conditional ignore needs one too: `#[cfg_attr(miri, ignore = "why")]`, not
# `#[cfg_attr(miri, ignore)]`. Spaces inside the brackets (`#[ ignore ]`)
# compile and are caught. Not caught: an attribute split across lines.
# Invoke: ./scripts/check_ignore_reasons.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$ROOT"

# No `//` before the attribute on its line, so a comment mentioning it is
# skipped. git grep reads tracked files only; the *.rs pathspec is a glob. Exit 1
# from git grep means "no match", which is success
# here; any other non-zero status is a real failure and must not read as clean.
status=0
hits=$(git grep -n -E '^([^/]|/[^/])*#\[[[:space:]]*(ignore[[:space:]]*\]|cfg_attr[[:space:]]*\(.*[[:space:],]ignore[[:space:]]*[,)])' -- '*.rs') || status=$?
if [ "$status" -gt 1 ]; then
  echo "check_ignore_reasons: git grep failed with status $status" >&2
  exit 1
fi

if [ -n "$hits" ]; then
  while IFS= read -r line; do
    echo "check_ignore_reasons: ${line%%:*}:$(echo "$line" | cut -d: -f2): bare #[ignore]; write #[ignore = \"<why, and how to run it>\"]" >&2
  done <<<"$hits"
  echo "check_ignore_reasons: $(printf '%s\n' "$hits" | wc -l | tr -d ' ') bare #[ignore] attribute(s)" >&2
  exit 1
fi
echo "check_ignore_reasons: no bare #[ignore] in tracked Rust source"
