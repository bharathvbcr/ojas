#!/usr/bin/env bash
# Self-test for check_ignore_reasons.sh. Builds a throwaway git repo and checks
# that the guard passes on reasoned ignores and on comments that mention
# `#[ignore]`, and fails, naming file:line, on a bare one (also one sharing a
# line with `#[test]`, or written `#[ignore ]`). Offline, no build.
# Invoke: ./scripts/check_ignore_reasons_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
repo="$tmp/work tree"
mkdir -p "$repo/scripts" "$repo/src"
cp "$HERE/check_ignore_reasons.sh" "$repo/scripts/"

cat >"$repo/src/lib.rs" <<'T'
//! The sweep is `#[ignore]`d; run it with `--ignored`.
/// A doc comment naming `#[ignore]` is not an attribute.
#[test]
#[ignore = "slow sweep; run with --ignored"]
fn reasoned() {}
#[test]
#[cfg_attr(miri, ignore = "too slow under the Miri interpreter")]
fn conditionally_reasoned() {}
#[cfg_attr(not(feature = "x"), ignore = "needs the x feature, run with --features x")]
fn conditionally_reasoned_negated() {}
// #[ignore]
// #[cfg_attr(miri, ignore)]
T

cd "$repo"
git init -q
git config user.email t@example.invalid
git config user.name t
git add -A
git commit -qm base

fail() { echo "check_ignore_reasons_test: FAIL: $1" >&2; exit 1; }

./scripts/check_ignore_reasons.sh >/dev/null 2>&1 || fail "clean tree was rejected"

expect_hit() { # label, source line, expected line number
  want=$(($(wc -l <src/lib.rs) + 1))
  printf '%s\n' "$2" >>src/lib.rs
  git add -A
  out=$(./scripts/check_ignore_reasons.sh 2>&1) && fail "$1: bare ignore was accepted"
  case "$out" in
    *"src/lib.rs:$want: bare #[ignore]"*) ;;
    *) fail "$1: output did not name src/lib.rs:$want: $out" ;;
  esac
  git checkout -q -- src/lib.rs 2>/dev/null || true
  git reset -q --hard
}

expect_hit "bracket spaces" "#[ ignore ]" 7
expect_hit "cfg_attr bare" "#[cfg_attr(miri, ignore)]" 7
expect_hit "cfg_attr bare, negated cfg" '#[cfg_attr(not(feature = "x"), ignore)]' 7
expect_hit "cfg_attr bare, bracket spaces" "#[ cfg_attr(miri, ignore) ]" 7
expect_hit "bare" "#[ignore]" 7
expect_hit "inline with test" "#[test] #[ignore]" 7
expect_hit "spaced" "#[ignore ]" 7
expect_hit "indented" "    #[ignore]" 7

echo "check_ignore_reasons_test: ok"
