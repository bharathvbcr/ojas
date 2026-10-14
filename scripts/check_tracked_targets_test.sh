#!/usr/bin/env bash
# Self-test for check_tracked_targets.sh. Builds a throwaway git repo (in a
# directory whose name has a space) holding one package, and checks that the
# guard passes when every target source is tracked and fails when a [[example]]
# path or an autodiscovered src/bin file is untracked. Offline, no build.
# Invoke: ./scripts/check_tracked_targets_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
repo="$tmp/work tree"
mkdir -p "$repo/scripts" "$repo/pkg/src/bin" "$repo/pkg/examples"
cp "$HERE/check_tracked_targets.sh" "$repo/scripts/"

cat >"$repo/Cargo.toml" <<'T'
[workspace]
members = ["pkg"]
resolver = "2"
T
cat >"$repo/pkg/Cargo.toml" <<'T'
[package]
name = "pkg"
version = "0.0.0"
edition = "2021"

[[example]]
name = "probe"
path = "examples/probe.rs"
T
echo 'pub fn f() {}' >"$repo/pkg/src/lib.rs"
echo 'fn main() {}' >"$repo/pkg/examples/probe.rs"

cd "$repo"
git init -q
git config user.email t@example.invalid
git config user.name t
git add -A
git commit -qm base

fail() { echo "check_tracked_targets_test: $*" >&2; exit 1; }
run() { ./scripts/check_tracked_targets.sh >"$tmp/out" 2>&1; }

run || { cat "$tmp/out" >&2; fail "guard rejected a fully tracked tree"; }

git rm -q --cached pkg/examples/probe.rs
if run; then fail "guard accepted an untracked [[example]] path"; fi
grep -q 'pkg/examples/probe.rs' "$tmp/out" || fail "failure does not name the path"
git add pkg/examples/probe.rs

echo 'fn main() {}' >pkg/src/bin/auto.rs
if run; then fail "guard accepted an untracked autodiscovered bin"; fi
grep -q 'pkg/src/bin/auto.rs' "$tmp/out" || fail "failure does not name the autobin"
git add pkg/src/bin/auto.rs

run || { cat "$tmp/out" >&2; fail "guard rejected the tree after tracking"; }
echo "check_tracked_targets_test: ok"
