#!/usr/bin/env bash
# Preflight for the sibling checkouts this workspace path-depends on.
#
# ojas does not stand alone. Members declare path dependencies that leave the
# repository: `ojas-metal`, `ojas-qwen35` and `gemma-metal` on `../../tessl`,
# and `ojas-gusset-engine` and `ojas-capi` on
# `../../../devtools/gusset/crates/gusset`. Cargo resolves every one of them
# before it builds, lints or tests anything, so a missing or misplaced sibling
# fails ahead of the first gate with an error about a manifest rather than
# about layout. This script says what is actually wrong, first, in one place.
#
# It exists because of three failures on 2026-11-01, each of which cost real
# time and none of which announced its own cause:
#
#   1. tessl's CI had seven `actions/checkout` steps and none checked out ojas,
#      while tessl's manifest has had mandatory path dependencies on
#      `../ojas/ojas-core` and `../ojas/ojas-io` for weeks. Every cargo job
#      failed before its own first step. A layout preflight in both repositories
#      would have named it the day the dependency landed.
#   2. An agent ran the MSRV probe from a git worktree, where `../../tessl`
#      resolves to nothing, and spent most of an hour reading a
#      missing-manifest error as an MSRV result. See the worktree check below:
#      that failure happens on every toolchain and says nothing about MSRV.
#   3. A claim about tessl's `rust-version` was read out of tessl's working
#      tree while it sat on another session's unpushed branch, and reported as
#      the state of the repository. On `origin/main` the field still said
#      something else. Hence the sibling state report: in this family of
#      checkouts, what is on disk beside you is frequently not what is
#      published.
#
# Invoke: ./scripts/check_siblings.sh
# Exits non-zero on the first hard failure. Nothing here is skippable: a check
# that cannot run must not report what a check that ran and passed reports.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# 1. Are we in a linked worktree? If so, stop: this workspace cannot build
#    from one, and the error cargo would give names the wrong cause.
#
# A linked worktree's `.git` is a regular file holding a gitdir pointer; a
# primary checkout's is a directory.
# ---------------------------------------------------------------------------
echo "==> checkout kind"
if [ -f .git ]; then
  echo "This is a linked git worktree: $ROOT" >&2
  echo >&2
  echo "ojas cannot be built from a worktree, at any path, and the reason is a" >&2
  echo "dependency cycle between fixed on-disk locations rather than anything" >&2
  echo "about this worktree:" >&2
  echo >&2
  echo "  ojas-metal/Cargo.toml  ->  tessl = { path = \"../../tessl\" }" >&2
  echo "  tessl/Cargo.toml       ->  ojas-core, ojas-io at \"../ojas/ojas-core\"" >&2
  echo >&2
  echo "Under .claude/worktrees/<lane>/ the first path resolves to nothing, and" >&2
  echo "cargo reports a missing tessl manifest on every toolchain — which says" >&2
  echo "nothing about MSRV, features or your changes. Moved to a sibling depth" >&2
  echo "where it does resolve, tessl then pulls the PRIMARY checkout's" >&2
  echo "ojas-core while this worktree's own ojas-core is a workspace member," >&2
  echo "and cargo refuses two different packages of one name:" >&2
  echo >&2
  echo "  error: package collision in the lockfile: packages ojas-core v0.1.0" >&2
  echo "  (<worktree>/ojas-core) and ojas-core v0.1.0 (<primary>/ojas-core)" >&2
  echo "  are different, but only one can be written to lockfile unambiguously" >&2
  echo >&2
  echo "Symlinking does not help; cargo canonicalises paths first." >&2
  echo >&2
  echo "Build and test from the primary checkout, one writer at a time. A" >&2
  echo "docs-only lane is fine in a worktree — it just must not run this gate," >&2
  echo "or cargo." >&2
  exit 1
fi
echo "primary checkout: $ROOT"

# ---------------------------------------------------------------------------
# 2. Every path dependency in every member manifest resolves to a real crate.
#
# Reported in full rather than stopping at the first, because a relocated
# sibling usually breaks several at once and the set is the diagnosis.
# ---------------------------------------------------------------------------
echo "==> path dependencies resolve"
missing=0
checked=0
external=0
siblings=""

# The manifest list comes from `git ls-files`, not a glob. ojas's manifests sit
# at more than one depth — arch02/burn-port and arch02/metal-native are two
# levels down — so `Cargo.toml */Cargo.toml` silently skipped them, and a check
# that looks at fewer files than it claims is this script's own failure mode,
# one level up from the one it exists to catch. Neither of those two carries an
# out-of-tree dependency today, so the answer was right by luck; the first one
# added would have been missed in silence. tessl's mirror of this script
# (tessl cd54859) reached the same conclusion independently.
#
# Tracked files only, which is what CI checks out. An untracked new manifest is
# not covered — the parser-sanity failure below is the backstop if the list ever
# comes back empty.
manifests="$(git ls-files '*Cargo.toml' 2>/dev/null || true)"
[ -n "$manifests" ] || fail "\`git ls-files '*Cargo.toml'\` returned nothing. Either this is not a git
       checkout or no manifest is tracked; both mean this check cannot see what
       it is meant to check. Fix that rather than ignoring it."

# Read on fd 3, one line per manifest, rather than `for manifest in $manifests`.
# Word splitting would break `my crate/Cargo.toml` into two non-existent paths,
# both of which the `-f` test below skips in silence — so a manifest whose path
# contains a space was simply not checked, and with it any out-of-tree
# dependency it declared. Caught by check_siblings_test.sh case 7; fd 3 keeps
# this loop's input clear of the inner heredoc below.
while IFS= read -r manifest <&3; do
  [ -n "$manifest" ] || continue
  [ -f "$manifest" ] || continue
  dir="$(dirname "$manifest")"
  # Only `path = "..."` entries that leave the member directory; a path with no
  # `..` is internal (a sub-crate or a test target) and cannot be a sibling.
  while IFS= read -r raw; do
    [ -n "$raw" ] || continue
    case "$raw" in *..*) ;; *) continue ;; esac
    target="$(cd "$dir" 2>/dev/null && cd "$raw" 2>/dev/null && pwd || true)"
    checked=$((checked + 1))
    if [ -z "$target" ] || [ ! -f "$target/Cargo.toml" ]; then
      echo "  MISSING  $manifest -> $raw" >&2
      missing=$((missing + 1))
      continue
    fi
    echo "  ok       $manifest -> $raw"
    # Record the owning repository root of anything outside this workspace, so
    # step 3 can report its state.
    case "$target" in
      "$ROOT"/*|"$ROOT") ;;
      *)
        # Leaves the repository, so it is one of the dependencies this script
        # exists for. Counted separately from `checked`: see the guard below.
        external=$((external + 1))
        repo="$(git -C "$target" rev-parse --show-toplevel 2>/dev/null || true)"
        [ -n "$repo" ] && siblings="$siblings$repo
"
        ;;
    esac
  done <<EOF_PATHS
$(grep -oE 'path[[:space:]]*=[[:space:]]*"[^"]+"' "$manifest" 2>/dev/null | sed -E 's/.*"([^"]+)".*/\1/' || true)
EOF_PATHS
done 3<<EOF_MANIFESTS
$manifests
EOF_MANIFESTS

# The guard is on `external`, not on `checked`. `checked` counts every `path`
# containing `..`, and most of those never leave the repository — `../ojas-core`
# and its siblings are in-tree workspace members. So `checked` is ~70 whatever
# happens, and a guard on it could never fire: a parser that stopped matching
# only the dependencies that LEAVE the tree (the whole point of this script)
# would still report dozens and pass.
#
# That is the same defect as the glob fixed in 6577c1a, one level subtler: not
# reading fewer files than claimed, but counting the wrong things and so
# claiming a coverage check that cannot fail. Found by the tessl mirror's
# verification, which demonstrated it there with a scratch repository.
#
# Concretely, the hole this closes: the grep at the top of the loop requires
# DOUBLE-quoted TOML (`path = "..."`). Rewrite one dependency with TOML's
# equally valid single quotes and this script stops seeing it — and before this
# change it would still have printed a reassuring two-digit total.
if [ "$external" -eq 0 ]; then
  fail "parsed $checked path dependencies but NONE of them leaves this
       repository. ojas path-depends on tessl (../../tessl) and gusset
       (../../../devtools/gusset/crates/gusset), so zero is impossible and the
       parser has stopped matching them — note it requires double-quoted TOML.
       This check is now blind to exactly what it exists to check. Fix the
       parser; do not ignore this."
fi
[ "$missing" -eq 0 ] || fail "$missing of $checked path dependencies do not resolve (see above)"
echo "$checked path dependencies resolve, $external of them leaving this repository"

# ---------------------------------------------------------------------------
# 3. What state are those siblings in?
#
# Not a hard failure: a dirty or branched sibling is normal while work is in
# flight. It is reported loudly because anything you read from such a checkout
# — a version, a line number, a field — is not what that repository publishes,
# which is exactly the mistake of 2026-11-01 (3) above.
# ---------------------------------------------------------------------------
echo "==> sibling checkout state"
printf '%s' "$siblings" | sort -u | while IFS= read -r repo; do
  [ -n "$repo" ] || continue
  branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  head="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo '?')"
  dirty="$(git -C "$repo" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  ahead="$(git -C "$repo" rev-list --count origin/main..HEAD 2>/dev/null || echo '?')"
  echo "  $repo"
  echo "    branch $branch at $head; $dirty uncommitted path(s); $ahead commit(s) not on origin/main"
  if [ "$branch" != "main" ] || [ "$dirty" != "0" ] || { [ "$ahead" != "0" ] && [ "$ahead" != "?" ]; }; then
    echo "    NOTE: this checkout does not match what it publishes. Cite origin/main," >&2
    echo "          or a commit, for any claim about this repository." >&2
  fi
done

# ---------------------------------------------------------------------------
# 4. The authoritative check: cargo itself resolves the graph.
#
# Steps 1-3 can all pass while resolution still fails — a version mismatch, a
# lockfile that disagrees, two packages of one name. `cargo metadata` answers
# that without compiling, and is not one of the subcommands the agent build-slot
# gate serialises, so it does not queue.
#
# --locked because the promise is about the committed lockfile, not about
# whatever a fresh resolve happens to pick today; --offline so a network
# failure cannot be read as a layout failure.
# ---------------------------------------------------------------------------
echo "==> cargo resolves the workspace (--locked --offline)"
if ! err="$(cargo metadata --locked --offline --format-version 1 2>&1 >/dev/null)"; then
  echo "$err" >&2
  fail "cargo cannot resolve this workspace. The layout checks above passed, so
       this is a manifest or lockfile problem, not a missing sibling."
fi
echo "resolved"

echo "==> siblings ok"
