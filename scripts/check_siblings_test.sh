#!/usr/bin/env bash
# Self-test for check_siblings.sh. Builds throwaway git repos and checks that
# the guard passes when the layout is sound and fails, with the right cause
# named, when it is not. Offline, no cargo — the script's own cargo step is
# reached only in the passing case, where it is expected to fail harmlessly on
# a fixture with no lockfile, so every case here asserts on the layout stages.
#
# Written because check_siblings.sh twice shipped a check that claimed more
# than it did: 6577c1a (a glob that read fewer manifests than it said) and
# be9e31a (a counter that made the zero-dependency guard unable to fire). Both
# were "exercised" by hand against the cases their author had thought of. This
# file is what makes that exercise repeatable, and it is wired into CI beside
# check_tracked_targets_test.sh.
#
# One repo path deliberately contains a space, as
# check_tracked_targets_test.sh does: it is the cheapest way to catch an
# unquoted expansion, and it caught one here.
#
# Invoke: ./scripts/check_siblings_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

pass=0
fail=0

# Lay out a fixture repository. $1 is its path; the caller writes manifests.
fixture() {
  local repo="$1"
  mkdir -p "$repo/scripts"
  cp "$HERE/check_siblings.sh" "$repo/scripts/"
  git -C "$repo" init -q
  git -C "$repo" config user.email t@t
  git -C "$repo" config user.name t
}

# Run the script in $1 and assert its layout verdict. $2 is "ok" when the
# layout stages must all pass, otherwise the text that must appear on stderr.
check() {
  local repo="$1" want="$2" name="$3" out rc
  git -C "$repo" add -A >/dev/null 2>&1 || true
  set +e
  out="$("$repo/scripts/check_siblings.sh" 2>&1)"
  rc=$?
  set -e
  if [ "$want" = "ok" ]; then
    # The layout stages must have passed. The cargo stage is allowed to fail:
    # these fixtures have no lockfile, and that is not what is under test.
    if printf '%s' "$out" | grep -q 'path dependencies resolve'; then
      echo "  PASS  $name"
      pass=$((pass + 1))
    else
      echo "  FAIL  $name — layout stages did not pass (rc=$rc)"
      printf '%s\n' "$out" | sed 's/^/        /'
      fail=$((fail + 1))
    fi
    return
  fi
  if [ "$rc" -eq 0 ]; then
    echo "  FAIL  $name — expected a non-zero exit, got 0"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1))
  elif printf '%s' "$out" | grep -q "$want"; then
    echo "  PASS  $name"
    pass=$((pass + 1))
  else
    echo "  FAIL  $name — exited $rc but did not say: $want"
    printf '%s\n' "$out" | sed 's/^/        /'
    fail=$((fail + 1))
  fi
}

echo "==> check_siblings.sh self-test"

# 1. A sound layout: one in-tree dependency and one that leaves the repository.
#    The repo path contains a space on purpose.
sound="$tmp/sound tree/repo"
mkdir -p "$tmp/sound tree/sibling"
fixture "$sound"
printf '%s\n' '[workspace]' 'members = ["a"]' > "$sound/Cargo.toml"
mkdir -p "$sound/a"
printf '%s\n' '[package]' 'name = "a"' '[dependencies]' \
  'ext = { path = "../../sibling" }' 'loc = { path = "../b" }' > "$sound/a/Cargo.toml"
mkdir -p "$sound/b"
printf '%s\n' '[package]' 'name = "b"' > "$sound/b/Cargo.toml"
printf '%s\n' '[package]' 'name = "sibling"' > "$tmp/sound tree/sibling/Cargo.toml"
check "$sound" ok "sound layout, and a path containing a space"

# 2. A dependency that leaves the repository and does not resolve. Must name
#    the manifest and the path, not merely fail.
gone="$tmp/gone"
fixture "$gone"
printf '%s\n' '[workspace]' 'members = ["a"]' > "$gone/Cargo.toml"
mkdir -p "$gone/a"
printf '%s\n' '[package]' 'name = "a"' '[dependencies]' \
  'ext = { path = "../../nowhere" }' > "$gone/a/Cargo.toml"
check "$gone" 'MISSING  a/Cargo.toml -> ../../nowhere' "an unresolvable sibling is named"

# 3. Single-quoted TOML must be parsed. `path = '...'` is as valid as
#    `path = "..."`, and the parser used to require double quotes — so one
#    dependency rewritten this way went unseen while the script still reported
#    success. It must now be seen, which here means named as unresolvable.
quoted="$tmp/quoted"
fixture "$quoted"
printf '%s\n' '[workspace]' 'members = ["a"]' > "$quoted/Cargo.toml"
mkdir -p "$quoted/a"
printf '%s\n' '[package]' 'name = "a"' '[dependencies]' \
  "ext = { path = '../../nowhere-quoted' }" > "$quoted/a/Cargo.toml"
check "$quoted" 'MISSING  a/Cargo.toml -> ../../nowhere-quoted' \
  "single-quoted TOML is parsed, not skipped"

# 4. The be9e31a guard, tested on its own terms rather than through a parser
#    weakness: a tree whose only `..` dependencies stay inside the repository.
#    `checked` is non-zero, so a guard on it could not fire; `external` is zero,
#    which here means this script is blind to the dependencies it exists for.
blind="$tmp/blind"
fixture "$blind"
printf '%s\n' '[workspace]' 'members = ["a","b"]' > "$blind/Cargo.toml"
mkdir -p "$blind/a" "$blind/b"
printf '%s\n' '[package]' 'name = "a"' '[dependencies]' \
  'loc = { path = "../b" }' > "$blind/a/Cargo.toml"
printf '%s\n' '[package]' 'name = "b"' > "$blind/b/Cargo.toml"
check "$blind" 'NONE of them leaves' "in-tree deps only cannot satisfy the guard"

# 4. The 6577c1a hole: a manifest below the first level must be read. The old
#    glob (Cargo.toml */Cargo.toml) could not see this one.
deep="$tmp/deep"
fixture "$deep"
printf '%s\n' '[workspace]' > "$deep/Cargo.toml"
mkdir -p "$deep/nested/crate"
printf '%s\n' '[package]' 'name = "deep"' '[dependencies]' \
  'ext = { path = "../../../nowhere-deep" }' > "$deep/nested/crate/Cargo.toml"
check "$deep" 'nested/crate/Cargo.toml' "a manifest two levels down is read"

# 5. No tracked manifest at all: the check cannot see what it is for.
empty="$tmp/empty"
fixture "$empty"
printf '%s\n' 'placeholder' > "$empty/README"
check "$empty" 'returned nothing' "no tracked manifest fails closed"

# 6. A linked worktree must refuse, because this workspace cannot build from
#    one at any path (ojas-metal -> tessl -> ojas-core pins it to one location).
wt_base="$tmp/wtbase"
fixture "$wt_base"
printf '%s\n' '[workspace]' > "$wt_base/Cargo.toml"
git -C "$wt_base" add -A >/dev/null 2>&1
git -C "$wt_base" commit -qm init
git -C "$wt_base" worktree add -q --detach "$tmp/wt" >/dev/null 2>&1
mkdir -p "$tmp/wt/scripts"
cp "$HERE/check_siblings.sh" "$tmp/wt/scripts/"
check "$tmp/wt" 'linked git worktree' "a linked worktree refuses"
git -C "$wt_base" worktree remove --force "$tmp/wt" >/dev/null 2>&1 || true

# 7. A manifest whose path INSIDE the repository contains a space. Case 1's
#    space is in the repo path, which `git ls-files` strips, so it never
#    reaches the loop that iterates the manifest list — this one does, and it
#    is where an unquoted expansion would split one path into two.
spacey="$tmp/spacey"
mkdir -p "$tmp/sibling-for-spacey"
fixture "$spacey"
printf '%s\n' '[workspace]' 'members = ["my crate"]' > "$spacey/Cargo.toml"
mkdir -p "$spacey/my crate"
printf '%s\n' '[package]' 'name = "mycrate"' '[dependencies]' \
  'ext = { path = "../../nowhere-spacey" }' > "$spacey/my crate/Cargo.toml"
check "$spacey" 'my crate/Cargo.toml' "a manifest path containing a space is read whole"

echo "==> $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
