#!/usr/bin/env bash
# Adversarial cases for check_tracked_targets.sh, on throwaway repos. Each case
# names the input and the verdict the guard must give. Offline, no build.
# Invoke: ./scripts/check_tracked_targets_adversarial_test.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
bad=0
note() { echo "check_tracked_targets_adversarial: FAIL: $*" >&2; bad=$((bad + 1)); }

mkrepo() { # $1 = dir; one package, one example at the path given in $2
  local repo="$1" path="$2"
  mkdir -p "$repo/scripts" "$repo/pkg/src" "$repo/pkg/examples"
  cp "$HERE/check_tracked_targets.sh" "$repo/scripts/"
  printf '[workspace]\nmembers = ["pkg"]\nresolver = "2"\n' >"$repo/Cargo.toml"
  printf '[package]\nname = "pkg"\nversion = "0.0.0"\nedition = "2021"\n\n[[example]]\nname = "probe"\npath = "%s"\n' "$path" >"$repo/pkg/Cargo.toml"
  echo 'pub fn f() {}' >"$repo/pkg/src/lib.rs"
  (cd "$repo" && git init -q && git config user.email t@example.invalid && git config user.name t)
}

# 1. A glob-character path: an untracked file named like a pathspec that
#    matches a tracked sibling must not pass because of the sibling.
r="$tmp/glob"
mkrepo "$r" 'examples/p*.rs'
echo 'fn main() {}' >"$r/pkg/examples/probe.rs"
(cd "$r" && git add -A && git commit -qm base)
echo 'fn main() {}' >"$r/pkg/examples/p*.rs"   # untracked, literal name
if (cd "$r" && ./scripts/check_tracked_targets.sh >/dev/null 2>&1); then
  note "untracked target 'examples/p*.rs' passed because the glob matched tracked probe.rs"
fi

# 2. Bracket pathspec: untracked 'a[1].rs' beside tracked 'a1.rs'.
r="$tmp/bracket"
mkrepo "$r" 'examples/a[1].rs'
echo 'fn main() {}' >"$r/pkg/examples/a1.rs"
(cd "$r" && git add -A && git commit -qm base)
echo 'fn main() {}' >"$r/pkg/examples/a[1].rs"
if (cd "$r" && ./scripts/check_tracked_targets.sh >/dev/null 2>&1); then
  note "untracked target 'examples/a[1].rs' passed because the class matched tracked a1.rs"
fi

# 3. A newline in an untracked path must still fail.
r="$tmp/newline"
mkrepo "$r" $'examples/x\ny.rs'
echo 'fn main() {}' >"$r/pkg/examples/ok.rs"
(cd "$r" && git add -A && git commit -qm base)
printf 'fn main() {}\n' >"$r/pkg/examples/x"$'\n'"y.rs"
if (cd "$r" && ./scripts/check_tracked_targets.sh >/dev/null 2>&1); then
  note "untracked target with a newline in its path passed"
fi

# 4. No cargo on PATH: loud failure, never a pass.
r="$tmp/nocargo"
mkrepo "$r" 'examples/probe.rs'
echo 'fn main() {}' >"$r/pkg/examples/probe.rs"
(cd "$r" && git add -A && git commit -qm base)
bin="$tmp/bin"; mkdir -p "$bin"
for t in bash git jq dirname cd pwd mktemp rm cat; do p="$(command -v "$t" 2>/dev/null)"; [ -x "$p" ] && ln -sf "$p" "$bin/$t"; done
if (cd "$r" && PATH="$bin" ./scripts/check_tracked_targets.sh >/dev/null 2>&1); then
  note "guard passed with cargo missing"
fi

# 5. jq missing: loud failure.
bin2="$tmp/bin2"; mkdir -p "$bin2"
for t in bash git cargo rustc dirname rm cat; do p="$(command -v "$t" 2>/dev/null)"; [ -x "$p" ] && ln -sf "$p" "$bin2/$t"; done
if (cd "$r" && PATH="$bin2" ./scripts/check_tracked_targets.sh >/dev/null 2>&1); then
  note "guard passed with jq missing"
fi

# 6. Symlinked checkout root, run from a subdirectory: still passes on a
#    tracked tree and still fails on an untracked one.
r="$tmp/real"
mkrepo "$r" 'examples/probe.rs'
echo 'fn main() {}' >"$r/pkg/examples/probe.rs"
(cd "$r" && git add -A && git commit -qm base)
ln -s "$r" "$tmp/alias"
if ! (cd "$tmp/alias/pkg/src" && ../../scripts/check_tracked_targets.sh >/dev/null 2>&1); then
  note "tracked tree failed through a symlinked root from a subdirectory"
fi
(cd "$r" && git rm -q --cached pkg/examples/probe.rs)
if (cd "$tmp/alias/pkg/src" && ../../scripts/check_tracked_targets.sh >/dev/null 2>&1); then
  note "untracked target passed through a symlinked root"
fi

[ "$bad" -eq 0 ] && echo "check_tracked_targets_adversarial: ok"
exit "$bad"
