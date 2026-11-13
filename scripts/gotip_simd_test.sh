#!/usr/bin/env bash
# Self-test for gotip.sh and simd_bench.sh run standalone: a gate that did not run
# must exit 3 (never 0), a failing probe must exit 1, and an internal rc of 3 must
# come out as 1. The real scripts run from a copied fixture tree (space and
# non-ASCII in its path) with `cargo`, `rustup`, `gotip` and `uname` stubbed on PATH.
# Invoke: ./scripts/gotip_simd_test.sh
# CI_LOCAL_UNDER_TEST=<dir> points it at another copy of the scripts (used to show
# the cases fail against the pre-change scripts); nothing else reads it.
# No -e: each check records its own status instead of aborting the run.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${CI_LOCAL_UNDER_TEST:-$HERE}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
ROOT="$tmp/r oot-ü"
STUBS="$tmp/stubs"
BIN="$tmp/bin"
LOG="$tmp/log"
mkdir -p "$ROOT/scripts" "$ROOT/go" "$STUBS" "$BIN"
for s in gotip.sh simd_bench.sh; do cp "$SRC/$s" "$ROOT/scripts/$s"; done

stub() { printf '#!/usr/bin/env bash\n%s\n' "$2" >"$STUBS/$1"; chmod +x "$STUBS/$1"; }
stub uname 'echo "${STUB_UNAME:-Darwin}"'
stub cargo 'echo "cargo $*" >>"$STUB_LOG"; exit "${STUB_CARGO_RC:-0}"'
stub rustup '[ "${STUB_RUSTUP_FAIL:-0}" = 1 ] && exit 1
if [ -n "${STUB_BIG_LIST:-}" ]; then
  echo "nightly-2026-10-05-aarch64-apple-darwin"; i=0; while [ $i -lt 30000 ]; do echo "filler-$i-aarch64-apple-darwin"; i=$((i + 1)); done; exit 0
fi
printf "%b" "${STUB_TOOLCHAINS-stable-aarch64-apple-darwin (default)\n}"'
stub gotip 'case "$1" in
  version) [ "${STUB_GOTIP_VERSION_FAIL:-0}" = 1 ] && { echo "gotip: not downloaded"; exit 2; }; echo "go version devel stub" ;;
  test) echo "gotip $* PKG_CONFIG_PATH=${PKG_CONFIG_PATH:-}" >>"$STUB_LOG"; exit "${STUB_GOTIP_TEST_RC:-0}" ;;
esac'

fails=0
out=""
rc=0
# run SCRIPT ENV... -- TOOLS...
run() {
  local script=$1 envs=() tools=() t
  shift
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  tools=("$@")
  rm -f "$BIN"/* "$tmp/reason"
  for t in ${tools[@]+"${tools[@]}"}; do ln -s "$STUBS/$t" "$BIN/$t"; done
  : >"$LOG"
  rc=0
  out="$(env -u OJAS_NOT_RUN_REASON_FILE PATH="$BIN:/usr/bin:/bin" STUB_LOG="$LOG" OJAS_NOT_RUN_REASON_FILE="$tmp/reason" \
    ${envs[@]+"${envs[@]}"} "$ROOT/scripts/$script" 2>&1)" || rc=$?
}
check() { # DESCRIPTION CONDITION-EXIT-CODE
  if [[ "$2" -ne 0 ]]; then
    echo "FAIL: $1"
    echo "$out" | tail -8 | sed 's/^/    | /'
    fails=$((fails + 1))
  else echo "ok: $1"; fi
}
has() { grep -qF -- "$1" <<<"$out"; }
log_has() { grep -qF -- "$1" "$LOG"; }
reason_has() { [[ -f "$tmp/reason" ]] && grep -qF -- "$1" "$tmp/reason"; }

# gotip.sh
run gotip.sh -- uname cargo
[[ "$rc" -eq 3 ]]; check "gotip.sh without gotip exits 3, never 0 (got $rc)" $?
has "NOT RUN: gotip is not on PATH" && reason_has "gotip is not on PATH"; check "gotip.sh without gotip: names it on stderr and in the reason file" $?

run gotip.sh -- uname cargo gotip
[[ "$rc" -eq 0 ]] && log_has "gotip test -a -tags gusset_pkgconfig -count=1 -timeout 20m ./..."; check "gotip.sh present: runs gotip test with -a (got $rc)" $?
log_has "cargo build --offline -p ojas-gusset-engine"; check "gotip.sh present: builds the staticlib first" $?
grep -q 'PKG_CONFIG_PATH=.*/go$' "$LOG"; check "gotip.sh on Darwin: PKG_CONFIG_PATH is go/" $?
run gotip.sh STUB_UNAME=Linux -- uname cargo gotip
grep -q 'PKG_CONFIG_PATH=.*/go/linux$' "$LOG"; check "gotip.sh on Linux: PKG_CONFIG_PATH is go/linux" $?
run gotip.sh STUB_UNAME=Plan9 -- uname cargo gotip
[[ "$rc" -eq 1 ]] && has "Plan9"; check "gotip.sh on an unsupported OS: exits 1 naming it (got $rc)" $?
run gotip.sh STUB_GOTIP_VERSION_FAIL=1 -- uname cargo gotip
[[ "$rc" -eq 3 ]] && has "not downloaded"; check "gotip.sh installed but not downloaded: exits 3 with its message (got $rc)" $?
run gotip.sh STUB_GOTIP_TEST_RC=3 -- uname cargo gotip
[[ "$rc" -eq 1 ]]; check "gotip.sh: a gotip test rc of 3 is remapped to 1 (got $rc)" $?
run gotip.sh STUB_GOTIP_TEST_RC=2 -- uname cargo gotip
[[ "$rc" -eq 2 ]]; check "gotip.sh: another failing rc passes through (got $rc)" $?
run gotip.sh STUB_CARGO_RC=3 -- uname cargo gotip
[[ "$rc" -eq 1 ]]; check "gotip.sh: a cargo build rc of 3 is remapped to 1 (got $rc)" $?

# simd_bench.sh
DATED="stable-aarch64-apple-darwin (default)\nnightly-2026-10-05-aarch64-apple-darwin\n"
run simd_bench.sh "STUB_TOOLCHAINS=stable-aarch64-apple-darwin (default)\nnightly-aarch64-apple-darwin\n" -- uname cargo rustup
[[ "$rc" -eq 3 ]] && has "nightly-2026-10-05 is not installed"; check "simd_bench.sh with only an undated nightly exits 3 (got $rc)" $?
reason_has "nightly-2026-10-05 is not installed"; check "simd_bench.sh: the reason file names the dated nightly" $?
run simd_bench.sh "STUB_TOOLCHAINS=nightly-2026-10-06-aarch64-apple-darwin\n" -- uname cargo rustup
[[ "$rc" -eq 3 ]]; check "simd_bench.sh with a different dated nightly exits 3 (got $rc)" $?
run simd_bench.sh "STUB_TOOLCHAINS=nightly-2026-10-050-aarch64-apple-darwin\n" -- uname cargo rustup
[[ "$rc" -eq 3 ]]; check "simd_bench.sh: a longer date is not a match (got $rc)" $?
run simd_bench.sh -- uname cargo
[[ "$rc" -eq 3 ]] && has "rustup not found"; check "simd_bench.sh without rustup exits 3 (got $rc)" $?
run simd_bench.sh STUB_RUSTUP_FAIL=1 -- uname cargo rustup
[[ "$rc" -eq 1 ]] && has "could not list toolchains"; check "simd_bench.sh with a failing 'rustup toolchain list' exits 1 (got $rc)" $?
run simd_bench.sh "STUB_TOOLCHAINS=$DATED" -- uname cargo rustup
[[ "$rc" -eq 0 ]] && log_has "cargo +nightly-2026-10-05 test -p ojas-simd --release --no-fail-fast"; check "simd_bench.sh with the dated nightly runs exactly the ojas-simd test (got $rc)" $?
run simd_bench.sh STUB_BIG_LIST=1 -- uname cargo rustup
[[ "$rc" -eq 0 ]] && log_has "cargo +nightly-2026-10-05 test"; check "simd_bench.sh: a long toolchain list is read whole (no SIGPIPE false negative) (got $rc)" $?
run simd_bench.sh "STUB_TOOLCHAINS=$DATED" STUB_CARGO_RC=3 -- uname cargo rustup
[[ "$rc" -eq 1 ]]; check "simd_bench.sh: a cargo rc of 3 is remapped to 1 (got $rc)" $?
run simd_bench.sh "STUB_TOOLCHAINS=$DATED" STUB_CARGO_RC=101 -- uname cargo rustup
[[ "$rc" -eq 101 ]]; check "simd_bench.sh: a test failure (101) passes through (got $rc)" $?
[[ ! -e "$ROOT/target" ]]; check "simd_bench.sh writes nothing into <root>/target" $?
[[ ! -e "$HERE/simd_bench.rs" ]]; check "scripts/simd_bench.rs is gone" $?

if [[ "$fails" -ne 0 ]]; then echo "gotip_simd_test: $fails check(s) failed" >&2; exit 1; fi
echo "gotip_simd_test OK"
