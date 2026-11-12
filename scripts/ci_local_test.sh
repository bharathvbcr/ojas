#!/usr/bin/env bash
# Self-test for ci_local.sh's lock, not-run verdict, waivers and signals. It runs
# the REAL ci_local.sh, gotip.sh and simd_bench.sh inside a copied fixture tree
# (`r oot-ü/`, so a space and a non-ASCII byte are exercised and the run can never
# recurse into the real repo) with every other script and every tool stubbed on
# PATH. It needs no build, no GPU and no lock.
# Invoke: ./scripts/ci_local_test.sh
# CI_LOCAL_UNDER_TEST=<dir> points it at another copy of the three scripts (used
# to show the cases fail against the pre-change scripts); nothing else reads it.
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
mkdir -p "$ROOT/scripts" "$ROOT/bench" "$ROOT/go" "$STUBS" "$BIN" "$tmp/home" "$tmp/tmpdir"

printf '[workspace.package]\nrust-version = "1.97"\n' >"$ROOT/Cargo.toml"
for s in ci_local.sh gotip.sh simd_bench.sh; do cp "$SRC/$s" "$ROOT/scripts/$s"; done
# Every other script ci_local calls: log its name and succeed.
for s in check_siblings.sh check_tracked_targets_test.sh check_tracked_targets_adversarial_test.sh \
  check_ignore_reasons_test.sh check_siblings_test.sh parity_check_test.sh gotip_simd_test.sh \
  ci_local_test.sh fmt_check.sh check_tracked_targets.sh check_ignore_reasons.sh; do
  printf '#!/usr/bin/env bash\necho "script %s" >>"$STUB_LOG"\n[ "${STUB_RC3_SCRIPT:-}" = "%s" ] && exit 3\nexit 0\n' "$s" "$s" >"$ROOT/scripts/$s"
  chmod +x "$ROOT/scripts/$s"
done
printf '#!/usr/bin/env bash\necho "script run_paired_test.sh" >>"$STUB_LOG"\n' >"$ROOT/bench/run_paired_test.sh"
chmod +x "$ROOT/bench/run_paired_test.sh"

stub() { printf '#!/usr/bin/env bash\n%s\n' "$2" >"$STUBS/$1"; chmod +x "$STUBS/$1"; }
stub uname 'echo "${STUB_UNAME:-Darwin}"'
stub cargo 'echo "cargo $*" >>"$STUB_LOG"
if [ -n "${STUB_CARGO_RC3:-}" ]; then case "$*" in *"$STUB_CARGO_RC3"*) exit 3 ;; esac; fi
case "$*" in *"${STUB_CARGO_SLEEP:-@@none@@}"*) sleep 2 ;; esac
case "$*" in *"${STUB_CARGO_HANG:-@@none@@}"*) sleep 600 ;; esac
if [ -n "${STUB_CARGO_FAIL:-}" ]; then case "$*" in *"$STUB_CARGO_FAIL"*) exit 101 ;; esac; fi'
stub cargo-deny 'exit 0'
stub go 'echo "go $* PKG_CONFIG_PATH=${PKG_CONFIG_PATH:-}" >>"$STUB_LOG"; echo "go version stub"'
stub rustup '[ "${STUB_RUSTUP_FAIL:-0}" = 1 ] && exit 1
printf "%b" "${STUB_TOOLCHAINS-stable-aarch64-apple-darwin (default)\nnightly-2026-10-05-aarch64-apple-darwin\n1.97-aarch64-apple-darwin\n}"'
stub gotip 'case "$1" in
  version) [ "${STUB_GOTIP_VERSION_FAIL:-0}" = 1 ] && { printf "gotip: not downloaded\nnightly: spoof\nthird line\n"; exit 2; }; echo "go version devel stub" ;;
  test) echo "gotip $*" >>"$STUB_LOG"; exit "${STUB_GOTIP_TEST_RC:-0}" ;;
esac'
stub mac_heavy.sh 'echo "mac_heavy $*" >>"$STUB_LOG"
d=${STUB_HEAVY_DEPTH:-0}
if [ "$d" -ge 3 ]; then echo "mac_heavy recursion" >>"$STUB_LOG"; exit 99; fi
export STUB_HEAVY_DEPTH=$((d + 1))
[ -n "${STUB_HEAVY_RC:-}" ] && exit "$STUB_HEAVY_RC"
export MAC_HEAVY_LABEL="$1"; shift; exec "$@"'

fails=0
out=""
rc=0
# A regression must fail a case, never hang the suite: every case runs under a
# watchdog (macOS has no `timeout`) that kills the whole process tree and makes
# the case's rc 124.
CASE_TIMEOUT="${CASE_TIMEOUT:-60}"
kill_tree() { # PID: stop it first (so it cannot spawn more), then its children, then it
  local c
  kill -STOP "$1" 2>/dev/null
  for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
  kill -KILL "$1" 2>/dev/null
  return 0
}
bounded_wait() { # PID: waits for it, sets rc (124 if the watchdog fired)
  local pid=$1 wd flag="$tmp/timedout.$1"
  (
    sleep "$CASE_TIMEOUT"
    : >"$flag"
    kill_tree "$pid"
  ) &
  wd=$!
  rc=0
  wait "$pid" 2>/dev/null || rc=$?
  kill_tree "$wd"
  wait "$wd" 2>/dev/null
  if [[ -e "$flag" ]]; then rc=124; fi
  return 0
}
# run ENV... -- TOOLS...: fresh PATH dir holding the named stubs, then the real ci_local.sh.
run() {
  local envs=() tools=() t
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  tools=("$@")
  rm -f "$BIN"/*
  for t in "${tools[@]}"; do ln -s "$STUBS/$t" "$BIN/$t"; done
  : >"$LOG"
  env -u MAC_HEAVY_LABEL -u OJAS_CI_LOCAL_ALLOW_NOT_RUN -u OJAS_MAC_HEAVY -u OJAS_CI_LOCAL_SENTINEL \
    PATH="$BIN:/usr/bin:/bin" HOME="$tmp/home" TMPDIR="$tmp/tmpdir" STUB_LOG="$LOG" ${envs[@]+"${envs[@]}"} \
    "$ROOT/scripts/ci_local.sh" >"$tmp/run.out" 2>&1 &
  bounded_wait $!
  out="$(cat "$tmp/run.out")"
  if [[ "$rc" -eq 124 ]]; then out="$out"$'\n'"TIMED OUT after ${CASE_TIMEOUT}s"; fi
}
BASE=(uname cargo cargo-deny go rustup)
HEAVY="OJAS_MAC_HEAVY=$STUBS/mac_heavy.sh"
check() { # DESCRIPTION CONDITION-EXIT-CODE
  if [[ "$2" -ne 0 ]]; then
    echo "FAIL: $1"
    echo "$out" | tail -8 | sed 's/^/    | /'
    fails=$((fails + 1))
  else echo "ok: $1"; fi
}
has() { grep -qF -- "$1" <<<"$out"; }
lacks() { ! has "$1"; }
log_has() { grep -qF -- "$1" "$LOG"; }
log_lacks() { ! log_has "$1"; }
last() { tail -n 1 <<<"$out"; }
bare_ok() { grep -qx 'ci:local OK' <<<"$out"; }
cnt() { grep -cF -- "$1" "$LOG"; }

# 1. Everything present on Darwin.
run "$HEAVY" -- "${BASE[@]}" gotip
[[ "$rc" -eq 0 ]]; check "1 all present: rc 0 (got $rc)" $?
[[ "$(last)" == "ci:local OK" ]]; check "1 all present: last line is exactly ci:local OK" $?
[[ "$(cnt 'mac_heavy ojas-ci-local')" -eq 1 ]]; check "1 all present: mac_heavy wrapper invoked once" $?
log_has "cargo test -p ojas-simd --features accelerate" && log_has "cargo clippy -p ojas-simd --features accelerate --all-targets -- -D warnings"
check "1 all present: accelerate test and clippy ran" $?
log_has "cargo test -p ojas-simd" && log_has "cargo +1.97 check --workspace --all-targets" && log_has "cargo +1.97 check -p ojas-cuda --features cuda --all-targets"
check "1 all present: plain ojas-simd test and +1.97 checks ran" $?
log_has "cargo test -p ojas-cuda --features cuda --no-run"; check "1 all present: cuda --no-run ran" $?
log_has "cargo +nightly-2026-10-05 test -p ojas-simd --release --no-fail-fast"; check "1 all present: dated nightly ojas-simd ran" $?
log_has "script run_paired_test.sh" && log_has "script ci_local_test.sh" && log_has "script gotip_simd_test.sh"; check "1 all present: self-tests ran (Darwin includes run_paired_test)" $?
has "ojas-hip check: not applicable on macOS" && log_lacks "ojas-hip"; check "1 all present: hip is printed not applicable on Darwin" $?

# 2. Already under mac_heavy: the wrapper is not taken again.
run "$HEAVY" MAC_HEAVY_LABEL=x -- "${BASE[@]}" gotip
[[ "$rc" -eq 0 ]] && [[ "$(cnt mac_heavy)" -eq 0 ]]; check "2 MAC_HEAVY_LABEL set: wrapper not invoked, rc 0" $?
has "already under mac_heavy (x)"; check "2 MAC_HEAVY_LABEL set: says so" $?

# 3. No wrapper anywhere: fail closed before any cargo.
run -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && has "OJAS_MAC_HEAVY" && log_lacks "cargo"; check "3 no wrapper: rc 1, names OJAS_MAC_HEAVY, no cargo (got $rc)" $?
run OJAS_MAC_HEAVY="$tmp/nonexistent" -- "${BASE[@]}" gotip
[[ "$rc" -eq 64 ]] && log_lacks "cargo"; check "3b OJAS_MAC_HEAVY not a file: rc 64 (got $rc)" $?

# 4. mac_heavy's own refusal passes through untouched.
run "$HEAVY" STUB_HEAVY_RC=76 -- "${BASE[@]}" gotip
[[ "$rc" -eq 76 ]] && lacks "ci:local"; check "4 wrapper rc 76: rc 76, no ci:local line (got $rc)" $?

# 5. gotip absent: INCOMPLETE, rc 3, later gates still ran.
run "$HEAVY" -- "${BASE[@]}"
[[ "$rc" -eq 3 ]]; check "5 gotip absent: rc 3 (got $rc)" $?
[[ "$(last)" == "ci:local INCOMPLETE"* ]] && [[ "$(last)" == *gotip* ]]; check "5 gotip absent: last line is INCOMPLETE and names gotip" $?
! bare_ok; check "5 gotip absent: no bare ci:local OK line" $?
log_has "cargo test --workspace --release --no-fail-fast"; check "5 gotip absent: cargo test --workspace still ran" $?

# 6. The same, waived by exact name.
run "$HEAVY" OJAS_CI_LOCAL_ALLOW_NOT_RUN=gotip -- "${BASE[@]}"
[[ "$rc" -eq 0 ]] && [[ "$(last)" == "ci:local OK WITH WAIVED SKIPS"* ]] && [[ "$(last)" == *gotip* ]]; check "6 gotip waived: rc 0, OK WITH WAIVED SKIPS naming gotip (got $rc)" $?
! bare_ok; check "6 gotip waived: no bare ci:local OK line" $?
run "$HEAVY" "OJAS_CI_LOCAL_ALLOW_NOT_RUN= gotip,,msrv " -- "${BASE[@]}"
[[ "$rc" -eq 0 ]]; check "6b waiver ' gotip,,msrv ': tolerated (got $rc)" $?
run "$HEAVY" OJAS_CI_LOCAL_ALLOW_NOT_RUN=GOTIP -- "${BASE[@]}"
[[ "$rc" -eq 64 ]]; check "6c waiver 'GOTIP': exact match only, rc 64 (got $rc)" $?
run "$HEAVY" "OJAS_CI_LOCAL_ALLOW_NOT_RUN=*" -- "${BASE[@]}"
[[ "$rc" -eq 64 ]] && log_lacks "cargo"; check "6d waiver '*': never a wildcard, rc 64 (got $rc)" $?
run "$HEAVY" OJAS_CI_LOCAL_ALLOW_NOT_RUN=msrv -- "${BASE[@]}"
[[ "$rc" -eq 3 ]] && [[ "$(last)" == *"INCOMPLETE"* ]] && [[ "$(last)" == *gotip* ]]; check "6e waiving a different step does not waive gotip (got $rc)" $?

# 7. A mistyped waiver name is a usage error before anything runs.
run "$HEAVY" OJAS_CI_LOCAL_ALLOW_NOT_RUN=gotpi -- "${BASE[@]}" gotip
[[ "$rc" -eq 64 ]] && log_lacks "cargo" && [[ "$(cnt mac_heavy)" -eq 0 ]]; check "7 waiver 'gotpi': rc 64, no cargo, no lock (got $rc)" $?

# 8. Dated nightly absent: gates after it still run, rc 3, names nightly.
run "$HEAVY" "STUB_TOOLCHAINS=stable-aarch64-apple-darwin (default)\nnightly-aarch64-apple-darwin\n1.97-aarch64-apple-darwin\n" -- "${BASE[@]}" gotip
[[ "$rc" -eq 3 ]] && [[ "$(last)" == "ci:local INCOMPLETE"* ]] && [[ "$(last)" == *nightly* ]] && [[ "$(last)" != *msrv* ]]; check "8 dated nightly absent: rc 3, INCOMPLETE names nightly only (got $rc)" $?
log_has "cargo test --workspace --release" && log_lacks "+nightly"; check "8 dated nightly absent: later gates ran, no nightly cargo" $?

# 9. rustup missing: both rustup gates are not run, with the reason.
run "$HEAVY" -- uname cargo cargo-deny go gotip
[[ "$rc" -eq 3 ]] && has "rustup not found" && [[ "$(last)" == *nightly* ]] && [[ "$(last)" == *msrv* ]]; check "9 rustup missing: rc 3, nightly and msrv not run, reason 'rustup not found' (got $rc)" $?

# 10. A failing probe is a failure, not a skip.
run "$HEAVY" STUB_RUSTUP_FAIL=1 -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && lacks "ci:local"; check "10 rustup toolchain list fails: rc 1, no verdict (got $rc)" $?

# 11. A failing gate ends the run with its own rc.
run "$HEAVY" STUB_CARGO_FAIL=clippy -- "${BASE[@]}" gotip
[[ "$rc" -eq 101 ]] && lacks "ci:local" && log_lacks "cargo deny" && log_lacks "cargo test"; check "11 clippy fails: rc 101, no verdict, no later gate (got $rc)" $?

# 12. gotip present: -a on its test.
run "$HEAVY" -- "${BASE[@]}" gotip
log_has "gotip test -a "; check "12 gotip test runs with -a" $?

# 13. gotip failing is a failure; its rc 3 must not read as not-run.
run "$HEAVY" STUB_GOTIP_TEST_RC=2 -- "${BASE[@]}" gotip
[[ "$rc" -eq 2 ]] && lacks "ci:local"; check "13 gotip test fails: rc passed through, no verdict (got $rc)" $?
run "$HEAVY" STUB_GOTIP_TEST_RC=3 -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && lacks "ci:local"; check "13b gotip test rc 3 is remapped to 1 (got $rc)" $?
run "$HEAVY" STUB_GOTIP_VERSION_FAIL=1 -- "${BASE[@]}" gotip
[[ "$rc" -eq 3 ]] && has "not downloaded"; check "13c gotip installed but not downloaded: not run, with its message (got $rc)" $?

# 14. Linux: no wrapper, linux .pc, hip checked, no Accelerate.
run STUB_UNAME=Linux -- "${BASE[@]}" gotip
[[ "$rc" -eq 0 ]] && [[ "$(last)" == "ci:local OK" ]]; check "14 Linux: rc 0, ci:local OK (got $rc)" $?
has "mac_heavy not used" && [[ "$(cnt mac_heavy)" -eq 0 ]]; check "14 Linux: wrapper not used" $?
grep -q 'PKG_CONFIG_PATH=.*/go/linux$' "$LOG"; check "14 Linux: PKG_CONFIG_PATH ends /go/linux" $?
log_has "cargo check -p ojas-hip --all-targets"; check "14 Linux: ojas-hip check ran" $?
log_lacks "accelerate" && has "accelerate: not applicable" && ! has "INCOMPLETE"; check "14 Linux: accelerate not run, printed not applicable, not a skip" $?
log_lacks "script run_paired_test.sh"; check "14 Linux: run_paired_test (Darwin-only) not run" $?
run "$HEAVY" -- "${BASE[@]}" gotip
grep -q 'PKG_CONFIG_PATH=.*/go$' "$LOG" && ! grep -q 'PKG_CONFIG_PATH=.*/go/linux' "$LOG"; check "14b Darwin: PKG_CONFIG_PATH is go/, not go/linux" $?
run STUB_UNAME=FreeBSD -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && has "FreeBSD"; check "14c unsupported OS: rc 1 naming it (got $rc)" $?

# 16. A hard gate that exits 3 is a FAILED gate, never INCOMPLETE (one place: the exit trap).
run "$HEAVY" STUB_RC3_SCRIPT=fmt_check.sh -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && [[ "$(last)" == "ci:local FAILED"* ]] && lacks "INCOMPLETE" && ! bare_ok && log_lacks "cargo clippy"; check "16 fmt_check.sh exits 3: rc 1, ci:local FAILED, no later gate (got $rc)" $?
run "$HEAVY" STUB_CARGO_RC3="clippy --workspace" -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && [[ "$(last)" == "ci:local FAILED"* ]] && lacks "INCOMPLETE" && log_lacks "cargo deny"; check "16 cargo clippy exits 3: rc 1, ci:local FAILED (got $rc)" $?
run "$HEAVY" STUB_CARGO_RC3="test --workspace" -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && [[ "$(last)" == "ci:local FAILED"* ]] && lacks "INCOMPLETE" && ! bare_ok; check "16 cargo test --workspace exits 3: rc 1, ci:local FAILED (got $rc)" $?
run STUB_UNAME=Linux STUB_CARGO_RC3="test --workspace" -- "${BASE[@]}" gotip
[[ "$rc" -eq 1 ]] && [[ "$(last)" == "ci:local FAILED"* ]]; check "16 the same off macOS (no lock path) (got $rc)" $?

# 17. A multi-line not-run reason is one line, so it cannot spoof another step and a waiver still matches.
run "$HEAVY" STUB_GOTIP_VERSION_FAIL=1 OJAS_CI_LOCAL_ALLOW_NOT_RUN=gotip -- "${BASE[@]}" gotip
[[ "$rc" -eq 0 ]] && [[ "$(last)" == "ci:local OK WITH WAIVED SKIPS"* ]] && [[ "$(last)" == *gotip* ]]; check "17 three-line gotip failure + waiver: rc 0, waived gotip named (got $rc)" $?
run "$HEAVY" STUB_GOTIP_VERSION_FAIL=1 OJAS_CI_LOCAL_ALLOW_NOT_RUN=nightly -- "${BASE[@]}" gotip
[[ "$rc" -eq 3 ]] && [[ "$(last)" == "ci:local INCOMPLETE"* ]]; check "17 a spoofed 'nightly:' continuation line is not a nightly skip (got $rc)" $?
[[ "$(grep -c '^ci:local' <<<"$out")" -eq 1 ]]; check "17 exactly one ci:local line" $?

# 18. A relative OJAS_MAC_HEAVY is refused; a directory is refused.
run OJAS_MAC_HEAVY="../stubs/mac_heavy.sh" -- "${BASE[@]}" gotip
[[ "$rc" -eq 64 ]] && has "absolute" && log_lacks "cargo"; check "18 relative OJAS_MAC_HEAVY: rc 64, says absolute (got $rc)" $?
run OJAS_MAC_HEAVY="$STUBS" -- "${BASE[@]}" gotip
[[ "$rc" -eq 64 ]] && log_lacks "cargo"; check "18 OJAS_MAC_HEAVY a directory: rc 64 (got $rc)" $?

# 19. A wrapper that exits 0 without running the gates is NOT RUN, never a quiet pass.
run "$HEAVY" STUB_HEAVY_RC=0 -- "${BASE[@]}" gotip
[[ "$rc" -eq 3 ]] && has "without running the gates" && log_lacks "cargo" && ! bare_ok; check "19 wrapper exits 0 without running: rc 3, says so (got $rc)" $?
run "$HEAVY" -- "${BASE[@]}" gotip
[[ "$rc" -eq 0 ]] && [[ "$(last)" == "ci:local OK" ]]; check "19 a wrapper that runs the gates: rc 0 as before (got $rc)" $?
run "$HEAVY" -- "${BASE[@]}"
[[ "$rc" -eq 3 ]] && [[ "$(last)" == "ci:local INCOMPLETE"* ]]; check "19 the inner INCOMPLETE (3) passes through the wrapper unchanged (got $rc)" $?
ls "$tmp/tmpdir"/ojas-ci-local-ran.* >/dev/null 2>&1 && check "19 no sentinel file is left behind" 1 || check "19 no sentinel file is left behind" 0

# 20. Go gate details a mutation run found unpinned.
run "$HEAVY" -- "${BASE[@]}" gotip
log_has "go test -a -tags gusset_pkgconfig -count=1 -timeout 20m ./..."; check "20 ci_local's own go test: -a and -timeout 20m" $?

# 21. The watchdog itself: a gate that hangs fails its case (rc 124), the suite does not hang.
saved_timeout="$CASE_TIMEOUT"
CASE_TIMEOUT=3
run "$HEAVY" STUB_CARGO_HANG=clippy -- "${BASE[@]}" gotip
CASE_TIMEOUT="$saved_timeout"
[[ "$rc" -eq 124 ]] && has "TIMED OUT"; check "21 a hanging gate is killed by the per-case watchdog (rc 124) (got $rc)" $?

# 15. SIGINT / SIGTERM mid-gate end the run with 128+signal, no verdict, no later gate.
# perl resets INT to its default before exec, so the script starts with the
# dispositions an interactive run has even when this test was backgrounded.
set -m
for sig in TERM INT; do
  want=$((128 + $(kill -l "$sig")))
  rm -f "$BIN"/*
  for t in "${BASE[@]}" gotip; do ln -s "$STUBS/$t" "$BIN/$t"; done
  : >"$LOG"
  outf="$tmp/sig.out"
  env -u MAC_HEAVY_LABEL -u OJAS_CI_LOCAL_ALLOW_NOT_RUN -u OJAS_MAC_HEAVY \
    PATH="$BIN:/usr/bin:/bin" HOME="$tmp/home" TMPDIR="$tmp/tmpdir" STUB_LOG="$LOG" "$HEAVY" STUB_CARGO_SLEEP=clippy \
    perl -e '$SIG{INT} = "DEFAULT"; exec @ARGV or die "exec: $!"' env \
    "$ROOT/scripts/ci_local.sh" >"$outf" 2>&1 &
  pid=$!
  sleep 0.8
  kill -"$sig" "$pid"
  bounded_wait "$pid"
  out="$(cat "$outf")"
  [[ "$rc" -eq "$want" ]] && lacks "ci:local" && log_lacks "cargo deny"; check "15 SIG$sig mid-gate: exit $want, no verdict, no later gate (got $rc)" $?
done
set +m

if [[ "$fails" -ne 0 ]]; then echo "ci_local_test: $fails check(s) failed" >&2; exit 1; fi
echo "ci_local_test OK"
