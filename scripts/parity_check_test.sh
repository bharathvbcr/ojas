#!/usr/bin/env bash
# Self-test for parity_check.sh: the closing banner, the fail-closed lock lookup on
# macOS, and the refused --no-gpu-lock. `uname`, `cargo` and `mac_heavy.sh` are
# stubs, so it needs no GPU, no fixtures, no build and no lock. It checks that a
# skipped Metal suite (OJAS_ALLOW_NO_GPU=1 off Darwin) never prints the SUCCESS line
# a run in which Metal executed prints, that a run where nothing executed says so,
# that failing or empty suites still stop the script before any banner, and that on
# Darwin the Metal suite never runs without mac_heavy.
# The script runs from a COPY inside `$tmp/r/ojas/scripts/`, with HOME under $tmp and
# MAC_HEAVY_LABEL / OJAS_MAC_HEAVY unset, so the real mac_heavy.sh and the real
# sibling checkouts can never be found, and a case run under ci_local or heavy.sh is
# not diverted into the in-place branch.
# Invoke: ./scripts/parity_check_test.sh
# PARITY_CHECK_UNDER_TEST=<file> runs the cases against another copy of the script
# (used to show they fail against the pre-change script); nothing else reads it.
# No -e: each check records its own status instead of aborting the run.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${PARITY_CHECK_UNDER_TEST:-$HERE/parity_check.sh}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/r/ojas/scripts" "$tmp/home"
PARITY="$tmp/r/ojas/scripts/parity_check.sh"
cp "$SRC" "$PARITY"
chmod +x "$PARITY"
LOG="$tmp/log"
: >"$tmp/fixture.safetensors"

cat >"$tmp/bin/uname" <<'T'
#!/usr/bin/env bash
echo "${STUB_UNAME:-Linux}"
T
cat >"$tmp/bin/cargo" <<'T'
#!/usr/bin/env bash
echo "cargo $*" >>"${STUB_LOG:-/dev/null}"
case "${STUB_CARGO:-ok}" in
  ok) echo "running 3 tests"; echo "test result: ok. 3 passed; 0 failed" ;;
  empty) echo "running 0 tests"; echo "test result: ok. 0 passed; 0 failed" ;;
  fail) echo "test result: FAILED. 1 passed; 2 failed"; exit 101 ;;
  silent) ;;                                   # prints nothing, exits 0
  failed_exit0) echo "test result: FAILED. 1 passed; 2 failed" ;;
  failed_then_ok) echo "test result: FAILED. 1 passed; 2 failed"; echo "test result: ok. 3 passed; 0 failed" ;;
  slow_zero) echo "running 0 tests"; echo "test result: ok. 0 passed; 0 failed"; sleep 1 ;;
  hang) sleep 600 ;;
esac
T
chmod +x "$tmp/bin/uname" "$tmp/bin/cargo"

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
# run NAME=VALUE... -- ARGS...: the script copy, with the stubs first on PATH.
run() {
  local envs=() args=()
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  args=("$@")
  : >"$LOG"
  env -u MAC_HEAVY_LABEL -u OJAS_MAC_HEAVY -u OJAS_ALLOW_NO_GPU \
    PATH="$tmp/bin:$PATH" HOME="$tmp/home" STUB_LOG="$LOG" ${envs[@]+"${envs[@]}"} \
    "$PARITY" --fixtures "$tmp/fixture.safetensors" ${args[@]+"${args[@]}"} >"$tmp/run.out" 2>&1 &
  bounded_wait $!
  out="$(cat "$tmp/run.out")"
  if [[ "$rc" -eq 124 ]]; then out="$out"$'\n'"TIMED OUT after ${CASE_TIMEOUT}s"; fi
}
check() { # DESCRIPTION CONDITION-EXIT-CODE
  if [[ "$2" -ne 0 ]]; then echo "FAIL: $1"; echo "$out" | tail -8 | sed 's/^/    | /'; fails=$((fails + 1)); else echo "ok: $1"; fi
}
has() { grep -qF -- "$1" <<<"$out"; }
lacks() { ! has "$1"; }
log_has() { grep -qF -- "$1" "$LOG"; }
log_lacks() { ! log_has "$1"; }
cnt() { grep -cF -- "$1" "$LOG"; }
no_cargo() { ! grep -q '^cargo ' "$LOG"; }
no_banner() { lacks "SUCCESS" && lacks "INCOMPLETE" && lacks "NOTHING RAN"; }

# mkwrapper FILE TAG: a stub mac_heavy.sh that logs `mac_heavy[TAG] <args>`, exports the
# label like the real one (mac_heavy.sh:475) and execs the command. STUB_HEAVY_RC makes
# it exit instead. A depth limit stops a regression that re-enters it from recursing.
mkwrapper() {
  mkdir -p "$(dirname "$1")"
  printf '#!/usr/bin/env bash\necho "mac_heavy[%s] $*" >>"$STUB_LOG"\nd=${STUB_HEAVY_DEPTH:-0}\nif [ "$d" -ge 3 ]; then echo "mac_heavy recursion" >>"$STUB_LOG"; exit 99; fi\nexport STUB_HEAVY_DEPTH=$((d + 1))\n[ -n "${STUB_HEAVY_RC:-}" ] && exit "$STUB_HEAVY_RC"\nexport MAC_HEAVY_LABEL="$1"; shift; exec "$@"\n' "$2" >"$1"
  chmod +x "$1"
}
# fresh: no wrapper anywhere any lookup could reach.
fresh() { rm -rf "$tmp/r/Lappi-decision" "$tmp/r/tessl" "$tmp/home/Code" "$tmp/mh"; }
SIB="$tmp/r/Lappi-decision/tools/mac_heavy.sh"
HOMEW="$tmp/home/Code/research/Lappi-decision/tools/mac_heavy.sh"

# 1. Everything ran: SUCCESS, naming both backends.
run STUB_UNAME=Darwin MAC_HEAVY_LABEL=t -- --backend all
check "all ran: exit 0" "$rc"
has "SUCCESS" && has "ran: cpu metal"; check "all ran: SUCCESS naming cpu and metal" $?

# 2. Metal skipped off Darwin: still exit 0 (documented clean skip) but no SUCCESS.
run STUB_UNAME=Linux OJAS_ALLOW_NO_GPU=1 -- --backend all
check "metal skipped: exit 0 (clean skip)" "$rc"
lacks "SUCCESS"; check "metal skipped: banner is not SUCCESS" $?
has "INCOMPLETE" && has "ran: cpu" && has "skipped: metal"; check "metal skipped: banner names ran cpu, skipped metal" $?

# 3. Only Metal requested and skipped: nothing executed.
run STUB_UNAME=Linux OJAS_ALLOW_NO_GPU=1 -- --backend metal
check "metal-only skipped: exit 0" "$rc"
lacks "SUCCESS"; check "metal-only skipped: banner is not SUCCESS" $?
has "NOTHING RAN"; check "metal-only skipped: banner says NOTHING RAN" $?

# 4. cpu only: SUCCESS, metal not claimed.
run STUB_UNAME=Linux -- --backend cpu
has "SUCCESS" && has "ran: cpu)"; check "cpu only: SUCCESS naming cpu only" $?

# 5. Off Darwin without the opt-in is a hard error, never a banner.
run STUB_UNAME=Linux OJAS_ALLOW_NO_GPU=0 -- --backend metal
[[ "$rc" -ne 0 ]]; check "metal without opt-in off Darwin: non-zero" $?
lacks "SUCCESS" && lacks "INCOMPLETE"; check "metal without opt-in: no banner" $?

# 6. A failing suite and an empty suite stop the script before any banner.
run STUB_UNAME=Linux STUB_CARGO=fail -- --backend cpu
[[ "$rc" -eq 101 ]] && lacks "SUCCESS"; check "failing suite: exit 101, no SUCCESS" $?
run STUB_UNAME=Linux STUB_CARGO=empty -- --backend cpu
[[ "$rc" -ne 0 ]] && lacks "SUCCESS"; check "zero-test suite: non-zero, no SUCCESS" $?

# 7. Exit 0 is not evidence: a silent cargo, a FAILED line, or a FAILED line followed
# by an ok line must not reach SUCCESS (or count as ran).
run STUB_UNAME=Linux STUB_CARGO=silent -- --backend cpu
[[ "$rc" -ne 0 ]] && lacks "SUCCESS"; check "silent exit 0: non-zero, no SUCCESS" $?
run STUB_UNAME=Linux STUB_CARGO=failed_exit0 -- --backend cpu
[[ "$rc" -ne 0 ]] && lacks "SUCCESS"; check "FAILED line with exit 0: non-zero, no SUCCESS" $?
run STUB_UNAME=Linux STUB_CARGO=failed_then_ok -- --backend cpu
[[ "$rc" -ne 0 ]] && lacks "SUCCESS"; check "FAILED then ok with exit 0: non-zero, no SUCCESS" $?

# 8. SIGTERM / SIGINT during a suite end the script with 128+signal and no banner,
# rather than letting it resume with its log deleted.
# A shell started with SIGINT ignored (any caller that backgrounds this test with `&`,
# a CI step, the b3 launcher) hands that disposition to its children, and a script that
# inherits an ignored SIGINT cannot trap it: the INT check below would then measure the
# caller, not parity_check.sh. perl resets INT to its default before exec'ing the script,
# so the script under test always starts with the dispositions an interactive run has.
set -m
for sig in TERM INT; do
  want=$((128 + $(kill -l "$sig")))
  outf="$tmp/sig.out"
  env -u MAC_HEAVY_LABEL -u OJAS_MAC_HEAVY \
    PATH="$tmp/bin:$PATH" HOME="$tmp/home" STUB_LOG="$LOG" STUB_UNAME=Linux STUB_CARGO=slow_zero \
    perl -e '$SIG{INT} = "DEFAULT"; exec @ARGV or die "exec: $!"' \
    "$PARITY" --fixtures "$tmp/fixture.safetensors" --backend cpu >"$outf" 2>&1 &
  pid=$!
  sleep 0.4
  kill -"$sig" "$pid"
  bounded_wait "$pid"
  out="$(cat "$outf")"
  [[ "$rc" -eq "$want" ]] && lacks "SUCCESS" && lacks "INCOMPLETE"; check "SIG$sig mid-suite: exit $want, no banner (got $rc)" $?
done
set +m

# 9. Darwin, Metal requested, no mac_heavy anywhere: fail closed BEFORE any suite
# (so --backend all cannot run the CPU suite and then refuse), naming OJAS_MAC_HEAVY.
fresh
for b in metal all; do
  run STUB_UNAME=Darwin -- --backend "$b"
  [[ "$rc" -eq 1 ]] && has "OJAS_MAC_HEAVY"; check "P1 no wrapper, --backend $b: rc 1 naming OJAS_MAC_HEAVY (got $rc)" $?
  no_cargo; check "P1 no wrapper, --backend $b: cargo never invoked" $?
  no_banner; check "P1 no wrapper, --backend $b: no banner" $?
done
run STUB_UNAME=Darwin -- --backend cpu
[[ "$rc" -eq 0 ]] && has "ran: cpu)"; check "P1 no wrapper, --backend cpu: the CPU suite needs no lock (got $rc)" $?

# 10. A bad OJAS_MAC_HEAVY is a usage error (64) before any suite.
fresh
mkdir -p "$tmp/mh/adir"
: >"$tmp/mh/plain"
chmod -x "$tmp/mh/plain"
mkwrapper "$tmp/mh/ok.sh" ok
for b in metal all; do
  for bad in "mh/ok.sh" "$tmp/mh/adir" "$tmp/mh/plain" "$tmp/mh/missing"; do
    run STUB_UNAME=Darwin OJAS_MAC_HEAVY="$bad" -- --backend "$b"
    [[ "$rc" -eq 64 ]] && no_cargo && no_banner && log_lacks "mac_heavy["; check "P2 OJAS_MAC_HEAVY=${bad#"$tmp"/} (--backend $b): rc 64, no cargo, no wrapper (got $rc)" $?
  done
done
run STUB_UNAME=Darwin OJAS_MAC_HEAVY="mh/ok.sh" -- --backend metal
has "absolute"; check "P2 a relative OJAS_MAC_HEAVY says it must be absolute" $?
run STUB_UNAME=Darwin OJAS_MAC_HEAVY="$tmp/mh/adir" -- --backend cpu
[[ "$rc" -eq 0 ]]; check "P2 a CPU-only run does not look at OJAS_MAC_HEAVY (got $rc)" $?

# 11. OJAS_MAC_HEAVY=<stub>: the stub is invoked once with the label, then the cargo test.
fresh
mkwrapper "$tmp/mh/ok.sh" explicit
run STUB_UNAME=Darwin OJAS_MAC_HEAVY="$tmp/mh/ok.sh" -- --backend metal
[[ "$rc" -eq 0 ]] && has "SUCCESS" && has "ran: metal"; check "P3 OJAS_MAC_HEAVY stub: rc 0, SUCCESS (got $rc)" $?
[[ "$(cnt 'mac_heavy[explicit] ojas-parity-check cargo test --release -p ojas-metal --test sam_fixtures_metal')" -eq 1 ]]; check "P3 the stub ran once with the label and the metal cargo test" $?
run STUB_UNAME=Darwin OJAS_MAC_HEAVY="$tmp/mh/ok.sh" -- --backend all
[[ "$rc" -eq 0 ]] && has "ran: cpu metal" && [[ "$(cnt 'mac_heavy[')" -eq 1 ]] && log_has "cargo test --release -p ojas-oracle --test sam_fixtures_cpu"; check "P3 --backend all: one lock for the metal suite only, CPU ran too (got $rc)" $?

# 12. Order: OJAS_MAC_HEAVY, then the sibling, then $HOME; the tessl candidate is gone.
fresh
mkwrapper "$tmp/mh/ok.sh" explicit
mkwrapper "$SIB" sibling
mkwrapper "$HOMEW" home
run STUB_UNAME=Darwin OJAS_MAC_HEAVY="$tmp/mh/ok.sh" -- --backend metal
log_has "mac_heavy[explicit]" && log_lacks "mac_heavy[sibling]" && log_lacks "mac_heavy[home]"; check "P4 OJAS_MAC_HEAVY beats sibling and HOME" $?
run STUB_UNAME=Darwin -- --backend metal
log_has "mac_heavy[sibling]" && log_lacks "mac_heavy[explicit]" && log_lacks "mac_heavy[home]"; check "P4 sibling beats HOME" $?
rm -rf "$tmp/r/Lappi-decision"
run STUB_UNAME=Darwin -- --backend metal
[[ "$rc" -eq 0 ]] && log_has "mac_heavy[home]"; check "P4 HOME candidate used when nothing else exists (got $rc)" $?
# A directory or a non-executable file at a candidate path is skipped, not used.
mkdir -p "$SIB"
mkdir -p "$(dirname "$HOMEW")"
chmod -x "$HOMEW"
run STUB_UNAME=Darwin -- --backend metal
[[ "$rc" -eq 1 ]] && no_cargo && log_lacks "mac_heavy["; check "P4 a directory and a non-executable file at the candidate paths are skipped (got $rc)" $?
fresh
mkwrapper "$tmp/r/tessl/tools/mac_heavy.sh" tessl
run STUB_UNAME=Darwin -- --backend metal
[[ "$rc" -eq 1 ]] && has "OJAS_MAC_HEAVY" && no_cargo && log_lacks "mac_heavy[tessl]"; check "P5 only tessl/tools/mac_heavy.sh present: rc 1, not used (got $rc)" $?

# 13. Already under mac_heavy: run in place, never take the lock twice.
fresh
mkwrapper "$SIB" sibling
run STUB_UNAME=Darwin MAC_HEAVY_LABEL=x -- --backend metal
[[ "$rc" -eq 0 ]] && has "SUCCESS" && log_lacks "mac_heavy[" && log_has "cargo test --release -p ojas-metal"; check "P6 MAC_HEAVY_LABEL set: runs in place, wrapper not invoked (got $rc)" $?
fresh
run STUB_UNAME=Darwin MAC_HEAVY_LABEL=x -- --backend all
[[ "$rc" -eq 0 ]] && has "ran: cpu metal"; check "P6 MAC_HEAVY_LABEL set and no wrapper anywhere: still runs in place (got $rc)" $?

# 14. Off Darwin there is no lookup at all.
fresh
mkwrapper "$HOMEW" home
run STUB_UNAME=Linux -- --backend cpu
[[ "$rc" -eq 0 ]] && log_lacks "mac_heavy["; check "P7 Linux cpu with a HOME wrapper present: no lookup (got $rc)" $?
run STUB_UNAME=Linux OJAS_ALLOW_NO_GPU=1 OJAS_MAC_HEAVY=relative-junk -- --backend metal
[[ "$rc" -eq 0 ]] && has "NOTHING RAN" && log_lacks "mac_heavy["; check "P7 Linux metal skip ignores a junk OJAS_MAC_HEAVY (got $rc)" $?
run STUB_UNAME=Linux -- --backend metal
[[ "$rc" -eq 1 ]] && has "require macOS" && no_cargo; check "P7 Linux metal without the opt-in is still the macOS-required error (got $rc)" $?

# 15. The wrapper's own refusal codes pass through with no banner.
fresh
mkwrapper "$SIB" sibling
for code in 75 76; do
  run STUB_UNAME=Darwin STUB_HEAVY_RC=$code -- --backend metal
  [[ "$rc" -eq $code ]] && no_banner; check "P8 wrapper exits $code: rc $code, no banner (got $rc)" $?
done

# 16. --no-gpu-lock is gone: an unknown option, cargo never ran.
for u in Darwin Linux; do
  run STUB_UNAME=$u MAC_HEAVY_LABEL=x -- --no-gpu-lock --backend all
  [[ "$rc" -eq 1 ]] && has "Unknown option: --no-gpu-lock" && no_cargo && no_banner; check "A2 --no-gpu-lock on $u: rc 1, unknown option, cargo never ran (got $rc)" $?
done
lacks "Do not require or attempt"; check "A2 the help text no longer offers --no-gpu-lock" $?

# 17. The watchdog itself: a hung suite fails its case (rc 124), the suite does not hang.
saved_timeout="$CASE_TIMEOUT"
CASE_TIMEOUT=3
run STUB_UNAME=Linux STUB_CARGO=hang -- --backend cpu
CASE_TIMEOUT="$saved_timeout"
[[ "$rc" -eq 124 ]] && has "TIMED OUT"; check "a hanging suite is killed by the per-case watchdog (rc 124) (got $rc)" $?

if [[ "$fails" -ne 0 ]]; then echo "parity_check_test: $fails check(s) failed" >&2; exit 1; fi
echo "parity_check_test OK"
