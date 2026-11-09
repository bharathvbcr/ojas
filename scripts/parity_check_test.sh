#!/usr/bin/env bash
# Self-test for parity_check.sh's closing banner. `uname` and `cargo` are stubbed
# on PATH, so it needs no GPU, no fixtures and no build: it checks that a skipped
# Metal suite (OJAS_ALLOW_NO_GPU=1 off Darwin) never prints the SUCCESS line a run
# in which Metal executed prints, that a run where nothing executed says so, and
# that failing or empty suites still stop the script before any banner.
# Invoke: ./scripts/parity_check_test.sh
# No -e: each check records its own status instead of aborting the run.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
: >"$tmp/fixture.safetensors"

cat >"$tmp/bin/uname" <<'T'
#!/usr/bin/env bash
echo "${STUB_UNAME:-Linux}"
T
cat >"$tmp/bin/cargo" <<'T'
#!/usr/bin/env bash
case "${STUB_CARGO:-ok}" in
  ok) echo "running 3 tests"; echo "test result: ok. 3 passed; 0 failed" ;;
  empty) echo "running 0 tests"; echo "test result: ok. 0 passed; 0 failed" ;;
  fail) echo "test result: FAILED. 1 passed; 2 failed"; exit 101 ;;
  silent) ;;                                   # prints nothing, exits 0
  failed_exit0) echo "test result: FAILED. 1 passed; 2 failed" ;;
  failed_then_ok) echo "test result: FAILED. 1 passed; 2 failed"; echo "test result: ok. 3 passed; 0 failed" ;;
  slow_zero) echo "running 0 tests"; echo "test result: ok. 0 passed; 0 failed"; sleep 1 ;;
esac
T
chmod +x "$tmp/bin/uname" "$tmp/bin/cargo"

fails=0
out=""
rc=0
# run NAME ENV... -- ARGS...: run the real script with the stubs first on PATH.
run() {
  local envs=() args=()
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  args=("$@")
  rc=0
  out="$(env PATH="$tmp/bin:$PATH" "${envs[@]}" "$HERE/parity_check.sh" \
    --fixtures "$tmp/fixture.safetensors" "${args[@]}" 2>&1)" || rc=$?
}
check() { # DESCRIPTION CONDITION-EXIT-CODE
  if [[ "$2" -ne 0 ]]; then echo "FAIL: $1"; echo "$out" | tail -8 | sed 's/^/    | /'; fails=$((fails + 1)); else echo "ok: $1"; fi
}
has() { grep -qF -- "$1" <<<"$out"; }
lacks() { ! has "$1"; }

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
  env PATH="$tmp/bin:$PATH" STUB_UNAME=Linux STUB_CARGO=slow_zero \
    perl -e '$SIG{INT} = "DEFAULT"; exec @ARGV or die "exec: $!"' \
    "$HERE/parity_check.sh" --fixtures "$tmp/fixture.safetensors" --backend cpu >"$outf" 2>&1 &
  pid=$!
  sleep 0.4
  kill -"$sig" "$pid"
  rc=0
  wait "$pid" || rc=$?
  out="$(cat "$outf")"
  [[ "$rc" -eq "$want" ]] && lacks "SUCCESS" && lacks "INCOMPLETE"; check "SIG$sig mid-suite: exit $want, no banner (got $rc)" $?
done
set +m

if [[ "$fails" -ne 0 ]]; then echo "parity_check_test: $fails check(s) failed" >&2; exit 1; fi
echo "parity_check_test OK"
