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

if [[ "$fails" -ne 0 ]]; then echo "parity_check_test: $fails check(s) failed" >&2; exit 1; fi
echo "parity_check_test OK"
