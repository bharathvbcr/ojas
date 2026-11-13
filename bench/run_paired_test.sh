#!/usr/bin/env bash
# Self-test for bench/run_paired.sh's argument and OUT_DIR checks, crash exit code
# and signal handling. The real script runs in place, always with OUT_DIR under a
# temp dir (space and non-ASCII in its name), never the repo's bench/out. The lane
# binaries, python, cargo, go, ioreg, sw_vers and sysctl are stubs, so it needs no
# GPU, no torch and no build. macOS only (run_paired.sh is): elsewhere it says so
# and exits 0, matching ci_local.sh's Darwin-only wiring.
# Invoke: ./bench/run_paired_test.sh
# RUN_PAIRED_UNDER_TEST=<script> points it at another copy of run_paired.sh (used
# to show the cases fail against the pre-change script); nothing else reads it.
# No -e: each check records its own status instead of aborting the run.
set -uo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
  echo "run_paired_test: not applicable off macOS (run_paired.sh needs Metal, BSD tail -r and stat -f)"
  exit 0
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${RUN_PAIRED_UNDER_TEST:-$HERE/run_paired.sh}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
BIN="$tmp/bin"
LOG="$tmp/log"
TGT="$tmp/tgt"
EXD="$TGT/release/examples"
OUTS="$tmp/o ut-ü"
mkdir -p "$BIN" "$EXD" "$OUTS"

stub() { printf '#!/usr/bin/env bash\n%s\n' "$2" >"$1"; chmod +x "$1"; }
lane='echo "lane $(basename "$0") $*" >>"$STUB_LOG"
: >"$STUB_STARTED"
[ "${STUB_SLEEP:-0}" != 0 ] && sleep "$STUB_SLEEP"
echo "{\"stub\":1}" >>"$OJAS_BENCH_OUT"
case "$0:${STUB_CRASH:-}" in
  *metal_vs_torch:metal) exit 3 ;;
  *metal_vs_torch:segv) kill -SEGV $$; sleep 5 ;;
  *metal_vs_torch:abort) kill -ABRT $$; sleep 5 ;;
esac
exit 0'
for b in metal_vs_torch wgpu_vs_torch decode_vs_torch; do stub "$EXD/$b" "$lane"; done
stub "$BIN/py" 'echo "py $*" >>"$STUB_LOG"
case "$2" in time) echo "{\"stub\":1}" >>"$OJAS_BENCH_OUT" ;; esac'
stub "$BIN/cargo" 'echo "cargo $*" >>"$STUB_LOG"; echo "cargo 0.0 stub"'
stub "$BIN/rustc" 'echo "rustc 0.0 stub"'
stub "$BIN/go" 'echo "go $*" >>"$STUB_LOG"; echo "# summary stub"; exit "${STUB_AGG_RC:-0}"'
stub "$BIN/ioreg" 'exit 0'
stub "$BIN/sw_vers" 'echo stub'
stub "$BIN/sysctl" 'echo stub'

fails=0
out=""
rc=0
CASE_TIMEOUT="${CASE_TIMEOUT:-60}"
kill_tree() { # PID: stop it first so it cannot spawn more, then its children, then it
  local c
  kill -STOP "$1" 2>/dev/null
  for c in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$c"; done
  kill -KILL "$1" 2>/dev/null
  return 0
}
bounded_wait() { # PID: sets rc (124 if the per-case watchdog fired)
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
# run ENV... -- ARGS...: the real script with the stubs first on PATH.
run() {
  local envs=()
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  : >"$LOG"
  rm -f "$tmp/started"
  env -u OUT_DIR PATH="$BIN:/usr/bin:/bin:/usr/sbin" PYTHON="$BIN/py" CARGO_TARGET_DIR="$TGT" \
    STUB_LOG="$LOG" STUB_STARTED="$tmp/started" ${envs[@]+"${envs[@]}"} /bin/bash "$SCRIPT" "$@" >"$tmp/run.out" 2>&1 &
  bounded_wait $!
  out="$(cat "$tmp/run.out")"
  if [[ "$rc" -eq 124 ]]; then out="$out"$'\n'"TIMED OUT after ${CASE_TIMEOUT}s"; fi
}
check() { # DESCRIPTION CONDITION-EXIT-CODE
  if [[ "$2" -ne 0 ]]; then
    echo "FAIL: $1"
    echo "$out" | tail -8 | sed 's/^/    | /'
    fails=$((fails + 1))
  else echo "ok: $1"; fi
}
has() { grep -qF -- "$1" <<<"$out"; }
lacks() { ! has "$1"; }
no_build() { ! grep -q '^cargo build' "$LOG"; }

# 1. A non-empty OUT_DIR is refused before anything is built.
d="$OUTS/nonempty"
mkdir -p "$d" && echo x >"$d/old.jsonl"
run OUT_DIR="$d" -- 1
[[ "$rc" -eq 64 ]] && no_build && has "not empty"; check "1 non-empty OUT_DIR: rc 64, no build (got $rc)" $?
[[ "$(cat "$d/old.jsonl")" == x ]] && [[ "$(ls -A "$d")" == old.jsonl ]]; check "1 non-empty OUT_DIR: left untouched" $?

# 2. OUT_DIR is a file.
echo x >"$OUTS/afile"
run OUT_DIR="$OUTS/afile" -- 1
[[ "$rc" -eq 64 ]] && no_build; check "2 OUT_DIR is a file: rc 64, no build (got $rc)" $?

# 3. An empty existing OUT_DIR is fine; a second run into it is refused.
d="$OUTS/empty"
mkdir -p "$d"
run OUT_DIR="$d" -- 1
[[ "$rc" -eq 0 ]] && [[ -s "$d/summary.md" ]]; check "3 empty OUT_DIR, all ok: rc 0 and summary written (got $rc)" $?
run OUT_DIR="$d" -- 1
[[ "$rc" -eq 64 ]] && no_build; check "3b a second run into the same OUT_DIR: rc 64 (got $rc)" $?
d="$OUTS/trail"
mkdir -p "$d"
run OUT_DIR="$d/" -- 1
[[ "$rc" -eq 0 ]]; check "3c OUT_DIR with a trailing slash: rc 0 (got $rc)" $?

# 4. A lane crash must not read as success even when the aggregate passes.
run OUT_DIR="$OUTS/crash" STUB_CRASH=metal STUB_AGG_RC=0 -- 1
[[ "$rc" -eq 1 ]] && has "crashed"; check "4 metal crash, aggregate 0: rc 1 and stderr names the crash (got $rc)" $?
grep -q '"status":"crash"' "$OUTS/crash/round1/ojas-metal.jsonl"; check "4 the crash row is recorded" $?

# 4b. A lane killed by a signal (SEGV, abort: a real Metal crash) is a crash like any
# other: recorded, the other lanes and rounds still run, the summary is written, rc 1.
for mode in segv abort; do
  run OUT_DIR="$OUTS/$mode" STUB_CRASH=$mode STUB_AGG_RC=0 -- 2
  [[ "$rc" -eq 1 ]] && has "crashed"; check "4b $mode in the metal lane: rc 1 and the crash is named (got $rc)" $?
  [[ -s "$OUTS/$mode/summary.md" ]]; check "4b $mode: the summary is still written" $?
  [[ -e "$OUTS/$mode/round1/ojas-wgpu.jsonl" ]] && [[ -e "$OUTS/$mode/round2/torch-mps.jsonl" ]]; check "4b $mode: the other lanes and the next round still ran" $?
  grep -q '"status":"crash","exit":1[0-9][0-9]' "$OUTS/$mode/round1/ojas-metal.jsonl"; check "4b $mode: the signal exit status is recorded in the crash row" $?
done

# 5. Aggregate failure with no crash.
run OUT_DIR="$OUTS/agg" STUB_AGG_RC=1 -- 1
[[ "$rc" -eq 1 ]] && lacks "crashed"; check "5 aggregate 1, no crash: rc 1 (got $rc)" $?
run OUT_DIR="$OUTS/agg2" STUB_AGG_RC=4 STUB_CRASH=metal -- 1
[[ "$rc" -eq 4 ]]; check "5b a non-zero aggregate status is never lowered (got $rc)" $?

# 6. ROUNDS must be an integer 1-999 without a leading zero.
for bad in 0 abc -1 1e3 0005 07 999999 "1 2" ""; do
  run OUT_DIR="$OUTS/rounds" -- "$bad"
  [[ "$rc" -eq 64 ]] && no_build && [[ ! -e "$OUTS/rounds" ]]; check "6 ROUNDS '$bad': rc 64, no build, no OUT_DIR created (got $rc)" $?
done

# 7. OUT_DIR shapes: only .DS_Store, a symlink to a non-empty dir, a leading dash.
d="$OUTS/dsstore"
mkdir -p "$d" && : >"$d/.DS_Store"
run OUT_DIR="$d" -- 1
[[ "$rc" -eq 64 ]] && no_build; check "7 OUT_DIR holding only .DS_Store: rc 64 (non-empty) (got $rc)" $?
mkdir -p "$OUTS/target-dir" && echo x >"$OUTS/target-dir/f"
ln -s "$OUTS/target-dir" "$OUTS/link"
run OUT_DIR="$OUTS/link" -- 1
[[ "$rc" -eq 64 ]] && no_build; check "7 OUT_DIR a symlink to a non-empty dir: rc 64 (got $rc)" $?
run OUT_DIR="-x" -- 1
[[ "$rc" -eq 64 ]] && no_build; check "7 OUT_DIR starting with '-': rc 64 (got $rc)" $?

# 8. SIGINT during a sleeping lane: 130, no summary line. perl resets INT to its
# default before exec so the script starts with an interactive run's dispositions.
set -m
: >"$LOG"
rm -f "$tmp/started"
outf="$tmp/sig.out"
env -u OUT_DIR PATH="$BIN:/usr/bin:/bin:/usr/sbin" PYTHON="$BIN/py" CARGO_TARGET_DIR="$TGT" \
  STUB_LOG="$LOG" STUB_STARTED="$tmp/started" STUB_SLEEP=3 OUT_DIR="$OUTS/sig" \
  perl -e '$SIG{INT} = "DEFAULT"; exec @ARGV or die "exec: $!"' /bin/bash "$SCRIPT" 1 >"$outf" 2>&1 &
pid=$!
for _ in $(seq 1 100); do [[ -e "$tmp/started" ]] && break; sleep 0.1; done
kill -INT "$pid"
bounded_wait "$pid"
out="$(cat "$outf")"
[[ -e "$tmp/started" ]] && [[ "$rc" -eq 130 ]] && lacks "summary:"; check "8 SIGINT during a lane: exit 130, no summary line (got $rc)" $?
set +m

if [[ "$fails" -ne 0 ]]; then echo "run_paired_test: $fails check(s) failed" >&2; exit 1; fi
echo "run_paired_test OK"
