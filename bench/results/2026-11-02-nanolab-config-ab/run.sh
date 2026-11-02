#!/bin/bash
# The interleaved driver of the nanolab config A/B (README.md here).
#
#   bash bench/results/2026-11-02-nanolab-config-ab/run.sh <binary> <out dir>
#
# <binary> is a release build of ojas-model's `nanolab_config_ab` example.
# Run it under the machine lock (tools/mac_heavy.sh in Lappi-decision): it is
# one GPU job for its whole length, so nothing else uses the GPU between arms.
#
# Every rep runs each arm once as its own process, in an order rotated by the
# rep index, so no arm always follows the same one; the B=1 and B=8 groups
# alternate which goes first. One `phase` process per batch size per rep.
# Rows are appended as they finish, so a killed run keeps what it measured.
# /usr/bin/time -l wraps every process for its peak memory footprint; the
# load average is read before and after each process.
set -u

BIN=$1
OUT=$2
REPS=${REPS:-5}
STEPS=${STEPS:-10}
PHASE_REPS=${PHASE_REPS:-8}

mkdir -p "$OUT/stderr"
ROWS="$OUT/rows.jsonl"
RUNS="$OUT/runs.tsv"
[ -f "$RUNS" ] || printf 'tag\trep\tstart_utc\tstatus\twall_s\tpeak_footprint_bytes\tload_before\tload_after\n' >"$RUNS"

{
  echo "date: $(date -u +%FT%TZ)"
  echo "git: $(git rev-parse HEAD) (dirty files: $(git status --porcelain | wc -l | awk '{print $1}'))"
  echo "example sha1: $(shasum ojas-model/examples/nanolab_config_ab.rs | awk '{print $1}')"
  echo "tessl: $(git -C ../tessl rev-parse HEAD 2>&1) (dirty files: $(git -C ../tessl status --porcelain 2>&1 | wc -l | awk '{print $1}'))"
  echo "rustc: $(rustc --version)"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
  echo "memory_bytes: $(sysctl -n hw.memsize)"
  echo "binary: $BIN $(stat -f '%Sm %z' "$BIN")"
  echo "reps: $REPS  steps: $STEPS  phase_reps: $PHASE_REPS"
  echo "mac_heavy_label: ${MAC_HEAVY_LABEL:-unset (not under the machine lock)}"
} >>"$OUT/env.txt"

load() { sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1, $2, $3}'; }

# tag batch ns5 ce_cols checkpoint
B1=(
  "b1-base 1 f32 8192 off"
  "b1-aa 1 f32 8192 off"
  "b1-ns5bf16 1 bf16 8192 off"
  "b1-ce1tile 1 f32 50304 off"
)
B8=(
  "b8-base 8 f32 8192 off"
  "b8-ns5bf16 8 bf16 8192 off"
  "b8-ce1tile 8 f32 50304 off"
  "b8-ckpt 8 f32 8192 blocks"
)

run_one() {
  local tag=$1 rep=$2
  shift 2
  local err="$OUT/stderr/$tag-r$rep.txt"
  local start lb la t0 t1 status peak
  start=$(date -u +%FT%TZ)
  lb=$(load)
  t0=$(date +%s)
  /usr/bin/time -l "$BIN" "$@" >>"$ROWS" 2>"$err"
  status=$?
  t1=$(date +%s)
  la=$(load)
  peak=$(awk '/peak memory footprint/ {print $1}' "$err")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$rep" "$start" "$status" "$((t1 - t0))" \
    "${peak:-?}" "$lb" "$la" >>"$RUNS"
  echo "$(date -u +%TZ) $tag r$rep status $status $((t1 - t0))s"
}

group() {
  local rep=$1
  shift
  local arms=("$@") n=$# i
  for ((i = 0; i < n; i++)); do
    # shellcheck disable=SC2086 # the arm string is the argument list
    set -- ${arms[$(((i + rep) % n))]}
    run_one "$1" "$rep" step "$2" "$3" "$4" "$5" "$STEPS" 1024 "$1:r$rep"
  done
}

for ((rep = 1; rep <= REPS; rep++)); do
  if ((rep % 2)); then
    group "$rep" "${B1[@]}"
    group "$rep" "${B8[@]}"
  else
    group "$rep" "${B8[@]}"
    group "$rep" "${B1[@]}"
  fi
  run_one phase-b1 "$rep" phase 1 "$PHASE_REPS" 1024
  run_one phase-b8 "$rep" phase 8 "$PHASE_REPS" 1024
done
echo "done: $ROWS"
