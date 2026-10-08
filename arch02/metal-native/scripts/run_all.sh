#!/usr/bin/env bash
# Orchestrator for the full Audit 7/8 result set.
#
# Tracks which stages have completed, so you can run one at a time, stop
# anywhere, and resume without redoing work.
#
#   ./scripts/run_all.sh status      what's done, what's left, time estimates
#   ./scripts/run_all.sh next        run the next pending stage and stop
#   ./scripts/run_all.sh quick       all stages under ~15 min (safe to babysit)
#   ./scripts/run_all.sh overnight   the long ones (champion x2 + 20k)
#   ./scripts/run_all.sh <stage>     run one stage by name (re-runs if done)
#   ./scripts/run_all.sh reset       clear completion markers
#
# Every stage logs to out/blog/logs/<stage>.log and marks out/blog/.done/<stage>.
set -uo pipefail
cd "$(dirname "$0")/.."
HERE=$(pwd)
OUT=out/blog
LOGS=$OUT/logs
DONE=$OUT/.done
mkdir -p "$LOGS" "$DONE"
B=./scripts/blog_results.sh

# stage | est | description
STAGES=(
  "env|1 min|Chip / macOS / Xcode / Metal / rustc / MLX versions"
  "gates|10 min|Full lib suite + 3 kernel parity gates (everything depends on these)"
  "baseline|5 min|Pre-Audit-7 vs winning stack, same session, same thermals"
  "glue|10 min|resid_glue atomics A/B, row_blocks 8-128 sweep"
  "profiles|5 min|bwd + fwd + optim section tables, before vs after"
  "variance|5 min|5 back-to-back reps: thermal + run-to-run spread"
  "mlx|10 min|MLX vs metal-native at matched 128M shapes"
  "champion|50 min|2000-step champion, seed 1337, all wins"
  "champion-seed2|50 min|2000-step champion, seed 2026 (needed for a quality claim)"
  "long|8 h|20k-step run - the quality proof"
)
QUICK=(env gates baseline glue profiles variance mlx)
OVERNIGHT=(champion champion-seed2 long)

name_of() { echo "${1%%|*}"; }
est_of()  { local r=${1#*|}; echo "${r%%|*}"; }
desc_of() { echo "${1##*|}"; }

status() {
  echo
  printf "%-18s %-8s %-9s %s\n" STAGE EST STATUS DESCRIPTION
  printf '%.0s-' {1..100}; echo
  for s in "${STAGES[@]}"; do
    local n; n=$(name_of "$s")
    local mark="pending"
    [ -f "$DONE/$n" ] && mark="DONE"
    printf "%-18s %-8s %-9s %s\n" "$n" "$(est_of "$s")" "$mark" "$(desc_of "$s")"
  done
  echo
  echo "next: $(next_stage || echo 'all stages complete')"
  echo "logs: $HERE/$LOGS"
  echo
  echo "Mac-only: every stage above runs on this machine and needs nothing else."
  echo "The story it supports is self-contained — same-session before/after, MLX"
  echo "vs hand-written kernels at matched shapes, and quality held across seeds."
  echo
  echo "OPTIONAL, deferred (needs the 3070 Ti): ./scripts/cuda_ref_128m.sh"
  echo "  Until that runs there is NO valid 128M CUDA number — do not quote the"
  echo "  sota-preset 1.9944 / 650-840 ms figures against 128M results."
  echo "  Mac ingest (when logs arrive): ./scripts/ingest_cuda128m.sh <cuda-logs-dir>"
  echo "  Status: logs/cuda128m/STATUS.md"
  echo "OPTIONAL (needs sudo, second terminal): ./scripts/blog_results.sh power"
}

next_stage() {
  for s in "${STAGES[@]}"; do
    local n; n=$(name_of "$s")
    [ -f "$DONE/$n" ] || { echo "$n"; return 0; }
  done
  return 1
}

run_stage() {
  local n=$1
  local found=""
  for s in "${STAGES[@]}"; do [ "$(name_of "$s")" = "$n" ] && found=$s; done
  if [ -z "$found" ]; then echo "unknown stage: $n"; return 1; fi
  echo
  echo "=============================================================="
  echo " stage: $n   (est $(est_of "$found"))"
  echo " $(desc_of "$found")"
  echo " started $(date '+%F %T')  ->  $LOGS/$n.log"
  echo "=============================================================="
  local t0=$SECONDS
  if "$B" "$n" 2>&1 | tee "$LOGS/$n.log"; then
    local mins=$(( (SECONDS - t0) / 60 ))
    touch "$DONE/$n"
    echo
    echo "--- stage '$n' finished in ${mins}m -> marked done"
  else
    echo
    echo "!!! stage '$n' FAILED (see $LOGS/$n.log). Not marked done." >&2
    return 1
  fi
}

case "${1:-status}" in
  status) status ;;
  reset)  rm -f "$DONE"/*; echo "cleared completion markers" ;;
  next)
    n=$(next_stage) || { echo "all stages complete"; exit 0; }
    run_stage "$n"
    echo; echo "run again for the next stage, or './scripts/run_all.sh status'"
    ;;
  quick)
    for n in "${QUICK[@]}"; do
      [ -f "$DONE/$n" ] && { echo "skip $n (done)"; continue; }
      run_stage "$n" || exit 1
    done
    echo; echo "quick stages complete. Overnight: ./scripts/run_all.sh overnight"
    ;;
  overnight)
    echo "Long stages (~10 h total). Tip: caffeinate -i ./scripts/run_all.sh overnight"
    for n in "${OVERNIGHT[@]}"; do
      [ -f "$DONE/$n" ] && { echo "skip $n (done)"; continue; }
      run_stage "$n" || echo "continuing despite failure in $n"
    done
    "$B" table 2>&1 | tee "$LOGS/table.log"
    ;;
  table) "$B" table 2>&1 | tee "$LOGS/table.log" ;;
  *) run_stage "$1" ;;
esac
