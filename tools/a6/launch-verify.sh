#!/bin/bash
# Detached launch of owner-verify-main.sh for the overnight slot (owner authorised ojas-a6 to run it,
# 2026-10-17). Survives the launching shell (nohup + disown). It writes:
#   logs/verify.pid       the detached runner's PID (kill only this PID, never by name)
#   logs/verify-run.out   the runner's own stdout/stderr
#   logs/VERIFY-DONE      sentinel with the exit code, written when the run ends
# MAC_HEAVY_MAX_LOAD defaults to 80 here; mac_heavy waits while load is above it.
set -u
T=/Users/bharath/Code/research/ojas/backups/a6-tools
mkdir -p "$T/logs"
rm -f "$T/logs/VERIFY-DONE"
MAXLOAD=${1:-80}
nohup bash -c "MAC_HEAVY_MAX_LOAD=$MAXLOAD bash '$T/owner-verify-main.sh' > '$T/logs/verify-run.out' 2>&1; echo \"exit=\$? at \$(date '+%H:%M:%S')\" > '$T/logs/VERIFY-DONE'" > /dev/null 2>&1 &
echo $! > "$T/logs/verify.pid"
disown
echo "started pid $(cat "$T/logs/verify.pid") with MAC_HEAVY_MAX_LOAD=$MAXLOAD"
