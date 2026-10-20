#!/bin/bash
# Detached launch of owner-verify-rerun.sh (owner authorised ojas-a6 to run verify, 2026-10-17).
# Writes logs/rerun/rerun.pid, logs/rerun/run.out and the logs/rerun/RERUN-DONE sentinel (exit code).
set -u
T=/Users/bharath/Code/research/ojas/backups/a6-tools
mkdir -p "$T/logs/rerun"
rm -f "$T/logs/rerun/RERUN-DONE"
MAXLOAD=${1:-80}
nohup bash -c "MAC_HEAVY_MAX_LOAD=$MAXLOAD bash '$T/owner-verify-rerun.sh' > '$T/logs/rerun/run.out' 2>&1; echo \"exit=\$? at \$(date '+%H:%M:%S')\" > '$T/logs/rerun/RERUN-DONE'" > /dev/null 2>&1 &
echo $! > "$T/logs/rerun/rerun.pid"
disown
echo "started rerun pid $(cat "$T/logs/rerun/rerun.pid") with MAC_HEAVY_MAX_LOAD=$MAXLOAD"
