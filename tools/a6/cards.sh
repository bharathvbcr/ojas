#!/bin/bash
# usage: cards.sh <task_id>...  — fetch each card into backups/a6-tools/card-<id>.json and print a digest
D=/Users/bharath/Code/research/ojas/backups/a6-tools
for id in "$@"; do
  bash "$D/gp.sh" gitpulse_get_task "{\"repo_path\":\"/Users/bharath/Code/research/ojas\",\"task_id\":\"$id\"}" 8 > "$D/card-$id.json"
done
python3 "$D/digest.py" "$@"
