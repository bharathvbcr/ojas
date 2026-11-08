#!/bin/bash
# Window 1: build old (6), mid (91), lead-1 (168), new (176), then pair all four.
set -u
LANE=/Users/bharath/Code/research/ojas/target-splitk
echo "# job1 start $(date -u +%Y-%m-%dT%H:%M:%SZ); $(sysctl -n vm.loadavg)"
built=(i006)
for spec in "i091 3ac4428 3c1f338" "i168 fcad865 d2c3ea8" "i176 286a444 e0ea9a0"; do
  set -- $spec
  if bash "$LANE/build.sh" "$1" "$2" "$3"; then built+=("$1"); else echo "# BUILD FAILED $1"; fi
done
echo "# built: ${built[*]}"
if [ "${#built[@]}" -ge 2 ]; then
  bash "$LANE/pair.sh" 4 "${built[@]}"
fi
echo "# job1 end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
