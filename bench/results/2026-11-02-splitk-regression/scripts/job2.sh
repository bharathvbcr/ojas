#!/bin/bash
# Window 2: i169 (d92a2e7), i175 (a1819b3, last point before tessl moves),
# and i169diag (d92a2e7 with the split-K part kernel's bf16_acc constant false;
# diagnostic only). Paired with the window-1 anchors i168 and i176.
set -u
LANE=/Users/bharath/Code/research/ojas/target-splitk
echo "# job2 start $(date -u +%Y-%m-%dT%H:%M:%SZ); $(sysctl -n vm.loadavg)"
built=(i168 i176)
for spec in "i169 d92a2e7 d2c3ea8" "i175 a1819b3 d2c3ea8"; do
  set -- $spec
  if bash "$LANE/build.sh" "$1" "$2" "$3"; then built+=("$1"); else echo "# BUILD FAILED $1"; fi
done
if bash "$LANE/build_diag.sh" i169diag d92a2e7 d2c3ea8; then built+=(i169diag); else echo "# BUILD FAILED i169diag"; fi
echo "# built: ${built[*]}"
bash "$LANE/pair.sh" 6 "${built[@]}"
echo "# job2 end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
