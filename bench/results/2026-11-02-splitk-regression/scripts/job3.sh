#!/bin/bash
# Window 3. d92a2e7 needs tessl 80caf39 (alloc_internal_buffer,
# SharedBuf::to_gpu_buffer), committed two minutes after it, so the co-dated
# d2c3ea8 does not compile there. Points:
#   i168x    fcad865 + 80caf39  cross build: tessl moves, ojas does not
#   i169     d92a2e7 + 80caf39  ojas moves to the lead commit
#   i169diag d92a2e7 + 80caf39  bf16_acc constant false in the split-K part kernel (diagnostic only)
#   i175     a1819b3 + 80caf39  last ojas point before HEAD's merge
# Anchors from window 1: i168 (fcad865 + d2c3ea8), i176 (286a444 + e0ea9a0).
set -u
LANE=/Users/bharath/Code/research/ojas/target-splitk
echo "# job3 start $(date -u +%Y-%m-%dT%H:%M:%SZ); $(sysctl -n vm.loadavg)"
built=(i168 i176)
for spec in "i168x fcad865 80caf39" "i169 d92a2e7 80caf39" "i175 a1819b3 80caf39"; do
  set -- $spec
  if bash "$LANE/build.sh" "$1" "$2" "$3"; then built+=("$1"); else echo "# BUILD FAILED $1"; fi
done
if bash "$LANE/build_diag.sh" i169diag d92a2e7 80caf39; then built+=(i169diag); else echo "# BUILD FAILED i169diag"; fi
echo "# built: ${built[*]}"
bash "$LANE/pair.sh" 6 "${built[@]}"
echo "# job3 end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
