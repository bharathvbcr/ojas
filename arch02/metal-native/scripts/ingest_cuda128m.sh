#!/usr/bin/env bash
# Mac-side ingest for same-shape CUDA 128M artifacts (Phase F).
#
# Usage:
#   ./scripts/ingest_cuda128m.sh /path/to/parameter-golf/logs/cuda128m
#   ./scripts/ingest_cuda128m.sh ./cuda128m_tarball_dir
#
# Copies into metal-native/logs/cuda128m, scores with score_cuda128m.py, and
# writes STATUS.md. Does not invent speed/BPB claims.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=logs/cuda128m
SRC=${1:-}
mkdir -p "$DEST"

if [ -z "$SRC" ]; then
  cat <<EOF
usage: $0 <dir-with-cuda-logs>

Expected contents from the 3070 Ti box (cuda_ref_128m.sh):
  .accum                          # GRAD_ACCUM_STEPS that fit (or "none")
  probe_accumN.log / probe_accumN/
  bench_accumN.log  / bench_accumN/
  quality_accumN.log / quality_accumN/

After copy, this script writes:
  $DEST/STATUS.md
  $DEST/summary.json
EOF
  exit 1
fi

if [ ! -d "$SRC" ]; then
  echo "source not a directory: $SRC" >&2
  exit 1
fi

echo "ingest: $SRC -> $DEST"
rsync -a --delete --exclude STATUS.md --exclude summary.json "$SRC"/ "$DEST"/

ACCUM=$(cat "$DEST/.accum" 2>/dev/null || echo "UNKNOWN")
python3 scripts/score_cuda128m.py "$DEST" --write-summary >/dev/null

HOST=$(uname -s)-$(uname -m)
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$DEST/STATUS.md" <<EOF
# CUDA 128M same-shape reference — ingest status

- **ingested_at_utc:** $NOW
- **ingest_host:** $HOST (Mac-side; not the CUDA runner)
- **source:** \`$SRC\`
- **GRAD_ACCUM_STEPS:** \`$ACCUM\`  ← **must be disclosed** next to every CUDA number
- **shape:** L24 / C768 / H24 / Hkv12 / MLP×3 / T256 / 4096 tok / seed 1337
- **scorer:** \`scripts/score_cuda128m.py\` → \`summary.json\`

## Claim discipline

- The sota-ladder CUDA figures (**1.9944 BPB**, **~650–840 ms**) are **not**
  128M twins. Do not quote them against arch02-128m.
- Do **not** claim Mac vs CUDA speed unless \`summary.json\` has real same-shape
  CUDA steady-state ms **and** a disclosed accum.
- CUDA \`val_bpb\` vs Metal \`final_ema_sliding_bpb\` use different eval paths;
  prefer one shared scorer before claiming quality parity.

## Next (if probe never ran on CUDA)

On the 3070 Ti box:

\`\`\`bash
cd parameter-golf   # or monorepo root; script finds train_gpt.py
../ojas/arch02/metal-native/scripts/cuda_ref_128m.sh probe
../ojas/arch02/metal-native/scripts/cuda_ref_128m.sh bench
../ojas/arch02/metal-native/scripts/cuda_ref_128m.sh quality
# then on Mac:
./scripts/ingest_cuda128m.sh /path/to/parameter-golf/logs/cuda128m
\`\`\`
EOF

echo "wrote $DEST/STATUS.md"
echo "DISCLOSE: GRAD_ACCUM_STEPS=$ACCUM"
ls -la "$DEST" | head -40
