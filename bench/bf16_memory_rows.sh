#!/usr/bin/env bash
# The bf16 tier's micro-batch memory grid (gp-bf16-compute-tier criterion 4):
# one `bf16_memory_rows` process per (backend, precision, batch), each under
# `/usr/bin/time -l` for its own page faults, swaps and peak footprint, with
# the machine's vm_stat pageout and swapout counters read before and after.
#
#   bash bench/bf16_memory_rows.sh <out.jsonl> [backends] [batches] [seq] [steps] [checkpoint]
#
# Defaults: backends "cpu wgpu metal", batches "1 8", seq 1024, steps 3,
# checkpoint off. Run it under Lappi-decision/tools/mac_heavy.sh. A row
# "fits without paging" when its peak charged bytes are within the budget,
# the process swapped nothing, and the machine's swapout counter did not
# move. The process's major page faults are kept as a column, not gated:
# on macOS they count file-backed page-ins too (a cold GPU process faults in
# its libraries and shader caches; 2026-10-10 rows). vm_stat pageouts are
# context, since other processes move them.
set -euo pipefail

OUT="${1:?usage: bf16_memory_rows.sh <out.jsonl> [backends] [batches] [seq] [steps] [checkpoint]}"
BACKENDS="${2:-cpu wgpu metal}"
BATCHES="${3:-1 8}"
SEQ="${4:-1024}"
STEPS="${5:-3}"
CKPT="${6:-off}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${OJAS_MANIFEST:-$ROOT/Cargo.toml}"

cargo build -q --release --manifest-path "$MANIFEST" -p ojas-model --example bf16_memory_rows
BIN="$(cargo metadata -q --format-version 1 --manifest-path "$MANIFEST" --no-deps \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/examples/bf16_memory_rows"

vm() { vm_stat | awk -v k="$1" -F: '$1 == k { gsub(/[ .]/, "", $2); print $2 }'; }
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ojas-bf16-rows.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

for backend in $BACKENDS; do
  for batch in $BATCHES; do
    for precision in f32 bf16; do
      po0=$(vm Pageouts); so0=$(vm Swapouts)
      status=0
      /usr/bin/time -l "$BIN" "$backend" "$precision" "$batch" "$SEQ" "$STEPS" "$CKPT" \
          >"$WORK/row.json" 2>"$WORK/time.txt" || status=$?
      po1=$(vm Pageouts); so1=$(vm Swapouts)
      faults=$(awk '/page faults/ { print $1 }' "$WORK/time.txt")
      swaps=$(awk '/ swaps/ { print $1 }' "$WORK/time.txt")
      rss=$(awk '/maximum resident set size/ { print $1 }' "$WORK/time.txt")
      foot=$(awk '/peak memory footprint/ { print $1 }' "$WORK/time.txt")
      if [ "$status" -ne 0 ]; then
        err=$(grep -v '^ ' "$WORK/time.txt" | head -3 | tr '"\n' "' " )
        printf '{"backend":"%s","precision":"%s","batch":%s,"seq":%s,"exit":%s,"error":"%s"}\n' \
            "$backend" "$precision" "$batch" "$SEQ" "$status" "$err" >>"$OUT"
        continue
      fi
      python3 - "$WORK/row.json" "$faults" "$swaps" "$rss" "$foot" "$((po1 - po0))" "$((so1 - so0))" >>"$OUT" <<'PY'
import json, sys
row = json.loads(open(sys.argv[1]).read())
row.update(process_page_faults=int(sys.argv[2]), process_swaps=int(sys.argv[3]),
           max_rss_bytes=int(sys.argv[4]), peak_footprint_bytes=int(sys.argv[5]),
           machine_pageouts_delta=int(sys.argv[6]), machine_swapouts_delta=int(sys.argv[7]))
row["fits_without_paging"] = (row["peak_charged_bytes"] <= row["budget_limit_bytes"]
                              and row["process_swaps"] == 0
                              and row["machine_swapouts_delta"] == 0)
print(json.dumps(row, sort_keys=True))
PY
      tail -1 "$OUT"
    done
  done
done
