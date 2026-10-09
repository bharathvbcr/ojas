#!/usr/bin/env bash
# Interleaved A/B of bpe_throughput binaries (bench/bpe_throughput.rs).
#
#   bash run.sh OUT_DIR ROUNDS label=BINARY [label=BINARY ...]
#
# Round r runs every lane once, in the listed order on odd rounds and reversed
# on even rounds. Each process times its own rows (5 inner rounds, min and
# median). `uptime` is recorded before and after every process in load.txt.
# VOCAB_DIR and CORPUS_BIN default to the GPT-2 table and nanolab's
# FineWeb-Edu val.bin on the machine that made this run.
set -euo pipefail
out=$1; rounds=$2; shift 2
VOCAB_DIR=${VOCAB_DIR:-/Users/bharath/Code/research/LocalModelBench/inference/audio/Step-Audio-EditX/funasr_detach/models/whisper/utils/assets/gpt2}
CORPUS_BIN=${CORPUS_BIN:-/Users/bharath/Code/research/MLSystemsLab/nanolab/data/HuggingFaceFW_fineweb-edu/val.bin}
INNER=${INNER:-5}
mkdir -p "$out"
lanes=("$@")
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  sw_vers 2>/dev/null || true
  rustc --version
  shasum -a 256 "$VOCAB_DIR/vocab.json" "$VOCAB_DIR/merges.txt" "$CORPUS_BIN"
  for lane in "${lanes[@]}"; do echo "lane ${lane%%=*}: ${lane#*=}"; done
} > "$out/env.txt"
: > "$out/load.txt"
for ((r = 1; r <= rounds; r++)); do
  order=("${lanes[@]}")
  if ((r % 2 == 0)); then
    order=()
    for ((i = ${#lanes[@]} - 1; i >= 0; i--)); do order+=("${lanes[$i]}"); done
  fi
  for lane in "${order[@]}"; do
    label=${lane%%=*}; bin=${lane#*=}
    echo "round $r $label before: $(uptime)" >> "$out/load.txt"
    "$bin" "$VOCAB_DIR" "$CORPUS_BIN" "$INNER" > "$out/r${r}_${label}.txt"
    echo "round $r $label after:  $(uptime)" >> "$out/load.txt"
  done
done
