#!/bin/bash
# The driver of the clip-scale-share measurement (README.md here; task ft-a306b417).
#
#   bash bench/results/2026-11-08-clip-scale-share/run.sh <out dir>
#
# Run it under the machine lock (tools/mac_heavy.sh in Lappi-decision): it is one GPU job for its
# whole length, so nothing else uses the GPU between parts. Three parts, each its own process,
# appended as they finish so a killed run keeps what it measured:
#
# 1. trainer: `nanolab_config_ab clip` at B=1 and B=8 (T=1024): three nanolab-124M trainers in one
#    process (forced clip, no scale, no-scale twin), steps interleaved in a rotated order.
#    TRAINER_REPS processes per batch size, alternating which batch size goes first.
# 2. rows: `metal_vs_torch` with BENCH_ROWS=clip_: `clip_grad_norm_full` (norm + scale) and
#    `clip_norm_only` (norm alone) over the same 170 nanolab gradients, ROWS_REPS processes.
#    The torch side is not timed; its reference directory supplies the generator check and each
#    row's parity check, and is written first when it lacks the clip specs.
# 3. provider: `gpu_memory_rows` arm `tape-provider-clip` on Qwen3.5-2B-Base, T=2048, rank 16
#    LoRA: 10 rounds of a step at grad_scale 0.5 and one at 1, alternating, one process.
#
# The binaries are release builds made before the run:
#   cargo build -j 2 --release -p ojas-model --example nanolab_config_ab
#   cargo build -j 2 --release -p ojas-metal --features metal --example metal_vs_torch
#   cargo test -j 2 -p ojas-qwen35 --release --test gpu_memory_rows --no-run
set -u

OUT=$1
TRAINER_REPS=${TRAINER_REPS:-3}
ROWS_REPS=${ROWS_REPS:-3}
B1_ROUNDS=${B1_ROUNDS:-30}
B8_ROUNDS=${B8_ROUNDS:-12}
REF=${REF:-bench/out/2026-11-09-clipnorm/ref}
PY=${PY:-/Users/bharath/.venvs/ml/bin/python}

TRAINER=target/release/examples/nanolab_config_ab
ROWS=target/release/examples/metal_vs_torch
PROVIDER=$(ls -t target/release/deps/gpu_memory_rows-* | grep -v '\.d$' | head -1)

mkdir -p "$OUT/stderr"
RUNS="$OUT/runs.tsv"
[ -f "$RUNS" ] || printf 'part\ttag\tstart_utc\tstatus\twall_s\tpeak_footprint_bytes\tload_before\tload_after\n' >"$RUNS"

{
  echo "date: $(date -u +%FT%TZ)"
  echo "git: $(git rev-parse HEAD) (dirty files: $(git status --porcelain | wc -l | awk '{print $1}'))"
  echo "nanolab_config_ab.rs sha1: $(shasum ojas-model/examples/nanolab_config_ab.rs | awk '{print $1}')"
  echo "gpu_memory_rows.rs sha1: $(shasum ojas-qwen35/tests/gpu_memory_rows.rs | awk '{print $1}')"
  echo "ojas_rows.rs sha1: $(shasum bench/ojas_rows.rs | awk '{print $1}')"
  echo "tessl: $(git -C ../tessl rev-parse HEAD 2>&1) (dirty files: $(git -C ../tessl status --porcelain 2>&1 | wc -l | awk '{print $1}'))"
  echo "rustc: $(rustc --version)"
  echo "os: $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "cpu: $(sysctl -n machdep.cpu.brand_string)"
  echo "memory_bytes: $(sysctl -n hw.memsize)"
  for b in "$TRAINER" "$ROWS" "$PROVIDER"; do echo "binary: $b $(stat -f '%Sm %z' "$b")"; done
  echo "trainer_reps: $TRAINER_REPS b1_rounds: $B1_ROUNDS b8_rounds: $B8_ROUNDS rows_reps: $ROWS_REPS ref: $REF"
  echo "mac_heavy_label: ${MAC_HEAVY_LABEL:-unset (not under the machine lock)}"
} >>"$OUT/env.txt"

load() { sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1, $2, $3}'; }

# part tag stdout-file command...
run_one() {
  local part=$1 tag=$2 dest=$3
  shift 3
  local err="$OUT/stderr/$tag.txt"
  local start lb la t0 t1 status peak
  start=$(date -u +%FT%TZ)
  lb=$(load)
  t0=$(date +%s)
  /usr/bin/time -l "$@" >>"$dest" 2>"$err"
  status=$?
  t1=$(date +%s)
  la=$(load)
  peak=$(awk '/peak memory footprint/ {print $1}' "$err")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$part" "$tag" "$start" "$status" "$((t1 - t0))" \
    "${peak:-?}" "$lb" "$la" >>"$RUNS"
  echo "$(date -u +%TZ) $tag status $status $((t1 - t0))s"
}

for ((rep = 1; rep <= TRAINER_REPS; rep++)); do
  if ((rep % 2)); then order="1 8"; else order="8 1"; fi
  for b in $order; do
    if [ "$b" = 1 ]; then rounds=$B1_ROUNDS; else rounds=$B8_ROUNDS; fi
    run_one trainer "trainer-b$b-r$rep" "$OUT/trainer.jsonl" "$TRAINER" clip "$b" "$rounds" 1024
  done
done

# The rows check their outputs against torch's reference. bench/out is gitignored, and the older
# reference directories (run5c and before) predate clip_norm_only, so write both clip specs here
# when they are missing (bench/torch_rows.py ref; MPS, so this too runs under the lock).
if [ ! -f "$REF/clip_norm_only.spec" ] || [ ! -f "$REF/clip_grad_norm_full.spec" ]; then
  (
    export BENCH_ROWS=clip_
    run_one rows "rows-ref" /dev/null "$PY" bench/torch_rows.py ref --ref "$REF"
  )
fi

for ((rep = 1; rep <= ROWS_REPS; rep++)); do
  (
    export OJAS_BENCH_REF=$REF OJAS_BENCH_OUT="$OUT/clip_rows.jsonl" BENCH_ROWS=clip_
    run_one rows "rows-r$rep" /dev/null "$ROWS"
  )
done

(
  export QWEN35_MEM_ARM=tape-provider-clip QWEN35_MEM_TOKENS=2048
  run_one provider provider-t2048 "$OUT/provider.txt" \
    "$PROVIDER" --ignored --exact --test-threads=1 --nocapture gpu_memory_row
)
grep '^ROW ' "$OUT/provider.txt" | sed 's/^ROW //' >"$OUT/provider.jsonl"
echo "done: $OUT"
