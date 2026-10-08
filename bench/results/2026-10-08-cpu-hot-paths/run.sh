#!/usr/bin/env bash
# Interleaved A/B of the CPU and tape hot paths: the tree before them
# (86a1096, with this tree's bench_ops.rs and tape_bench.rs copied in so both
# run the same cases), the tree after, and torch 2.13 CPU on the nanolab rows.
#
#   bash run.sh BASE_BIN_DIR AFTER_BIN_DIR ROUNDS OUT_DIR
#
# Each *_BIN_DIR holds `bench_ops` and `tape_bench`, the release test binaries
# of `cargo test -p ojas-cpu --release --test bench_ops --no-run` and
# `cargo test -p ojas-autograd --release --test tape_bench --no-run` built in
# that tree. Odd rounds run base, after, torch; even rounds torch, after,
# base. `vm.loadavg` is recorded before every lane. Every lane is 6 threads.
# TAPE_ONLY=1 runs only the tape_bench lanes (base and after).
set -euo pipefail
base=$1 after=$2 rounds=$3 out=$4
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
mkdir -p "$out"
OPS=${OPS:-accum_768x768,accum_50304x768,embedding,embedding_qwen,permute,permute_qwen,muon_768x768,muon_2048x768,muon_3072x768,muon_2048x2048,muon_6144x2048,conv1d,gated_rms,rope_partial,gdn,mul,add,silu,adamw_50304x768}
TORCH_OPS=${TORCH_OPS:-embedding,permute,muon_2048x768,mul,add,silu,adamw_50304x768}
DUMP=${DUMP:-$out/dump}
python=${PYTHON:-python3}

# Inputs for the torch lane come from the after tree's dump, and its outputs
# are the parity reference.
if [ ! -d "$DUMP" ]; then
    OJAS_BENCH_MODE=dump OJAS_BENCH_DIR="$DUMP" OJAS_BENCH_OPS="$TORCH_OPS" OJAS_BENCH_THREADS=6 \
        "$after/bench_ops" --ignored --nocapture --test-threads=1 > "$out/dump.txt" 2>&1
fi
"$python" "$repo/ojas-cpu/benches/torch_ops.py" --threads 6 --dir "$DUMP" --ops "$TORCH_OPS" --parity \
    > "$out/parity.txt" 2>&1

lane() {
    local round=$1 name=$2
    echo "round=$round lane=$name loadavg=$(sysctl -n vm.loadavg)" >> "$out/load.txt"
    case $name in
        base|after)
            # libtest writes `test name ... ` before a test's first line of
            # output, so each marker is matched anywhere on its line.
            local dir=$base
            [ "$name" = after ] && dir=$after
            [ -n "${TAPE_ONLY:-}" ] || OJAS_BENCH_MODE=time OJAS_BENCH_OPS="$OPS" OJAS_BENCH_THREADS=6 \
                "$dir/bench_ops" --ignored --nocapture --test-threads=1 2>&1 \
                | grep -oE 'OJAS_(OP|META) .*' > "$out/ops-$round-$name.txt"
            "$dir/tape_bench" --ignored --nocapture --test-threads=1 2>&1 \
                | grep -oE 'OJAS_TAPE .*' > "$out/tape-$round-$name.txt"
            ;;
        torch)
            "$python" "$repo/ojas-cpu/benches/torch_ops.py" --threads 6 --dir "$DUMP" --ops "$TORCH_OPS" --time \
                2>&1 | grep -oE 'TORCH_(OP|META) .*' > "$out/ops-$round-torch.txt"
            ;;
    esac
}

for round in $(seq 1 "$rounds"); do
    if [ $((round % 2)) -eq 1 ]; then order="base after torch"; else order="torch after base"; fi
    for name in $order; do
        [ -n "${TAPE_ONLY:-}" ] && [ "$name" = torch ] && continue
        lane "$round" "$name"
    done
done
echo "done rounds=$rounds out=$out" >> "$out/load.txt"
