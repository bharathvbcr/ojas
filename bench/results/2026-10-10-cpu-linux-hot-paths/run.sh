#!/usr/bin/env bash
# Interleaved A/B of the 2026-10-10 CPU hot-path changes on Linux x86_64:
# the tree before them (edeaf73, with this tree's bench_ops.rs copied in so
# both run the same cases) against this tree. No torch lane.
#
#   bash run.sh BASE_BENCH_OPS AFTER_BENCH_OPS ROUNDS OUT_DIR
#
# Each argument is the release test binary of
# `cargo test -p ojas-cpu --release --test bench_ops --no-run` built in that
# tree. Odd rounds run base then after, even rounds after then base, each
# lane at THREADS (default 4,1). /proc/loadavg is recorded before every lane.
# Then both binaries dump every case at the first thread count and the
# outputs are compared byte for byte (bits.txt).
set -euo pipefail
base=$1 after=$2 rounds=$3 out=$4
mkdir -p "$out"
OPS=${OPS:-linear_dec_qkvo,linear_dec_up,linear_dec_down,linear_ce,linear_ce_whole_vocab,cached_attn_dec,cached_attn_dec_short,add,mul,embedding,gdn_decay}
THREADS=${THREADS:-4,1}

lane() {
    local round=$1 name=$2 bin=$3
    echo "round=$round lane=$name loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$out/load.txt"
    OJAS_BENCH_MODE=time OJAS_BENCH_OPS="$OPS" OJAS_BENCH_THREADS="$THREADS" \
        "$bin" --ignored --nocapture --test-threads=1 2>&1 \
        | grep -oE 'OJAS_(OP|META) .*' > "$out/ops-$round-$name.txt"
}

for round in $(seq 1 "$rounds"); do
    if [ $((round % 2)) = 1 ]; then
        lane "$round" base "$base"
        lane "$round" after "$after"
    else
        lane "$round" after "$after"
        lane "$round" base "$base"
    fi
done

# One case at a time, so at most one case's inputs and outputs are on disk.
first=${THREADS%%,*}
echo "bench_ops dump at $first threads, Fast: base against after" > "$out/bits.txt"
total=0 differ=0
for op in ${OPS//,/ }; do
    for name in base after; do
        bin=$base
        [ "$name" = after ] && bin=$after
        rm -rf "$out/dump-$name"
        OJAS_BENCH_MODE=dump OJAS_BENCH_DIR="$out/dump-$name" OJAS_BENCH_OPS="$op" OJAS_BENCH_THREADS="$first" \
            "$bin" --ignored --nocapture --test-threads=1 > /dev/null 2>&1
    done
    while IFS= read -r f; do
        total=$((total + 1))
        if ! cmp -s "$out/dump-base/$f" "$out/dump-after/$f"; then
            differ=$((differ + 1))
            echo "DIFFERS $f" >> "$out/bits.txt"
        fi
    done < <(cd "$out/dump-after" && find . -type f -name '*.bin' | sort)
    rm -rf "$out/dump-base" "$out/dump-after"
done
echo "$total files, $differ differ" >> "$out/bits.txt"
cat "$out/bits.txt"
