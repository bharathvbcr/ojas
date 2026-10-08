#!/bin/bash
# GPU step-throughput A/B (task ft-2c183b7103c95088ae62ea0da43aebc3): the
# paired-bench rows on an old and a new ojas tree, Metal and wgpu, with the
# torch-mps lane in the same rounds. Run under mac_heavy.sh.
#
#   OLD_REV=<rev> NEW_REV=<rev> ROUNDS=5 bash run_ab.sh
#
# Both trees are `git archive` exports into target-gpu-step-ab/ab/{old,new}
# (gitignored), each built --release with its own CARGO_TARGET_DIR against
# the one tessl checkout (symlinked as ab/tessl, the relative path the
# manifests name). ojas-capi and ojas-gusset-engine leave the exported
# member lists: their gusset path does not resolve there and the bench uses
# neither. Both trees run the NEW tree's bench/ojas_rows.rs, so the row code
# is identical and only the library differs (the old tree gets the same
# file copied in; it uses only the Backend trait both trees share).
#
# Each round runs old-metal, new-metal, old-wgpu, new-wgpu and torch-mps;
# odd rounds in that order, even rounds reversed. One JSONL per lane and
# round, as run_paired.sh writes; `agg.py` reads them.
set -u
REPO=/Users/bharath/Code/research/ojas
D=$REPO/target-gpu-step-ab
OLD_REV=${OLD_REV:-f3f8988}
NEW_REV=${NEW_REV:-HEAD}
ROUNDS=${ROUNDS:-5}
OUT=${OUT:-$D/out}
PY=${PYTHON:-/opt/homebrew/opt/python@3.14/bin/python3.14}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2} PYTHONDONTWRITEBYTECODE=1 RIPGREP_CONFIG_PATH=
export BENCH_ITERS=${BENCH_ITERS:-20} BENCH_WARMUP=${BENCH_WARMUP:-5}
export BENCH_ROWS=${BENCH_ROWS:-floor_silu_1,sweep_silu_n1,rms_norm_fwd,rms_qk_norm_fwd,rope_fwd,vres_fwd,residual_add,permute_bthd_bhtd,silu_fwd,mul_fwd,gate_,cross_entropy_bwd,embed_bwd,linear_qkv_fwd,linear_up_fwd,linear_down_fwd,linear_ce_,clip_grad_norm_full,adamw_full,muon_768x768,accumulate_grad,sdpa_,block_fwd,decode_attn_kv1024}

mkdir -p "$D/ab" "$OUT"
[ -e "$D/ab/tessl" ] || ln -s /Users/bharath/Code/research/tessl "$D/ab/tessl"
for t in old new; do
    rev=$OLD_REV; [ $t = new ] && rev=$NEW_REV
    rm -rf "$D/ab/$t"; mkdir -p "$D/ab/$t"
    git -C "$REPO" archive "$rev" | tar -x -C "$D/ab/$t"
    sed -i '' -e '/"ojas-capi",/d' -e '/"ojas-gusset-engine",/d' "$D/ab/$t/Cargo.toml"
    echo "$t: $(git -C "$REPO" rev-parse "$rev")" >> "$OUT/env.txt"
done
cp "$D/ab/new/bench/ojas_rows.rs" "$D/ab/old/bench/ojas_rows.rs"
for t in old new; do
    echo "build $t" >&2
    CARGO_TARGET_DIR=$D/target-$t cargo build --release --manifest-path "$D/ab/$t/Cargo.toml" \
        -p ojas-metal -p ojas-wgpu --example metal_vs_torch --example wgpu_vs_torch \
        > "$OUT/build-$t.log" 2>&1 || { echo "build $t failed"; tail -30 "$OUT/build-$t.log"; exit 1; }
done
{
    echo "tessl: $(git -C /Users/bharath/Code/research/tessl rev-parse HEAD) (dirty files: $(git -C /Users/bharath/Code/research/tessl status --porcelain | wc -l | tr -d ' '))"
    echo "rustc: $(rustc -V)"
    echo "os: $(sw_vers -productVersion) cpu: $(sysctl -n machdep.cpu.brand_string)"
    echo "rounds: $ROUNDS iters: $BENCH_ITERS warmup: $BENCH_WARMUP rows: $BENCH_ROWS"
} >> "$OUT/env.txt"
"$PY" "$D/ab/new/bench/torch_rows.py" env --out "$OUT/env_torch.json" >&2
"$PY" "$D/ab/new/bench/torch_rows.py" ref --ref "$OUT/ref" 2> "$OUT/ref.log" || { echo "torch ref failed"; tail "$OUT/ref.log"; exit 1; }

lane() { # round lane
    local f="$OUT/round$1/$2.jsonl"
    printf '{"round":%s,"lane":"%s","when":"before","time":"%s","load":"%s","gpu_util_pct":%s}\n' "$1" "$2" \
        "$(date +%H:%M:%S)" "$(uptime | sed 's/.*load averages*: //')" \
        "$(ioreg -r -d 1 -w 0 -c IOAccelerator | rg -o '"Device Utilization %"=[0-9]+' | head -1 | rg -o '[0-9]+$' || echo null)" \
        >> "$OUT/load.jsonl"
    export OJAS_BENCH_REF="$OUT/ref" OJAS_BENCH_OUT="$f"
    case "$2" in
        old-metal) "$D/target-old/release/examples/metal_vs_torch" ;;
        new-metal) "$D/target-new/release/examples/metal_vs_torch" ;;
        old-wgpu) "$D/target-old/release/examples/wgpu_vs_torch" ;;
        new-wgpu) "$D/target-new/release/examples/wgpu_vs_torch" ;;
        torch-mps) "$PY" "$D/ab/new/bench/torch_rows.py" time --out "$f" ;;
    esac 2>> "$OUT/round$1/$2.log" || printf '{"row":"_process","status":"crash"}\n' >> "$f"
}

LANES="old-metal new-metal old-wgpu new-wgpu torch-mps"
for r in $(seq 1 "$ROUNDS"); do
    mkdir -p "$OUT/round$r"
    order=$LANES
    [ $((r % 2)) -eq 0 ] && order=$(printf '%s\n' $LANES | tail -r | tr '\n' ' ')
    for l in $order; do lane "$r" "$l"; done
    echo "round $r/$ROUNDS done ($order)" >&2
done
echo "done: $OUT" >&2
