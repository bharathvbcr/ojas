#!/bin/bash
# e15 CPU Muon A/B: base (42d5a9f, git archive in backups/e15-tools/base) against this branch,
# ojas-cpu/tests/bench_ops.rs muon rows, 6 threads, rounds alternating which binary goes first.
# Every cargo under mac_heavy (heavy.sh). usage (via launch.sh): muonab.sh <log-dir>
set -u
L="$1"
R=/Users/bharath/Code/research
W=$R/ojas/.gitpulse/worktrees/route-ojas-metal-fast-autocast-to-tessl-6223aeac
T=$W/backups/e15-tools
H="$T/heavy.sh"
A=$R/ojas-e15
B=$R/ojas-e15-base
made=""
for pair in "$A=$W" "$B=$T/base"; do
  link=${pair%%=*}; dest=${pair#*=}
  if [ ! -e "$link" ] && [ ! -L "$link" ]; then ln -s "$dest" "$link" && made="$made $link"; fi
  if [ "$(readlink "$link")" != "$dest" ]; then echo "alias $link is not ours; stopping"; exit 2; fi
done
echo "aliases made:$made"
export OJAS_BENCH_OPS=muon_768x768,muon_768x768_bf16,muon_2048x768,muon_2048x768_bf16,muon_3072x768,muon_3072x768_bf16
export OJAS_BENCH_THREADS=6
run() { # run <side> <round>
  local side="$1" r="$2" man tgt
  if [ "$side" = new ]; then man="$A/Cargo.toml"; tgt="$T/target"; else man="$B/Cargo.toml"; tgt="$T/target-base"; fi
  TGT="$tgt" bash "$H" "muon-$side-$r" 7200 "$L/$side-$r.log" \
    cargo test --manifest-path "$man" -p ojas-cpu --release --test bench_ops -- --ignored --nocapture --test-threads=1
  echo "$side round $r exit=$? load=$(sysctl -n vm.loadavg)"
}
for r in 1 2 3 4 5; do
  if [ $((r % 2)) = 1 ]; then run base "$r"; run new "$r"; else run new "$r"; run base "$r"; fi
done
grep "OJAS_OP" "$L"/new-*.log "$L"/base-*.log > "$L/ops.txt"
for link in $made; do rm "$link"; echo "removed $link"; done
