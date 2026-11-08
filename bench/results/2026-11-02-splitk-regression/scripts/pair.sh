#!/bin/bash
# pair.sh <rounds> <labelA> <labelB> [labelC ...]
# Runs the prebuilt probe binaries back to back, one process each, rotating
# the starting binary every round (A B, B A, ...). Each process: 3 warm-ups,
# 15 timed rounds. Prints the probe lines with the round number in front.
set -u
ROUNDS=$1
shift
BINS=/Users/bharath/Code/research/ojas/target-splitk/bins
n=$#
labels=("$@")
echo "# pair $(date -u +%Y-%m-%dT%H:%M:%SZ) rounds $ROUNDS labels ${labels[*]}; $(sysctl -n vm.loadavg)"
for r in $(seq 1 "$ROUNDS"); do
  for s in $(seq 0 $((n - 1))); do
    if [ $((r % 2)) -eq 1 ]; then i=$s; else i=$((n - 1 - s)); fi
    l=${labels[$i]}
    "$BINS/$l" 3 15 "$l" | sed "s/^/r$r\t/"
  done
done
echo "# pair done $(date -u +%Y-%m-%dT%H:%M:%SZ); $(sysctl -n vm.loadavg)"
