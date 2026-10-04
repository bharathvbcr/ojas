#!/bin/bash
# Short slot at -j 2: ojas-metal clippy alone, with the staged bias sum.
set -uo pipefail
dir=/Users/bharath/Code/research/ojas/target-baseline/gate-ab
uptime
cargo clippy --release -j 2 --all-targets --features metal -p ojas-metal \
  --manifest-path /Users/bharath/Code/research/ojas/Cargo.toml \
  --target-dir /Users/bharath/Code/research/ojas/target-lane-bench -- -D warnings > "$dir/stage-clippy-2.log" 2>&1
echo "clippy exit=$?"
grep -E '^(error|warning)|-->' "$dir/stage-clippy-2.log" | head -20
uptime
