#!/usr/bin/env bash
# Compile and run scripts/simd_bench.rs with nightly. It is not part of the
# Cargo workspace, so the stable MSRV and `cargo clippy --all-targets` never
# see `std::simd`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if ! rustup toolchain list | grep -q '^nightly'; then
  echo "nightly toolchain is not installed; simd gate skipped" >&2
  exit 1
fi

OUT="$ROOT/target/simd_bench"
mkdir -p "$ROOT/target"
rustup run nightly rustc --edition 2021 "$ROOT/scripts/simd_bench.rs" -o "$OUT"
"$OUT"
