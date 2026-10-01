#!/usr/bin/env bash
# Local gates for this tree. There is no .github workflow and no remote.
# Invoke: ./scripts/ci_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> fmt"
cargo fmt --all --check

echo "==> clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> cuda check"
# cudarc loads the driver dynamically. This host can compile the feature
# without a CUDA toolkit. It does not execute a kernel.
cargo check -p ojas-cuda --features cuda

echo "==> test (release, serialized)"
cargo test --workspace --release -- --test-threads=1

# HIP is not compile-checked here. hip-runtime-sys 0.1.2 panics in its
# build script when hip/hip_runtime_api.h is absent, which it is on this Mac.

echo "ci:local OK"
