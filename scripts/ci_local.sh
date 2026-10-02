#!/usr/bin/env bash
# Local gates for this tree. .github/workflows/test.yml exists but has never
# run, because the repository has no remote.
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

echo "==> nightly simd (not a Cargo target)"
./scripts/simd_bench.sh

echo "==> go (stable)"
if ! command -v go >/dev/null 2>&1; then
  echo "ERROR: go is not installed; the stable Go gate cannot be skipped" >&2
  exit 1
fi
go version
cargo build -p ojas-gusset-engine
(
  cd "$ROOT/go"
  PKG_CONFIG_PATH="$PWD" go test -a -tags gusset_pkgconfig -count=1 -timeout 10m ./...
)

echo "==> gotip (optional)"
./scripts/gotip.sh

echo "==> test (release, serialized)"
cargo test --workspace --release -- --test-threads=1

# HIP is not compile-checked here. hip-runtime-sys 0.1.2 panics in its
# build script when hip/hip_runtime_api.h is absent, which it is on this Mac.

echo "ci:local OK"
