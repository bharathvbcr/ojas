#!/usr/bin/env bash
# Local gates for this tree, mirroring .github/workflows/test.yml (Linux and
# Go jobs) so a push is not the first place a gate runs.
# Invoke: ./scripts/ci_local.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> fmt"
./scripts/fmt_check.sh

echo "==> cargo target sources are tracked"
./scripts/check_tracked_targets.sh

echo "==> clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> cuda check"
# cudarc loads the driver dynamically. This host can compile the feature
# without a CUDA toolkit. It does not execute a kernel. The lib-only lints
# check ojas-cuda's default surface: any target that builds its tests turns
# on `harness` (its dev-dependency on itself).
cargo clippy -p ojas-cuda --features cuda --all-targets -- -D warnings
cargo clippy -p ojas-cuda --features cuda -- -D warnings
cargo clippy -p ojas-cuda -- -D warnings

echo "==> nightly simd (not a Cargo target)"
./scripts/simd_bench.sh

echo "==> go (stable)"
if ! command -v go >/dev/null 2>&1; then
  echo "ERROR: go is not installed; the stable Go gate cannot be skipped" >&2
  exit 1
fi
go version
echo "==> site (reference pages and docs mirror must match the markdown)"
go run scripts/sitegen/main.go -check
echo "==> bench charts (site/assets/plots/bench and docs/bench-plots.md must match bench/results)"
go test -count=1 scripts/benchreport/main.go scripts/benchreport/main_test.go
go run scripts/benchreport/main.go plots -check
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

echo "==> rustdoc (intra-doc links resolve; no link to a private item)"
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

echo "==> ojas-metal without the metal feature (stub build, every target)"
cargo clippy -p ojas-metal --no-default-features --all-targets -- -D warnings

echo "ci:local OK"
