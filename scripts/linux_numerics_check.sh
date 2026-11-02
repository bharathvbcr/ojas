#!/usr/bin/env bash
# Do the pinned bit-digests hold off macOS, and off aarch64?
#
# ojas-cpu/tests/softmax_bits.rs pins exact FNV-1a digests of f32 bit patterns
# for both `Numerics::Exact` and `Numerics::Fast` (:209 EXACT, :333 FAST). It
# carries no `cfg` gating of any kind, so it asserts one set of bits for every
# platform the workspace builds on. Those digests were produced on an
# Apple-silicon Mac; CI's Linux job runs x86-64 (`ubuntu-latest`).
#
# ANSWERED for amd64, 2026-11-01, and not by this script. Read this before
# spending a VM on it.
#
# The header used to say the question "has been *inferred* and never measured".
# That was wrong: CI's `linux (fmt, clippy, test)` job runs
# `cargo test --workspace --release` on ubuntu-24.04, which includes ojas-cpu,
# so it has measured this on every push for as long as the test has existed.
#
# The answer was not the predicted pass:
#
#   Exact digests on x86-64   hold
#   Fast digests on x86-64    DO NOT hold ("Fast softmax dim=7 amp=1 moved")
#
# and that one test was the sole reason main was red (six of seven jobs green).
#
# The cause is the ARCHITECTURE, not the OS or libm. CI moves both at once and
# cannot separate them; running the same test on both targets of one
# Apple-silicon machine can, because `x86_64-apple-darwin` runs under Rosetta
# with Apple's libm on both sides:
#
#   aarch64-apple-darwin   exact ok   fast ok
#   x86_64-apple-darwin    exact ok   fast FAILED   <- same case as CI
#
# Fast is not broken on x86-64: its worst departure from Exact is 8.94e-8
# (~1 f32 ulp), the same value measured on aarch64. It rounds the last bit
# differently, which is what `Numerics::Fast` is documented to be allowed to do
# (ojas-core/src/backend.rs:32-36: agreement is "a stated tolerance rather than
# bitwise"). Fixed in 32e4642 by pinning the Fast bits per architecture and
# adding the tolerance check the softmax path had never had.
#
# WHAT IS STILL WORTH RUNNING HERE: linux/arm64 only — same architecture as the
# recorded digests, different OS and libm, so it isolates "did we pin Apple's
# libm?" Nothing has measured that. linux/amd64 adds nothing now; CI covers it,
# and the architecture question is settled above.
#
# Invoke: ./scripts/linux_numerics_check.sh [--arch arm64|amd64|both]
#
# Reuses linux_device_check.sh's container layout: the three sibling checkouts
# are mounted at the relative paths the manifests expect, because
# `cargo test -p ojas-cpu` still resolves the whole workspace, and ojas-metal
# path-depends on tessl even though it is a stub off macOS.

set -euo pipefail

ROOT_LOGICAL="$(dirname "$0")/.."
ROOT="$(cd "$ROOT_LOGICAL" && pwd -P)"
TESSL="$(cd "${OJAS_TESSL:-$ROOT_LOGICAL/../tessl}" && pwd -P)"
GUSSET="$(cd "${OJAS_GUSSET:-$ROOT_LOGICAL/../../devtools/gusset}" && pwd -P)"
IMAGE="${OJAS_LINUX_IMAGE:-docker.io/library/rust:1-bookworm}"
PULL="${OJAS_LINUX_PULL:-missing}"
# A cold Rust build in a container, and under amd64 emulation a slow one. The
# same ceiling mac_heavy.sh uses, for the same reason: on 2026-10-17 parallel
# builds from several agent worktrees starved syspolicyd until the WindowServer
# watchdog killed the GUI session.
LOAD_CAP="${OJAS_LINUX_LOAD_CAP:-12}"

ARCHES="both"
[ "${1:-}" = "--arch" ] && { ARCHES="${2:?--arch needs a value}"; }
case "$ARCHES" in
  arm64|amd64) ;;
  both) ARCHES="arm64 amd64" ;;
  *) echo "ERROR: --arch takes arm64, amd64 or both; got '$ARCHES'" >&2; exit 2 ;;
esac

command -v podman >/dev/null || {
  echo "ERROR: podman is not installed (brew install podman); this gate cannot be skipped" >&2
  exit 1
}

# Fail closed and say which, rather than letting `podman run` fail later with a
# connection error that reads like a container problem.
if ! podman info --format '{{.Host.Arch}}' >/dev/null 2>&1; then
  echo "ERROR: podman is installed but its machine is not reachable." >&2
  echo "       Start it with:  podman machine start" >&2
  echo "       On an Apple-silicon host that machine is a 16 GiB VM, so start it" >&2
  echo "       when the machine is otherwise quiet — see the load check below." >&2
  exit 1
fi

load1="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}' || echo 0)"
if awk -v l="$load1" -v c="$LOAD_CAP" 'BEGIN{exit !(l>c)}'; then
  echo "ERROR: 1-minute load is $load1, above the cap of $LOAD_CAP." >&2
  echo "       This runs a cold Rust build in a container (slower still under" >&2
  echo "       amd64 emulation). Wait for a quiet machine, or raise the cap" >&2
  echo "       deliberately with OJAS_LINUX_LOAD_CAP=<n> if you know why." >&2
  exit 1
fi

fail=0
for arch in $ARCHES; do
  target="$ROOT/target/linux-numerics-$arch"
  mkdir -p "$target"
  echo "==> linux/$arch: pinned softmax and attention bit digests"
  # --arch is honoured by podman via qemu-user for the non-native one. The
  # digests are what is under test, so the test is never allowed to be skipped
  # or filtered: it runs by name and its exit status is the answer.
  if podman run --rm --pull="$PULL" --arch "$arch" \
      -v "$ROOT:/work/code/research/ojas:ro" \
      -v "$TESSL:/work/code/research/tessl:ro" \
      -v "$GUSSET:/work/code/devtools/gusset:ro" \
      -v "$target:/work/target" \
      -v "ojas-linux-cargo-$arch:/usr/local/cargo/registry" \
      -e CARGO_TARGET_DIR=/work/target \
      -e CARGO_TERM_COLOR=never \
      -w /work/code/research/ojas \
      "$IMAGE" bash -c '
        set -euo pipefail
        uname -srm
        cargo --version
        # Both tests, not just Fast: if Exact moved too, the cause is the
        # softmax reduction order rather than a fast-math approximation, and
        # that is a different bug with a different fix.
        cargo test --locked --release -p ojas-cpu --test softmax_bits -- --test-threads=1
      '; then
    echo "    linux/$arch: digests HOLD"
  else
    echo "    linux/$arch: digests MOVED (or the run failed; read the output above)" >&2
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "At least one platform disagrees with the pinned digests." >&2
  echo "Do NOT re-pin to whatever this host produced: that would make the test" >&2
  echo "assert the platform it last ran on and silently stop being a parity" >&2
  echo "gate. softmax_bits.rs:7 says these digests are the oracle the Metal" >&2
  echo "parity tests compare against, so a per-platform digest set needs the" >&2
  echo "Metal side considered in the same change." >&2
  exit 1
fi

echo "==> digests hold on every platform checked ($ARCHES)"
