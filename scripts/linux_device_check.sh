#!/usr/bin/env bash
# The Linux device checks, on a real Linux kernel in a podman container:
#
#   1. The ojas-device suite: the sysfs, cgroup v2 and PSI readers.
#   2. The live probe there, through ojas-capi's ignored
#      `linux_runner_admission_follows_the_live_probe`: the architecture
#      `probe_system` reads from /sys/class/drm, pressure `Normal` from PSI in
#      an idle container, and an allocating opcode admitted by the engine's
#      dispatcher.
#   3. E_PRESSURE admission under a real stall. A second container splits
#      its cgroup in two: `hog`, whose memory.high sits far below what a
#      background allocator asks for, and `init`, which runs the test. The
#      kernel throttles the allocator and charges the stall to `hog` and to
#      every ancestor, so the container root's PSI rises while the test
#      itself is never throttled; the probe reads it as an ancestor of its
#      own cgroup. (A test inside the throttled cgroup stalled with it and
#      never printed; that reclaim took its code pages, as anonymous memory
#      has no swap, is the likely cause, not a traced one.)
#      The allocator is bounded: each pass, `tail` holds OJAS_PRESSURE_HOG
#      bytes of /dev/zero (it has no newline) and then lets them go, under
#      the container's 1 GiB memory.max. The same test, expecting
#      `critical`, waits for the probe to read Critical (at most 60 s) and
#      then requires NEW to be refused with ojas:E_PRESSURE while FREE still
#      runs; it gets at most 300 s in all. Splitting the cgroup needs it
#      writable, which podman gives with --systemd=always.
#
# Any step that cannot run fails the script; nothing is skipped.
#
# The tree is mounted read-only beside its siblings, in the layout
# .github/workflows/test.yml uses (ojas, ../tessl, ../../devtools/gusset),
# and built with --locked. Build output goes to target/linux-device-check
# and the cargo registry to the named volume ojas-linux-cargo.
#
# Invoke from a checkout whose parent holds tessl and whose grandparent
# holds devtools/gusset, or name them:
#
#   ./scripts/linux_device_check.sh
#
# Environment:
#   OJAS_LINUX_IMAGE  image with cargo, bash and coreutils (default
#                     docker.io/library/rust:1-bookworm; the runs recorded in
#                     docs/adaptive-resources.md used a local image instead,
#                     so the default has not been run)
#   OJAS_LINUX_PULL   pull policy (default missing; never on an offline host)
#   OJAS_TESSL        tessl checkout (default ../tessl)
#   OJAS_GUSSET       gusset checkout (default ../../devtools/gusset)
#   OJAS_PRESSURE_HIGH  memory.high of the allocator's cgroup, bytes (default 128 MiB)
#   OJAS_PRESSURE_HOG   what the allocator holds per pass, bytes (default 512 MiB;
#                       above memory.high, under the 1 GiB memory.max)
# The container scripts are single-quoted on purpose: they expand inside.
# shellcheck disable=SC2016
set -euo pipefail

ROOT_LOGICAL="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$ROOT_LOGICAL" && pwd -P)"
TESSL="$(cd "${OJAS_TESSL:-$ROOT_LOGICAL/../tessl}" && pwd -P)"
GUSSET="$(cd "${OJAS_GUSSET:-$ROOT_LOGICAL/../../devtools/gusset}" && pwd -P)"
IMAGE="${OJAS_LINUX_IMAGE:-docker.io/library/rust:1-bookworm}"
PULL="${OJAS_LINUX_PULL:-missing}"
HIGH="${OJAS_PRESSURE_HIGH:-134217728}"
HOG="${OJAS_PRESSURE_HOG:-536870912}"
MAX=1073741824
TARGET="$ROOT/target/linux-device-check"
TEST=tests::linux_runner_admission_follows_the_live_probe

command -v podman >/dev/null || { echo "ERROR: podman is not installed" >&2; exit 1; }
for n in "$HIGH" "$HOG"; do
  [[ "$n" =~ ^[0-9]+$ ]] || { echo "ERROR: not a byte count: $n" >&2; exit 1; }
done
if (( HOG <= HIGH || HOG + (128 << 20) > MAX )); then
  echo "ERROR: the allocator's $HOG bytes must exceed memory.high ($HIGH) and leave 128 MiB under memory.max ($MAX)" >&2
  exit 1
fi
mkdir -p "$TARGET"

# One container per phase, on the same image, mounts and target.
run() {
  podman run --rm --pull="$PULL" \
    -v "$ROOT:/work/code/research/ojas:ro" \
    -v "$TESSL:/work/code/research/tessl:ro" \
    -v "$GUSSET:/work/code/devtools/gusset:ro" \
    -v "$TARGET:/work/target" \
    -v ojas-linux-cargo:/usr/local/cargo/registry \
    -e CARGO_TARGET_DIR=/work/target \
    -e CARGO_TERM_COLOR=never \
    -w /work/code/research/ojas \
    "$@"
}

echo "==> image $IMAGE; kernel and build"
run "$IMAGE" bash -c '
  set -euo pipefail
  uname -srm
  cargo --version
  cargo test --locked --release -p ojas-device
  cargo run --locked --release -p ojas-device --example profile -- 8589934592
  cargo test --locked --release -p ojas-capi --lib --no-run 2>&1 | tee /dev/stderr |
    sed -n "s/^ *Executable unittests src\/lib.rs (\(.*\))$/\1/p" > /work/target/capi-test-binary
  test -s /work/target/capi-test-binary
'
BIN="$(cat "$TARGET/capi-test-binary")"
case "$BIN" in
  /work/target/*) ;;
  *) echo "ERROR: the capi test binary is not under the target dir: $BIN" >&2; exit 1 ;;
esac

echo "==> idle container: the live probe reads Normal, and NEW is admitted"
run -e OJAS_EXPECT_PRESSURE=normal "$IMAGE" \
  "$BIN" --ignored --exact "$TEST" --nocapture --test-threads=1

echo "==> stalled sibling cgroup (memory.high $HIGH): Critical, and NEW is E_PRESSURE"
run --systemd=always --memory="$MAX" --memory-swap="$MAX" \
  -e OJAS_EXPECT_PRESSURE=critical -e BIN="$BIN" -e TEST="$TEST" -e HIGH="$HIGH" -e HOG="$HOG" \
  "$IMAGE" bash -c '
    set -euo pipefail
    cg=/sys/fs/cgroup
    test "$(cat /proc/self/cgroup)" = "0::/"
    # A cgroup with children holds no processes: this shell, the only one,
    # moves out first, and everything it starts follows it.
    mkdir "$cg/init" "$cg/hog"
    echo "$$" > "$cg/init/cgroup.procs"
    test -z "$(cat "$cg/cgroup.procs")"
    echo +memory > "$cg/cgroup.subtree_control"
    echo "$HIGH" > "$cg/hog/memory.high"
    test "$(cat "$cg/hog/memory.high")" = "$HIGH"
    (echo "$BASHPID" > "$cg/hog/cgroup.procs"
     while :; do head -c "$HOG" /dev/zero | tail > /dev/null; done) &
    hog=$!
    test "$(cat /proc/self/cgroup)" = "0::/init"
    status=0
    timeout 300 "$BIN" --ignored --exact "$TEST" --nocapture --test-threads=1 || status=$?
    # The loop stops here; a pass still running ends with the container.
    kill "$hog"
    exit "$status"
  '

echo "linux device check OK"
