#!/usr/bin/env bash
# Local gates for this tree, mirroring .github/workflows/test.yml (Linux and
# Go jobs) so a push is not the first place a gate runs.
# Invoke: ./scripts/ci_local.sh
#
# Exit codes:
#   0    every gate ran and passed (last line `ci:local OK`), or every gate that
#        could not run is named in OJAS_CI_LOCAL_ALLOW_NOT_RUN (last line
#        `ci:local OK WITH WAIVED SKIPS (...)`)
#   3    every gate that ran passed, but a gate could NOT RUN because a tool or
#        toolchain is absent (gotip, the dated nightly, the MSRV toolchain).
#        Last line `ci:local INCOMPLETE (...)`. This is not a pass.
#   64   bad usage: an unknown name in OJAS_CI_LOCAL_ALLOW_NOT_RUN, or an
#        OJAS_MAC_HEAVY that is not an absolute path to an executable file
#   >=128  a signal (130 INT, 143 TERM)
#   else a failed gate's own exit code (1 for ci_local's own errors)
# Only the verdict (and the not-run branches that feed it) exits 3. A hard gate
# (fmt, clippy, test, ...) that returns 3 is a FAILED gate: the exit trap prints
# `ci:local FAILED (...)` and exits 1, at that one place for every step.
# A not-run reason is flattened to one line, so it cannot spoof another step.
# When the lock wrapper returns 0 but the gates never started (they write a
# sentinel file at start), the run is NOT RUN: INCOMPLETE, exit 3.
# Signals: INT/TERM sent to this process are forwarded (as TERM) to the lock
# wrapper's child and then exit 130/143; a signal sent only to a child of a
# plain (unlocked or in-place) step takes effect when that step ends, while
# Ctrl-C reaches the whole process group at once.
#
# OJAS_CI_LOCAL_ALLOW_NOT_RUN: comma- or space-separated, exact, case-sensitive
# names from `gotip nightly msrv`. A waived gate still runs when it can; the
# waiver only lets the verdict be 0 when it cannot.
#
# Lock: on macOS the whole run takes mac_heavy (OJAS_MAC_HEAVY, else
# ../Lappi-decision/tools/mac_heavy.sh, else
# ~/Code/research/Lappi-decision/tools/mac_heavy.sh) and fails closed (exit 1)
# when none exists. The sibling-path lookup is macOS-only. mac_heavy's own
# environment applies (MAC_HEAVY_MAX_LOAD default 12, a 4 h wait that exits 76),
# so on a busy host set MAC_HEAVY_MAX_LOAD; its exit codes (75, 76, ...) pass
# through with no `ci:local` line. When MAC_HEAVY_LABEL is already set the run
# is already under mac_heavy (99-tools/ci.sh, heavy.sh) and goes in place, so
# the lock is never taken twice. Known trade-off: a stale MAC_HEAVY_LABEL
# inherited from an unrelated shell also skips the lock (parity_check.sh and
# tessl do the same). There is no unlock flag.
#
# Deliberate differences from test.yml: the Linux-only lavapipe opt-in
# (test.yml:115), the macos job's `--exclude ojas-metal` (here the real Metal
# tests run), the go matrix (one OS here, chosen by `uname -s`), and the deny
# job (cargo-deny-action there, the installed cargo-deny here). The MSRV step
# adds a second artifact set to the target dir (disk). Not applicable gates
# (ojas-hip on macOS, Accelerate on Linux) are printed and are not skips.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Waivers: validated first, before the lock or any gate.
KNOWN_STEPS="gotip nightly msrv"
WAIVED=" "
set -f
for name in $(printf '%s' "${OJAS_CI_LOCAL_ALLOW_NOT_RUN:-}" | tr ',' ' '); do
  case " $KNOWN_STEPS " in
    *" $name "*) WAIVED="$WAIVED$name " ;;
    *)
      set +f
      echo "ERROR: OJAS_CI_LOCAL_ALLOW_NOT_RUN names unknown step '$name' (known: $KNOWN_STEPS; names are exact)" >&2
      exit 64
      ;;
  esac
done
set +f

# Cleanup, and the one place that keeps 3 meaning "not run": an exit 3 that did
# not come from the verdict below (a hard gate such as fmt, clippy or test that
# returned 3) is a FAILED gate and is reported as 1, never as INCOMPLETE.
REASON_FILE=""
SENTINEL=""
VERDICT_3=0
# shellcheck disable=SC2329  # trap handler
on_exit() {
  local rc=$?
  [ -z "$REASON_FILE" ] || rm -f "$REASON_FILE"
  [ -z "$SENTINEL" ] || rm -f "$SENTINEL"
  if [ "$rc" -eq 3 ] && [ "$VERDICT_3" -ne 1 ]; then
    echo "ci:local FAILED (a gate exited 3, which ci_local reserves for NOT RUN; reporting 1)" >&2
    exit 1
  fi
}
trap on_exit EXIT

# Signals: forward TERM to the lock wrapper child (if any), wait for it, then
# exit 128+n. INT is forwarded as TERM because a background child ignores INT.
child=""
# shellcheck disable=SC2329  # trap handler
on_signal() { # CODE
  if [ -n "$child" ]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  exit "$1"
}
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

# 2. Lock.
if [ "$(uname -s)" = Darwin ]; then
  if [ -n "${MAC_HEAVY_LABEL:-}" ]; then
    echo "==> lock: already under mac_heavy ($MAC_HEAVY_LABEL)"
    # Prove to the process that took the lock that the gates really started.
    if [ -n "${OJAS_CI_LOCAL_SENTINEL:-}" ] && [ -f "$OJAS_CI_LOCAL_SENTINEL" ]; then
      printf 'started\n' >"$OJAS_CI_LOCAL_SENTINEL"
    fi
  else
    wrapper=""
    if [ -n "${OJAS_MAC_HEAVY:-}" ]; then
      case "$OJAS_MAC_HEAVY" in
        /*) ;;
        *)
          echo "ERROR: OJAS_MAC_HEAVY=$OJAS_MAC_HEAVY must be an absolute path (a relative one would resolve against the repo root, not your cwd)" >&2
          exit 64
          ;;
      esac
      if [ ! -f "$OJAS_MAC_HEAVY" ] || [ ! -x "$OJAS_MAC_HEAVY" ]; then
        echo "ERROR: OJAS_MAC_HEAVY=$OJAS_MAC_HEAVY is not an executable file" >&2
        exit 64
      fi
      wrapper="$OJAS_MAC_HEAVY"
    else
      for candidate in \
        "$ROOT/../Lappi-decision/tools/mac_heavy.sh" \
        "${HOME:-/nonexistent}/Code/research/Lappi-decision/tools/mac_heavy.sh"; do
        if [ -f "$candidate" ] && [ -x "$candidate" ]; then
          wrapper="$candidate"
          break
        fi
      done
    fi
    if [ -z "$wrapper" ]; then
      echo "ERROR: mac_heavy.sh not found; ci_local takes the machine lock on macOS and will not run without it. Set OJAS_MAC_HEAVY to its path." >&2
      exit 1
    fi
    echo "==> lock: taking mac_heavy via $wrapper"
    SENTINEL="$(mktemp "${TMPDIR:-/tmp}/ojas-ci-local-ran.XXXXXX")"
    export OJAS_CI_LOCAL_SENTINEL="$SENTINEL"
    bash "$wrapper" ojas-ci-local "$ROOT/scripts/ci_local.sh" "$@" &
    child=$!
    wrapper_rc=0
    wait "$child" || wrapper_rc=$?
    child=""
    if [ "$wrapper_rc" -eq 0 ] && [ ! -s "$SENTINEL" ]; then
      VERDICT_3=1
      echo "ci:local INCOMPLETE (not run: the lock wrapper $wrapper exited 0 without running the gates)" >&2
      exit 3
    fi
    # The wrapped run already printed its own verdict (and mapped a stray 3 to 1);
    # a 3 here is that verdict, so it passes through.
    if [ "$wrapper_rc" -eq 3 ]; then VERDICT_3=1; fi
    exit "$wrapper_rc"
  fi
else
  echo "==> lock: not macOS; mac_heavy not used"
fi

# 3. Not-run bookkeeping. A step that cannot run exits 3 and (for the scripts)
# writes its reason to OJAS_NOT_RUN_REASON_FILE.
REASON_FILE="$(mktemp "${TMPDIR:-/tmp}/ojas-ci-local.XXXXXX")"
export OJAS_NOT_RUN_REASON_FILE="$REASON_FILE"
SKIPS="" # newline-joined "name: reason" (a string: an empty array breaks `set -u` on bash 3.2)

# optional_step NAME cmd...: rc 3 records NAME as not run; any other non-zero
# rc (including a signal) ends the run with it.
optional_step() {
  local name=$1 rc=0 reason
  shift
  : >"$REASON_FILE"
  "$@" || rc=$?
  case "$rc" in
    0) ;;
    3)
      # One line: a multi-line reason would split into fake "name: ..." entries.
      reason="$(tr '\r\n' '  ' <"$REASON_FILE")"
      SKIPS="$SKIPS$name: ${reason:-exited 3 without a reason}"$'\n'
      echo "==> NOT RUN ($name): ${reason:-exited 3 without a reason}" >&2
      ;;
    *) exit "$rc" ;;
  esac
}

# not_run REASON: record the reason and return 3 (only for the explicit not-run branches).
# shellcheck disable=SC2329  # called through optional_step
not_run() {
  printf '%s\n' "$1" >"$REASON_FILE"
  return 3
}

# msrv_gate: test.yml's msrv job. Every command is checked explicitly: this runs
# under `||`, where `set -e` does not apply.
# shellcheck disable=SC2329  # called through optional_step
msrv_gate() {
  local msrv list rc=0
  msrv="$(sed -n 's/^rust-version = "\([0-9.]*\)"/\1/p' Cargo.toml)" || return 1
  if [ -z "$msrv" ]; then
    echo "ERROR: no rust-version in Cargo.toml" >&2
    return 1
  fi
  if ! command -v rustup >/dev/null 2>&1; then
    not_run "rustup not found; the MSRV $msrv gate needs it"
    return
  fi
  if ! list="$(rustup toolchain list)"; then
    echo "ERROR: could not list toolchains ('rustup toolchain list' failed)" >&2
    return 1
  fi
  case $'\n'"$list" in
    *$'\n'"$msrv"-*) ;;
    *)
      not_run "toolchain $msrv is not installed (rustup toolchain install $msrv --profile minimal)"
      return
      ;;
  esac
  cargo "+$msrv" check --workspace --all-targets || rc=$?
  if [ "$rc" -eq 0 ]; then
    cargo "+$msrv" check -p ojas-cuda --features cuda --all-targets || rc=$?
  fi
  if [ "$rc" -eq 3 ]; then
    echo "ERROR: cargo exited 3, which ci_local reserves for NOT RUN; reporting 1" >&2
    rc=1
  fi
  return "$rc"
}

# First, because every gate below it needs cargo to resolve this workspace, and
# this workspace path-depends on sibling checkouts (tessl, gusset). When one is
# missing or this is a worktree, cargo fails with a manifest error that names
# the wrong cause; this says what is actually wrong instead.
echo "==> sibling checkouts and path dependencies"
./scripts/check_siblings.sh

echo "==> script self-tests"
./scripts/check_tracked_targets_test.sh
./scripts/check_tracked_targets_adversarial_test.sh
./scripts/check_ignore_reasons_test.sh
./scripts/check_siblings_test.sh
./scripts/parity_check_test.sh
./scripts/gotip_simd_test.sh
./scripts/ci_local_test.sh
if [ "$(uname -s)" = Darwin ]; then
  ./bench/run_paired_test.sh
fi

echo "==> fmt"
./scripts/fmt_check.sh

echo "==> cargo target sources are tracked"
./scripts/check_tracked_targets.sh

echo "==> no bare #[ignore]"
./scripts/check_ignore_reasons.sh

echo "==> clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> cargo-deny (deny.toml)"
if ! command -v cargo-deny >/dev/null 2>&1; then
  echo "ERROR: cargo-deny is not installed (cargo install cargo-deny --locked); the deny gate cannot be skipped" >&2
  exit 1
fi
cargo deny check

echo "==> cuda check"
# cudarc loads the driver dynamically. This host can compile the feature
# without a CUDA toolkit. It does not execute a kernel. The lib-only lints
# check ojas-cuda's default surface: any target that builds its tests turns
# on `harness` (its dev-dependency on itself).
cargo clippy -p ojas-cuda --features cuda --all-targets -- -D warnings
cargo clippy -p ojas-cuda --features cuda -- -D warnings
cargo clippy -p ojas-cuda -- -D warnings
cargo test -p ojas-cuda --features cuda --no-run

echo "==> ojas-simd (plain; Accelerate on macOS)"
cargo test -p ojas-simd
if [ "$(uname -s)" = Darwin ]; then
  cargo test -p ojas-simd --features accelerate
  cargo clippy -p ojas-simd --features accelerate --all-targets -- -D warnings
else
  echo "==> ojas-simd accelerate: not applicable (Accelerate is macOS-only)"
fi

echo "==> nightly simd (ojas-simd on the dated nightly)"
optional_step nightly ./scripts/simd_bench.sh

echo "==> msrv (cargo check at the declared rust-version)"
optional_step msrv msrv_gate

echo "==> go (stable)"
if ! command -v go >/dev/null 2>&1; then
  echo "ERROR: go is not installed; the stable Go gate cannot be skipped" >&2
  exit 1
fi
go version
case "$(uname -s)" in
  Darwin) PC_DIR="$ROOT/go" ;;
  Linux) PC_DIR="$ROOT/go/linux" ;;
  *)
    echo "ERROR: ci_local.sh supports Darwin and Linux, not $(uname -s)" >&2
    exit 1
    ;;
esac
echo "==> site (reference pages and docs mirror must match the markdown)"
go run scripts/sitegen/main.go -check
echo "==> bench charts (site/assets/plots/bench and docs/bench-plots.md must match bench/results)"
go test -count=1 scripts/benchreport/main.go scripts/benchreport/main_test.go
go run scripts/benchreport/main.go plots -check
cargo build -p ojas-gusset-engine
(
  cd "$ROOT/go"
  PKG_CONFIG_PATH="$PC_DIR" go test -a -tags gusset_pkgconfig -count=1 -timeout 20m ./...
)

echo "==> gotip (optional)"
optional_step gotip ./scripts/gotip.sh

echo "==> test (release, serialized)"
# --no-fail-fast to match test.yml's Linux job. Without it cargo stops at the
# first test binary that fails, so the crates after it never run and a red
# result says nothing about them — and this script exists precisely so a push
# is not the first place a gate runs. If it reported less than CI does, it
# would send exactly the drift it is meant to prevent.
cargo test --workspace --release --no-fail-fast -- --test-threads=1

echo "==> rustdoc (intra-doc links resolve; no link to a private item)"
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
# The workspace run above builds ojas-simd with Accelerate on macOS (feature
# unification), so the plain feature set is documented on its own, and on macOS the
# accelerate set too. Off macOS that feature is a compile_error!.
RUSTDOCFLAGS="-D warnings" cargo doc -p ojas-simd --no-deps
if [ "$(uname -s)" = Darwin ]; then
  RUSTDOCFLAGS="-D warnings" cargo doc -p ojas-simd --no-deps --features accelerate
fi

echo "==> ojas-metal without the metal feature (stub build, every target)"
cargo clippy -p ojas-metal --no-default-features --all-targets -- -D warnings

# HIP: hip-runtime-sys 0.1.2 panics in its build script when
# hip/hip_runtime_api.h is absent, which it is on macOS. That is not a skip.
if [ "$(uname -s)" = Darwin ]; then
  echo "==> ojas-hip check: not applicable on macOS (hip-runtime-sys needs hip_runtime_api.h)"
else
  echo "==> ojas-hip check"
  cargo check -p ojas-hip --all-targets
fi

# Verdict. `ci:local OK` is printed only when nothing was left unrun.
if [ -z "$SKIPS" ]; then
  echo "ci:local OK"
  exit 0
fi
all_list=""
unwaived_list=""
while IFS= read -r entry; do
  [ -n "$entry" ] || continue
  all_list="${all_list:+$all_list; }$entry"
  case "$WAIVED" in
    *" ${entry%%:*} "*) ;;
    *) unwaived_list="${unwaived_list:+$unwaived_list; }$entry" ;;
  esac
done <<<"$SKIPS"
if [ -z "$unwaived_list" ]; then
  echo "ci:local OK WITH WAIVED SKIPS (not run: $all_list; waived by OJAS_CI_LOCAL_ALLOW_NOT_RUN)"
  exit 0
fi
echo "ci:local INCOMPLETE (every gate that ran passed; not run: $unwaived_list)"
VERDICT_3=1
exit 3
