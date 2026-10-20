#!/usr/bin/env bash
# GitPulse ft-b5e64557 phase A: history-preserving import of Lappi's qd-train moving set into ojas-train.
# Spec: Lappi AUDIT/ojas-train-move-2026-10-14/fable-ruling.md ("Handed to ojas", Phase A) and
#       Lappi HANDOFF/ojas-train-move-2026-10-14.md.
#
# Usage: owner-phase-a-import.sh <lappi-freeze-sha> [--execute]
#   Default is a DRY RUN: prints every command and the gate results, changes nothing in ojas.
#   --execute performs the import on a new branch gp/a6-phase-a-import.
# Environment: OJAS_DIR (default /Users/bharath/Code/research/ojas), LAPPI_DIR (default
#   /Users/bharath/Code/research/Lappi-decision; only ever cloned from, never touched).
# This script runs no cargo. It runs git in ojas and in a throwaway clone only.
set -euo pipefail

OJAS="${OJAS_DIR:-/Users/bharath/Code/research/ojas}"
LAPPI="${LAPPI_DIR:-/Users/bharath/Code/research/Lappi-decision}"
FILTER_REPO="/opt/homebrew/bin/git-filter-repo"
BRANCH="gp/a6-phase-a-import"
LANE1_BRANCH="gp/a6-train"
CARD="ft-b5e64557"

# Moving modules (Lappi crates/qd-train/src/<name>.rs). lib.rs is NOT imported.
SRC_FILES=(step trainer schedule recipe adamw mock run_control pyjson)

# Moving paths under crates/qd-train/tests/. Each entry: "<path relative to tests/>|<reason>".
TEST_PATHS=(
  "adamw_decay_sensitive.rs|imports only qd_train::{adamw,recipe,step} (all moving) plus common::{crate_dir,pins}; oracle for the decay-sensitive AdamW fixture"
  "adamw_oracle.rs|imports only qd_train::{adamw,recipe,step} plus common::{fhex,fhex_list,trainer_oracle}; Python-oracle parity for AdamW (ruling Q5: adamw_* oracles move with their fixtures)"
  "schedule_oracle.rs|imports only qd_train::schedule plus common::{fhex,fhex_list,trainer_oracle}; Python-oracle parity for the LR schedule (ruling Q5)"
  "pyjson_oracle.rs|ruling Q5 names pyjson_oracle as moving; imports qd_train::{pyjson,run_control} (moving) BUT ALSO qd_train::ledger::isoformat_utc (NOT moving): UNSURE, phase A must re-point or cut that one import"
  "common|helper module the moving tests declare with 'mod common'; mod.rs imports NON-moving qd_train::{files,held_out,shards,objective} and pins.rs is self-contained: UNSURE, phase A must split mod.rs down to crate_dir/fhex/fhex_list/trainer_oracle/ToyBatch-free helpers"
  "fixtures/trainer-oracle.json|read by common::trainer_oracle(), used by adamw_oracle, schedule_oracle, pyjson_oracle"
  "fixtures/adamw-decay-sensitive|fixture directory (golden/inputs/manifest/preregistration/table) read by adamw_decay_sensitive via common::pins"
)

# Considered and EXCLUDED (stay in Lappi); printed in the dry run so the owner can overrule.
EXCLUDED_TESTS=(
  "trainer_loop.rs|tests the moving trainer but imports qd_train::objective (LetterSpanObjective), NOT moving: UNSURE, excluded; re-import later if objective moves or the test is rewritten against the mock"
  "optimizer_table.rs|imports ft_data, objective, span_head; ruling Q3 says it stays a Lappi test against the moved API"
  "ft_data.rs, head_init.rs, span_head.rs, door.rs, v4.rs, parity.rs, export.rs, ledger_oracle.rs, caller_records_are_held_out.rs, shard_header_parity.rs|import shards/held_out/ft_data/span_head/export/ledger/tristate/supervision/np_random (not moving)"
  "fixture_pins.rs|checks fixture pins for Lappi's shard/span-head fixtures via common::{fixture,scratch}; not a moving module's test"
)

EXECUTE=0
SHA=""
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) if [ -z "$SHA" ]; then SHA="$arg"; else echo "unexpected argument: $arg" >&2; exit 2; fi ;;
  esac
done

fail() { echo "FATAL: $*" >&2; exit 1; }
say()  { printf '%s\n' "$*"; }
# run: print the command; execute it only with --execute.
run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [ "$EXECUTE" -eq 1 ]; then "$@"; fi
}

TMP=""
cleanup() {
  local rc=$?
  if [ -n "$TMP" ] && [ -d "$TMP" ]; then rm -rf -- "$TMP"; fi
  if [ "$rc" -ne 0 ]; then
    echo "FAILED (exit $rc). Temp clone removed." >&2
    if [ -f "$OJAS/.git/MERGE_HEAD" ] 2>/dev/null; then
      echo "ojas has a merge in progress on $(git -C "$OJAS" branch --show-current 2>/dev/null)." >&2
      echo "Recover: git -C '$OJAS' merge --abort && git -C '$OJAS' switch main && git -C '$OJAS' branch -D $BRANCH" >&2
    fi
  fi
}
trap cleanup EXIT

if [ "$EXECUTE" -eq 1 ]; then say "MODE: EXECUTE"; else say "MODE: DRY RUN (nothing in ojas changes; pass --execute to act)"; fi
say "ojas:  $OJAS"
say "lappi: $LAPPI (cloned only; never modified)"
say ""

say "== Moving source files =="
for m in "${SRC_FILES[@]}"; do say "  crates/qd-train/src/$m.rs -> ojas-train/src/lappi/$m.rs"; done
say "  (Lappi lib.rs is NOT imported; ojas-train/src/lib.rs and sha256.rs are never overwritten)"
say "== Moving test paths =="
for t in "${TEST_PATHS[@]}"; do say "  crates/qd-train/tests/${t%%|*}  <- ${t#*|}"; done
say "== Excluded tests (stay in Lappi) =="
for t in "${EXCLUDED_TESTS[@]}"; do say "  ${t%%|*}  <- ${t#*|}"; done
say ""
say "== UNSURE items needing the owner's eye (flagged above) =="
say "  1. pyjson_oracle.rs also imports qd_train::ledger (not moving)."
say "  2. tests/common/mod.rs mixes moving helpers with shard/held_out/objective helpers."
say "  3. trainer_loop.rs is excluded because it imports qd_train::objective."
say "  The branch does not compile after the import regardless; phase A rewires these."
say ""

say "== Gates =="
# Gate 1: freeze sha given, 40 hex.
[ -n "$SHA" ] || fail "gate 1: no freeze sha given. Usage: $0 <lappi-freeze-sha> [--execute]"
case "$SHA" in
  *[!0-9a-fA-F]*) fail "gate 1: freeze sha contains non-hex characters: $SHA" ;;
esac
[ "${#SHA}" -eq 40 ] || fail "gate 1: freeze sha must be 40 hex characters, got ${#SHA}"
SHA="$(printf '%s' "$SHA" | tr 'A-F' 'a-f')"
say "gate 1 PASS: freeze sha $SHA is 40 hex characters"

[ -x "$FILTER_REPO" ] || fail "git-filter-repo not found at $FILTER_REPO"
[ -d "$LAPPI/.git" ] || fail "Lappi checkout not found at $LAPPI"
[ -d "$OJAS/.git" ] || [ -f "$OJAS/.git" ] || fail "ojas checkout not found at $OJAS"

# Fresh clone of Lappi into a new temp dir.
TMP="$(mktemp -d "${TMPDIR:-/tmp}/ojas-phase-a.XXXXXX")"
CLONE="$TMP/lappi"
FILTERED_NOTE="$TMP/lappi (filtered in place; this clone is disposable)"
run_clone() { git clone --no-local --quiet -- "$LAPPI" "$CLONE"; }
printf '+ git clone --no-local -- %q %q\n' "$LAPPI" "$CLONE"
run_clone   # read-only with respect to Lappi, so it also runs in the dry run (needed for gate 2)

git -C "$CLONE" cat-file -e "$SHA^{commit}" 2>/dev/null \
  || fail "gate 2: $SHA is not a commit in the Lappi clone (is it on a pushed/local branch of $LAPPI?)"

# Gate 2: freeze sha is an ancestor of origin/main (or main) in the clone.
GATE2=""
for ref in origin/main main; do
  if git -C "$CLONE" rev-parse --verify --quiet "$ref^{commit}" >/dev/null \
     && git -C "$CLONE" merge-base --is-ancestor "$SHA" "$ref"; then GATE2="$ref"; break; fi
done
[ -n "$GATE2" ] || fail "gate 2: $SHA is not an ancestor of origin/main or main in the Lappi clone"
say "gate 2 PASS: $SHA is an ancestor of $GATE2 in the clone"

# Gate 3: lane 1 (gp/a6-train) is merged into ojas main.
git -C "$OJAS" rev-parse --verify --quiet "$LANE1_BRANCH^{commit}" >/dev/null \
  || fail "gate 3: ojas branch $LANE1_BRANCH does not exist"
git -C "$OJAS" merge-base --is-ancestor "$LANE1_BRANCH" main \
  || fail "gate 3: $LANE1_BRANCH is not an ancestor of ojas main (lane 1 not merged)"
say "gate 3 PASS: $LANE1_BRANCH is an ancestor of ojas main"

# Gate 4: ojas worktree fully clean, untracked files included.
DIRTY="$(git -C "$OJAS" status --porcelain --untracked-files=all)"
if [ -n "$DIRTY" ]; then
  printf '%s\n' "$DIRTY" | head -40 >&2
  fail "gate 4: ojas working tree is not clean (untracked files count; peers' dirty files will block this, which is intended). Run from a clean checkout/worktree of ojas via OJAS_DIR."
fi
say "gate 4 PASS: git status --porcelain (untracked included) is empty in ojas"

# Gate 5 (part 1): the new branch must not exist.
if git -C "$OJAS" rev-parse --verify --quiet "refs/heads/$BRANCH" >/dev/null; then
  fail "gate 5: branch $BRANCH already exists; refusing (delete or rename it yourself)"
fi
say "gate 5a PASS: branch $BRANCH does not exist"
say ""

say "== Import =="
LIB_BLOB="$(git -C "$CLONE" rev-parse "$SHA:crates/qd-train/src/lib.rs")"
say "Lappi lib.rs blob at freeze: $LIB_BLOB"
for m in "${SRC_FILES[@]}"; do
  git -C "$CLONE" cat-file -e "$SHA:crates/qd-train/src/$m.rs" || fail "missing at freeze: crates/qd-train/src/$m.rs"
done
for t in "${TEST_PATHS[@]}"; do
  p="${t%%|*}"
  git -C "$CLONE" cat-file -e "$SHA:crates/qd-train/tests/$p" || fail "missing at freeze: crates/qd-train/tests/$p"
done

FR_ARGS=(--refs refs/heads/freeze)
for m in "${SRC_FILES[@]}"; do FR_ARGS+=(--path "crates/qd-train/src/$m.rs"); done
for t in "${TEST_PATHS[@]}"; do FR_ARGS+=(--path "crates/qd-train/tests/${t%%|*}"); done
FR_ARGS+=(--path-rename "crates/qd-train/src/:ojas-train/src/lappi/"
          --path-rename "crates/qd-train/tests/:ojas-train/tests/lappi/")

printf '+ git -C %q branch freeze %q\n' "$CLONE" "$SHA"
printf '+ (cd %q && %q' "$CLONE" "$FILTER_REPO"; printf ' %q' "${FR_ARGS[@]}"; printf ' --force)\n'
printf '+ git -C %q switch -c %q main\n' "$OJAS" "$BRANCH"
printf '+ git -C %q fetch %q freeze\n' "$OJAS" "$CLONE"
printf '+ git -C %q merge --allow-unrelated-histories --no-commit FETCH_HEAD\n' "$OJAS"
printf '+ write %q (freeze sha, lib.rs blob, file list, date, not-compiling note)\n' "$OJAS/ojas-train/LAPPI-IMPORT.md"
printf '+ git -C %q add ojas-train/LAPPI-IMPORT.md && git commit (message ends with the Co-Authored-By line)\n' "$OJAS"
printf '+ git -C %q log --follow -- ojas-train/src/lappi/trainer.rs   (verify Lappi commits, print count)\n' "$OJAS"

if [ "$EXECUTE" -ne 1 ]; then
  say ""
  say "DRY RUN complete: gates 1-4 and 5a passed; nothing in ojas was changed. Re-run with --execute."
  exit 0
fi

git -C "$CLONE" branch freeze "$SHA"
( cd "$CLONE" && "$FILTER_REPO" "${FR_ARGS[@]}" --force )

# The filtered tree must not carry anything that would overwrite ojas-train's own files.
FILTERED_FILES="$(git -C "$CLONE" ls-tree -r --name-only freeze)"
printf '%s\n' "$FILTERED_FILES"
if printf '%s\n' "$FILTERED_FILES" | rg -q '^ojas-train/src/(lib|sha256)\.rs$'; then
  fail "filtered history contains ojas-train/src/lib.rs or sha256.rs; refusing"
fi
if printf '%s\n' "$FILTERED_FILES" | rg -qv '^ojas-train/(src|tests)/lappi/'; then
  fail "filtered history contains paths outside ojas-train/{src,tests}/lappi/; refusing"
fi
for m in "${SRC_FILES[@]}"; do
  printf '%s\n' "$FILTERED_FILES" | rg -qx "ojas-train/src/lappi/$m\.rs" || fail "filtered history lacks $m.rs"
done

git -C "$OJAS" switch -c "$BRANCH" main
# Gate 5 (part 2): not on main after switching.
CUR="$(git -C "$OJAS" branch --show-current)"
[ "$CUR" = "$BRANCH" ] && [ "$CUR" != "main" ] || fail "gate 5: current ojas branch is '$CUR', expected $BRANCH (not main)"
say "gate 5b PASS: ojas is on $CUR"

git -C "$OJAS" fetch --quiet -- "$CLONE" freeze
IMPORT_TIP="$(git -C "$OJAS" rev-parse FETCH_HEAD)"
git -C "$OJAS" merge --allow-unrelated-histories --no-commit FETCH_HEAD

DATE_UTC="$(date -u +%Y-%m-%d)"
{
  printf '# Lappi import (phase A, card %s)\n\n' "$CARD"
  printf 'History-preserving import of Lappi `crates/qd-train` into `ojas-train` (git-filter-repo, path rewrite).\n\n'
  printf -- '- Lappi freeze sha: `%s`\n' "$SHA"
  printf -- '- Lappi `crates/qd-train/src/lib.rs` blob sha at the freeze: `%s` (lib.rs was NOT imported)\n' "$LIB_BLOB"
  printf -- '- Imported on: %s (UTC)\n' "$DATE_UTC"
  printf -- '- Rewritten filtered tip (second parent of the import merge): `%s`\n\n' "$IMPORT_TIP"
  printf '## Files (as they stand in ojas-train)\n\n'
  printf '%s\n' "$FILTERED_FILES" | sed 's/^/- `/; s/$/`/'
  printf '\n## Moving tests and why\n\n'
  for t in "${TEST_PATHS[@]}"; do printf -- '- `crates/qd-train/tests/%s`: %s\n' "${t%%|*}" "${t#*|}"; done
  printf '\n## State\n\n'
  printf 'This branch does not compile. Phase A still has to rewire the `qd_train::` paths in these files\n'
  printf 'and add `pub mod lappi` to `ojas-train/src/lib.rs`, then move `pyjson` onto `ojas_io::JsonValue`,\n'
  printf 'add SHA-256 per ruling Q4, and fix the flagged tests (pyjson_oracle imports `qd_train::ledger`;\n'
  printf '`tests/common/mod.rs` imports non-moving shard/held_out/objective modules).\n'
} > "$OJAS/ojas-train/LAPPI-IMPORT.md"

git -C "$OJAS" add ojas-train/LAPPI-IMPORT.md
git -C "$OJAS" commit --quiet -F - <<EOF
ojas-train: import Lappi qd-train moving set with history (phase A, $CARD)

git-filter-repo import of crates/qd-train/src/{step,trainer,schedule,recipe,adamw,mock,run_control,pyjson}.rs
and the tests that import only them (plus common/ and two fixtures), at Lappi freeze $SHA,
renamed to ojas-train/src/lappi/ and ojas-train/tests/lappi/. Lappi's lib.rs is not imported;
ojas-train/src/lib.rs and sha256.rs are untouched. See ojas-train/LAPPI-IMPORT.md.

This commit does not compile: phase A rewires qd_train:: paths and adds pub mod lappi next.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF

# Verify history preservation: Lappi commits visible through --follow.
TOTAL=0
LAPPI_COUNT=0
while IFS= read -r h; do
  [ -n "$h" ] || continue
  TOTAL=$((TOTAL + 1))
  if git -C "$OJAS" merge-base --is-ancestor "$h" "$IMPORT_TIP"; then LAPPI_COUNT=$((LAPPI_COUNT + 1)); fi
done < <(git -C "$OJAS" log --follow --format=%H -- ojas-train/src/lappi/trainer.rs)
say "git log --follow ojas-train/src/lappi/trainer.rs: $TOTAL commits, $LAPPI_COUNT of them Lappi commits (ancestors of $IMPORT_TIP)"
[ "$LAPPI_COUNT" -ge 1 ] || fail "history check: --follow shows no Lappi commits for ojas-train/src/lappi/trainer.rs"
git -C "$OJAS" log --follow --oneline -5 -- ojas-train/src/lappi/trainer.rs

NEW_SHA="$(git -C "$OJAS" rev-parse HEAD)"
say ""
say "== Owner next steps =="
say "1. Review: git -C '$OJAS' show --stat $NEW_SHA ; do NOT merge to main yet (branch does not compile)."
say "2. Post to the GitPulse log for $CARD:"
say "   phase A import done on $BRANCH @ $NEW_SHA: Lappi qd-train moving set imported with history at freeze $SHA (lib.rs blob $LIB_BLOB), $LAPPI_COUNT Lappi commits visible via git log --follow on ojas-train/src/lappi/trainer.rs; branch does not compile until the qd_train:: rewire and pub mod lappi land; flagged: pyjson_oracle (ledger import), tests/common/mod.rs (non-moving helpers), trainer_loop.rs excluded (objective)."
say "3. Continue phase A in an ojas session on $BRANCH: rewire paths, pub mod lappi, pyjson onto ojas_io::JsonValue, SHA-256 (Q4), groups.rs, ParamSpec."
say "4. Lappi's half (depend on ojas-train, delete its moved copies, switch qd-train) is a Lappi session's job, and starts only after the owner's pin-bump yes."
