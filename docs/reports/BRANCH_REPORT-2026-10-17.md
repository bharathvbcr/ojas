# Branch consolidation, 2026-10-17 (session ojas-1c)

This pass made local changes only. Nothing was pushed, and no remote branch was touched. This file is new. `BRANCH_REPORT.md` (the 2026-10-16 pass) was left as is, because peer session ojas-f2 was live in this checkout.

- **main:** `881888e` → `67dc393`. One branch was merged (`gp/w11-scratch`). It had no conflicts.
- **Deleted:** 3 branches and 4 worktrees. Every one had a backup ref first.
- **Kept:** 3 branches.
  - Two belong to live session ojas-f2.
  - One is blocked on tessl.
- **Lappi pin:** this pass moves main further from Lappi's pin. Lappi main `2733015` pins ojas `5b67cdd`. ojas-f2 has already asked the owner whether to re-bump it. The bump was not made here.

## Inventory (start of pass, against main `881888e`)

All tips are by BHARATH CHANDRA VADDARAM.

**Live sessions** (GitPulse Insights, all facets `ok`):

- 26207 is ojas-f2, in the main checkout.
- 39222 is a busy claude session in the main checkout.
- 97075 is a stale `ugrep` from 2026-10-15, in a removed `.gitpulse/worktrees` path.
- tessl-54 is in the tessl repo.

**Collision risk:** 0 overlapping files across 8 worktrees.

| Branch | Behind/ahead | Last commit | Merged? | Worktree (dirty) | Session | Task | Class |
|---|---|---|---|---|---|---|---|
| `gp/w11-scratch` | 38/2 | 10-16 23:36 | no | `ojas-w11-scratch` (clean) | none | ft-81e49280 ready (tests for it) | **mergeable** |
| `gp/w11-oracle` | 30/0 | 10-16 23:24 | yes | `ojas-w11-oracle` (clean) | none | ft-ab2f1daf review | **merged** |
| `gp/w11-readout-sync` | 37/0 | 10-16 23:22 | yes | `ojas-w11-readout-sync` (clean) | none | ft-9c4ae3c8 review | **merged** |
| `gp/ft-audit2-brief-sync` | 1/0 | 10-17 07:53 | yes | `ojas-ft-brief-sync` (clean) | ojas-f2 owns | none | merged, **owner asked to keep** |
| `gp/lora-step-rows` | 0/1 (`0f13b37`)¹ | 10-17 ~08:05 | no | `ojas-lora-rows` (3 dirty at start, then clean) | ojas-f2 live | ft-2bbced18 in_progress | **in use** |
| `gp/clip-sq-norm` | 64/1 | 10-16 23:36 | no | `ojas-clipnorm` (clean) | ojas-c7 ended | ft-2c183b71 ready (clip item) | **blocked** (tessl dependency) |
| detached `ojas-merge-pass2` @ `829a571` | n/a | 10-17 00:11 | no | clean | none (no session claims it) | none | **superseded** |

¹ At first read the branch sat at `881888e` while a mac_heavy test was running in its worktree. ojas-f2 then committed `0f13b37`. Fable verified the new tip.

**Superseded proof for `829a571`:** its tree `94a571e` is identical to `git merge-tree 5b67cdd cc9f5ad`. That means it is the same `gp/w11-scratch` merge, built on an older main. Fable verified this.

**Classification review:** Fable reviewed the classification and agreed on every branch. It confirmed three things:

- The `#[ignore = "BREAK: ..."]` convention is already established on main (`ojas-cpu/tests/redteam_ops_heap.rs:710`).
- tessl is a path dependency (`ojas-metal/Cargo.toml:22`).
- `tessl::sq_norm` exists only on tessl's `gp/clip-sq-norm` branch (`2c02982`).

## Backups

- **Bundle:** `/Users/bharath/Code/research/ojas-backups/2026-10-17/ojas-branches.bundle`. It holds 15 heads (every branch plus the backup refs), and `git bundle verify` succeeded.
- **Ref list:** `/Users/bharath/Code/research/ojas-backups/2026-10-17/backup-refs.txt`.

| Backup ref | Commit |
|---|---|
| `refs/backup/2026-10-17/main` | `881888e` |
| `refs/backup/2026-10-17/gp/w11-scratch` | `cc9f5ad` |
| `refs/backup/2026-10-17/gp/w11-oracle` | `5eb4995` |
| `refs/backup/2026-10-17/gp/w11-readout-sync` | `1fef9ff` |
| `refs/backup/2026-10-17/gp/ft-audit2-brief-sync` | `b68971c` |
| `refs/backup/2026-10-17/gp/lora-step-rows` | `0f13b37` |
| `refs/backup/2026-10-17/gp/clip-sq-norm` | `2146553` |
| `refs/backup/2026-10-17/detached/ojas-merge-pass2` | `829a571` |

## Pending work (Phase 3)

Nothing was committed.

- **`ojas-lora-rows`:** its dirty files belonged to live session ojas-f2.
- **Other branch worktrees:** all clean.
- **Main checkout, not committed:**
  - `M .gitignore` is Manvi's `.devcouncil/*` block. Committing it is the owner's decision.
  - Untracked reports: `AUDIT_REPORT.md`, `BRANCH_REPORT.md`, `BUILD_REPORT*.md`, and `backups/`.
  - Five untracked briefs, `tasks/gp-{exact-libm-determinism-and-tape-capi-hardening,gpu-vision-stack-and-vit-sdpa,metal-fast-lane-epilogue-and-cpu-bf16-muon,repo-hygiene-gemma-metal-and-devcouncil-sync,uma-zero-copy-io-and-budget-preflight}.md`. They come from an unknown session, and that session's work is not this pass's to commit.

## Merged

| Branch | Merge commit | CI |
|---|---|---|
| `gp/w11-scratch` (`06fc333`, `cc9f5ad`) | `67dc393` (`--no-ff`, parents `881888e` + `cc9f5ad`) | `ci:local OK` under `mac_heavy.sh` (label `ojas-merge-w11-scratch`, exit 0): 2,295 passed, 0 failed, 69 ignored |

About the merge:

- **What it contains:** tests only, 3 files, +153 lines, no `src/` changes.
  - The two failing-first wgpu tests are ignored as `BREAK: fails on current code until ft-81e49280 lands`.
  - The new Metal test `scratch_stacking.rs` runs and passes.
- **How it landed:** the merge was built and CI'd in the detached worktree `ojas-merge-pass2`, which has its own target dir. main was then fast-forwarded to `67dc393` after confirming it was still at `881888e`.
- **Conflicts:** none. `git merge-tree` came back clean, so there was no resolution to review.

## Branches deleted

Each one was merged into main, matched its backup ref, and had a clean worktree. Branches were deleted with `git branch -d`, and `-D` was never needed. Worktrees were removed with `git worktree remove`.

| Branch / worktree | Was | Backup ref |
|---|---|---|
| `gp/w11-oracle` + `../ojas-w11-oracle` | `5eb4995` | `refs/backup/2026-10-17/gp/w11-oracle` |
| `gp/w11-readout-sync` + `../ojas-w11-readout-sync` | `1fef9ff` | `refs/backup/2026-10-17/gp/w11-readout-sync` |
| `gp/w11-scratch` + `../ojas-w11-scratch` | `cc9f5ad` | `refs/backup/2026-10-17/gp/w11-scratch` |
| worktree `../ojas-merge-pass2` (detached, superseded `829a571`) | `829a571` | `refs/backup/2026-10-17/detached/ojas-merge-pass2` |

## Branches kept

| Branch | Reason |
|---|---|
| `main` | Trunk. It is 143 commits ahead of `origin/main` and was not pushed. |
| `gp/lora-step-rows` (`0f13b37`) | Owned by live session ojas-f2, and its card ft-2bbced18 is in_progress. ojas-f2 is waiting on the owner for two things: whether to merge it, and whether to re-bump Lappi's pin. |
| `gp/ft-audit2-brief-sync` (`b68971c`) | Merged. ojas-f2 asked that the branch and its worktree be left for the owner to delete. To delete it: `git worktree remove ../ojas-ft-brief-sync && git branch -d gp/ft-audit2-brief-sync`. |
| `gp/clip-sq-norm` (`2146553`) | Blocked. See below. |

There are no release branches. The only tag is `v0.1.0`.

## Blocked

**`gp/clip-sq-norm`** needs three things before it can merge:

1. **tessl's `gp/clip-sq-norm` (`2c02982`) has to land on tessl main.**
   - Until then, the ojas branch does not build. It calls `tessl::sq_norm`, and tessl is a path dependency that does not have it yet.
   - tessl-54 is merging it now. Its merge is `--no-ff` onto `42f1d52`, built in `../tessl-merge-clip` with no conflicts, and its CI and Fable review are pending. tessl-54 will report the result to ojas-1c.
2. **The diff needs a code review.** Its commit message says no one has reviewed it. The diff is +266/−180 across `ojas_backend.metal`, `device.rs` and `link.rs`.
3. **`ci:local` has to pass** on the merge.

A trial `git merge-tree main gp/clip-sq-norm` is already textually clean.

**Status at the end of this pass:** tessl's merge is not on tessl main yet.

- Its first `ci:local` failed in `qwen35_adamw_offsets_are_widened`. tessl-54 traced this to a stale text needle in `tests/shader_index_arithmetic.rs`, not a real narrowing.
- tessl-54 fixed the needle on the merge line (`tessl-merge-clip` @ `5cdf05d`) and is rerunning `ci:local`.
- It will fast-forward tessl main only while no ojas mac_heavy job holds the lock or is queued. ojas-1c told it none is.

**Update: the owner said "continue with merge".**

- **Opus review of 2146553:** approved the clip code (semantics, error propagation, no dangling kernel references, API fit with tessl 2c02982). It raised two blockers:
  - tessl has to land first.
  - The generated docs were stale (`plots -check` and `sitegen -check`).
- **Two claims, wrong after merge:**
  - `README.md` called the replaced kernel's 7.07 ms "current"; the shipped `sq_norm` takes 7.45 ms.
  - The "about 1.2× torch" ratio is about 1.13× with `sq_norm`.
- **Meanwhile on main:** ojas-f2 merged `gp/lora-step-rows` (`4da214b`). ojas-1c found that merge left `plots -check` red. ojas-f2 fixed it in `04277f8` and is pinning Lappi to `04277f8`.
- **Merge staged in `../ojas-merge-clip`:**
  - `baf527a`: the merge, `--no-ff` onto `04277f8`, no conflicts.
  - `a797c1d`: corrects both claims and regenerates `docs/bench-plots.md` and `site/`. Both generator checks are clean.
- **Still to do:**
  1. tessl main gets `sq_norm`.
  2. ojas-f2 reports its Lappi pin verified.
  3. `ci:local` runs on `a797c1d`.
  4. Fast-forward main if it's green. Landing this needs one more Lappi pin bump after ojas-f2's.

## Verification

- **CI:** a final `ci:local` on main means CI on `67dc393`, the exact commit main was fast-forwarded to. It passed (the run above) and was not repeated.
- **`git fsck`:** exit 0, with dangling objects only.
- **GitPulse Insights after the pass:** every facet is `ok`.
  - Live sessions 26207 and 39222 are both in the main checkout, which is intact.
  - The remaining worktrees are main, `ojas-clipnorm`, `ojas-ft-brief-sync` and `ojas-lora-rows`, all of them clean except main, which has the 11 owner files listed above.
  - No session lost its branch.

## Restore commands

```sh
cd /Users/bharath/Code/research/ojas
git branch gp/w11-oracle       refs/backup/2026-10-17/gp/w11-oracle        # 5eb4995
git branch gp/w11-readout-sync refs/backup/2026-10-17/gp/w11-readout-sync  # 1fef9ff
git branch gp/w11-scratch      refs/backup/2026-10-17/gp/w11-scratch       # cc9f5ad
git worktree add ../ojas-merge-pass2 --detach refs/backup/2026-10-17/detached/ojas-merge-pass2  # 829a571
# If the refs are gone, use the bundle:
git fetch /Users/bharath/Code/research/ojas-backups/2026-10-17/ojas-branches.bundle \
  'refs/backup/2026-10-17/*:refs/backup/2026-10-17/*'
# To undo the merge on main:
git branch -f main refs/backup/2026-10-17/main  # 881888e (only with main not checked out, or use reset from the main checkout)
```
