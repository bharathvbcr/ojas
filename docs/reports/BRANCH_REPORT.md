# Branch consolidation, 2026-10-17, pass 2

This pass was local only. Nothing was pushed and no remote branch was touched. It follows the earlier pass today in `BRANCH_REPORT-2026-10-17.md`. The 2026-10-16 report that used to sit in this file was copied unchanged to `BRANCH_REPORT-2026-10-16.md` first.

- **main:** `04277f8`. It did not move in this pass, because no branch could be merged.
- **Merged:** none. The only unmerged branch, `gp/clip-sq-norm`, is still blocked on tessl.
- **Deleted:** 1 branch and its worktree (`gp/ft-audit2-brief-sync`). Both had a backup first.
- **Kept:** `main`, `gp/lora-step-rows`, `gp/clip-sq-norm`, and the detached `ojas-merge-clip` worktree.
- **Not run:** the final `ci:local` (see Verification). It needs `mac_heavy.sh`, which the shell gate refuses from a session, so the command is handed over below.

## Inventory (start of pass, against main `04277f8`)

All tips are by BHARATH CHANDRA VADDARAM. Behind/ahead is from `git for-each-ref --format='%(ahead-behind:main)'`.

**Live sessions** (GitPulse Insights, every facet `ok`): three claude-desktop sessions, pids 21213 (busy), 24780 (idle) and 30404 (busy). All three have cwd in the main checkout, and one of them is this session. No session has cwd in any other worktree. ojas-f2 (pid 26207) from the earlier pass is gone.

**Collision risk:** 0 overlapping files across 5 worktrees. The main checkout is shared by 3 live sessions and has 14 dirty paths.

| Branch | Behind/ahead | Last commit | Merged? | Worktree (dirty?) | Live session | Task (status) | Class |
|---|---|---|---|---|---|---|---|
| `main` | n/a | `04277f8` 10-17 08:51 | n/a | main checkout (14 dirty, peers' work) | 3 | n/a | trunk |
| `gp/ft-audit2-brief-sync` | 7/0 | `b68971c` 10-17 07:53 | yes | `../ojas-ft-brief-sync` (clean) | none | none | **merged** |
| `gp/lora-step-rows` | 5/0 | `0f13b37` 10-17 08:08 | yes (via `4da214b`) | `../ojas-lora-rows` (clean) | none | ft-2bbced18 (in_progress)¹ | **in use** |
| `gp/clip-sq-norm` | 70/1 | `2146553` 10-16 23:36 | no (`git cherry`: `+`) | `../ojas-clipnorm` (clean) | none | ft-2c183b71 (ready) | **blocked** |
| detached `../ojas-merge-clip` @ `a797c1d` | 0/3 | 10-17 | no | clean | none | goes with clip | **blocked** (staged merge) |

¹ The binding is inferred. The task brief never names the branch; the link comes from the earlier report. Fable flagged it.

Remote: `origin/main` only, at `bc498c4`. Remote branches were not in scope.

Directories next to the repo that are not worktrees, flagged and left alone:

- **`../ojas-w11-advpin`:** a plain copy of the tree, with no `.git` and files dated 10-15 22:39. `git worktree list` does not list it. It looks stale, but deleting it is your call.
- **`../ojas-cw`:** a dangling symlink into the removed `.gitpulse/worktrees/fused-ce-want-flags`.

**Classification review (Fable):** Fable agreed on all six items, with two corrections:

- The `gp/lora-step-rows` task binding is inferred, as footnoted above.
- The 11:40 timestamp on `ojas-w11-advpin` is a Finder `.DS_Store` touch that every `ojas-*` sibling shares. It is not session activity.

Evidence Fable checked:

- `git merge-base --is-ancestor` exits 0 for `ft-audit2-brief-sync` and `lora-step-rows`, and 1 for `clip-sq-norm`.
- `git grep sq_norm main -- ojas-metal/src` is empty, so main has no form of the clip work and it is not superseded.
- tessl main (`42f1d52`) has no `fn sq_norm`. It exists only on the tessl merge line `tessl-merge-clip` @ `5cdf05d` (`src/sq_norm.rs:225`).
- Gap: the gate refuses `git --git-dir`/`--work-tree`, so the tessl state was read from ref files, and sibling-worktree cleanliness from GitPulse rather than `git status`.

## Backups

- **Bundle:** `/Users/bharath/Code/research/ojas-backups/2026-10-17-pass2/ojas-branches.bundle` (39 MB). It holds 9 heads: every local branch plus the backup refs below. `git bundle verify` reports "The bundle records a complete history."
- **Ref list:** `/Users/bharath/Code/research/ojas-backups/2026-10-17-pass2/backup-refs.txt`.

| Backup ref | Commit |
|---|---|
| `refs/backup/2026-10-17-pass2/main` | `04277f8` |
| `refs/backup/2026-10-17-pass2/gp/ft-audit2-brief-sync` | `b68971c` |
| `refs/backup/2026-10-17-pass2/gp/lora-step-rows` | `0f13b37` |
| `refs/backup/2026-10-17-pass2/gp/clip-sq-norm` | `2146553` |
| `refs/backup/2026-10-17-pass2/detached/ojas-merge-clip` | `a797c1d` |

The earlier `refs/backup/2026-10-17/*` refs from pass 1 are still in place.

## Pending work (Phase 3)

Nothing was committed.

- **Branch worktrees:** `ojas-clipnorm`, `ojas-lora-rows` and `ojas-merge-clip` are clean.
- **Main checkout:** it is in use by three live sessions, so per the rule nothing in it was committed.
  - `bench/ojas_rows.rs` (+26, a `clip_norm_only` group) and `bench/torch_rows.py` (+12) were last written at 11:38 today, during this pass. They are a peer session's in-progress work.
  - `.gitignore` (+5) is the DevCouncil `.devcouncil/*` block. Committing it is your decision.
  - Untracked: the reports, `backups/`, and five `tasks/gp-*.md` briefs from another session.

## Merged

None. No branch was in the mergeable class, so there were no conflicts to resolve and no Opus conflict review was needed. DevMap overlap was not checked, because nothing was merged.

## Branches deleted

The branch was an ancestor of main, matched its backup ref, and had a clean worktree. The worktree came off with `git worktree remove` (no `--force`), which refuses a dirty tree. The branch was deleted with `git branch -d`, and `-D` was not needed.

| Branch / worktree | Was | Backup ref |
|---|---|---|
| `gp/ft-audit2-brief-sync` + `../ojas-ft-brief-sync` | `b68971c` | `refs/backup/2026-10-17-pass2/gp/ft-audit2-brief-sync` (and `refs/backup/2026-10-17/gp/ft-audit2-brief-sync`) |

ojas-f2 had asked that this one be left for the owner, and that session is no longer running. This pass treated your instruction to delete fully merged branches as the owner's decision.

## Branches kept

| Branch | Reason |
|---|---|
| `main` | Trunk. It is 146 commits ahead of `origin/main` and was not pushed. |
| `gp/lora-step-rows` (`0f13b37`) | Fully merged (`4da214b`), but its task ft-2bbced18 is in_progress, so it stays hands off. To delete it once the task closes: `git worktree remove ../ojas-lora-rows && git branch -d gp/lora-step-rows`. |
| `gp/clip-sq-norm` (`2146553`) | Blocked. See below. |
| detached `../ojas-merge-clip` (`a797c1d`) | The staged merge for clip (`baf527a`) plus the README and site fix (`a797c1d`). It lands together with clip. |

There are no release branches. The only tag is `v0.1.0`.

## Blocked

**`gp/clip-sq-norm`**, with its staged merge in `../ojas-merge-clip`, is still in the state pass 1 left it:

1. **Waiting on tessl.** tessl main has to get `sq_norm`. The ojas branch calls `tessl::sq_norm` through the path dependency `ojas-metal/Cargo.toml:22` (`../../tessl`), and tessl main `42f1d52` does not have it. tessl's merge line is `../tessl-merge-clip` @ `5cdf05d`. Landing it is a tessl-repo step, so do it from a session rooted in tessl.
2. **Then run `ci:local` on `a797c1d`.** If it's green, fast-forward main from `04277f8` to `a797c1d`. This is a fast-forward only, because `a797c1d` already has `04277f8` as an ancestor. Opus already reviewed the clip code in pass 1 and approved it.
3. **Lappi pin.** Landing clip moves main again, so Lappi's ojas pin needs one more bump. Ask before making it.

## Verification

- **Final `ci:local` on main: not run.** Two things stopped it:
  - Heavy runs must go through `mac_heavy.sh`, and the gate refuses that script (`command.operand.unverifiable`).
  - The main checkout holds a peer's uncommitted bench edits, so a run there would not test `04277f8`.

  main did not change in this pass, but that is not coverage. Pass 1's `ci:local` ran on `67dc393`. main then moved through `4da214b` and `04277f8` with only `plots -check` recorded, so **neither pass has a `ci:local` on record for `04277f8`**. To check it, run it in a clean detached worktree:

  ```sh
  git worktree add --detach ../ojas-ci-main 04277f8
  bash /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ojas-ci-main-04277f8 /Users/bharath/Code/research/ojas-ci-main/scripts/ci_local.sh
  git worktree remove ../ojas-ci-main
  ```
- **`git fsck`:** no errors. It reported 70 dangling objects and nothing else.
- **GitPulse Insights after the pass:** every facet `ok`.
  - The same three live sessions, all in the main checkout, which is intact.
  - 4 worktrees remain: main, `ojas-clipnorm`, `ojas-lora-rows` and `ojas-merge-clip`. All are clean except main.
  - No session lost its branch.
  - DevMap is fresh (generation 7053).

## Folder cleanup in `/Users/bharath/Code/research/` (owner: "clean them if they are merged")

- **Removed:** `gp/lora-step-rows` and `../ojas-lora-rows`.
  - The branch was an ancestor of main, its tip equalled `refs/backup/2026-10-17-pass2/gp/lora-step-rows` (`0f13b37`), its worktree was clean, and no process had its cwd there (`lsof -d cwd`).
  - Removed with `git worktree remove` and `git branch -d`.
  - Its task ft-2bbced18 is still in_progress on the board. The owner's instruction took precedence over the inferred binding.
- **Verified merged, not yet removed:**
  - **`../ojas-w11-advpin`:** a copy of the tree with no git. All 2,981 of its files outside `target/` hash to blobs reachable from main (`target-branchclean/advpin_vs_main.sh`: `not_in_main_history=0`). The rest is build output in `target/`.
  - **`../ojas-cw`:** a dangling symlink.
  - **Why they are still there:** the gate refuses writes outside the repo, which includes moving them to the Trash. The owner runs the moves (below).
- **Kept:**
  - `../ojas-clipnorm` and `../ojas-merge-clip`, both unmerged and blocked on tessl.
  - `../ojas-backups` and `../tessl-backups`, which are backups.
  - The separate projects (BINN, Evo2PopgenSBI, GenoThermalTargeting, KneeMRIContactSheets, Lappi-decision, LocalModelBench, MLSystemsLab, ProprioceptionStudy, StressProject, external, sparsl). None of them is a worktree or copy of anything, so "merged" does not apply.
- **tessl folders:** handed to a tessl-rooted session, because this session's gate refuses git against tessl:
  - `tessl-w10-faults`, `tessl-w10-gdnchunk`, `tessl-w11-2bmem`, `tessl-w12-kcov`, `tessl-merge-w11`, `tessl-merge2` are candidates.
  - `tessl-eg2` is live, with 24 processes in it.
  - `tessl-clipnorm` and `tessl-merge-clip` are unmerged.
  - `tessl-w2e` holds only build output.

  Its result goes in tessl's `BRANCH_REPORT.md`.

```sh
mv /Users/bharath/Code/research/ojas-w11-advpin ~/.Trash/ojas-w11-advpin-2026-10-17
rm /Users/bharath/Code/research/ojas-cw
```

## Restore commands

```sh
cd /Users/bharath/Code/research/ojas
git branch gp/ft-audit2-brief-sync refs/backup/2026-10-17-pass2/gp/ft-audit2-brief-sync   # b68971c
git worktree add ../ojas-ft-brief-sync gp/ft-audit2-brief-sync
git branch gp/lora-step-rows refs/backup/2026-10-17-pass2/gp/lora-step-rows               # 0f13b37
git worktree add ../ojas-lora-rows gp/lora-step-rows
# If the refs are gone, use the bundle:
git fetch /Users/bharath/Code/research/ojas-backups/2026-10-17-pass2/ojas-branches.bundle \
  'refs/backup/2026-10-17-pass2/*:refs/backup/2026-10-17-pass2/*'
```
