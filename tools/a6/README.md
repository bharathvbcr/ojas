# tools/a6: ojas-a6 orchestration scripts (2026-10-17 to 2026-10-19), kept as a record

These are the scripts session ojas-a6 used to build, verify and land the LoRA-2B and quantised-arm work: phase A (Lappi qd-train imported into ojas-train) and phase B (the Qwen3.5 Tape `StepProvider`). The run's report is `docs/reports/BUILD_REPORT-a6.md`.

**They are a record, not a supported tool.**
- They hard-code absolute paths (`/Users/bharath/Code/research/...`, and `backups/a6-tools/` as the working copy and log directory).
- Several target worktrees that no longer exist (`../ojas-a6-*`, `../ojas-phase-a`, `../ojas-phase-b`).

Card **ft-0e1d7ad5** replaces them with a tracked, gate-aware verification harness. Until then, read a script before reusing it.

The tessl commit names in these scripts' comments (`bf091a2`, `a37a0c2`, `41d8866`) predate tessl's 2026-10-18 history rewrite. Their current names are in the table at the top of `docs/reports/BUILD_REPORT-a6.md`.

| Group | Files | What they do |
|---|---|---|
| Board | `gp.sh` | GitPulse MCP over stdio; `@file` for JSON args |
| Board | `cards.sh`, `digest.py`, `summ.py` | read cards |
| Board | `dups.py`, `new-cards.py`, `final-*.py` | board writes |
| Verification | `launch-verify.sh`, `launch-rerun.sh` | detached nohup launchers, with a pid file and DONE sentinel |
| Verification | `owner-verify-main.sh`, `owner-verify-rerun.sh`, `phase-*-verify.sh` | step lists under `mac_heavy.sh`, with a per-run `CARGO_TARGET_DIR` and perl alarms |
| Commits | `phase-a-commit.sh`, `phase-b-commit.sh` | commit only in one named worktree, on one branch |
| Commits | `owner-commit-q1.sh`, `owner-merge.sh`, `owner-batch.sh` | the early owner-run batches; obsolete after the lane worktrees were removed |
| Phase A | `owner-phase-a-import.sh` | the history-preserving import of Lappi qd-train, via git-filter-repo on a fresh clone |
| Bench | `owner-bench-stepcost.sh` | the ft-2bbced18 A/B rows (not yet run) |
| Diagnostics | `load-sources.py` | CPU load grouped by working directory |
| Diagnostics | `newest.py` | newest file per worktree, used before deleting worktrees |
| Hand-offs | `LAPPI-WORK-ORDER.md`, `HANDOFF-for-lappi.md` | what Lappi must do, with the ojas contracts |
| Hand-offs | `lane-rules.md` | the rules every lane agent followed |

The pitfalls these scripts hit (a keep-alive `touch` that recreated a deleted target directory as a file, an ignored pathspec aborting `git add`, the default `MAC_HEAVY_MAX_LOAD` of 12, the cargo-slot gate, ENOSPC) are listed in `docs/reports/BUILD_REPORT-a6.md`.
