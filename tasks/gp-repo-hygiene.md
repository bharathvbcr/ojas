---
id: "gp-repo-hygiene"
title: "Repo hygiene: untrack scan.txt and analyze.py, drop the dead CI tessl step, give every #[ignore] a reason"
status: backlog
priority: 3
severity: low
type: chore
owner: "unassigned"
due: "none"
labels:
  - "housekeeping"
  - "ci"
  - "tests"
repositories:
  - "ojas"
planned_files:
  - "scan.txt"
  - "analyze.py"
  - ".github/workflows/test.yml"
  - "ojas-cpu/tests/redteam.rs"
  - "ojas-qwen35/tests/gpu_parity.rs"
  - "ojas-core/src/exp_exact.rs"
acceptance_criteria:
  - "scan.txt and analyze.py are no longer tracked"
  - "The four dead CI steps are removed and CI stays green"
  - "rg '#\\[ignore\\]$' finds no bare ignores in the workspace"
  - "redteam.rs either asserts on before_prefix_budget or no longer computes it"
---

# Task brief v1

## Title
Repo hygiene: untrack scan.txt and analyze.py, drop the dead CI tessl step, give every #[ignore] a reason

Task: gp-repo-hygiene
Type: chore
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: housekeeping, ci, tests

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). - `scan.txt` and `analyze.py` are tracked at the repo root. `scan.txt` is in `.gitignore:99`, but the rule has no effect while the file is tracked, so it needs `git rm --cached`. `analyze.py` is a throwaway grep script that points at a non-existent `ojas-engine/src/lib.rs`.
- The CI step "Point the absolute tessl path at the checkout" appears 4 times (`.github/workflows/test.yml:63,114,178,230`). It only acts when `ojas-metal/Cargo.toml` has an absolute tessl path, but both tessl deps are now relative (`../../tessl`), so it does nothing.
- 20 `#[ignore]` attributes carry no reason string (e.g. `ojas-qwen35/tests/gpu_parity.rs:139,152,184,227,382,497,583`, `ojas-core/src/exp_exact.rs:151,223,246`, several `ojas-cpu/tests/bench_*.rs`). Give each `#[ignore = "..."]`.
- `ojas-cpu/tests/redteam.rs:486` has `let _ = before_prefix_budget;`: the budget snapshot is taken and thrown away, so the comparison it was meant for never happens (`docs/audit-resources.md:47`, M3). Assert on it or delete it.

Local-only, not part of this task: 18 `target-*` directories and a stray `ojas_cpu-*.rcgu.o` at the root (about 3.3 GB) are already gitignored; delete them when no lane is using them.

## Acceptance criteria
- [ ] scan.txt and analyze.py are no longer tracked
- [ ] The four dead CI steps are removed and CI stays green
- [ ] rg '#\[ignore\]$' finds no bare ignores in the workspace
- [ ] redteam.rs either asserts on before_prefix_budget or no longer computes it

## Planned files
- scan.txt
- analyze.py
- .github/workflows/test.yml
- ojas-cpu/tests/redteam.rs
- ojas-qwen35/tests/gpu_parity.rs
- ojas-core/src/exp_exact.rs
