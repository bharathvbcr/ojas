---
id: "gp-repo-hygiene-gemma-metal-and-devcouncil-sync"
title: "Fix gemma-metal build.rs panic, add CI/DevCouncil verification config, and converge duplicated safetensors/Muon/KV crates"
status: ready
priority: 2
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "ci"
  - "hygiene"
  - "gemma-metal"
  - "devcouncil"
  - "dedup"
  - "audit-2026-10-16"
repositories:
  - "ojas"
  - "tessl"
planned_files:
  - "gemma-metal/build.rs"
  - "Cargo.toml"
  - "scripts/ci_local.sh"
  - ".devcouncil/config.yaml"
  - "ojas-io/src/safetensors.rs"
  - "ojas-infer/src/lib.rs"
acceptance_criteria:
  - "Cherry-pick 0906ccc from gp/w9-icb to fix gemma-metal/build.rs:41 panicking on missing gelu.h (renamed to act.h in tessl), and gate gemma-metal and arch02 builds under mac_heavy.sh / scripts/ci_local.sh (BRANCH_REPORT.md)"
  - "Hold fb907e3 (replacing tessl icb_stub()) on gp/w9-icb until tessl gp/wave10 lands on tessl main, then merge and verify"
  - "Configure .devcouncil/config.yaml in ojas and tessl so dev verify runs cleanly and gates moving GitPulse cards from review to done (BUILD_REPORT.md Section 6)"
  - "Remove the dangling /Users/bharath/Code/research/ojas-cw symlink and converge duplicated safetensors (ojas-io vs tessl/qd-metal/qd-export), Muon (ojas-cpu/ojas-metal vs metal-native/burn-port), and KV/sampling implementations (docs/pytorch-parity-plan.md Section 2)"
---

# Task brief v1

## Title
Fix gemma-metal build.rs panic, add CI/DevCouncil verification config, and converge duplicated safetensors/Muon/KV crates

Task: gp-repo-hygiene-gemma-metal-and-devcouncil-sync
Type: chore
Status: ready
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: ci, hygiene, gemma-metal, devcouncil, dedup, audit-2026-10-16

## Repositories
- ojas
- tessl

## Description
Consolidates the repository build, verification tooling, and cross-crate deduplication gaps documented in `ojas/BRANCH_REPORT.md` (2026-10-16), `ojas/BUILD_REPORT.md` (Section 6), and `ojas/docs/pytorch-parity-plan.md` (Section 2):

1. **Broken `gemma-metal/build.rs:41` on `main` (`BRANCH_REPORT.md`):** `gemma-metal/build.rs:41` currently panics on `ojas` `main` because `tessl` renamed `gelu.h` to `act.h`. Commit `0906ccc` on `gp/w9-icb` fixes the header path, while `gemma-metal` and `arch02` are currently excluded from `Cargo.toml` workspace members and never built by `scripts/ci_local.sh`.
2. **Missing DevCouncil Verification Config (`BUILD_REPORT.md`):** `dev verify` currently fails with `"DevCouncil state is unavailable in this directory"` because `.devcouncil/config.yaml` is not configured and verification mode is `off`, preventing reviewed cards from advancing to `done`.
3. **Workspace Duplication & Stale Symlink (`pytorch-parity-plan.md`, `BRANCH_REPORT.md`):** `/Users/bharath/Code/research/ojas-cw` dangles after worktree cleanup, and duplicate implementations of `safetensors`, `Muon`, and KV/sampling remain spread across `ojas`, `tessl`, `metal-native`, `gemma-metal`, and dead vendored `Rust_MLKit/crates/tessl`.

## Acceptance criteria
- [ ] Cherry-pick 0906ccc from gp/w9-icb to fix gemma-metal/build.rs:41 panicking on missing gelu.h (renamed to act.h in tessl), and gate gemma-metal and arch02 builds under mac_heavy.sh / scripts/ci_local.sh (BRANCH_REPORT.md)
- [ ] Hold fb907e3 (replacing tessl icb_stub()) on gp/w9-icb until tessl gp/wave10 lands on tessl main, then merge and verify
- [ ] Configure .devcouncil/config.yaml in ojas and tessl so dev verify runs cleanly and gates moving GitPulse cards from review to done (BUILD_REPORT.md Section 6)
- [ ] Remove the dangling /Users/bharath/Code/research/ojas-cw symlink and converge duplicated safetensors (ojas-io vs tessl/qd-metal/qd-export), Muon (ojas-cpu/ojas-metal vs metal-native/burn-port), and KV/sampling implementations (docs/pytorch-parity-plan.md Section 2)

## Planned files
- gemma-metal/build.rs
- Cargo.toml
- scripts/ci_local.sh
- .devcouncil/config.yaml
- ojas-io/src/safetensors.rs
- ojas-infer/src/lib.rs
