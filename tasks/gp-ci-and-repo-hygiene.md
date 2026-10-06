---
id: "gp-ci-and-repo-hygiene"
title: "Unify CI Automation, Test Parity, Multi-Threaded Readback Accounting, and Repo Hygiene"
status: ready
priority: 1
severity: high
type: chore
owner: "unassigned"
due: "none"
labels:
  - "ci"
  - "tests"
  - "testing"
  - "concurrency"
  - "budget"
  - "metal"
  - "build"
  - "housekeeping"
  - "robustness"
repositories:
  - "ojas"
planned_files:
  - ".github/workflows/test.yml"
  - "scan.txt"
  - "analyze.py"
  - "ojas-cpu/tests/redteam.rs"
  - "ojas-qwen35/tests/gpu_parity.rs"
  - "ojas-core/src/exp_exact.rs"
  - "ojas-wgpu/tests/muon.rs"
  - "ojas-wgpu/tests/permute.rs"
  - "ojas-wgpu/tests/residency.rs"
  - "ojas-wgpu/tests/parity.rs"
  - "ojas-autograd/tests/device_tape.rs"
  - "ojas-autograd/tests/wgpu_tape.rs"
  - "ojas-autograd/Cargo.toml"
  - "ojas-capi/src/tests.rs"
  - "ojas-capi/src/engine.rs"
  - "ojas-metal/tests/backend_contract.rs"
  - "ojas-data/src/bpe.rs"
  - "ojas-data/tests/"
  - "ojas-metal/Cargo.toml"
  - "ojas-qwen35/Cargo.toml"
  - "ojas-capi/Cargo.toml"
  - "ojas-gusset-engine/Cargo.toml"
  - "README.md"
  - "docs/status.md"
acceptance_criteria:
  - "scan.txt and analyze.py are no longer tracked in git"
  - "The four dead CI steps in .github/workflows/test.yml are removed and CI stays green"
  - "rg finds no bare ignores across workspace (every #[ignore] has an explicit documented reason)"
  - "redteam.rs either asserts on before_prefix_budget or no longer computes it"
  - "Migrate GPU test readback assertions from process-wide device_readbacks() to Budget::device_readbacks() scoped per budget tree (resolve remaining ojas-capi/src/tests.rs site)"
  - "Verify all GPU tests (wgpu, autograd, capi, metal) pass cleanly under default multi-threaded test execution (cargo test without --test-threads=1)"
  - "Remove --test-threads=1 restriction from README.md"
  - "A decision and protocol are recorded for Metal GPU test execution (runner or manual protocol), and results are recorded in docs/status.md with a commit SHA"
  - "Pin tessl and gusset revisions in CI workflows and document pin bump procedure"
  - "Add optional target-gated ojas-metal dev-dependency on macOS in ojas-autograd and wire into G6 seeded backward tests"
  - "De-duplicate test helpers (Resident, Rng, lin, ce) across ojas-autograd test suites"
  - "Pin ojas_engine_init double initialization mutex safety with a dedicated test"
  - "A scheduled CI job runs the CPU --ignored slow/soak tests"
---

# Task brief v1

## Title
Unify CI Automation, Test Parity, Multi-Threaded Readback Accounting, and Repo Hygiene

Task: gp-ci-and-repo-hygiene
Type: chore
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: ci, tests, testing, concurrency, budget, metal, build, housekeeping, robustness

## Repositories
- ojas

## Description
Consolidated task combining test infrastructure, CI automation, concurrency safety, dependency pinning, and repository hygiene:
- Repo hygiene and dead step cleanup (`gp-repo-hygiene`)
- Resource audit test gaps (`gp-test-gaps-from-audits`)
- Multi-threaded readback accounting migration to `Budget::device_readbacks` (`gp-gpu-test-budget-readbacks`)
- Metal CI testing protocol (`gp-metal-ci-coverage`)
- Pinned sibling dependency revisions (`gp-pin-sibling-deps`)
- Autograd Metal tape test dev-dependency (`gp-autograd-metal-tape-tests`)

### Audit status and progress notes (2026-10-05)
1. **Housekeeping:** `scan.txt` (listed in `.gitignore`) and `analyze.py` (throwaway script) are untracked at repo root. Four dead CI steps remain in `.github/workflows/test.yml`. Bare `#[ignore]` attributes exist without explanatory strings.
2. **Readback Race Elimination:** Global `AtomicU64` `device_readbacks()` caused test download races under parallel `cargo test`. `Budget::device_readbacks()` resolved this for wgpu, Metal, and autograd. Only one site remains: `ojas-capi/src/tests.rs:1796-1800`. Once migrated, `README.md` can drop `--test-threads=1`.
3. **Metal CI Coverage:** No Metal GPU tests run in CI; macOS job only compiles. Establish runner protocol or dedicated validation mechanism.
4. **Sibling Dependency Pinning:** `tessl` and `gusset` paths float against unpinned local checkouts; pin tested git revisions in CI.
5. **Autograd Metal Dev-Dependency:** `ojas-autograd` lacks target-gated `ojas-metal` dev-dependency on macOS, skipping G6 backward tests on Metal.

### Execution plan
- **Phase 1 (Hygiene & Readbacks):** Remove tracked `scan.txt` / `analyze.py`. Migrate `ojas-capi/src/tests.rs` to tree-scoped budget reader, verify parallel `cargo test`, and update README. Clean up dead CI steps.
- **Phase 2 (Dependency & CI Pinning):** Pin `tessl` and `gusset` in CI. Add scheduled slow/soak test workflow. Document Metal test runner protocol.
- **Phase 3 (Test Coverage & Gaps):** Add Metal dev-dependency to `ojas-autograd`, wire G6 seeded backward tests, de-duplicate autograd helpers, and add double `ojas_engine_init` test.

## Acceptance criteria
- [ ] scan.txt and analyze.py are no longer tracked in git
- [ ] The four dead CI steps in .github/workflows/test.yml are removed and CI stays green
- [ ] rg '#\[ignore\]$' finds no bare ignores across workspace (every ignore has an explicit reason)
- [ ] redteam.rs either asserts on before_prefix_budget or no longer computes it
- [ ] Migrate GPU test readback assertions from process-wide device_readbacks() to Budget::device_readbacks() scoped per budget tree (resolve remaining ojas-capi/src/tests.rs site)
- [ ] Verify all GPU tests (wgpu, autograd, capi, metal) pass cleanly under default multi-threaded test execution (cargo test without --test-threads=1)
- [ ] Remove --test-threads=1 restriction from README.md
- [ ] A decision and protocol are recorded for Metal GPU test execution (runner or manual protocol), and results are recorded in docs/status.md with a commit SHA
- [ ] Pin tessl and gusset revisions in CI workflows and document pin bump procedure
- [ ] Add optional target-gated ojas-metal dev-dependency on macOS in ojas-autograd and wire into G6 seeded backward tests
- [ ] De-duplicate test helpers (Resident, Rng, lin, ce) across ojas-autograd test suites
- [ ] Pin ojas_engine_init double initialization mutex safety with a dedicated test
- [ ] A scheduled CI job runs the CPU --ignored slow/soak tests

## Planned files
- .github/workflows/test.yml
- scan.txt
- analyze.py
- ojas-cpu/tests/redteam.rs
- ojas-qwen35/tests/gpu_parity.rs
- ojas-core/src/exp_exact.rs
- ojas-wgpu/tests/muon.rs
- ojas-wgpu/tests/permute.rs
- ojas-wgpu/tests/residency.rs
- ojas-wgpu/tests/parity.rs
- ojas-autograd/tests/device_tape.rs
- ojas-autograd/tests/wgpu_tape.rs
- ojas-autograd/Cargo.toml
- ojas-capi/src/tests.rs
- ojas-capi/src/engine.rs
- ojas-metal/tests/backend_contract.rs
- ojas-data/src/bpe.rs
- ojas-data/tests/
- ojas-metal/Cargo.toml
- ojas-qwen35/Cargo.toml
- ojas-capi/Cargo.toml
- ojas-gusset-engine/Cargo.toml
- README.md
- docs/status.md
