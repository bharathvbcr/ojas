---
id: "gp-test-gaps-from-audits"
title: "Close the test and robustness gaps the resource audits recorded"
status: backlog
priority: 3
severity: low
type: chore
owner: "unassigned"
due: "none"
labels:
  - "tests"
  - "ci"
  - "robustness"
repositories:
  - "ojas"
planned_files:
  - "ojas-capi/src/engine.rs"
  - "ojas-data/src/bpe.rs"
  - "ojas-data/tests/"
  - ".github/workflows/"
  - "ojas-metal/src/"
acceptance_criteria:
  - "Each bullet is either fixed with a test, or closed with a recorded reason"
  - "A scheduled CI job runs the CPU --ignored slow/soak tests"
---

# Task brief v1

## Title
Close the test and robustness gaps the resource audits recorded

Task: gp-test-gaps-from-audits
Type: chore
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
Labels: tests, ci, robustness

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). Small, independent gaps; each can land on its own:
- No test calls `ojas_engine_init` twice (`docs/audit-resources.md:57`, S5). `install_engine` guards it with a mutex (`ojas-capi/src/engine.rs:89-103`); pin that with a test.
- The tokenizer is checked against tiktoken on only 20 strings, and the test returns early without comparing ids when the rank tables are absent (`ojas-data/src/bpe.rs:28` is `"verified-20-strings"`; `docs/audit-close.md:75-77`).
- Rarely-run paths never exercised: a failed thread spawn, a real GPU hang, a successful 1 GiB checkpoint, a full 32 MiB tokenizer encode (`docs/audit-resources.md:90-103`).
- No CI job runs the CPU slow/soak `#[ignore]` tests (`ojas-core/tests/redteam_decode.rs:252`, `ojas-cpu/tests/redteam_linear.rs:535`, `redteam_linear_budget.rs:292,583`, `pool_stress.rs:211`, `ojas-cpu/src/pool.rs:928`) or the oracle golden fixture (`ojas-oracle/tests/golden.rs:356`). A nightly `--ignored` job for the CPU ones is cheap.
- Metal: no start-up check that the MPP pipelines accept 128 threads per threadgroup (`docs/pytorch-parity-plan.md:183`). tessl checks this for some kernels; `ojas-metal/src` does not.

## Acceptance criteria
- [ ] Each bullet is either fixed with a test, or closed with a recorded reason
- [ ] A scheduled CI job runs the CPU --ignored slow/soak tests

## Planned files
- ojas-capi/src/engine.rs
- ojas-data/src/bpe.rs
- ojas-data/tests/
- .github/workflows/
- ojas-metal/src/
