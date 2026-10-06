---
id: "gp-metal-ci-coverage"
title: "Run the Metal GPU tests somewhere other than a developer's laptop"
status: backlog
priority: 1
severity: high
type: chore
owner: "unassigned"
due: "none"
labels:
  - "ci"
  - "metal"
  - "needs-your-words"
repositories:
  - "ojas"
planned_files:
  - ".github/workflows/test.yml"
  - "scripts/"
  - "docs/status.md"
acceptance_criteria:
  - "A decision is recorded: self-hosted runner, or a manual run protocol"
  - "ojas-metal's tests and the ojas-qwen35 gpu_* tests run under that mechanism, and the result is recorded with a commit SHA"
  - "docs/status.md says which mechanism produced the Metal numbers it quotes"
---

# Task brief v1

## Title
Run the Metal GPU tests somewhere other than a developer's laptop

Task: gp-metal-ci-coverage
Type: chore
Status: backlog
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: ci, metal, needs-your-words

## Repositories
- ojas

## Description
Found by the 2026-10-05 gap audit of 9668bfa (code inspection; nothing executed). No Metal GPU code runs in CI. The macos job only builds `ojas-metal`'s tests and doesn't run them, and `OJAS_ALLOW_NO_GPU=1` turns other crates' Metal tests into SKIP (`.github/workflows/test.yml`, macos job). The 8 GPU tests in `ojas-qwen35/tests/gpu_parity.rs` are `#[ignore]`d. GitHub's hosted macOS runners have no usable Metal 4 device, so this is a known runner limitation. It is also the largest coverage gap in the repo: the default backend on Apple silicon is verified only by local runs.

This needs a decision from you: a self-hosted Apple-silicon runner (cost, security of running PR code), or a recorded manual protocol (a script that runs the Metal suites and writes a dated result file that status.md cites).

## Acceptance criteria
- [ ] A decision is recorded: self-hosted runner, or a manual run protocol
- [ ] ojas-metal's tests and the ojas-qwen35 gpu_* tests run under that mechanism, and the result is recorded with a commit SHA
- [ ] docs/status.md says which mechanism produced the Metal numbers it quotes

## Planned files
- .github/workflows/test.yml
- scripts/
- docs/status.md
