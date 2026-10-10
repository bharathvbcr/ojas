---
id: "gp-imported-research-trees"
title: "Decide the imported research trees arch02/ and gemma-metal/: 932 tracked files and 18-20 MB outside the workspace, never built in CI, absolute local paths, a Python HTTP server, .wip files"
status: backlog
priority: 2
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "housekeeping"
  - "ci"
  - "python-policy"
  - "needs-your-words"
repositories:
  - "ojas"
planned_files:
  - "arch02/"
  - "gemma-metal/"
  - "Cargo.toml"
  - ".github/workflows/test.yml"
  - "scripts/sitegen/main.go"
acceptance_criteria:
  - "A recorded decision per tree: its own repository (history preserved, link left in README) or kept here as a workspace member/exclude with a CI build job"
  - "Whatever stays builds on a fresh clone: no absolute /Users paths in any tracked Cargo.toml"
  - "Whatever stays follows the language policy: the Python HTTP server and analysis scripts are removed, ported, or confined to throwaway status with the reason; duplicate scripts and .wip files are gone"
---

# Task brief v1

## Title
Decide the imported research trees arch02/ and gemma-metal/: 932 tracked files and 18-20 MB outside the workspace, never built in CI, absolute local paths, a Python HTTP server, .wip files

Task: gp-imported-research-trees
Type: chore
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: housekeeping, ci, python-policy, needs-your-words

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949). Both research trees came in with 1fc6a03 [A, C]:
- **Size:** 932 tracked files and 108,608 lines (about 16M plus 3.8M on disk).
- **Outside the build:** each has its own `[workspace]`, CI never builds it, and `rg -uu` finds no ojas code that references it.
- **Machine-specific:** dependency paths are absolute (/Users/bharath/Code/research/tessl at arch02/metal-native/Cargo.toml:29 and gemma-metal/Cargo.toml:15), so neither builds on any other machine.
- **Unaudited dependencies:** arch02/burn-port pulls burn 0.21, candle-core, serde and rand, which the root lockfile and any dependency audit never see.
- **Python policy:** there are 27 Python files. gemma-metal/bench/serve_dflash.py is an HTTP server, a shipped-service shape the language policy forbids. gemma-metal/bench/ddtree_core.py is duplicated under bench/results/.
- **Leftovers:** two tracked .wip files (gemma-metal/src/verify_batch_impl.rs.wip, kernels/gemm_q4_mlx.metal.wip).
- **Mislabelled on the site:** sitegen publishes gemma-metal/README.md as an ojas crate page.

The decision is the user's: move each to its own repository, or make it a workspace member or exclude with a CI job.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- The crate-page criterion was owned twice; gp-docs-pipeline-single-source owns it now.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P2. Note for the decision: gemma-metal holds the only existing quantizer, which the gated gp-ft-quantized-frozen-base and gp-ft-other-base-architectures briefs depend on.
- Fine-tune briefs that depend on this one: gp-ft-quantized-frozen-base, gp-ft-other-base-architectures.

## Acceptance criteria
- [ ] A recorded decision per tree: its own repository (history preserved, link left in README) or kept here as a workspace member/exclude with a CI build job
- [ ] Whatever stays builds on a fresh clone: no absolute /Users paths in any tracked Cargo.toml
- [ ] Whatever stays follows the language policy: the Python HTTP server and analysis scripts are removed, ported, or confined to throwaway status with the reason; duplicate scripts and .wip files are gone

## Planned files
- arch02/
- gemma-metal/
- Cargo.toml
- .github/workflows/test.yml
- scripts/sitegen/main.go
