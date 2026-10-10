---
id: "gp-long-runs-and-quiet-benches"
title: "Owed long runs and quiet benchmarks: 124M v1 acceptance run, real-2B Qwen3.5 Tape run and headroom, quiet GPU-vs-torch rerun, typed-storage parity, embedding regression"
status: backlog
priority: 2
severity: medium
type: test
owner: "unassigned"
due: "none"
labels:
  - "bench"
  - "acceptance"
  - "metal"
  - "qwen35"
  - "measurement"
repositories:
  - "ojas"
planned_files:
  - "bench/results/"
  - "docs/status.md"
  - "docs/bench-gpu-vs-torch.md"
  - "docs/typed-storage-plan.md"
  - "ojas-qwen35/README.md"
  - "ojas-qwen35/tests/gpu_tape_2b.rs"
  - "README.md"
acceptance_criteria:
  - "The v1 acceptance run is executed and recorded: Metal 124M (ModelSpec::nanolab_124m) against PyTorch nanolab per docs/framework-design.md section 10 (fp32, compile off, B=4 K=4 T=1024, warmup 30, cosine over 300 steps, shared init.safetensors and dumped batch starts, NS5 patched to fp32), with every pass criterion in that table reported as met or missed; results committed under bench/results/ and README/docs/status.md stop saying no full 124M run is recorded"
  - "A full quiet GPU-vs-torch re-run is committed under bench/results/ with spread per row; rows still over 10% are marked as such in docs"
  - "parity.sh, block_ab.sh and the sample profile are run for typed storage (after gp-docs-pipeline-single-source recovers or rewrites the scripts from the deleted target-matmul/), and the result is recorded in docs/typed-storage-plan.md"
  - "The ~9% embedding-forward regression in typed storage step 1 is explained or fixed, with the run cited"
  - "During the 124M run, waits and memory-cap commits per step are counted by trigger (Link Wait counters), closing the inferred figure in gp-gpu-runtime-hardening"
---

# Task brief v1

## Title
Owed long runs and quiet benchmarks: 124M v1 acceptance run, real-2B Qwen3.5 Tape run and headroom, quiet GPU-vs-torch rerun, typed-storage parity, embedding regression

Task: gp-long-runs-and-quiet-benches
Type: test
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: bench, acceptance, metal, qwen35, measurement

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949). These items are execution, not code. They take hours of GPU time or a quiet machine, so they were split out of the briefs that held them. Without this split, those briefs could not close, and these runs would stay invisible behind code work. Queue every run through Lappi-decision/tools/mac_heavy.sh.

Sources of each item:
- **124M v1 acceptance run:** from gp-oracle-and-hardening-coverage, per docs/framework-design.md section 10.
- **Real-2B Tape run and 2B save/load and long-sequence headroom:** from gp-autograd-and-model-primitives. ojas-qwen35/tests/gpu_tape_2b.rs:96 exists but has never run, and ojas-qwen35/README.md:230-235 says 'Not measured'.
- **Quiet GPU-vs-torch rerun, typed-storage parity.sh/block_ab.sh and the ~9% embedding-forward regression:** from gp-docs-and-bench-parity.
- **The 124M wait split by trigger:** from gp-gpu-runtime-hardening's 'Not measured' line, measured under gp-device-memory-probes.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- The real-2B Metal Tape run moved to gp-qwen35-2b-metal-tape-run at P1, because it is on the fine-tune path. The other runs stay P2.
- Fine-tune briefs that depend on this one: gp-ft-step-memory-and-throughput.

## Acceptance criteria
- [ ] The v1 acceptance run is executed and recorded: Metal 124M (ModelSpec::nanolab_124m) against PyTorch nanolab per docs/framework-design.md section 10 (fp32, compile off, B=4 K=4 T=1024, warmup 30, cosine over 300 steps, shared init.safetensors and dumped batch starts, NS5 patched to fp32), with every pass criterion in that table reported as met or missed; results committed under bench/results/ and README/docs/status.md stop saying no full 124M run is recorded
- [ ] A full quiet GPU-vs-torch re-run is committed under bench/results/ with spread per row; rows still over 10% are marked as such in docs
- [ ] parity.sh, block_ab.sh and the sample profile are run for typed storage (after gp-docs-pipeline-single-source recovers or rewrites the scripts from the deleted target-matmul/), and the result is recorded in docs/typed-storage-plan.md
- [ ] The ~9% embedding-forward regression in typed storage step 1 is explained or fixed, with the run cited
- [ ] During the 124M run, waits and memory-cap commits per step are counted by trigger (Link Wait counters), closing the inferred figure in gp-gpu-runtime-hardening

## Planned files
- bench/results/
- docs/status.md
- docs/bench-gpu-vs-torch.md
- docs/typed-storage-plan.md
- ojas-qwen35/README.md
- ojas-qwen35/tests/gpu_tape_2b.rs
- README.md
