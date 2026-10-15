---
id: "gp-long-runs-and-quiet-benches"
title: "Owed long runs and quiet benchmarks: 124M v1 acceptance run, quiet GPU-vs-torch rerun, typed-storage parity, embedding regression"
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
Owed long runs and quiet benchmarks: 124M v1 acceptance run, quiet GPU-vs-torch rerun, typed-storage parity, embedding regression

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
- ~~Real-2B Tape run and 2B save/load and long-sequence headroom~~: struck 2026-10-15 (Fable). The 2B Tape gate is gp-qwen35-2b-metal-tape-run; the provider save/load row is retired there; long-sequence headroom is the Tape memory rows (bench/results/2026-10-11-bf16-frozen-base, 2026-10-14-bf16-tied-table).
- **Quiet GPU-vs-torch rerun, typed-storage parity.sh/block_ab.sh and the ~9% embedding-forward regression:** from gp-docs-and-bench-parity.
- **The 124M wait split by trigger:** from gp-gpu-runtime-hardening's 'Not measured' line, measured under gp-device-memory-probes.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- The real-2B Metal Tape run moved to gp-qwen35-2b-metal-tape-run at P1, because it is on the fine-tune path. The other runs stay P2.
- Fine-tune briefs that depend on this one: gp-ft-step-memory-and-throughput.

### Handed over by gp-device-memory-probes (2026-10-10)
- `WaitCounts` gained `freed` (frees past the memory cap wait for a commit). One release run of 3 steps per tree, not interleaved, at nanolab 124M B=4 T=1024 K=4 on a 16 GiB budget: HEAD mem_cap 41, total 47; with the change mem_cap 38, freed 7, total 51; 3.65-3.66 s a step on both. The quiet interleaved run here should carry the `freed` column.
- `ojas-model/tests/metal_waits.rs` `nanolab_124m_step_waits_by_trigger` (ignored, release) fails on HEAD b146559 and with the change alike: 4 upload waits a step where it asserts 0. The ids and targets are 16 KiB each and fit `inline_upload_fits` (ojas-metal/src/device.rs), so another per-micro-batch upload is the likely one, perhaps one over the 64 KiB inline limit; inferred, not traced. `WaitCounts` records no sizes, so `upload` needs instrumenting to find it.

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
- README.md

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Fine-tune scope only (the real-2B items). The real-2B Tape run was split to gp-qwen35-2b-metal-tape-run on 2026-10-09; this brief still lists gpu_tape_2b.rs and ojas-qwen35/README.md in its planned files [A]. Propose: strike the 2B item here so it is not counted twice.
- Headroom: the only figure is README:213 (1.34 GB at the provider, 128 tokens); it is not comparable to the Tape rows (16-23 GB of 51.5 GB) [A].

### Fable rulings (2026-10-15)
Fable ruled on the audit's open decisions on 2026-10-15; the owner decided two items directly (Lappi pin bump, hand-rolled sha256).
- **Struck:** the real-2B Tape run, the 2B save/load row and long-sequence headroom, with gpu_tape_2b.rs and ojas-qwen35/README.md from the planned files. They live in gp-qwen35-2b-metal-tape-run (the save/load row is retired there).
