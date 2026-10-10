---
id: "gp-qwen35-above-2b"
title: "Qwen3.5 above 2B in ojas: GDN value-head grouping, untied head, newer config keys, bf16 storage with a measured memory model"
status: backlog
priority: 2
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "qwen35"
  - "cuda"
  - "metal"
  - "bf16"
  - "memory"
  - "checkpoint"
repositories:
  - "ojas"
planned_files:
  - "ojas-qwen35/src/step.rs"
  - "ojas-qwen35/src/state.rs"
  - "ojas-qwen35/src/config.rs"
  - "ojas-model/src/load.rs"
  - "ojas-cuda/src/gdn_plan.rs"
  - "ojas-oracle/"
acceptance_criteria:
  - "Native GDN key-head repetition in forward and backward on the CUDA plan (published rule), with f64 goldens that include backward and do not pre-expand K; fail-first test on a 16/32 head config"
  - "Untied LM head: its own parameter and gradient entry; tied path bit-identical"
  - "Newer config keys classified (accepted with meaning, or refused with a named reason); CUDA RMSNorm backward tiled past 4096"
  - "bf16 storage for weights, gradient bank and saved per-layer inputs with f32 accumulation; moments f32 or bf16-with-compensation, selectable and recorded in describe() and state headers"
  - "Measured peak allocation at 2B against the 16 B/param baseline; a pre-registered loss-divergence bound against the f32 arm; bit-exact save/resume"
  - "The official 4B Base config.json is pinned by sha in fixtures and parses"
  - "ojas-qwen35 reads, saves and loads state per entry through tessl's host-mapped reads once tessl ships them; staging_tensors/device_copies are deleted; the 2B gradient-read, save_state and load_state peaks are measured and recorded against the 1.34 GB headroom baseline"
  - "save_state checks free disk space (ojas_device::available_disk_bytes, as ojas-model's checkpoint does) before writing and refuses with a typed error that names the bytes needed and available; a test proves the refusal writes nothing"
  - "Qwen35Step::open sets tessl's pool cache cap explicitly (step.rs:341 leaves the 2 GiB default, which tessl's step_bytes charges against admission) and records it in describe(); the 2B maximum admissible sequence length is measured before and after"
  - "Loading state opens each shard once (StateIndex::read_entry reopens SafeTensors per entry, state.rs:~455), folded into the per-entry read rework"
  - "The tied-head check in ojas-model/src/load.rs:122 reports the real header error (missing entry, dtype, shape, I/O) instead of mapping every failure to 'untied head not supported'"
  - "4B is out of the fine-tune track (Fable D4, user-approved 2026-10-09); the 4B criteria stay dormant until a box campaign needs them"
---

# Task brief v1

## Title
Qwen3.5 above 2B in ojas: GDN value-head grouping, untied head, newer config keys, bf16 storage with a measured memory model

Task: gp-qwen35-above-2b
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: high
Owner: unassigned
Due: none
Labels: qwen35, cuda, metal, bf16, memory, checkpoint

## Repositories
- ojas

## Description
Mirrors GitPulse board card ft-c57c98e0bfcbce549ce3f9057160c63b (backlog, p1, updated 2026-10-07). It had no brief in tasks/. The card's eleven criteria are carried verbatim and nothing was re-audited against code in this pass.

Sibling session ojas-7c (fine-tune track) will send additive criteria if the 4B base is chosen: bf16 or quantized frozen-base storage at the 4B shape, and a measured LoRA-step memory row at sequence length 2048.

Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Criteria from the fine-tune track (2026-10-09)
Sent by sibling session ojas-7c (QLoRA/LoRA fine-tune of Lappi on Mac). The code claims were checked by this session [V]: ojas-model/src/trainer.rs:858-861 refuses a trainable parameter with no gradient; ojas-qwen35/src/names.rs:286-296 refuses a tower split across files; clip_scale (ojas-core/src/backend.rs:170-174) returns max_norm/(norm+eps) clamped to 1, so max_norm 0 zeroes every gradient. The HF schedule formula is to be pinned against the installed transformers source, not memory.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P1, conditional on the user choosing the 4B base.
- Fine-tune briefs that depend on this one: gp-ft-end-to-end-acceptance (if 4B).

### Fine-tune decisions (2026-10-09)
Fable ruled on the fine-tune track's decisions and the user approved them (2026-10-09), as relayed by sibling session ojas-7c. The record is in that session's scratchpad: fable-ft-decisions-2026-10-09.md, sections D1-D6.
- Tape path for LoRA.
- bf16 frozen base; no 4/8-bit.
- 2B-Base, with a 0.8B-Base smoke; 4B is out of the track.
- Lappi's data shape: rows up to 9,638 tokens, a 35,403-token budget, grad_accum 1.
- Lappi's own LRSchedule.
- The trainer moves into a new ojas-train crate.
Checked by this session [V]: the local Qwen3.5-2B snapshot exists (4.3 GB). ojas-qwen35/src/config.rs:503-508 still refuses unequal GDN value and key heads with 'tessl's gdn_train has no head grouping', while ../tessl/src/qwen35_train.rs now groups heads.
- Priority 1 -> 2: 4B is out of the fine-tune track.
- The two '(Only if the 4B base is chosen)' criteria are replaced by a dormancy note.
- The stale GDN value-head grouping refusal message (ojas-qwen35/src/config.rs:503-508) is fixed as a true-message item in gp-ft-training-path-correctness (P0). Criterion 1 here (native key-head repetition on the CUDA plan) is unchanged.

## Acceptance criteria
- [ ] Native GDN key-head repetition in forward and backward on the CUDA plan (published rule), with f64 goldens that include backward and do not pre-expand K; fail-first test on a 16/32 head config
- [ ] Untied LM head: its own parameter and gradient entry; tied path bit-identical
- [ ] Newer config keys classified (accepted with meaning, or refused with a named reason); CUDA RMSNorm backward tiled past 4096
- [ ] bf16 storage for weights, gradient bank and saved per-layer inputs with f32 accumulation; moments f32 or bf16-with-compensation, selectable and recorded in describe() and state headers
- [ ] Measured peak allocation at 2B against the 16 B/param baseline; a pre-registered loss-divergence bound against the f32 arm; bit-exact save/resume
- [ ] The official 4B Base config.json is pinned by sha in fixtures and parses
- [ ] ojas-qwen35 reads, saves and loads state per entry through tessl's host-mapped reads once tessl ships them; staging_tensors/device_copies are deleted; the 2B gradient-read, save_state and load_state peaks are measured and recorded against the 1.34 GB headroom baseline
- [ ] save_state checks free disk space (ojas_device::available_disk_bytes, as ojas-model's checkpoint does) before writing and refuses with a typed error that names the bytes needed and available; a test proves the refusal writes nothing
- [ ] Qwen35Step::open sets tessl's pool cache cap explicitly (step.rs:341 leaves the 2 GiB default, which tessl's step_bytes charges against admission) and records it in describe(); the 2B maximum admissible sequence length is measured before and after
- [ ] Loading state opens each shard once (StateIndex::read_entry reopens SafeTensors per entry, state.rs:~455), folded into the per-entry read rework
- [ ] The tied-head check in ojas-model/src/load.rs:122 reports the real header error (missing entry, dtype, shape, I/O) instead of mapping every failure to 'untied head not supported'
- [ ] 4B is out of the fine-tune track (Fable D4, user-approved 2026-10-09); the 4B criteria stay dormant until a box campaign needs them

## Planned files
- ojas-qwen35/src/step.rs
- ojas-qwen35/src/state.rs
- ojas-qwen35/src/config.rs
- ojas-model/src/load.rs
- ojas-cuda/src/gdn_plan.rs
- ojas-oracle/
