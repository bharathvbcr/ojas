---
id: "gp-ft-step-throughput-followups"
title: "LoRA step-cost follow-ups at 2B: per-call GDN backward workspace, attention layout copies doubled by checkpoint replay, host RoPE uploads, fused CE logits computed 4 times, one sequence per provider launch"
status: backlog
priority: 2
severity: low
type: performance
owner: "unassigned"
due: "none"
labels:
  - "fine-tune"
  - "performance"
  - "metal"
  - "qwen35"
repositories:
  - "ojas"
planned_files:
  - "ojas-metal/src/device.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-model/src/qwen35/forward.rs"
  - "ojas-qwen35/src/step.rs"
  - "bench/"
acceptance_criteria:
  - "Every item lands with a committed interleaved A/B, min-of-N benchmark at the 2B shape (T=2048 and T=9,638, micro-batch 1) under bench/results/, run under mac_heavy.sh"
  - "The GDN backward reuses its workspace across layers and calls instead of allocating about 2.7 GB per call, 18 calls per step [A]"
  - "The four q/k/v/output layout copies per attention layer, doubled by checkpoint replay, are removed or fused [A]"
  - "RoPE tables are device-resident, so no host upload forces a GPU wait (about 24 per forward) [A]"
  - "The fused linear CE computes its logits 3 times instead of 4 and, when a weight gradient is wanted, accumulates it once per column tile rather than once per row tile (ojas-metal/src/device.rs:2684-2741) [A]; with the LoRA-2B head (17-way, frozen embedding) this affects only the report-only full-vocab CE and the full-FT paths"
  - "The provider runs several sequences per Metal launch instead of one step per sequence with a host read-back each [A]"
---

# Task brief v1

## Title
LoRA step-cost follow-ups at 2B: per-call GDN backward workspace, attention layout copies doubled by checkpoint replay, host RoPE uploads, fused CE logits computed 4 times, one sequence per provider launch

Task: gp-ft-step-throughput-followups
Type: performance
Status: backlog
Priority: 2 (Normal)
Severity: low
Owner: unassigned
Due: none
Labels: fine-tune, performance, metal, qwen35

## Repositories
- ojas

## Description
Filed 2026-10-09 by splitting gp-ft-step-memory-and-throughput along Fable's priority line (Lappi AUDIT/lora-2b-2026-10-09/fable-ft-decisions.md, 'Task priorities'). These costs were found by reading the code paths; none was measured. They matter for longer runs and the full-FT path, not for deciding D1.

Related: gp-gpu-step-throughput, gp-ft-step-memory-and-throughput. Umbrella: gp-ft-end-to-end-acceptance.

Labels: [V] re-read at d431949 by the writer of this brief (2026-10-09 fine-tune audit); [A] read by an audit subagent at d431949, not re-read; [U] unverified, needs a run or an upstream source read. No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] Every item lands with a committed interleaved A/B, min-of-N benchmark at the 2B shape (T=2048 and T=9,638, micro-batch 1) under bench/results/, run under mac_heavy.sh
- [ ] The GDN backward reuses its workspace across layers and calls instead of allocating about 2.7 GB per call, 18 calls per step [A]
- [ ] The four q/k/v/output layout copies per attention layer, doubled by checkpoint replay, are removed or fused [A]
- [ ] RoPE tables are device-resident, so no host upload forces a GPU wait (about 24 per forward) [A]
- [ ] The fused linear CE computes its logits 3 times instead of 4 and, when a weight gradient is wanted, accumulates it once per column tile rather than once per row tile (ojas-metal/src/device.rs:2684-2741) [A]; with the LoRA-2B head (17-way, frozen embedding) this affects only the report-only full-vocab CE and the full-FT paths
- [ ] The provider runs several sequences per Metal launch instead of one step per sequence with a host read-back each [A]

## Planned files
- ojas-metal/src/device.rs
- ojas-metal/src/backend.rs
- ojas-model/src/qwen35/forward.rs
- ojas-qwen35/src/step.rs
- bench/

### Fine-tune track audit (2026-10-15, at 88be893)
Read-only re-audit of the fine-tune track at main 88be893. No cargo, GPU or device run was made in this audit.
Labels: [V] re-read by the writer of this note; [A] read by an audit subagent at 88be893, not re-read; [I] inferred.
DevMap (main checkout, generation 6968, fresh) and rg were used; Lappi was read at 4b0b3b8.
- Status: backlog, nothing landed; header accurate [A].
- GDN backward workspace: still per call (device.rs:2186) and charged; the brief's 2.7 GB does not reproduce: 0.34 GB at T = 2,048 and 1.34 GB at T = 9,638 from backend.rs:774-791 with H = 16 [A].
- Attention layout copies: 4 permutes per attention layer, 6 layers (forward.rs:536, 542); a few hundred MB, low value [A].
- RoPE tables device-resident: obsolete in practice; both GPU harnesses call `Qwen35Tables::upload` (forward.rs:72-79) [A].
- Fused CE "logits 4 times": wording wrong and the cite stale (:2684-2741 is now clip_norm). Per tile it is 2 logits GEMMs (pass 0, pass 1 unless one tile) plus gx and gw GEMMs, once per step (tape.rs:970) [A].
- Provider multi-sequence launches: off the LoRA path (grad_accum 1 on the Tape) [A].
- None of this gates the acceptance run.
