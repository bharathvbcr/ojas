---
id: "gp-metal-red-rows-vs-torch"
title: "Lift the remaining red rows against PyTorch on Metal (grad accumulation, CE backward, hd64 attention, block forward)"
status: ready
priority: 2
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "metal"
  - "torch-2.15"
  - "perf"
  - "research-2026-10-20"
repositories:
  - "ojas"
  - "tessl"
planned_files:
  - "docs/bench-gpu-vs-torch.md"
acceptance_criteria:
  - "Before starting, rows already owned by gp-gpu-step-throughput are removed from this task's scope"
  - "LM-head row re-measured after its fix"
  - "Gradient accumulation (0.57x), cross-entropy backward (0.61x), attention at head dim 64 (0.65-0.72x) and block forward (0.66x) each at or above 0.9x torch 2.15 in the quiet paired re-run, or each with a documented cause"
---

# Task brief v1

## Title
Lift the remaining red rows against PyTorch on Metal (grad accumulation, CE backward, hd64 attention, block forward)

Task: gp-metal-red-rows-vs-torch
Type: perf
Status: ready
Priority: 2 (Normal)
Severity: medium
Source: research session 2026-10-20 (article audit gaps)

## Description
The slowest rows in the last full GPU-vs-torch round:
- gradient accumulation: 0.57×
- cross-entropy backward: 0.61×
- attention at head dim 64: 0.65–0.72×
- block forward: 0.66×

The LM-head fix has landed but hasn't been re-measured. `gp-gpu-step-throughput` may already own some of these rows, so trim the overlap before starting. The quiet scorecard re-run belongs to `gp-long-runs-and-quiet-benches`. Use its numbers rather than starting a separate round.

## Acceptance criteria
- [ ] Before starting, rows already owned by gp-gpu-step-throughput are removed from this task's scope
- [ ] LM-head row re-measured after its fix
- [ ] Gradient accumulation (0.57x), cross-entropy backward (0.61x), attention at head dim 64 (0.65-0.72x) and block forward (0.66x) each at or above 0.9x torch 2.15 in the quiet paired re-run, or each with a documented cause
