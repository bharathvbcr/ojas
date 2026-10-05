---
id: "gp-activation-checkpointing"
title: "Add Selective Activation Checkpointing to Autograd Tape"
status: ready
priority: 1
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "autograd"
  - "memory"
  - "scaling"
repositories:
  - "ojas"
planned_files:
  - "ojas-autograd/src/tape.rs"
  - "ojas-model/src/block.rs"
  - "ojas-model/src/trainer.rs"
acceptance_criteria:
  - "Implement block-level activation checkpointing in Tape, releasing internal layer activations during the forward pass"
  - "Recompute block forward activations on demand during the backward adjoint traversal"
  - "Enable sequence length scaling past T=2048 without exceeding resident memory budgets"
  - "Unit tests verifying gradient equality between standard and checkpointed runs"
---

# Task brief v1

## Title
Add Selective Activation Checkpointing to Autograd Tape

Task: gp-activation-checkpointing
Type: feature
Status: ready
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: autograd, memory, scaling

## Repositories
- ojas

## Description
Ojas currently materializes all intermediate activations on the autograd Tape during forward execution and holds them until backward completes. For deep models or sequence lengths T > 2048, activation memory dominates and quickly leads to CapacityExceeded errors or unified memory paging.

This task introduces block-level selective activation recomputation (gradient checkpointing), storing only block boundaries and recomputing internal block activations on demand during the backward pass.

## Acceptance criteria
- [ ] Implement block-level activation checkpointing in Tape, releasing internal layer activations during the forward pass
- [ ] Recompute block forward activations on demand during the backward adjoint traversal
- [ ] Enable sequence length scaling past T=2048 without exceeding resident memory budgets
- [ ] Unit tests verifying gradient equality between standard and checkpointed runs

## Planned files
- ojas-autograd/src/tape.rs
- ojas-model/src/block.rs
- ojas-model/src/trainer.rs
