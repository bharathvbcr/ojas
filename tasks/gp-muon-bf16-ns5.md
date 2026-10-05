---
id: "gp-muon-bf16-ns5"
title: "Implement BF16 Newton-Schulz 5 Kernel Path for Muon"
status: backlog
priority: 2
severity: low
type: feature
owner: "unassigned"
due: "none"
labels:
  - "optimizer"
  - "muon"
  - "parity"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-cpu/src/optim.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_backend.metal"
acceptance_criteria:
  - "Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer"
  - "Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps"
  - "Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization"
  - "Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace"
---

# Task brief v1

## Title
Implement BF16 Newton-Schulz 5 Kernel Path for Muon

Task: gp-muon-bf16-ns5
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: low
Owner: unassigned
Due: none
Labels: optimizer, muon, parity

## Repositories
- ojas

## Description
Upstream nanolab and PyTorch Muon execute the quintic Newton-Schulz iteration in bf16 (X = G.bfloat16()), whereas Ojas executes NS5 strictly in f32. While FP32 is numerically more accurate, it causes a drift of up to 1.8e-4 nats over the first 5 steps compared to stock nanolab, and runs slower than batched BF16 iterations.

This task adds an optional BF16 compute path for Muon Newton-Schulz iterations to achieve exact bit-for-bit trajectory parity with stock nanolab runs.

## Acceptance criteria
- [ ] Add optional bf16 compute path for Newton-Schulz 5 quintic iterations in Muon optimizer
- [ ] Align update magnitudes exactly with PyTorch and nanolab stock optimizer steps
- [ ] Measure throughput improvement from BF16 batched matrix multiplications during orthogonalization
- [ ] Golden trace parity test verifying convergence within 1e-4 nats against stock nanolab trace

## Planned files
- ojas-core/src/backend.rs
- ojas-cpu/src/optim.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_backend.metal
