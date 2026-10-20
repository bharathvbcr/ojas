---
id: "gp-gpu-vision-stack-and-vit-sdpa"
title: "GPU Vision training stack on Metal: MNIST/CIFAR readers, vision ops, CNN conv2d/pool/BatchNorm, bidirectional ViT SDPA, and MPS benchmarks"
status: backlog
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "vision"
  - "cnn"
  - "vit"
  - "metal"
  - "kernel"
  - "benchmarks"
  - "audit-2026-10-16"
repositories:
  - "ojas"
  - "tessl"
planned_files:
  - "ojas-data/src/vision.rs"
  - "ojas-core/src/backend.rs"
  - "ojas-autograd/src/tape.rs"
  - "ojas-cpu/src/vision.rs"
  - "ojas-metal/src/backend.rs"
  - "ojas-metal/src/device.rs"
  - "ojas-metal/kernels/ojas_vision.metal"
  - "ojas-metal/build.rs"
  - "bench/ojas_rows.rs"
  - "docs/op-coverage.md"
acceptance_criteria:
  - "G-V4 (Wave 1): Implement std-only MNIST idx and CIFAR-10 binary readers with Budget-bounded parsing, seeded f32 batch sampler, random crop/flip augmentations, and an end-to-end Metal training step"
  - "G-V1 (Wave 4): Add vision op set on Backend + Tape with Metal forward/backward kernels in ojas_vision.metal (ReLU/GELU/Tanh, bias-add, LayerNorm, softmax, reduce, recorded-copy slice/concat, and frozen-input dW-only linear backward VIS-F), fix docs/op-coverage.md:102 and docs/pytorch-parity-plan.md:47, and benchmark MLP 784-512-10 step on MNIST at batch 128 vs torch MPS"
  - "G-V2 (Wave 5): Implement CNN ops on Metal (NCHW external layout conv2d forward/backward, max/avg/global pooling, BatchNorm forward/backward with CPU parity oracle) and benchmark LeNet-5 (MNIST) and ResNet-18-style (CIFAR-10 32x32) steps at batch 128 vs torch MPS"
  - "G-V3 (Wave 6): Add mask-mode argument to SDPA methods supporting bidirectional attention for ViT on Metal, and benchmark ViT-tiny (patch 4, D=192, 4 layers) on CIFAR-10 at batch 128 vs torch MPS"
  - "G-BENCH (Wave 3): Record paired torch MPS baseline rows (MLP/LeNet-5/ResNet-18/ViT-tiny at batch 128 fp32), vision goldens, top_k=None sampler row, and Metal upload-wait count across a 4-micro-batch CIFAR step (filing unwaited tessl upload API if wait exceeds 10% per E14)"
---

# Task brief v1

## Title
GPU Vision training stack on Metal: MNIST/CIFAR readers, vision ops, CNN conv2d/pool/BatchNorm, bidirectional ViT SDPA, and MPS benchmarks

Task: gp-gpu-vision-stack-and-vit-sdpa
Type: feature
Status: backlog
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: vision, cnn, vit, metal, kernel, benchmarks, audit-2026-10-16

## Repositories
- ojas
- tessl

## Description
Syncs the complete GPU vision training track (`G-V4`, `G-V1`, `G-V2`, `G-V3`, `G-BENCH`) and Fable decisions (`E1`-`E5`, `E10`, `E13`, `E14`, `E18`) from `ojas/AUDIT_REPORT.md` (2026-10-16) into `tasks/` so `tasks/*.md` stays aligned with the GitPulse board (per Decision 19):

- **Delivery Target:** Training runs on the GPU (`ojas-metal` over `tessl`) with the CPU implementation as the parity oracle (`wgpu` may return `Unsupported`, and `ojas-cuda` must compile).
- **Kernel Home (`E5`):** Vision shaders live in a dedicated `ojas-metal/kernels/ojas_vision.metal` file rather than growing `ojas_backend.metal`.
- **Acceptance Models (`E18`):** MLP 784-512-10 on MNIST, LeNet-5 on MNIST, ResNet-18-style on CIFAR-10 (32x32), and ViT-tiny (patch 4, D=192, 4 layers) on CIFAR-10, all at batch 128 in `fp32`.

## Acceptance criteria
- [ ] G-V4 (Wave 1): Implement std-only MNIST idx and CIFAR-10 binary readers with Budget-bounded parsing, seeded f32 batch sampler, random crop/flip augmentations, and an end-to-end Metal training step
- [ ] G-V1 (Wave 4): Add vision op set on Backend + Tape with Metal forward/backward kernels in ojas_vision.metal (ReLU/GELU/Tanh, bias-add, LayerNorm, softmax, reduce, recorded-copy slice/concat, and frozen-input dW-only linear backward VIS-F), fix docs/op-coverage.md:102 and docs/pytorch-parity-plan.md:47, and benchmark MLP 784-512-10 step on MNIST at batch 128 vs torch MPS
- [ ] G-V2 (Wave 5): Implement CNN ops on Metal (NCHW external layout conv2d forward/backward, max/avg/global pooling, BatchNorm forward/backward with CPU parity oracle) and benchmark LeNet-5 (MNIST) and ResNet-18-style (CIFAR-10 32x32) steps at batch 128 vs torch MPS
- [ ] G-V3 (Wave 6): Add mask-mode argument to SDPA methods supporting bidirectional attention for ViT on Metal, and benchmark ViT-tiny (patch 4, D=192, 4 layers) on CIFAR-10 at batch 128 vs torch MPS
- [ ] G-BENCH (Wave 3): Record paired torch MPS baseline rows (MLP/LeNet-5/ResNet-18/ViT-tiny at batch 128 fp32), vision goldens, top_k=None sampler row, and Metal upload-wait count across a 4-micro-batch CIFAR step (filing unwaited tessl upload API if wait exceeds 10% per E14)

## Planned files
- ojas-data/src/vision.rs
- ojas-core/src/backend.rs
- ojas-autograd/src/tape.rs
- ojas-cpu/src/vision.rs
- ojas-metal/src/backend.rs
- ojas-metal/src/device.rs
- ojas-metal/kernels/ojas_vision.metal
- ojas-metal/build.rs
- bench/ojas_rows.rs
- docs/op-coverage.md
