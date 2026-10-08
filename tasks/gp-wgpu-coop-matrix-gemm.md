---
id: "gp-wgpu-coop-matrix-gemm"
title: "wgpu Cooperative-Matrix GEMM Behind a Narrow Unsafe Exemption"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "wgpu"
  - "performance"
  - "gemm"
repositories:
  - "ojas"
planned_files:
  - "ojas-wgpu/src/lib.rs"
  - "ojas-wgpu/src/context.rs"
  - "ojas-wgpu/Cargo.toml"
  - "ojas-kernels/src/wgsl/"
  - "ojas-wgpu/tests/bench.rs"
  - "docs/pytorch-parity-plan.md"
acceptance_criteria:
  - "ojas-wgpu changes #![forbid(unsafe_code)] to #![deny(unsafe_code)]; exactly one #[allow(unsafe_code)] exists, on the ExperimentalFeatures::enabled() call in context.rs, with a // SAFETY: comment; `rg -n 'allow\\(unsafe_code\\)' ojas-wgpu` returns that one line"
  - "The path is behind an opt-in cargo feature, off by default; with it off the build, tests and GEMM kernels are unchanged"
  - "With the feature on, an adapter without Features::EXPERIMENTAL_COOPERATIVE_MATRIX or without a matching configuration uses the portable GEMM, and a test covers that selection"
  - "Throughput target: at least 4.5 TFLOP/s GPU-resident on the existing `linear fwd 2048x2048x2048` row (f32, the 8x8x8 f32 configuration) on this M5 Pro, measured with `cargo test -p ojas-wgpu --release --features <feature> --test bench -- --ignored --nocapture --test-threads=1`, interleaved A/B against the portable kernel, min of N; torch MPS reference is 5.2 TFLOP/s (docs/pytorch-parity-plan.md; re-measure torch at the same shape before comparing)"
  - "Results stay within the Fast tier (1e-4 relative) of CPU on the ojas-oracle parity harness; an f16-input variant, if added, documents its own tolerance against an f64 reference"
---

# Task brief v1

## Title
wgpu Cooperative-Matrix GEMM Behind a Narrow Unsafe Exemption

Task: gp-wgpu-coop-matrix-gemm
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
Labels: wgpu, performance, gemm

## Repositories
- ojas

## Description
Filed by `gp-backend-architecture-decisions`: the user approved (2026-10-08) a narrow unsafe exemption in `ojas-wgpu` for the cooperative-matrix extension. wgpu 30 exposes `Features::EXPERIMENTAL_COOPERATIVE_MATRIX` behind `unsafe ExperimentalFeatures::enabled()`; this M5 Pro reports 8x8x8 configurations for f32, f16, and f16 with an f32 accumulator. wgpu GEMM is about 1.6-3.2 TFLOP/s today against torch's 5.2. A GEMM and SDPA design exists in the wgpu lane's round-4 report; this task covers GEMM only.

The 4.5 TFLOP/s target was set by the decision task (about 87% of torch MPS, and 1.4x the best portable result). Revise it here, with a reason, if the measured matrix-unit ceiling on this adapter is lower.

## Acceptance criteria
- [ ] ojas-wgpu changes #![forbid(unsafe_code)] to #![deny(unsafe_code)]; exactly one #[allow(unsafe_code)] exists, on the ExperimentalFeatures::enabled() call in context.rs, with a // SAFETY: comment; `rg -n 'allow\(unsafe_code\)' ojas-wgpu` returns that one line
- [ ] The path is behind an opt-in cargo feature, off by default; with it off the build, tests and GEMM kernels are unchanged
- [ ] With the feature on, an adapter without Features::EXPERIMENTAL_COOPERATIVE_MATRIX or without a matching configuration uses the portable GEMM, and a test covers that selection
- [ ] Throughput target: at least 4.5 TFLOP/s GPU-resident on the existing `linear fwd 2048x2048x2048` row (f32, the 8x8x8 f32 configuration) on this M5 Pro, measured with `cargo test -p ojas-wgpu --release --features <feature> --test bench -- --ignored --nocapture --test-threads=1`, interleaved A/B against the portable kernel, min of N; torch MPS reference is 5.2 TFLOP/s (docs/pytorch-parity-plan.md; re-measure torch at the same shape before comparing)
- [ ] Results stay within the Fast tier (1e-4 relative) of CPU on the ojas-oracle parity harness; an f16-input variant, if added, documents its own tolerance against an f64 reference

## Planned files
- ojas-wgpu/src/lib.rs
- ojas-wgpu/src/context.rs
- ojas-wgpu/Cargo.toml
- ojas-kernels/src/wgsl/
- ojas-wgpu/tests/bench.rs
- docs/pytorch-parity-plan.md
