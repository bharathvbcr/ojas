---
id: "gp-backend-wrapper-forwarding"
title: "Backend wrappers silently drop overrides: forward_backend!, Autocast and Gated each hand-list trait methods and none forwards causal_sdpa_backward_recompute"
status: ready
priority: 1
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "backend"
  - "autograd"
  - "capi"
  - "refactor"
  - "tests"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/backend.rs"
  - "ojas-core/src/autocast.rs"
  - "ojas-capi/src/gate.rs"
acceptance_criteria:
  - "A failing test first: a backend that overrides causal_sdpa_backward_recompute (a test double recording the call) is reached through &B, Arc<B>, Autocast<B> and Gated<B>, and the override is not called today"
  - "Every provided Backend method is forwarded by every wrapper, enforced mechanically (one generating macro, or a completeness test that fails when a new trait method is not forwarded); hand-maintained parallel lists are gone"
  - "Autocast's per-op precision decisions and Gated's admission checks stay explicit where they differ from plain forwarding, and the tests for both pass unchanged"
  - "The test wrapper Probe<B> (ojas-model/tests/common/mod.rs:713) forwards every Backend method or is replaced by the generated wrapper: it misses 19 today (including scale_grad, bf16_operands and all hybrid ops), so trainer and checkpoint tests that wrap CpuBackend in it silently run trait defaults instead of the CPU overrides"
---

# Task brief v1

## Title
Backend wrappers silently drop overrides: forward_backend!, Autocast and Gated each hand-list trait methods and none forwards causal_sdpa_backward_recompute

Task: gp-backend-wrapper-forwarding
Type: bug
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: backend, autograd, capi, refactor, tests

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949).

Adding a Backend op means editing three hand-written forwarding lists: `forward_backend!` (ojas-core/src/backend.rs:1480, for `&B` and `Arc<B>`), `Autocast` (ojas-core/src/autocast.rs:353) and `Gated` (ojas-capi/src/gate.rs:168). A provided method that a backend overrides is silently replaced by the trait default when the backend is reached through any wrapper.

- **Confirmed instance [V]:** `causal_sdpa_backward_recompute` (backend.rs:751) appears in none of the three lists (rg finds only its definition, a shapes.rs doc and one Metal call). It is harmless today only because no backend overrides it.
- **Fix the class, not the one method:** a test that enumerates trait methods and proves every wrapper forwards each one, or a single macro that generates all three wrappers from one list. A new trait method should fail the build or the test until it is forwarded.

Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- CONFIRMED and UNDERSTATED. Of the 33 default-bodied trait methods, the three production wrappers each miss only causal_sdpa_backward_recompute. But the test wrapper Probe<B> misses 19, so tests that use it exercise trait defaults instead of the backend they claim to test. Severity is raised to high [A].
- Corrected: the brief said 'rg finds only its definition'; causal_sdpa_backward_recompute has about 60 call sites in tests [A, C].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P1: LoRA and the frozen bf16 base add Backend methods, which must reach every wrapper.
- Fine-tune briefs that depend on this one: gp-ft-frozen-params-and-lora, gp-ft-bf16-frozen-base.

## Acceptance criteria
- [ ] A failing test first: a backend that overrides causal_sdpa_backward_recompute (a test double recording the call) is reached through &B, Arc<B>, Autocast<B> and Gated<B>, and the override is not called today
- [ ] Every provided Backend method is forwarded by every wrapper, enforced mechanically (one generating macro, or a completeness test that fails when a new trait method is not forwarded); hand-maintained parallel lists are gone
- [ ] Autocast's per-op precision decisions and Gated's admission checks stay explicit where they differ from plain forwarding, and the tests for both pass unchanged
- [ ] The test wrapper Probe<B> (ojas-model/tests/common/mod.rs:713) forwards every Backend method or is replaced by the generated wrapper: it misses 19 today (including scale_grad, bf16_operands and all hybrid ops), so trainer and checkpoint tests that wrap CpuBackend in it silently run trait defaults instead of the CPU overrides

## Planned files
- ojas-core/src/backend.rs
- ojas-core/src/autocast.rs
- ojas-capi/src/gate.rs
