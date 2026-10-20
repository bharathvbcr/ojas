---
id: "gp-exact-libm-determinism-and-tape-capi-hardening"
title: "Replace platform libm in Numerics::Exact, harden Tape shape/Var/replay checks, guard Go C-ABI micro-batches, and add Miri/TSan gates"
status: ready
priority: 1
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "determinism"
  - "numerics"
  - "autograd"
  - "capi"
  - "simd"
  - "safety"
  - "audit-2026-10-16"
repositories:
  - "ojas"
planned_files:
  - "ojas-core/src/lib.rs"
  - "ojas-cpu/src/fused_ce.rs"
  - "ojas-cpu/src/gdn.rs"
  - "ojas-autograd/src/tape.rs"
  - "ojas-capi/src/lib.rs"
  - "ojas-capi/src/train.rs"
  - "ojas-simd/src/arch.rs"
  - ".github/workflows/test.yml"
acceptance_criteria:
  - "Route Exact-tier cross-entropy loss and GDN softplus through a platform-free log_sum_exp_exact / deterministic ln1p implementation instead of platform libm so macOS and Linux produce bit-identical digests, and enable the Exact digest assertion in the Linux CI job (G-LIBM / CS-1)"
  - "Harden Tape backward in ojas-autograd/src/tape.rs to reject backend gradients whose shape mismatches the forward Var instead of keeping them (G-TAPE / CA-4)"
  - "Bump a generation counter on Tape::clear() so stale Var accessors are refused, make checkpoint replay panic-safe, and read Autocast state from the recorded forward node rather than ambient backward state (G-TAPE / CA-2, CA-3, CA-5)"
  - "Bound micro-batch allocations in ojas-capi so a public Go call passing 16,385 1-row micro-batches under a heap ceiling returns OjasError::CapacityExceeded instead of aborting the process (F-BD63 / CG-1)"
  - "Add Miri and ThreadSanitizer (TSan) CI gates covering unsafe impl Send/Sync over raw pointers in ojas-simd and Handoff::drop in ojas-cpu, and reconcile the MSRV comment in ojas-simd/src/arch.rs:9 (1.82 vs Cargo.toml 1.97) (F-3902 / CS-3, CS-4)"
---

# Task brief v1

## Title
Replace platform libm in Numerics::Exact, harden Tape shape/Var/replay checks, guard Go C-ABI micro-batches, and add Miri/TSan gates

Task: gp-exact-libm-determinism-and-tape-capi-hardening
Type: bug
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: determinism, numerics, autograd, capi, simd, safety, audit-2026-10-16

## Repositories
- ojas

## Description
Consolidates the cross-machine reproducibility, autograd `Tape` fail-open, C-ABI heap-ceiling abort, and `unsafe` concurrency verification gaps from `ojas/AUDIT_REPORT.md` (`G-LIBM`, `G-TAPE`, `F-BD63`, `F-3902`):

1. **Cross-Machine `libm` Drift (`G-LIBM` / `CS-1`, Risk #8):** `README.md:99` promises bit-identical results across machines under `Numerics::Exact`, but Exact-tier cross-entropy loss and GDN `softplus` call platform `libm`, which drifts by up to 1 ULP between Darwin `libSystem` and Linux `glibc` while the Linux CI job skips the digest check.
2. **Autograd `Tape` Fail-Open Class (`G-TAPE` / `CA-2`..`CA-5`):** A wrong-shaped gradient returned by a backend is currently kept on the `Tape`, `clear()` does not invalidate stale `Var` handles, and checkpoint replay reads ambient `Autocast` state during backward instead of the forward node's recorded state.
3. **Go C-ABI Process Abort (`F-BD63` / `CG-1`, Risk #10):** Calling into `ojas-capi` from Go with 16,385 1-row micro-batches under a tight heap ceiling triggers an unguarded allocation abort instead of returning `OjasError::CapacityExceeded`.
4. **Missing Miri / TSan Gates on `ojas-simd` (`F-3902` / `CS-3`, Risk #9):** `unsafe impl Send/Sync` over raw pointers in `ojas-simd` and `Handoff::drop` in `ojas-cpu` (`CS-4`) have no Miri or TSan CI coverage.

## Acceptance criteria
- [ ] Route Exact-tier cross-entropy loss and GDN softplus through a platform-free log_sum_exp_exact / deterministic ln1p implementation instead of platform libm so macOS and Linux produce bit-identical digests, and enable the Exact digest assertion in the Linux CI job (G-LIBM / CS-1)
- [ ] Harden Tape backward in ojas-autograd/src/tape.rs to reject backend gradients whose shape mismatches the forward Var instead of keeping them (G-TAPE / CA-4)
- [ ] Bump a generation counter on Tape::clear() so stale Var accessors are refused, make checkpoint replay panic-safe, and read Autocast state from the recorded forward node rather than ambient backward state (G-TAPE / CA-2, CA-3, CA-5)
- [ ] Bound micro-batch allocations in ojas-capi so a public Go call passing 16,385 1-row micro-batches under a heap ceiling returns OjasError::CapacityExceeded instead of aborting the process (F-BD63 / CG-1)
- [ ] Add Miri and ThreadSanitizer (TSan) CI gates covering unsafe impl Send/Sync over raw pointers in ojas-simd and Handoff::drop in ojas-cpu, and reconcile the MSRV comment in ojas-simd/src/arch.rs:9 (1.82 vs Cargo.toml 1.97) (F-3902 / CS-3, CS-4)

## Planned files
- ojas-core/src/lib.rs
- ojas-cpu/src/fused_ce.rs
- ojas-cpu/src/gdn.rs
- ojas-autograd/src/tape.rs
- ojas-capi/src/lib.rs
- ojas-capi/src/train.rs
- ojas-simd/src/arch.rs
- .github/workflows/test.yml
