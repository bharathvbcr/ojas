---
id: "gp-uma-zero-copy-io-and-budget-preflight"
title: "Eliminate safetensors double-copy host transients, enforce child-first Budget::check_room, and wire zero-copy UMA weight loading"
status: ready
priority: 1
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "memory"
  - "uma"
  - "safetensors"
  - "io"
  - "budget"
  - "qwen35"
  - "audit-2026-10-16"
repositories:
  - "ojas"
  - "tessl"
planned_files:
  - "ojas-core/src/budget.rs"
  - "ojas-io/src/safetensors.rs"
  - "ojas-model/src/qwen35/load.rs"
  - "ojas-model/src/trainer.rs"
  - "ojas-qwen35/src/state.rs"
  - "ojas-qwen35/src/names.rs"
  - "ojas-metal/src/device.rs"
acceptance_criteria:
  - "Eliminate the two whole-tensor host copies (~2 GB uncharged transient per large tensor during Qwen3.5-2B load) in ojas-io typed safetensors reads (G-IOREAD / DI-4)"
  - "Pre-check the total byte requirement from the safetensors header against Budget before reading payloads in Qwen3.5 load_hf (G-IOREAD / MD-1)"
  - "Support page-aligned mmap wrapping into Metal shared buffers (newBufferWithBytesNoCopy -> tessl BufferKind::External) for zero-copy UMA weight ingestion on Apple Silicon"
  - "Fix Budget::check_room (ojas-core/src/budget.rs:150) so preflight checks are read-only and validate child capacity before reserving parent bytes, preventing a rejected preflight in one session from poisoning a sibling trainer (G-BUDGET / CA-1)"
  - "Fix a_real_oversize_gdn_workspace_is_capacity_exceeded (ojas-metal/src/device.rs:3264) so oversize GDN workspaces fail closed with OjasError::CapacityExceeded (ft-cab0fdf1)"
  - "Once tessl gp/wave10 lands on main, convert ojas-qwen35/src/state.rs to per-entry host mapping (ft-dc1fa0a3 AC3), load sharded checkpoints through CheckpointView to retire the single-file Unsupported branch at ojas-qwen35/src/names.rs:292-296 (ft-02256073), and skip binding frozen moment slots in ojas-model/src/trainer.rs (ft-2bbced18)"
---

# Task brief v1

## Title
Eliminate safetensors double-copy host transients, enforce child-first Budget::check_room, and wire zero-copy UMA weight loading

Task: gp-uma-zero-copy-io-and-budget-preflight
Type: bug
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: memory, uma, safetensors, io, budget, qwen35, audit-2026-10-16

## Repositories
- ojas
- tessl

## Description
Consolidates the unified memory, `Budget` governance, and `safetensors` I/O gaps identified in `ojas/AUDIT_REPORT.md` (`G-IOREAD`, `G-BUDGET`) and `ojas/BUILD_REPORT.md` (`ft-cab0fdf1`, `ft-02256073`, `ft-dc1fa0a3` AC3, `ft-2bbced18`):

1. **Uncharged Host Transients & Missing Header Budget Check (`G-IOREAD` / `DI-4`, `MD-1`):** Typed `safetensors` reads in `ojas-io` currently hold two whole-tensor copies in host RAM (~2 GB of uncharged transient memory per large tensor during Qwen3.5-2B loading under the 32 GiB `memguard` cap), and `Qwen3.5 load_hf` does not check the `Budget` from the header before allocating. On Apple Silicon UMA, page-aligned `safetensors` payloads can also be mapped directly into `tessl`'s `BufferKind::External` via `newBufferWithBytesNoCopy`.
2. **Parent-First `Budget::check_room` Poisoning (`G-BUDGET` / `CA-1`):** `Budget::check_room` (`ojas-core/src/budget.rs:150`) reserves parent bytes before checking child capacity, so a failed preflight in one session can transiently starve a concurrent trainer.
3. **Oversize GDN Workspace & Sharded 2B State (`ft-cab0fdf1`, `ft-02256073`, `ft-2bbced18`):** `ojas-metal/src/device.rs:3264` fails `a_real_oversize_gdn_workspace_is_capacity_exceeded` on `main`, `ojas-model/src/trainer.rs` still binds frozen moment slots with `g.param` (computing and discarding weight gradients), and `ojas-qwen35` awaits `tessl` `CheckpointView::tensor_info` and per-entry host mapping.

## Acceptance criteria
- [ ] Eliminate the two whole-tensor host copies (~2 GB uncharged transient per large tensor during Qwen3.5-2B load) in ojas-io typed safetensors reads (G-IOREAD / DI-4)
- [ ] Pre-check the total byte requirement from the safetensors header against Budget before reading payloads in Qwen3.5 load_hf (G-IOREAD / MD-1)
- [ ] Support page-aligned mmap wrapping into Metal shared buffers (newBufferWithBytesNoCopy -> tessl BufferKind::External) for zero-copy UMA weight ingestion on Apple Silicon
- [ ] Fix Budget::check_room (ojas-core/src/budget.rs:150) so preflight checks are read-only and validate child capacity before reserving parent bytes, preventing a rejected preflight in one session from poisoning a sibling trainer (G-BUDGET / CA-1)
- [ ] Fix a_real_oversize_gdn_workspace_is_capacity_exceeded (ojas-metal/src/device.rs:3264) so oversize GDN workspaces fail closed with OjasError::CapacityExceeded (ft-cab0fdf1)
- [ ] Once tessl gp/wave10 lands on main, convert ojas-qwen35/src/state.rs to per-entry host mapping (ft-dc1fa0a3 AC3), load sharded checkpoints through CheckpointView to retire the single-file Unsupported branch at ojas-qwen35/src/names.rs:292-296 (ft-02256073), and skip binding frozen moment slots in ojas-model/src/trainer.rs (ft-2bbced18)

## Planned files
- ojas-core/src/budget.rs
- ojas-io/src/safetensors.rs
- ojas-model/src/qwen35/load.rs
- ojas-model/src/trainer.rs
- ojas-qwen35/src/state.rs
- ojas-qwen35/src/names.rs
- ojas-metal/src/device.rs
