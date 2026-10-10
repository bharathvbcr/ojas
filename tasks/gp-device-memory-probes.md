---
id: "gp-device-memory-probes"
title: "Honest device memory: uncharged Metal hybrid-op copies and wgpu staging, Linux PSI runner, plan-down visible to callers"
status: ready
priority: 1
severity: high
type: bug
owner: "unassigned"
due: "none"
labels:
  - "budget"
  - "memory"
  - "metal"
  - "wgpu"
  - "capi"
  - "robustness"
repositories:
  - "ojas"
planned_files:
  - "ojas-capi/src/load.rs"
  - "ojas-capi/src/profile.rs"
  - "ojas-metal/src/memory.rs"
  - "ojas-wgpu/src/context.rs"
  - "ojas-device/src/plan.rs"
  - "go/api.go"
acceptance_criteria:
  - "A failing test first: a Metal session whose caller budget exceeds the device's recommended working set is planned down to device_room (or refused), not run on the caller's number"
  - "ojas-capi load and profile pass the session backend's MemoryProbe to ResourcePlan::derive instead of `&[] as &[NoProbe]`"
  - "wgpu implements MemoryProbe from what the adapter reports (limits/allocator info), or the plan records 'unknown' explicitly and that is visible in the profile output; never silently the default budget"
  - "Unified-memory devices take the shared_budget path, proven by a test on Apple silicon"
  - "The Go profile/describe output shows which probe was used and the room it reported"
  - "Linux reports memory pressure: PSI and architecture code exists (ojas-device/src/system.rs:76-91, 197-254); a committed podman (or CI) Linux runner proves E_PRESSURE admission and probe_architecture there, instead of the one-off manual run recorded in the doc"
  - "Idle GPU pools are bounded and charged or trimmable: MetalBackend's Worker::open (ojas-metal/src/device.rs:368-409) never calls set_pool_cache_cap_bytes, so tessl's 2 GiB default pool cache sits outside the Budget (only the legacy gpu.rs:325 session caps it, at 64 MiB); Metal gains a trim like wgpu's trim_pool, and the cap is recorded in the plan"
  - "The Metal budget charges what the device actually allocates, or the gap is bounded and documented: ojas-metal/src/backend.rs:298-300 reserves elems*4 while the pool can round up to 2x (docs/framework-design.md:155, docs/adaptive-resources.md:253)"
  - "wgpu recovers from an allocation OOM by recycling the pool and retrying once (exists, ojas-wgpu/src/context.rs:928-955); the error class it ends with is fixed in gp-oom-error-class"
  - "A wgpu Queue::drop that times out (DROP_WAIT 2 s, ojas-wgpu/src/context.rs:47) no longer leaks a queue, a device and a thread (docs/pytorch-parity-plan.md:196), or the leak is bounded, counted and reported"
  - "A failing budget test first: Metal hybrid ops charge the copies they make for offset views. Worker::whole (ojas-metal/src/device.rs:2123-2130) calls fresh(v.n) (uncharged, :762-771) for any operand with off != 0 (uses at :2158, :2183, :2215, :2249), while the forwards charge 0 scratch (backend.rs:1576) and the backwards only `part`; ojas-metal/tests/hybrid.rs:261 tests the bits of offset operands but not the budget"
  - "wgpu staging memory is charged and reported: the 64 MiB staging cache and the next-power-of-two readback buffer (span+16 rounded up, context.rs:1386, :1610-1612; a 256 MiB read allocates 512 MiB) count against the session budget or are bounded and documented; a read larger than about half of max_buffer_size, which fails outright today, is chunked"
  - "A session budget planned down to the device room is visible in the LOAD reply (ojas-capi/src/engine.rs:122-127 drops it today); MetalBackend::with_budget, documented as test-only but used by ojas-capi/src/model.rs:248 on the production path, has its doc corrected"
  - "Metal's documented device-memory bound (backend.rs:28-29: live + rounding + pool cap + slab) accounts for buffers freed since the last waited commit: a MetalBuffer drop posts Free and releases its budget charge at once (backend.rs:115-118), while the buffer returns to tessl's pool only at the next waited commit (device.rs:684-690, 736-741); the bound is corrected or the free is deferred until then"
---

# Task brief v1

## Title
Honest device memory: uncharged Metal hybrid-op copies and wgpu staging, Linux PSI runner, plan-down visible to callers

Task: gp-device-memory-probes
Type: bug
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: budget, memory, metal, wgpu, capi, robustness

## Repositories
- ojas

## Description
Gap audit 2026-10-07 (rg + Read; DevMap unavailable that session, so "no caller" is a text search).

- `ojas-capi/src/load.rs:282` and `ojas-capi/src/profile.rs:129` always call `ResourcePlan::derive(..., &[] as &[NoProbe])` [V]. `NoProbe` is defined at `profile.rs:194`.
- `impl MemoryProbe` exists only in `ojas-metal/src/memory.rs`, `ojas-device` (plan + example) and the capi `NoProbe` [V]. Nothing outside ojas-metal hands `MetalMemory` to a plan [V by rg]. wgpu has no probe at all [V].
- So `device_room` and `shared_budget` never apply in production: Metal and wgpu sessions run on the caller's budget or `DEFAULT_BUDGET_BYTES` [I from the above].

This is a correctness gap in the adaptive-resources design (`docs/adaptive-resources.md`), not a performance item: a budget that never consults the device can admit a plan the device cannot hold.

**Added by the second gap audit 2026-10-07** (same concern, widened from "probes" to "honest GPU memory accounting"; [V] re-read by the auditor; [A] read by an audit subagent, not re-read):
- **Linux pressure is blind [V]:** `#[cfg(not(target_os = "macos"))] { MemoryPressure::Unknown }` (system.rs:96-99).
- **Pool caches outside the budget [V]:** `set_pool_cache_cap_bytes` appears only at ojas-metal/src/gpu.rs:325 (the legacy tiny session); tessl's 2 GiB default (tessl/src/runtime.rs:44) is [A]. The Qwen35Step half of the same omission is on ft-c57c98e0.
- **wgpu trim has no caller [V]:** `pub fn trim_pool` (context.rs:444) is referenced only by its own doc line (:51).
- **Metal OOM says live: 0 [V]:** device.rs:654.
- **Metal logical-byte charge, wgpu OOM path detail, Queue::drop leak [A]** (DROP_WAIT = 2 s at context.rs:47 [V]).

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Most of this brief landed in 00adde8 (merged c2d4549, 2026-10-08) after it was written [A]:
  - Metal budget past the working set is planned down (test at ojas-capi/src/tests.rs:2109-2143).
  - capi passes the real probe (ojas-capi/src/model.rs:179-245, profile.rs:211-220).
  - wgpu room reports Unknown explicitly (ojas-wgpu/src/memory.rs:33-35).
  - The unified-memory path, the Go profile output, the Metal pool cap and trim (device.rs:410-414, plan.rs:144,189) and the wgpu Queue::drop leak (context.rs:84-180, 574-610) are done.
- Metal charge vs pool round-up is bounded and documented (backend.rs:11-29). The brief's '2x' claim was wrong: rounding applies only up to 1 MiB [A].
- The OOM classification items (Metal retry never fires, live bytes, wgpu/CUDA error kind) moved to the new gp-oom-error-class, which fixes the class across backends.
- Not run: the unified-memory and Go-profile ticks are on code read; no device run was made [U].

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Both new uncharged-memory items are CONFIRMED. The wgpu staging item is understated: large reads fail outright [A].
- The 124M wait split by trigger had three owners. It now has one, gp-long-runs-and-quiet-benches, and is dropped here.
- New: the Metal memory bound leaves out buffers freed but not yet recycled [A, low confidence on tessl's pool accounting].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P1: uncharged Metal copies make the fine-tune's memory plan wrong.
- Fine-tune briefs that depend on this one: gp-ft-bf16-frozen-base, gp-ft-trainer-loop, gp-ft-step-memory-and-throughput.

## Acceptance criteria
- [x] A failing test first: a Metal session whose caller budget exceeds the device's recommended working set is planned down to device_room (or refused), not run on the caller's number
- [x] ojas-capi load and profile pass the session backend's MemoryProbe to ResourcePlan::derive instead of `&[] as &[NoProbe]`
- [x] wgpu implements MemoryProbe from what the adapter reports (limits/allocator info), or the plan records 'unknown' explicitly and that is visible in the profile output; never silently the default budget
- [x] Unified-memory devices take the shared_budget path, proven by a test on Apple silicon
- [x] The Go profile/describe output shows which probe was used and the room it reported
- [ ] Linux reports memory pressure: PSI and architecture code exists (ojas-device/src/system.rs:76-91, 197-254); a committed podman (or CI) Linux runner proves E_PRESSURE admission and probe_architecture there, instead of the one-off manual run recorded in the doc
- [x] Idle GPU pools are bounded and charged or trimmable: MetalBackend's Worker::open (ojas-metal/src/device.rs:368-409) never calls set_pool_cache_cap_bytes, so tessl's 2 GiB default pool cache sits outside the Budget (only the legacy gpu.rs:325 session caps it, at 64 MiB); Metal gains a trim like wgpu's trim_pool, and the cap is recorded in the plan
- [x] The Metal budget charges what the device actually allocates, or the gap is bounded and documented: ojas-metal/src/backend.rs:298-300 reserves elems*4 while the pool can round up to 2x (docs/framework-design.md:155, docs/adaptive-resources.md:253)
- [ ] wgpu recovers from an allocation OOM by recycling the pool and retrying once (exists, ojas-wgpu/src/context.rs:928-955); the error class it ends with is fixed in gp-oom-error-class
- [x] A wgpu Queue::drop that times out (DROP_WAIT 2 s, ojas-wgpu/src/context.rs:47) no longer leaks a queue, a device and a thread (docs/pytorch-parity-plan.md:196), or the leak is bounded, counted and reported
- [ ] A failing budget test first: Metal hybrid ops charge the copies they make for offset views. Worker::whole (ojas-metal/src/device.rs:2123-2130) calls fresh(v.n) (uncharged, :762-771) for any operand with off != 0 (uses at :2158, :2183, :2215, :2249), while the forwards charge 0 scratch (backend.rs:1576) and the backwards only `part`; ojas-metal/tests/hybrid.rs:261 tests the bits of offset operands but not the budget
- [ ] wgpu staging memory is charged and reported: the 64 MiB staging cache and the next-power-of-two readback buffer (span+16 rounded up, context.rs:1386, :1610-1612; a 256 MiB read allocates 512 MiB) count against the session budget or are bounded and documented; a read larger than about half of max_buffer_size, which fails outright today, is chunked
- [ ] A session budget planned down to the device room is visible in the LOAD reply (ojas-capi/src/engine.rs:122-127 drops it today); MetalBackend::with_budget, documented as test-only but used by ojas-capi/src/model.rs:248 on the production path, has its doc corrected
- [ ] Metal's documented device-memory bound (backend.rs:28-29: live + rounding + pool cap + slab) accounts for buffers freed since the last waited commit: a MetalBuffer drop posts Free and releases its budget charge at once (backend.rs:115-118), while the buffer returns to tessl's pool only at the next waited commit (device.rs:684-690, 736-741); the bound is corrected or the free is deferred until then

## Planned files
- ojas-capi/src/load.rs
- ojas-capi/src/profile.rs
- ojas-metal/src/memory.rs
- ojas-wgpu/src/context.rs
- ojas-device/src/plan.rs
- go/api.go
