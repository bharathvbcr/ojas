---
id: "gp-device-memory-probes"
title: "Wire GPU memory probes into ResourcePlan and make GPU memory accounting honest: blind budgets, uncharged pool caches, no wgpu OOM retry, Linux pressure"
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
  - "Linux reports memory pressure: probe_pressure returns Unknown on every non-macOS OS (ojas-device/src/system.rs:89-100), so E_PRESSURE admission (ojas-capi/src/engine.rs:146-157 refuses only on Critical) can never fire there; PSI (/proc/pressure/memory, cgroup memory.pressure or memory.events) feeds it, and probe_architecture stops being Unknown on Linux (system.rs:71-83); proven in the podman Linux runner"
  - "Idle GPU pools are bounded and charged or trimmable: MetalBackend's Worker::open (ojas-metal/src/device.rs:368-409) never calls set_pool_cache_cap_bytes, so tessl's 2 GiB default pool cache sits outside the Budget (only the legacy gpu.rs:325 session caps it, at 64 MiB); Metal gains a trim like wgpu's trim_pool, and the cap is recorded in the plan"
  - "The Metal budget charges what the device actually allocates, or the gap is bounded and documented: ojas-metal/src/backend.rs:298-300 reserves elems*4 while the pool can round up to 2x (docs/framework-design.md:155, docs/adaptive-resources.md:253)"
  - "wgpu recovers from an allocation OOM the way Metal does (recycle and retry once, ojas-metal/src/device.rs:641-648): today ojas-wgpu/src/context.rs:476-511 fails at once while up to POOL_CAP_BYTES (512 MiB, :52) of idle pooled buffers stay held, and trim_pool (:444) has no caller in the library"
  - "Metal's device-OOM error reports live bytes (device.rs:651-655 builds CapacityExceeded with live: 0; wgpu fills it from budget.live_bytes(), context.rs:464-472)"
  - "A wgpu Queue::drop that times out (DROP_WAIT 2 s, ojas-wgpu/src/context.rs:47) no longer leaks a queue, a device and a thread (docs/pytorch-parity-plan.md:196), or the leak is bounded, counted and reported"
---

# Task brief v1

## Title
Wire GPU memory probes into ResourcePlan and make GPU memory accounting honest: blind budgets, uncharged pool caches, no wgpu OOM retry, Linux pressure

Task: gp-device-memory-probes
Type: bug
Status: ready
Priority: 1 (High)
Severity: high
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
