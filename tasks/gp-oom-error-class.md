---
id: "gp-oom-error-class"
title: "Device out-of-memory is misclassified on every GPU backend: Metal's retry never fires on tessl's real error, and wgpu and CUDA OOM reach callers as Backend instead of CapacityExceeded"
status: in_progress
priority: 0
severity: medium
type: bug
owner: "unassigned"
due: "none"
labels:
  - "metal"
  - "wgpu"
  - "cuda"
  - "errors"
  - "budget"
  - "capi"
  - "robustness"
repositories:
  - "ojas"
planned_files:
  - "ojas-metal/src/device.rs"
  - "ojas-wgpu/src/context.rs"
  - "ojas-cuda/src/error.rs"
  - "ojas-cuda/src/backend.rs"
  - "ojas-core/src/error.rs"
  - "ojas-capi/src/lib.rs"
acceptance_criteria:
  - "A failing test first, per backend, driven through the real allocation path (not an injected string; device.rs:725-731 injects the matcher's own text): on Metal a tessl newBuffer failure ('newBuffer({key}) failed', ../tessl/src/runtime.rs:262-263) takes the recycle-and-retry path and, if still short, returns CapacityExceeded; tessl's permanent 'rounded buffer size exceeds device limit' (:253) is refused at once with the device limit as the cap, not retried twice; the :249 failure and gdn_bwd's retry-then-metal_err path (device.rs:2072-2077) are classified the same way"
  - "Allocation failure is classified at the point of detection on Metal and wgpu; no backend decides 'out of memory' by substring matching on another crate's message (tessl exposes a typed error, or ojas matches its real text behind one function tested against tessl's real path)"
  - "wgpu's post-retry OOM (ojas-wgpu/src/context.rs:950-954) and bind-group OOM (:1309-1318) map to CapacityExceeded with live bytes from the budget, and the two tests that pin them as Backend (:2071-2074 and the post-retry test) are changed with the reason; CUDA's existing CudaError::Capacity (error.rs:97-110, runtime.rs:574-586) survives From<CudaError> for OjasError (error.rs:213-227, a one-arm fix), and check_free's refusal (backend.rs:163-164) uses it"
  - "The C ABI and Go already map CapacityExceeded to E_CAPACITY / ErrCapacity (ojas-capi/src/lib.rs:71,99; go/ffi.go:96; asserted at go/api_test.go:208,237); a Go test now drives a real device OOM on Metal to ErrCapacity"
  - "Metal's OOM error reports live bytes (already done at device.rs:707-712 via current_allocated_bytes; unreachable only because of criterion 1's mismatch)"
---

# Task brief v1

## Title
Device out-of-memory is misclassified on every GPU backend: Metal's retry never fires on tessl's real error, and wgpu and CUDA OOM reach callers as Backend instead of CapacityExceeded

Task: gp-oom-error-class
Type: bug
Status: ready
Priority: 0 (Urgent)
Severity: medium
Owner: unassigned
Due: none
Labels: metal, wgpu, cuda, errors, budget, capi, robustness

## Repositories
- ojas

## Description
Filed by the 2026-10-09 task audit (at d431949). One bug class seen on three backends: a device allocation failure is not recognised as one, so the recycle-and-retry path does not run and callers (C ABI, Go) get a generic `E_BACKEND` instead of `E_CAPACITY`.
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- **Metal retry never fires [V].** `exhausted()` (ojas-metal/src/device.rs:394-396) matches `"newBuffer failed"`, but tessl produces `format!("newBuffer({key}) failed")` (../tessl/src/runtime.rs:262-263). A real OOM therefore skips the waited-commit recycle and the retry, and surfaces as `Backend`. The only test injects the string the matcher expects, so it passes against the wrong contract.
- **wgpu final OOM is Backend [A].** After its one retry, ojas-wgpu/src/context.rs:950-954 returns `OjasError::Backend`.
- **CUDA OOM is Backend [A].** `From<CudaError> for OjasError` (ojas-cuda/src/error.rs:213-227) special-cases only device-lost; its test pins OOM as Backend. In `CudaBackend::upload` the budget refusal stays `CapacityExceeded` but the device-free refusal (`check_free`, backend.rs:163-164) becomes Backend.
- **Fix the class at the owner:** a typed allocation-failure kind produced at the point of detection (the way `DeviceLost` was done in gp-gpu-runtime-hardening), not substring matching on a sibling crate's message. Metal needs tessl to expose a typed OOM, or ojas matches tessl's real text behind one function with a test that calls tessl's real allocation path.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- The Metal mismatch is CONFIRMED [V]. Severity is lowered to medium: every Metal op reserves against the session Budget first (backend.rs:9-11), so this path fires only when the caller's budget exceeds real free device memory. gp-device-memory-probes now plans that budget down to the device room.
- Corrections: CUDA already has a typed capacity error, the C and Go mappings already exist, and Metal already reports live bytes. Criteria 2-5 are rewritten to the real remaining work [A].
- Missed instances, now included: tessl's 'exceeds device limit' string is matched as retryable although it is permanent; wgpu bind-group OOM; a second wgpu test pins the wrong kind [A].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P0: Metal out-of-memory must be retried and reported as capacity for a long fine-tune run.
- Fine-tune briefs that depend on this one: gp-ft-bf16-frozen-base, gp-ft-trainer-loop.

## Acceptance criteria
- [ ] A failing test first, per backend, driven through the real allocation path (not an injected string; device.rs:725-731 injects the matcher's own text): on Metal a tessl newBuffer failure ('newBuffer({key}) failed', ../tessl/src/runtime.rs:262-263) takes the recycle-and-retry path and, if still short, returns CapacityExceeded; tessl's permanent 'rounded buffer size exceeds device limit' (:253) is refused at once with the device limit as the cap, not retried twice; the :249 failure and gdn_bwd's retry-then-metal_err path (device.rs:2072-2077) are classified the same way
- [ ] Allocation failure is classified at the point of detection on Metal and wgpu; no backend decides 'out of memory' by substring matching on another crate's message (tessl exposes a typed error, or ojas matches its real text behind one function tested against tessl's real path)
- [ ] wgpu's post-retry OOM (ojas-wgpu/src/context.rs:950-954) and bind-group OOM (:1309-1318) map to CapacityExceeded with live bytes from the budget, and the two tests that pin them as Backend (:2071-2074 and the post-retry test) are changed with the reason; CUDA's existing CudaError::Capacity (error.rs:97-110, runtime.rs:574-586) survives From<CudaError> for OjasError (error.rs:213-227, a one-arm fix), and check_free's refusal (backend.rs:163-164) uses it
- [ ] The C ABI and Go already map CapacityExceeded to E_CAPACITY / ErrCapacity (ojas-capi/src/lib.rs:71,99; go/ffi.go:96; asserted at go/api_test.go:208,237); a Go test now drives a real device OOM on Metal to ErrCapacity
- [x] Metal's OOM error reports live bytes (already done at device.rs:707-712 via current_allocated_bytes; unreachable only because of criterion 1's mismatch)

## Planned files
- ojas-metal/src/device.rs
- ojas-wgpu/src/context.rs
- ojas-cuda/src/error.rs
- ojas-cuda/src/backend.rs
- ojas-core/src/error.rs
- ojas-capi/src/lib.rs

## Progress 2026-10-12 (branch gitpulse/oom-error-class)

- Criterion 1 (Metal): partial. Cause confirmed: tessl's real failures are `buffer request N exceeds host/device allocation limit` (runtime.rs:269-270, which `exhausted()` also missed), `rounded buffer size exceeds device limit` (:273-275) and `newBuffer({key}) failed` (:283); the old matcher read `"newBuffer failed"`. `classify_alloc` (ojas-metal/src/device.rs) now sorts them into Limit (permanent, refused at once, no waited commit), Exhausted (recycle, trim, retry) and Other; `Worker::retrying` serves `alloc`, `fresh_from` and gdn backward's workspace (`gdn_workspace`). Real-path tests: `a_real_oversize_allocation_is_capacity_exceeded_without_a_retry`, `a_real_oversize_gdn_workspace_is_capacity_exceeded`, `classify_alloc_reads_tessls_failures`. The newBuffer failure cannot be forced on a 64 GiB Mac without resident memory (allocations are lazy; 30 GiB succeeds), so that arm is pinned by `classify_alloc_matches_the_text_tessl_builds` (reads tessl's source) and the injected test now injects tessl's real text. No Backend op reaches the device limit: every kernel indexes with 32 bits (max 16 GiB f32), so the real test is at the Worker.
- Criterion 2: partial. wgpu is typed at detection (error scope filtered to OutOfMemory). Metal cannot be typed without a tessl change: tessl needs a typed alloc error (`Limit { requested, max_buffer_length }` / `Device { bytes }`) and `GpuRuntime::max_buffer_length()` (the cap is `recommended_working_set` until then). `classify_alloc` is the single place that reads tessl's text; delete it when tessl lands the type. tessl was not edited.
- Criterion 3: wgpu `new_buffer` post-retry and bind-group OOM now `CapacityExceeded` (via `out_of_memory`, which keeps `DeviceLost`); the budget is not plumbed into the context so `live` is the allocator's reserved bytes (0 if unreported) and `cap` the binding cap. The post-retry test now expects CapacityExceeded; new `a_bind_group_out_of_memory_is_capacity_exceeded`. CUDA: `CudaError::Capacity` survives `From<CudaError> for OjasError` (new `capacity_of`, `check_free` uses it); compile-checked only, GPU tests cannot run here.
- Criterion 4 (Go real device OOM on Metal): blocked, see criterion 1 (no public op can ask for more than the device limit; a Go test would need a tens-of-GiB resident allocation). The C ABI and Go mappings already exist.
- Criterion 5: unchanged, now reachable.

- VERIFICATION STATUS: nothing above has been compiled or run. Every cargo invocation queued behind the machine lock, held for over 2 hours by another tree's `fg-post` job (pid 84257, ojas-wt-wgpu-folded-grid-tails). Not set to review for that reason. Needed before review: `cargo test -p ojas-metal --features metal --lib -- a_real_oversize classify_alloc a_device_refusal`, the existing metal retry tests in backend.rs, `cargo test -p ojas-wgpu --lib out_of_memory` and `bind_group_out_of_memory`, `cargo test -p ojas-cuda --lib`, `cargo check -p ojas-cuda --features cuda --tests`. Fail-first runs against unmodified code were also not made.
