---
id: "gp-simd-safe-api-soundness"
title: "ojas-simd exposes safe functions whose memory safety rests on caller discipline: a lifetime-free Send raw-pointer band with a safe write(), and write_chunk/commit paths that set_len over lanes a closure may not have written"
status: ready
priority: 1
severity: medium
type: bug
owner: "unassigned"
due: "none"
labels:
  - "simd"
  - "unsafe"
  - "soundness"
  - "cpu"
repositories:
  - "ojas"
planned_files:
  - "ojas-simd/src/lib.rs"
  - "ojas-simd/src/arch.rs"
  - "ojas-cpu/src/layout.rs"
  - "ojas-cpu/src/pointwise.rs"
acceptance_criteria:
  - "The band API is sound for safe callers: either the functions are `unsafe fn` with their contract stated, or the band carries a lifetime (PhantomData<&'a mut [f32]>) that stops it outliving the call, and write() returns a proof token body must hand back before commit"
  - "write_chunk / into_vec cannot set_len over unwritten lanes from safe code (a completion count checked before set_len, or the functions become unsafe)"
  - "A compile-fail test (trybuild or a doc test with compile_fail) shows a band cannot escape the closure; Exact-tier outputs and the A/B timings of the callers are unchanged"
---

# Task brief v1

## Title
ojas-simd exposes safe functions whose memory safety rests on caller discipline: a lifetime-free Send raw-pointer band with a safe write(), and write_chunk/commit paths that set_len over lanes a closure may not have written

Task: gp-simd-safe-api-soundness
Type: bug
Status: ready
Priority: 1 (High)
Severity: medium
Owner: unassigned
Due: none
Labels: simd, unsafe, soundness, cpu

## Repositories
- ojas

## Description
Filed by the second audit (2026-10-09) [A, read in full: 48/48 unsafe lines in ojas-simd].

- **Band escape:** `with_nanolab_token_bands` (ojas-simd/src/lib.rs:891-932) hands `body` two `NanolabTokenBand`s. A band is a raw `*const`/`*mut` pair with no lifetime, is `unsafe impl Send` (arch.rs:674), and has a safe `pub fn write(self)` (arch.rs:657-665).
  - Safe code can move a band into a detached thread or an outer Option and call write() after dst was reallocated or dropped: a use-after-free.
  - A `body` that returns Ok(true) without writing both bands (e.g. after mem::forget(worker)) makes commit_nanolab_spare set_len over uninitialized lanes (lib.rs:927-930).
- **Unwritten lanes:** `ReservedF32::write_chunk` (arch.rs:1263-1294, re-exported lib.rs:1093) is safe. Its only guard is a doc rule that the closure stores every lane, and into_vec then set_lens (arch.rs:1317).
- **Today's callers are correct:**
  - ojas-cpu/src/layout.rs:342-357 always joins, with HandoffJoin on unwind (:37-48), and the pool's join waits (pool.rs:392-401).
  - pointwise.rs:535 writes every lane.
- **Why it matters:** this is how `forbid(unsafe_code)` ojas-cpu gets a persistent pool writing borrowed outputs. Whether to adopt unsafe or a dependency for it is still the user's decision (recorded in gp-cpu-autograd-hot-paths).

Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

## Acceptance criteria
- [ ] The band API is sound for safe callers: either the functions are `unsafe fn` with their contract stated, or the band carries a lifetime (PhantomData<&'a mut [f32]>) that stops it outliving the call, and write() returns a proof token body must hand back before commit
- [ ] write_chunk / into_vec cannot set_len over unwritten lanes from safe code (a completion count checked before set_len, or the functions become unsafe)
- [ ] A compile-fail test (trybuild or a doc test with compile_fail) shows a band cannot escape the closure; Exact-tier outputs and the A/B timings of the callers are unchanged

## Planned files
- ojas-simd/src/lib.rs
- ojas-simd/src/arch.rs
- ojas-cpu/src/layout.rs
- ojas-cpu/src/pointwise.rs
