---
id: "gp-data-and-checkpoint-robustness"
title: "Robustify Data Loading (u32 TokenBin), Checkpoint Durability, and Host Memory Safety"
status: ready
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "data"
  - "tokenizer"
  - "vocab"
  - "checkpoint"
  - "robustness"
  - "gusset"
  - "capi"
repositories:
  - "ojas"
planned_files:
  - "ojas-data/src/tokens.rs"
  - "ojas-data/src/sampler.rs"
  - "ojas-model/src/trainer.rs"
  - "ojas-model/src/checkpoint.rs"
  - "ojas-core/src/checkpoint.rs"
  - "ojas-io/src/"
  - "docs/checkpoint-v1.md"
  - "ojas-gusset-engine/src/lib.rs"
  - "ojas-capi/src/"
  - "go/"
acceptance_criteria:
  - "Support u32 little-endian token binary files alongside u16 headerless streams (verified met)"
  - "Extend TokenBin and BatchSampler to stream tokens for vocabularies exceeding 65,535 (verified met)"
  - "Retain backwards compatibility and zero-copy slicing for existing u16 token bins (verified met)"
  - "Unit tests verifying window extraction, epoch shuffling, and boundary invariants on large vocabulary datasets (verified met)"
  - "Add explicit assertion on visited_starts in batch_sampler_end_to_end_with_qwen_u32_tokens for complete u32 epoch coverage"
  - "The rng_state layout is specified and versioned, and a test proves save, resume, and N more steps match an uninterrupted run bit for bit on CPU"
  - "Save checks free space against expected size and refuses before writing"
  - "Device-tensor save streams in pieces, or the decision not to is recorded with the reason"
  - "Large allocations on the Gusset engine path use fallible allocation (try_reserve) and surface OjasError::OutOfMemory to Go rather than aborting"
  - "A Go test drives an allocation failure under a low memory ceiling and receives a clean error instead of a process abort"
---

# Task brief v1

## Title
Robustify Data Loading (u32 TokenBin), Checkpoint Durability, and Host Memory Safety

Task: gp-data-and-checkpoint-robustness
Type: feature
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: data, tokenizer, vocab, checkpoint, robustness, gusset, capi

## Repositories
- ojas

## Description
Consolidated task covering data input streams, checkpoint persistence durability, and host process FFI memory safety:
- 32-bit TokenBin token streaming for large vocabularies (`gp-tokenbin-u32`)
- Checkpoint save/resume hardening (`gp-checkpoint-hardening`)
- Gusset fallible allocation to prevent Go host aborts (`gp-gusset-alloc-abort`)

### Audit status and progress notes (2026-10-05)
1. **u32 TokenBin Status:** Landed in 1f26a4b (status in review). `open_headerless_u32`, `read_into_u32`, and FineWeb u32 headers implemented; sampler reads via `read_into_u32`. Remaining nit: `batch_sampler_end_to_end_with_qwen_u32_tokens` in `ojas-data/tests/u32_tokens.rs:165,179` builds `visited_starts` but never asserts on it. Add the assertion and mark u32 token loading complete.
2. **Checkpoint Hardening:** `docs/checkpoint-v1.md` records three durability gaps: (a) `rng_state` layout is not versioned or checked for bit-exact step resumption across CPU; (b) checkpoint write does not verify available disk space before writing, risking corrupt partial saves on full disks; (c) device tensors are saved as single contiguous buffers rather than streaming in chunks.
3. **Gusset Memory Safety:** In `ojas-gusset-engine/src/lib.rs`, large allocations currently allocate infallibly. If Rust hits an OOM condition, it aborts the entire Go host process. Large allocations must use fallible allocations (`try_reserve`) to bubble `OjasError::OutOfMemory` up to Go safely.

### Execution plan
- **Phase 1 (TokenBin Review Completion):** Add `visited_starts` assertion to `u32_tokens.rs` test and verify `cargo test -p ojas-data --release`.
- **Phase 2 (Checkpoint Hardening):** Version `rng_state` layout with bit-exact resume test. Add free disk space pre-check and streaming device tensor save.
- **Phase 3 (Gusset Fallible Allocation):** Transition engine allocation paths to `try_reserve`, returning structured OOM errors to Go with a low-ceiling test.

## Acceptance criteria
- [x] Support u32 little-endian token binary files alongside u16 headerless streams (verified met)
- [x] Extend TokenBin and BatchSampler to stream tokens for vocabularies exceeding 65,535 (verified met)
- [x] Retain backwards compatibility and zero-copy slicing for existing u16 token bins (verified met)
- [x] Unit tests verifying window extraction, epoch shuffling, and boundary invariants on large vocabulary datasets (verified met)
- [ ] Add explicit assertion on visited_starts in batch_sampler_end_to_end_with_qwen_u32_tokens for complete u32 epoch coverage
- [ ] The rng_state layout is specified and versioned, and a test proves save, resume, and N more steps match an uninterrupted run bit for bit on CPU
- [ ] Save checks free space against expected size and refuses before writing
- [ ] Device-tensor save streams in pieces, or the decision not to is recorded with the reason
- [ ] Large allocations on the Gusset engine path use fallible allocation (try_reserve) and surface OjasError::OutOfMemory to Go rather than aborting
- [ ] A Go test drives an allocation failure under a low memory ceiling and receives a clean error instead of a process abort

## Planned files
- ojas-data/src/tokens.rs
- ojas-data/src/sampler.rs
- ojas-model/src/trainer.rs
- ojas-model/src/checkpoint.rs
- ojas-core/src/checkpoint.rs
- ojas-io/src/
- docs/checkpoint-v1.md
- ojas-gusset-engine/src/lib.rs
- ojas-capi/src/
- go/
