---
id: "gp-data-io-scaling"
title: "Data and IO scaling: buffer reuse and prefetch in the sampler, multi-shard datasets with rank partitioning, more safetensors dtypes, sharded index in ojas-io, run identity on resume"
status: backlog
priority: 3
severity: low
type: feature
owner: "unassigned"
due: "none"
labels:
  - "data"
  - "io"
  - "safetensors"
  - "performance"
  - "distributed"
repositories:
  - "ojas"
planned_files:
  - "ojas-data/src/tokens.rs"
  - "ojas-data/src/sampler.rs"
  - "ojas-io/src/safetensors.rs"
  - "ojas-qwen35/src/names.rs"
acceptance_criteria:
  - "read_into_u32 and next_batch reuse caller-owned buffers (no fresh raw/row Vec per window or batch); allocation counts pinned by a test"
  - "An optional background prefetch of the next batch exists, bounded to one batch of memory and cancellable, with resume determinism unchanged (g9 forty-steps test still bit-exact)"
  - "BatchSampler can read a multi-shard dataset and partition windows by (rank, world_size); DataCursor stops overloading `shard` to hold the epoch; the checkpoint format change is versioned"
  - "ojas-io safetensors accepts U32 (token ids are u32 now), I32, U8, BOOL and F64 with exact round-trip tests; F8 is accepted or refused with a named reason"
  - "model.safetensors.index.json parsing moves from ojas-qwen35/src/names.rs:210-263 into ojas-io and ojas-qwen35 calls it"
  - "Resume checks run identity, not only config and seed: the checkpoint records the token bin's length, token width and a content fingerprint, and resume_from (ojas-model/src/checkpoint.rs:513-545) refuses a grown or swapped bin, which today silently changes the window count and so the shuffle order (check_setup, trainer.rs:499-520, only checks the cursor is inside one epoch)"
  - "tokenizer_hash is computed from the loaded tokenizer (or its vocab/merges files) instead of being caller-supplied and defaulting to zeros (ojas-capi/src/train.rs:70, :105-106; go/api.go TrainConfig), so the equality check at checkpoint.rs:538 compares something real; a test resumes with a different tokenizer and is refused. Lands with the versioned checkpoint change above"
---

# Task brief v1

## Title
Data and IO scaling: buffer reuse and prefetch in the sampler, multi-shard datasets with rank partitioning, more safetensors dtypes, sharded index in ojas-io, run identity on resume

Task: gp-data-io-scaling
Type: feature
Status: backlog
Priority: 3 (Low)
Severity: low
Labels: data, io, safetensors, performance, distributed

## Repositories
- ojas

## Description
Gap audit 2026-10-07 (rg + Read). Separate from `gp-data-and-checkpoint-robustness` (in progress), which covers u32 TokenBin, checkpoint durability and host memory safety, not throughput or scale-out. V = verified.

- `ojas-data/src/tokens.rs:291-294`: `read_into_u32` allocates a fresh `raw` buffer per call [V].
- `ojas-data/src/sampler.rs:248-252`: B positioned reads and a fresh `row` per batch, no prefetch [V].
- `sampler.rs:149`: `BatchSampler::new` takes one `&TokenBin`; `DataCursor.shard` is reused for the epoch (`:207-211`) [V]. No rank/world partitioning.
- `ojas-io/src/safetensors.rs:25-61`: only F32, BF16, F16, I64, U16 [V].
- Sharded-index handling lives in `ojas-qwen35/src/names.rs:210-263` [V].

**Added by the second gap audit 2026-10-07** ([V] re-read by the auditor; [A] read by an audit subagent, not re-read):
- **tokenizer_hash is whatever the caller sends [V]:** ojas-capi/src/train.rs:70 documents the default as zeros and :105-106 copies the caller's bytes; checkpoint.rs:538 compares saved against configured, so two zero hashes always match.
- **No token-bin identity in the checkpoint [A]:** resume_from checks config, seed and tokenizer_hash only (checkpoint.rs:513-545).
