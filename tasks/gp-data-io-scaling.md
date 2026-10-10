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
  - "read_into_u32 and next_batch reuse caller-owned buffers (fresh raw per call at ojas-data/src/tokens.rs:314-317; fresh row at sampler.rs:252); read_into and read_into_u32 (tokens.rs:231-283 vs 289-327, ~45 near-identical lines) become one body generic over width; allocation counts pinned by a test"
  - "An optional background prefetch of the next batch exists, bounded to one batch of memory and cancellable, with resume determinism unchanged (g9 forty-steps test still bit-exact)"
  - "BatchSampler can read a multi-shard dataset and partition windows by (rank, world_size); DataCursor stops overloading `shard` to hold the epoch (sampler.rs:195, :205-210); the checkpoint format change is versioned"
  - "model.safetensors.index.json parsing moves from ojas-qwen35/src/names.rs:210-263 into ojas-io and ojas-qwen35 calls it"
  - "Resume checks run identity, not only config, seed and tokenizer hash (resume_from, ojas-model/src/checkpoint.rs:514-545): the checkpoint records the token bin's length, token width and a content fingerprint, and resume refuses a mismatch"
  - "tokenizer_hash is computed from the loaded tokenizer (or its vocab/merges files) instead of being caller-supplied and defaulting to zeros (ojas-capi/src/train.rs:70, :105-106; go/api.go TrainConfig), so the equality check at checkpoint.rs:538 compares something real; a test resumes with a different tokenizer and is refused. Lands with the versioned checkpoint change above"
  - "Library open paths refuse non-regular files before reading, as TokenBin::open does since 8b63fa9: SafeTensors::open (ojas-io/src/safetensors.rs:175-179) and read_checkpoint (ojas-io/src/checkpoint.rs:53) can block on a FIFO; TokenBin::open's metadata-then-open race (tokens.rs:109-115) is closed by checking the opened handle (the C ABI is already safe through open_nofollow)"
  - "Resume refuses a checkpoint saved under a different numerics tier or backend (or records and states the change), since resume_from checks only train JSON, seed and tokenizer hash (ojas-model/src/checkpoint.rs:514-545) and 'bit for bit' otherwise holds only by caller discipline"
  - "A headerless token bin cannot be read at the wrong width silently: only len % width is checked (ojas-data/src/tokens.rs:125-137), so a u32 bin opened as u16, or a FineWeb file opened headerless, reads plausible tokens; the width is recorded or detected"
---

# Task brief v1

## Title
Data and IO scaling: buffer reuse and prefetch in the sampler, multi-shard datasets with rank partitioning, more safetensors dtypes, sharded index in ojas-io, run identity on resume

Task: gp-data-io-scaling
Type: feature
Status: backlog
Priority: 3 (Low)
Severity: low
Owner: unassigned
Due: none
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

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- All criteria open [A]; line references refreshed. New: the FIFO and TOCTOU open paths, and the duplicated TokenBin read bodies [A].

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- New: resume does not check numerics tier or backend; a token bin read at the wrong width passes silently [A].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Criterion 'ojas-io safetensors accepts U32, I32, U8, BOOL and F64 ... F8 decided' moved to gp-ft-quantized-frozen-base, at ojas-7c's request; that brief owns safetensors dtypes now.
- Fine-tune briefs that depend on this one: gp-ft-quantized-frozen-base.

## Acceptance criteria
- [ ] read_into_u32 and next_batch reuse caller-owned buffers (fresh raw per call at ojas-data/src/tokens.rs:314-317; fresh row at sampler.rs:252); read_into and read_into_u32 (tokens.rs:231-283 vs 289-327, ~45 near-identical lines) become one body generic over width; allocation counts pinned by a test
- [ ] An optional background prefetch of the next batch exists, bounded to one batch of memory and cancellable, with resume determinism unchanged (g9 forty-steps test still bit-exact)
- [ ] BatchSampler can read a multi-shard dataset and partition windows by (rank, world_size); DataCursor stops overloading `shard` to hold the epoch (sampler.rs:195, :205-210); the checkpoint format change is versioned
- [ ] model.safetensors.index.json parsing moves from ojas-qwen35/src/names.rs:210-263 into ojas-io and ojas-qwen35 calls it
- [ ] Resume checks run identity, not only config, seed and tokenizer hash (resume_from, ojas-model/src/checkpoint.rs:514-545): the checkpoint records the token bin's length, token width and a content fingerprint, and resume refuses a mismatch
- [ ] tokenizer_hash is computed from the loaded tokenizer (or its vocab/merges files) instead of being caller-supplied and defaulting to zeros (ojas-capi/src/train.rs:70, :105-106; go/api.go TrainConfig), so the equality check at checkpoint.rs:538 compares something real; a test resumes with a different tokenizer and is refused. Lands with the versioned checkpoint change above
- [ ] Library open paths refuse non-regular files before reading, as TokenBin::open does since 8b63fa9: SafeTensors::open (ojas-io/src/safetensors.rs:175-179) and read_checkpoint (ojas-io/src/checkpoint.rs:53) can block on a FIFO; TokenBin::open's metadata-then-open race (tokens.rs:109-115) is closed by checking the opened handle (the C ABI is already safe through open_nofollow)
- [ ] Resume refuses a checkpoint saved under a different numerics tier or backend (or records and states the change), since resume_from checks only train JSON, seed and tokenizer hash (ojas-model/src/checkpoint.rs:514-545) and 'bit for bit' otherwise holds only by caller discipline
- [ ] A headerless token bin cannot be read at the wrong width silently: only len % width is checked (ojas-data/src/tokens.rs:125-137), so a u32 bin opened as u16, or a FineWeb file opened headerless, reads plausible tokens; the width is recorded or detected

## Planned files
- ojas-data/src/tokens.rs
- ojas-data/src/sampler.rs
- ojas-io/src/safetensors.rs
- ojas-qwen35/src/names.rs
