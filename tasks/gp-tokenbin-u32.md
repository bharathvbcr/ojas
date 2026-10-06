---
id: "gp-tokenbin-u32"
title: "Support 32-bit Token Streams in TokenBin for Large Vocabularies"
status: review
priority: 1
severity: high
type: feature
owner: "unassigned"
due: "none"
labels:
  - "data"
  - "tokenizer"
  - "vocab"
repositories:
  - "ojas"
planned_files:
  - "ojas-data/src/tokens.rs"
  - "ojas-data/src/sampler.rs"
  - "ojas-model/src/trainer.rs"
acceptance_criteria:
  - "Support u32 little-endian token binary files alongside u16 headerless streams"
  - "Extend TokenBin and BatchSampler to stream tokens for vocabularies exceeding 65,535 (e.g. Qwen 248k, Llama 128k)"
  - "Retain backwards compatibility and zero-copy slicing for existing u16 token bins"
  - "Unit tests verifying window extraction, epoch shuffling, and boundary invariants on large vocabulary datasets"
---

# Task brief v1

## Title
Support 32-bit Token Streams in TokenBin for Large Vocabularies

Task: gp-tokenbin-u32
Type: feature
Status: review
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: data, tokenizer, vocab

## Repositories
- ojas

## Description
ojas_data::TokenBin is currently hardcoded to read headerless little-endian u16 token streams. This caps the vocabulary size at 65,535. While sufficient for nanolab GPT-2 (V=50,304), it physically prevents loading datasets for modern models such as Qwen (V=248,320) or Llama 3 (V=128,256).

This task extends TokenBin and the data pipeline to support 32-bit (u32) token representations while maintaining zero-copy backwards compatibility with legacy u16 token files.

### Progress (audit of 9668bfa, 2026-10-05, code inspection only; tests not executed)
Landed in 1f26a4b; moved to review. A reviewer should run `cargo test -p ojas-data --release` before marking it done.
- u32: `open_headerless_u32` (`ojas-data/src/tokens.rs:69`), `read_into_u32` (:269), FineWeb u32 headers (:97-160). `read_into` into u16 now errors instead of truncating (:248-258).
- The sampler always reads through `read_into_u32` (`ojas-data/src/sampler.rs:175-177`). u16 `open_headerless` is unchanged (:46).
- "Zero-copy" was never true: the reader used `read_exact_at` before and after this change, so there was nothing to retain.
- Tests: `ojas-data/tests/u32_tokens.rs`, plus unit tests at `tokens.rs:626-783`.
- **Remaining nit:** `batch_sampler_end_to_end_with_qwen_u32_tokens` builds `visited_starts` (`u32_tokens.rs:165,179`) but never asserts on it, so u32 epoch coverage is only covered indirectly. Add the assertion.

## Acceptance criteria
- [x] Support u32 little-endian token binary files alongside u16 headerless streams
- [x] Extend TokenBin and BatchSampler to stream tokens for vocabularies exceeding 65,535 (e.g. Qwen 248k, Llama 128k)
- [x] Retain backwards compatibility and zero-copy slicing for existing u16 token bins
- [x] Unit tests verifying window extraction, epoch shuffling, and boundary invariants on large vocabulary datasets

## Planned files
- ojas-data/src/tokens.rs
- ojas-data/src/sampler.rs
- ojas-model/src/trainer.rs
