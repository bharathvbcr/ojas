---
id: "gp-tokenbin-u32"
title: "Support 32-bit Token Streams in TokenBin for Large Vocabularies"
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
Status: ready
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

## Acceptance criteria
- [ ] Support u32 little-endian token binary files alongside u16 headerless streams
- [ ] Extend TokenBin and BatchSampler to stream tokens for vocabularies exceeding 65,535 (e.g. Qwen 248k, Llama 128k)
- [ ] Retain backwards compatibility and zero-copy slicing for existing u16 token bins
- [ ] Unit tests verifying window extraction, epoch shuffling, and boundary invariants on large vocabulary datasets

## Planned files
- ojas-data/src/tokens.rs
- ojas-data/src/sampler.rs
- ojas-model/src/trainer.rs
