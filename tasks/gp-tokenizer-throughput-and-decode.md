---
id: "gp-tokenizer-throughput-and-decode"
title: "Tokenizer: BPE throughput (per-pre-token allocation, BTreeMap lookups, no word cache, no benchmark), lossy/streaming detokenize, special-token ids"
status: backlog
priority: 2
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "tokenizer"
  - "data"
  - "performance"
  - "capi"
  - "go"
repositories:
  - "ojas"
planned_files:
  - "ojas-data/src/bpe.rs"
  - "ojas-data/src/tokens.rs"
  - "ojas-capi/src/tokenize.rs"
  - "ojas-capi/src/train.rs"
  - "go/api.go"
  - "bench/"
acceptance_criteria:
  - "A tokenizer throughput benchmark (encode_ordinary MB/s and tokens/s on a real GPT-2 vocab over a fixed multi-MB English text, plus decode) is committed under bench/ and lands first; every later item cites it. Today the only performance check is a 2 s wall-clock assertion on the 6-token test vocab (bpe.rs:939-947)"
  - "encode reuses its working buffers across pre-tokens (today each encode() call at bpe.rs:137-157 allocates ids/next/prev/alive/at Vecs and a BTreeSet), and an optional bounded pre-token cache is added or rejected with numbers"
  - "encoder (bpe.rs:33) and merges (bpe.rs:35) move from BTreeMap to a hash map or a sorted-array lookup if the benchmark shows a win; the merge priority order and output ids stay bit-identical, pinned by the existing tiktoken-equality test"
  - "Detokenize has a lossy mode (U+FFFD) or a bytes/streaming decoder so a generation that stops mid-codepoint at max_new_tokens can be shown; today decode_ordinary refuses non-UTF-8 (bpe.rs:274) and ojas-capi/src/tokenize.rs passes the error through"
  - "ojas-capi/src/tokenize.rs:8-9 claims decode refuses text over the cap; decode_ordinary (bpe.rs:253) has no cap. The comment is corrected or decode output is capped, with a test"
  - "Callers can look up special-token ids (EOS `<|endoftext|>` = 50256 for GPT-2) from the loaded tokenizer, and Go exposes it so GenerateIDs Stop ids need not be hard-coded; encode_ordinary keeps treating specials as ordinary text (bpe.rs:232)"
  - "TokenBin::open refuses a non-regular file the way SafeTensors::from_file does (tokens.rs:319 opens any path), and a from_file(File) constructor lets ojas-capi/src/train.rs:125-136 stop reopening through fd_path"
---

# Task brief v1

## Title
Tokenizer: BPE throughput (per-pre-token allocation, BTreeMap lookups, no word cache, no benchmark), lossy/streaming detokenize, special-token ids

Task: gp-tokenizer-throughput-and-decode
Type: perf
Status: backlog
Priority: 2 (Normal)
Severity: medium
Labels: tokenizer, data, performance, capi, go

## Repositories
- ojas

## Description
Second gap audit 2026-10-07. Labels: [V] re-read by the auditor; [A] read by an audit subagent, not re-read.

No card on the board owns the tokenizer's speed or its decode behaviour. gp-ci-toolchain-reproducibility owns only the silent-pass tiktoken test; gp-data-and-checkpoint-robustness (review) owns u32 token bins.

- **Allocation per pre-token [V]:** `encode_ordinary` (bpe.rs:235) calls `encode` (bpe.rs:137) once per regex pre-token; `encode` builds fresh vectors and a `BTreeSet<(u32, usize)>` (bpe.rs:156) every call.
- **Ordered maps on the hot path [V]:** `encoder: BTreeMap<String, u32>` (bpe.rs:33) is queried once per character; `merges: BTreeMap<(u32,u32),(u32,u32)>` (bpe.rs:35).
- **No benchmark [A]:** nothing under bench/ or any benches dir calls `encode_ordinary`. The speed cost above is inferred until the benchmark exists.
- **Decode fails on partial UTF-8 [V]:** `String::from_utf8(bytes)` at bpe.rs:274.
- **Doc/code mismatch [V]:** ojas-capi/src/tokenize.rs:8-9 says both directions refuse text over the cap.
- **Specials [V]:** bpe.rs:232, "Special tokens are ordinary text". Go accepts Stop ids (go/api.go:493-495 [A]) but cannot ask the tokenizer for EOS.
- **TokenBin open [V]:** tokens.rs:319 `File::open(path)` with no regular-file check.
