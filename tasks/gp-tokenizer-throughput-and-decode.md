---
id: "gp-tokenizer-throughput-and-decode"
title: "Tokenizer: BPE throughput (per-pre-token allocation, BTreeMap lookups, no word cache, no benchmark), lossy/streaming detokenize, special-token ids"
status: done
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
Status: done
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
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

### Close-out audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

Closed by 77f8f42, b6a8d33, 7302dd2 and 8b63fa9 (merged in d431949). Every criterion met on code read [A]; no test run in this audit [U].
1. **Benchmark:** bench/bpe_throughput.rs via ojas-data/examples/bpe_throughput.rs, with results in bench/results/2026-10-08-bpe-throughput/. The commit reports 5.65 to 31.4 MB/s.
2. **Buffer reuse and pre-token cache:** MergeScratch (bpe.rs:87) and the cache (:369-391).
3. **Lookups:** merges are a HashMap (:55). The encoder stays a BTreeMap (:40), but it is off the hot path because the byte table (:380) replaces it there.
4. **Lossy and bytes decode:** decode_bytes (:400) and decode_ordinary_lossy (:439), exposed as C ABI mode 3.
5. **Decode output cap:** refuse_decode_len (:333, :405).
6. **Special-token ids:** piece_id (:348), with Go TokenID and EndOfText (go/api.go:507-529).
7. **TokenBin:** refuses non-regular files (tokens.rs:109-128), and capi passes the opened file (train.rs:136).

Follow-ups filed elsewhere:
- Lossy output exceeding the cap, and the BPE allocations the heap ceiling cannot refuse: gp-capi-go-surface.
- The TokenBin metadata/open race: gp-data-io-scaling.
- Vendoring the untracked ojas-data/testdata/gpt2/: gp-ci-toolchain-reproducibility.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Weak claim: the only real-vocab pin for the hash-map and cache change, EOS 50256 and the lossy-decode case is `verified_20_strings_match_tiktoken_encode_ordinary`. It returns early in CI because ojas-data/testdata/ is untracked. No test targets the pre-token cache. The real GPT-2 round trip exists only in bench/bpe_throughput.rs, which never runs. Owned by gp-ci-toolchain-reproducibility (vendoring) and gp-test-suite-integrity (cache test) [A].

## Acceptance criteria
- [x] A tokenizer throughput benchmark (encode_ordinary MB/s and tokens/s on a real GPT-2 vocab over a fixed multi-MB English text, plus decode) is committed under bench/ and lands first; every later item cites it. Today the only performance check is a 2 s wall-clock assertion on the 6-token test vocab (bpe.rs:939-947)
- [x] encode reuses its working buffers across pre-tokens (today each encode() call at bpe.rs:137-157 allocates ids/next/prev/alive/at Vecs and a BTreeSet), and an optional bounded pre-token cache is added or rejected with numbers
- [x] encoder (bpe.rs:33) and merges (bpe.rs:35) move from BTreeMap to a hash map or a sorted-array lookup if the benchmark shows a win; the merge priority order and output ids stay bit-identical, pinned by the existing tiktoken-equality test
- [x] Detokenize has a lossy mode (U+FFFD) or a bytes/streaming decoder so a generation that stops mid-codepoint at max_new_tokens can be shown; today decode_ordinary refuses non-UTF-8 (bpe.rs:274) and ojas-capi/src/tokenize.rs passes the error through
- [x] ojas-capi/src/tokenize.rs:8-9 claims decode refuses text over the cap; decode_ordinary (bpe.rs:253) has no cap. The comment is corrected or decode output is capped, with a test
- [x] Callers can look up special-token ids (EOS `<|endoftext|>` = 50256 for GPT-2) from the loaded tokenizer, and Go exposes it so GenerateIDs Stop ids need not be hard-coded; encode_ordinary keeps treating specials as ordinary text (bpe.rs:232)
- [x] TokenBin::open refuses a non-regular file the way SafeTensors::from_file does (tokens.rs:319 opens any path), and a from_file(File) constructor lets ojas-capi/src/train.rs:125-136 stop reopening through fd_path

## Planned files
- ojas-data/src/bpe.rs
- ojas-data/src/tokens.rs
- ojas-capi/src/tokenize.rs
- ojas-capi/src/train.rs
- go/api.go
- bench/
