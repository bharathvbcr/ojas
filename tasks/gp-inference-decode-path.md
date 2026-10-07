---
id: "gp-inference-decode-path"
title: "Inference decode path: per-token uploads and full-vocab readback, O(V log V) sampler, CpuGpt duplicate block, model-level sliding window, decode benchmark, KV-cache failure atomicity"
status: backlog
priority: 2
severity: medium
type: perf
owner: "unassigned"
due: "none"
labels:
  - "inference"
  - "decode"
  - "performance"
  - "kv-cache"
  - "long-context"
repositories:
  - "ojas"
planned_files:
  - "ojas-infer/src/device.rs"
  - "ojas-infer/src/sample.rs"
  - "ojas-infer/src/kernels.rs"
  - "ojas-infer/src/gpt.rs"
  - "ojas-model/src/spec.rs"
  - "ojas-model/src/block.rs"
  - "ojas-wgpu/src/backend.rs"
  - "ojas-metal/src/backend.rs"
  - "bench/ojas_rows.rs"
acceptance_criteria:
  - "A decode tokens/second benchmark (prefill + N decode steps, CPU/Metal/wgpu) is committed under bench/ and a torch-mps reference row exists; this lands first and every later item cites it"
  - "DeviceDecoder::forward stops rebuilding and uploading RoPE rows every call (device-resident table or on-device generation); per-token host uploads counted before and after"
  - "Greedy decode reads back the argmax id (4 bytes) rather than 4*vocab bytes; sampling with temperature reads back only what the sampler needs"
  - "Device-produced U32 ids (on-GPU argmax/sampling) can feed embedding on Metal and wgpu without readback and re-upload; the token-range check moves to the device or is proven by construction (today it reads a host shadow kept at upload)"
  - "sample() uses a partial select: greedy is O(V) with no allocation of V indices, top-k is O(V + k log k); tie order (logit desc, index asc) and NonFinite refusals unchanged and pinned by tests at V=248320"
  - "CpuGpt's host fast path (kernels.rs linear/attend_one/dot, gpt.rs:366-455) is either replaced by DeviceDecoder<CpuBackend> or stops computing the vocab head for non-final prompt tokens and uses CpuBackend threads under Numerics::Fast; the Exact index-order contract is kept"
  - "Sliding windows reach above the Backend trait: ModelSpec carries a window, Graph::sdpa / cached_attention_forward / DeviceDecoder pass it, and the KV cache becomes a ring buffer for windowed layers; training and decode agree with the f64 reference"
  - "Batch > 1 decode and paged KV are either implemented or recorded as out of scope with the reason"
  - "A failing test first: a host forward_token that errors in the final norm or vocab head leaves cache.len() unchanged, as the doc at gpt.rs:364-365 promises; today gpt.rs:450 sets cache.len = pos + 1 before the fallible norm/head at :451-453"
  - "Host prompt prefill is all-or-nothing like DeviceDecoder (device.rs:30-32, :237): HostStep::forward (gpt.rs:514-520) no longer leaves tokens 0..k in the cache when token k fails"
  - "KvCache gains reset() and truncate(len) (DeviceDecoder has reset() at device.rs:133), so a caller can recover after a failure or roll back to a prefix for regenerate; DeviceDecoder::generate documents, as CpuGpt does (gpt.rs:479-480), that the last emitted token is not yet in the cache"
  - "CPU decode stops re-copying the whole cache per token: cached_attention_forward re-packs the full K/V prefix into new scratch on one thread (ojas-cpu/src/kv.rs:146-160), and kv_cache_write copies the whole cache out and back (kv.rs:214-237); per-token bytes moved are counted before and after on the decode benchmark"
  - "wgpu cached attention splits a grid over max_compute_workgroups_per_dimension (Tq*H > 65535) instead of refusing it (ojas-wgpu/src/backend.rs:2643-2649)"
  - "Readback of a device tensor into a caller buffer (read_f32_into or equal) is added if the decode benchmark shows the transient double copy matters (typed-storage-plan.md:88; tensor.rs:29 offers only read_bytes, and the test at tensor.rs:2179 pins the doubled charge), or the deferral is re-recorded with the measured number"
  - "Hybrid layers can carry recurrent state through the trait for chunked prefill and decode: chunked_gdn takes and returns its state (Tape::chunked_gdn passes initial_state: None and drops the final state, ojas-autograd/src/tape.rs:489-519, :818) and causal conv1d takes its tail (today 'from a zero state', :521-523); or the decision that only the ojas-qwen35 provider decodes hybrid layers is recorded"
  - "Seeded temperature/top-k sampling gives the same ids on every platform: sample.rs:143 uses std (z - top).exp(); it moves to ojas-core's exp_exact or the cross-platform risk is pinned by the sampler golden in gp-oracle-and-hardening-coverage"
---

# Task brief v1

## Title
Inference decode path: per-token uploads and full-vocab readback, O(V log V) sampler, CpuGpt duplicate block, model-level sliding window, decode benchmark, KV-cache failure atomicity

Task: gp-inference-decode-path
Type: perf
Status: backlog
Priority: 2 (Normal)
Severity: medium
Labels: inference, decode, performance, kv-cache, long-context

## Repositories
- ojas

## Description
Gap audit 2026-10-07 (rg + Read, DevMap unavailable). V = verified, I = inferred. `docs/pytorch-parity-plan.md:119,129` sets the target "a decode path with few dispatches per token", but nothing in bench/ times decode.

- **Per-token host traffic [V]:** `ojas-infer/src/device.rs:159` builds RoPE rows on the host and uploads them each call, `:160` uploads ids, `:199` uploads an index tensor on prefill, `:206` reads back `4*vocab` bytes even for greedy.
- **U32 ids must originate on the host [V]:** wgpu reads a host shadow for the range check (`ojas-wgpu/src/backend.rs:505-511`); Metal keeps `MetalBuffer::ids` (`ojas-metal/src/backend.rs:70-73`).
- **Sampler sorts the full vocab every token [V]:** `ojas-infer/src/sample.rs:124-133` collects and sorts V indices, then returns `ranked[0]` when temperature is 0.
- **CpuGpt duplicate block [V]:** scalar single-threaded `linear`/`attend_one`/`dot` (`kernels.rs:29-153`) and a second copy of the block (`gpt.rs:366-455`) beside `ojas_model::block_with`; prefill computes the full head for every prompt token (`gpt.rs:514-519` -> `:453`).
- **Sliding window stops at the trait [V]:** `Backend::causal_sdpa_*` takes `window` (`ojas-core/src/backend.rs:663-709`); `ojas-model/src/spec.rs` has no window field (rg: no hits). The kernel side is `gp-attention-kernels` (in progress); this card is the model and decoder side only.
- **Batch of one [V]:** `DeviceDecoder`/`KvCache` fixed at batch 1 (`device.rs:66`).

**Added by the second gap audit 2026-10-07** ([V] re-read by the auditor; [A] read by an audit subagent, not re-read):
- **Cache length advanced before the fallible tail [V]:** gpt.rs:450 `cache.len = pos + 1;` then `rms_norm` and `apply_linear` over the vocab at :451-453, both of which return `Err`.
- **Prefill atomicity, missing reset/truncate, CPU KV repack, wgpu grid refusal, readback double copy, hybrid state, platform exp in the sampler [A].** The CPU KV repack is the path CpuGpt would take if its item above picks the `DeviceDecoder<CpuBackend>` branch. The first three items stay open if CpuGpt keeps its own host path.
