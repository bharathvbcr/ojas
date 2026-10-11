---
id: "gp-inference-decode-path"
title: "Decode path: device top-k is never wired so sampling reads back the full vocab every token; after-change decode bench; batch>1 decision; hybrid recurrent state"
status: ready
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
  - "A failing test first: DeviceDecoder overrides Forward::forward_top so top-k / temperature sampling, and Go GenerateIDs at temperature 0 (which reaches forward_top(k=1) through Pick::Sample, generate.rs:94 -> decode.rs:75-81), select on the device and read back only the k leaders. Today it does not (impl at ojas-infer/src/device.rs:330-350), the default at decode.rs:35-37 reads the whole [vocab] row, topk_rows has no caller, and the doc at decode.rs:33-34 claims an override that does not exist. The override handles topk_rows returning -inf leaders (draw refuses non-finite, sample.rs:188) and k > 1024 (GPU refusal, backend.rs:1306-1308) with a host fallback. A readback-bytes counter pins k*8 bytes per token"
  - "Every decode speedup cites an after-change run of bench/decode_rows.rs committed beside bench/results/2026-10-08-decode-before/ (none exists for 7f0a9f3, e12d347, 84ab777); the GQA decode row from gp-docs-and-bench-parity lands here"
  - "Batch > 1 decode and paged KV are either implemented or recorded as out of scope with the reason"
  - "CPU decode byte counts before and after the in-place KV change (ojas-cpu/src/kv.rs:141-258) are recorded"
  - "Readback of a device tensor into a caller buffer (read_f32_into or equal) is added if the decode benchmark shows the transient double copy matters (tensor.rs:29 offers only read_bytes), or the deferral is re-recorded with the measured number"
  - "Hybrid layers can carry recurrent state through the trait for chunked prefill and decode: chunked_gdn takes and returns its state (Tape passes initial_state: None, ojas-autograd/src/tape.rs:851,1230) and causal conv1d takes its tail (today 'from a zero state', :98, :537); or the decision that only the ojas-qwen35 provider decodes hybrid layers is recorded"
  - "check_token is defined once (a default method on the Forward trait, decode.rs:22) instead of four copies (ojas-infer/src/device.rs:189,339; gpt.rs:318,472)"
  - "One greedy rule: sample_token at temperature 0 ranks with total_cmp so +0.0 beats -0.0 (sample.rs:236-240), while argmax_token/argmax_rows take the lowest index on ties (gpt.rs:497-500) as torch.argmax does; both rules are pinned by tests (tests/sampling.rs:517-519 and :708-719). The decision is recorded, and the losing test is changed with the reason, before any temperature-0 path is rerouted to argmax_rows (which also refuses -inf where the sampler treats it as a mask)"
---

# Task brief v1

## Title
Decode path: device top-k is never wired so sampling reads back the full vocab every token; after-change decode bench; batch>1 decision; hybrid recurrent state

Task: gp-inference-decode-path
Type: perf
Status: ready
Priority: 2 (Normal)
Severity: medium
Owner: unassigned
Due: none
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

### Re-audit (2026-10-09, at d431949)
Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- Closed by the inference-decode-path merge (8060846, repaired in c5aeee1) and dropped from the criteria list [A]:
  - RoPE uploaded once (device.rs:115-118).
  - Greedy reads back 4 bytes (forward_greedy, device.rs:228-255).
  - Device U32 ids feed embedding with the range proven (Metal Ids::Below; wgpu backend.rs:1597-1630).
  - Partial-select sampler (sample.rs:128-160).
  - CpuGpt duplicate block deleted (kernels.rs gone; gpt.rs uses run_pieces).
  - Sliding window above the trait with a ring KV cache (spec.rs:70, cache.rs:20-60).
  - Cache length unchanged on a norm or head failure (test gpt.rs:857).
  - All-or-nothing prefill.
  - reset and truncate.
  - In-place CPU KV (code).
  - wgpu grid split (backend.rs:2831-2837).
  - Sampler uses exp_exact (sample.rs:205).
- Found [V]: forward_top is defined only as the trait default in ojas-infer/src/decode.rs and nothing overrides it, so the device top-k path is dead. Raised to priority 1 as a bug.

### Second audit (2026-10-09)
Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

- forward_top not being overridden is CONFIRMED [V]. The old criterion 2 (route temperature 0 to argmax_rows) is folded into criterion 1, because fixing forward_top fixes the cost and rerouting would change outputs [A].
- New: greedy tie semantics disagree between the sampler and argmax, and tests pin both [A].

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Lowered P1 -> P2: fine-tune evaluation scores with a forward pass, not decode.

## Acceptance criteria
- [ ] A failing test first: DeviceDecoder overrides Forward::forward_top so top-k / temperature sampling, and Go GenerateIDs at temperature 0 (which reaches forward_top(k=1) through Pick::Sample, generate.rs:94 -> decode.rs:75-81), select on the device and read back only the k leaders. Today it does not (impl at ojas-infer/src/device.rs:330-350), the default at decode.rs:35-37 reads the whole [vocab] row, topk_rows has no caller, and the doc at decode.rs:33-34 claims an override that does not exist. The override handles topk_rows returning -inf leaders (draw refuses non-finite, sample.rs:188) and k > 1024 (GPU refusal, backend.rs:1306-1308) with a host fallback. A readback-bytes counter pins k*8 bytes per token
- [ ] Every decode speedup cites an after-change run of bench/decode_rows.rs committed beside bench/results/2026-10-08-decode-before/ (none exists for 7f0a9f3, e12d347, 84ab777); the GQA decode row from gp-docs-and-bench-parity lands here
- [ ] Batch > 1 decode and paged KV are either implemented or recorded as out of scope with the reason
- [ ] CPU decode byte counts before and after the in-place KV change (ojas-cpu/src/kv.rs:141-258) are recorded
- [ ] Readback of a device tensor into a caller buffer (read_f32_into or equal) is added if the decode benchmark shows the transient double copy matters (tensor.rs:29 offers only read_bytes), or the deferral is re-recorded with the measured number
- [ ] Hybrid layers can carry recurrent state through the trait for chunked prefill and decode: chunked_gdn takes and returns its state (Tape passes initial_state: None, ojas-autograd/src/tape.rs:851,1230) and causal conv1d takes its tail (today 'from a zero state', :98, :537); or the decision that only the ojas-qwen35 provider decodes hybrid layers is recorded
- [ ] check_token is defined once (a default method on the Forward trait, decode.rs:22) instead of four copies (ojas-infer/src/device.rs:189,339; gpt.rs:318,472)
- [ ] One greedy rule: sample_token at temperature 0 ranks with total_cmp so +0.0 beats -0.0 (sample.rs:236-240), while argmax_token/argmax_rows take the lowest index on ties (gpt.rs:497-500) as torch.argmax does; both rules are pinned by tests (tests/sampling.rs:517-519 and :708-719). The decision is recorded, and the losing test is changed with the reason, before any temperature-0 path is rerouted to argmax_rows (which also refuses -inf where the sampler treats it as a mask)

## Planned files
- ojas-infer/src/device.rs
- ojas-infer/src/sample.rs
- ojas-infer/src/kernels.rs
- ojas-infer/src/gpt.rs
- ojas-model/src/spec.rs
- ojas-model/src/block.rs
- ojas-wgpu/src/backend.rs
- ojas-metal/src/backend.rs
- bench/ojas_rows.rs

### Resolution (2026-10-10)
Labels: [C] command output in this session; [M] measured (bench/results/2026-10-10-decode-after/); [I] inferred from code.

1. **forward_top, done, then amended by measurement (see "Decision" below).** `DeviceDecoder::forward_top` overrides the trait default: `topk_rows` on the device, `k` values and `k` ids read back (two readbacks, `8k` bytes), `k` clamped to the vocabulary, `-inf` leaders dropped (`sample::leading_candidates`, which refuses an all-masked row as `ranked_candidates` does), host fallback on `Unsupported` (`k > 1024`), and a lone leader's id fed back from the device. Red first [C]: `tests/device.rs` `check_top` failed on wgpu and Metal with `(36, 13824)` readbacks against `(72, 288)`; `tests/sampling.rs` failed with `(12, 1104)` against `(0, 0)`. Green [C] on wgpu, Metal and CPU, and Go `go test ./...` passes (its CPU-vs-device `GenerateIDs` at temperature 0), rerun after the greedy-rule commit against tessl `de4747a`. The decode.rs doc now names the override. Final state (`cefc864`): device selection only for one leader; `k` > 1 reads the row back and ranks it on the host. Red [C]: `check_top` saw (72, 1440) readbacks against the new (36, 13824); green on Metal, wgpu and CPU.
2. **After-change bench, done; the speedup is not there at k = 50.** [M] `base/` (`de5efaf`) and `after/` (`2b68d11`) runs of `decode_rows.rs` beside `2026-10-08-decode-before/`, plus a 6-round interleaved A/B (`ab/`). Top-50 step readback 201,216 B to 400 B on Metal and wgpu (0 B on CPU). Time per token at `k` = 50: +0.5 ms (+4.6%) on Metal, +3% on wgpu (within noise), because the `topk_rows` kernels make `k` passes over the row. A per-`k` probe [M] (`probes/`) finds no crossover: the device selection is slower than reading the row back and selecting on the host at every `k` from 1 to 1024, on Metal (about 17 µs per pass) and wgpu (about 20 µs). Runs used tessl `de4747a`; tessl `d0b1792` breaks `ojas-metal`'s build (`CeWorkspace::new` takes a sixth argument, E0061), an upstream break left out of this task. The GQA decode rows (`gen_gqa4_*`, 4 KV heads, torch twin `Config(n_kv_head=4)`) land here. The bench's `_gen_traffic` step had a u64 underflow since `7f0a9f3` (it panicked the lane); fixed in `de5efaf`. `ab-k1/` [M]: at `k` = 1 (new row `gen_temp0_p32_n32`, `867c921`) a step reads back 8 B instead of 201,216 B and uploads nothing; its time is below that run's noise (control-subtracted quiet rounds −0.21 to +0.39 ms/token), so no speedup is claimed at any `k`.
3. **Batch > 1 and paged KV: out of scope, recorded** in `ojas-infer/README.md` "Scope", with the reason (single-request consumers; the trait already takes `[B, ...]`).
4. **CPU KV byte counts, recorded** [I] in the results README: about 12.98 MB moved per decode step before `e12d347` (whole-cache copy out and back per K/V write, prefix repack in attention) against 73,728 B after, for nanolab 124M at a 64-slot cache.
5. **Readback into a caller buffer: deferral re-recorded with the number** [M]: 7.2–9.4 µs per 201 KB row on Metal (0.1% of a step), within `read_bytes` noise on wgpu; `docs/typed-storage-plan.md` updated.
6. **Hybrid recurrent state: decision recorded** in `ojas-infer/README.md` "Scope". The brief's alternative ("only the ojas-qwen35 provider decodes hybrid layers") does not match the code: `ojas-qwen35` is training only, and Qwen3.5 decode with carried state is tessl's (`tessl/src/qwen35.rs` `gdn_recurrent`, `conv1d_silu` state_in/state_out). The trait's GDN already takes `initial_state` and returns `final_state`; conv1d has no tail.
7. **check_token, done.** One body (`decode::check_token`); `Forward::check_token` is a default method over `Forward::vocab`. `CpuGpt::forward_tokens` and `DeviceDecoder::run`, which are not trait calls, call the same function.
8. **One greedy rule, done.** The argmax rule won: `+0.0` and `-0.0` are one value and the lowest index wins (`torch.argmax`), via `ojas_core::rank_logits` in the sampler and the `topk_rows` default, and the same key in the Metal and WGSL `topk_rows` kernels. The losing tests changed with the reason in their docs: `sampling.rs` `partial_select_keeps_the_full_sort_order_at_qwen_vocab` (and its oracle) and the CPU, wgpu and Metal `topk_rows` tests. No temperature-0 path was rerouted to `argmax_rows`.

Decision (2026-10-10, made by a Fable agent at the user's direction, "for decisions ask fable"): device selection only at `k` = 1, with the row read back for `k` > 1 on every backend (`cefc864`). Criterion 1's "every `k` <= 1024" is amended with the measured reason: `ab/` +0.5 ms/token at `k` = 50 on Metal, and the probe has the device slower at every `k` from 2 to 1024. `k` = 1 stays on the device: 8 B, no upload, id kept on the device as `forward_greedy`'s is, and no measurable time cost (`ab-k1/`). No kernel card now. A single-pass `topk_rows` is deferred until a trigger: a discrete-GPU backend, or a measured readback well above 0.1 ms per step. Its acceptance would be a committed interleaved A/B on that device in which device top-50 beats the host row by more than the control's noise.
