---
id: "gp-capi-go-surface"
title: "C ABI and Go API: persistent decode sessions, eval-loss op, TrainConfig parity, typed error sentinels, payload-parser fuzzing, failure containment, handle lifetime"
status: backlog
priority: 2
severity: medium
type: feature
owner: "unassigned"
due: "none"
labels:
  - "capi"
  - "go"
  - "gusset"
  - "api"
  - "errors"
  - "fuzzing"
repositories:
  - "ojas"
planned_files:
  - "ojas-capi/src/engine.rs"
  - "ojas-capi/src/generate.rs"
  - "ojas-capi/src/train.rs"
  - "ojas-capi/src/session.rs"
  - "ojas-capi/src/wire.rs"
  - "ojas-capi/src/lib.rs"
  - "go/api.go"
  - "go/ffi.go"
  - "go/api_test.go"
  - "go/governor_test.go"
acceptance_criteria:
  - "A decode-session opcode keeps a DeviceDecoder and its KV cache alive across SAMPLE calls, so a multi-turn Go chat appends tokens instead of re-prefilling history; sessions are bounded, freed explicitly, and charged to the budget"
  - "An eval/forward-loss opcode returns held-out loss without an optimizer step, backed by ojas_model::forward_loss; Go exposes it"
  - "The C TRAIN_FIELDS and Go TrainConfig carry activations (activation checkpointing), ignore_index, chunk (CE chunk) and git_sha from ojas_model::TrainConfig; Go ModelSpec carries tie_embeddings; each field has a round-trip test and an out-of-range refusal test"
  - "Unknown id, bad argument and unsupported request get typed ojas:E_* kinds chosen at the point of production (as DeviceLost already is); Go exposes sentinels usable with errors.Is; go/ffi.go no longer string-matches 'workers still running' and Go tests stop matching message text where a sentinel exists"
  - "A cargo-fuzz (or equivalent bounded) target covers wire::Reader / Fields::read and dispatch with hostile payloads; any crash found gets a regression test"
  - "Payload decode allocates fallibly: Reader::values (wire.rs:70), Fields::u32s (wire.rs:204), tokenize encode/decode (tokenize.rs:62,75), train token batches (train.rs:252-253) and the SAMPLE result (engine.rs:179) use try_reserve and return OjasError::OutOfMemory; a Go test under SetHeapCeiling drives each path to a clean error instead of an abort (go/heap_ceiling_test.go covers only NewModel and OpenTrainer)"
  - "The last-error channel is per call, not one global Mutex<String> (session.rs:232): a failing call's message cannot be replaced by another caller's, or cleared by engineReset or a successful set_model_root (session.rs:296); Go's setRoot then lastCError pair (ffi.go:442-455) is serialized with engineInit/engineReset or replaced; a race test proves it"
  - "dispatch (engine.rs:107-188) catches a handler panic per call and poisons only that model (E_POISONED, session.rs:210-213), so one panic no longer makes Go reset the engine and drop every model (ffi.go:480-483)"
  - "A payload above the gusset buffer budget (64 MiB default, ffi.go:79) is refused up front with ErrCapacity and a message naming the size, not gusset.ErrBufferBudget's 'Free buffers before allocating more'; Tokenize/Detokenize/LoadTokenizer get the same up-front size check the path APIs have (api.go:452-483)"
  - "A save refused for lack of disk space (today OjasError::OutOfRange, ojas-model/src/checkpoint.rs:79-84, 466-480) gets its own error kind and a Go sentinel usable with errors.Is"
  - "Tokenize/Detokenize stop taking the model state lock (tokenize.rs:52; LoadTokenizer :36 may keep it) since the tokenizer is immutable after load, so they no longer return ErrBusy during TrainStep/GenerateIDs; TOKENIZE checks cancellation during a long encode, not only once before it (:47)"
  - "Large calls copy less across the boundary: Go encodes payloads straight into the gusset buffer and decodes results from the view before Free (ffi.go:487-506, api.go:631-640); Rust builds token tensors without an intermediate Vec where it can (train.rs:252); copies per call counted before and after"
  - "Go wraps model ids in a Model type with Close and runtime.AddCleanup, so a forgotten id stops holding its share of the memory ceiling and one of the 64 session slots; session() keeps the tensors count LOAD/NEW/RESUME return (api.go:625-628)"
  - "SetModelRoot either uses its ctx and the call gate or drops the parameter (api.go:53-58), and changing the root while models are open is refused or pinned per session, so a later SAVE or LoadTokenizer cannot resolve relative paths against a new root (session.rs:282-298, train.rs:299)"
  - "The Go module builds outside this directory layout: the replace of gusset with ../../../devtools/gusset (go/go.mod:7) and the .pc files hard-coding the sibling checkout and target/debug (no Linux release .pc) give way to a documented build a fresh clone can follow"
---

# Task brief v1

## Title
C ABI and Go API: persistent decode sessions, eval-loss op, TrainConfig parity, typed error sentinels, payload-parser fuzzing, failure containment, handle lifetime

Task: gp-capi-go-surface
Type: feature
Status: backlog
Priority: 2 (Normal)
Severity: medium
Labels: capi, go, gusset, api, errors, fuzzing

## Repositories
- ojas

## Description
Gap audit 2026-10-07 (rg + Read, DevMap unavailable). V = verified, I = inferred.

- **No decode session [V]:** `ojas-capi/src/generate.rs:93-95` builds a new `DeviceDecoder` per call; `ojas-infer/src/device.rs:66-79` allocates and uploads a zeroed KV cache per layer each time. Weights are not re-uploaded (Metal/wgpu `upload` return the tensor unchanged), but the history is re-prefilled every turn.
- **No evaluation op [V]:** the opcode table (`ojas-capi/src/engine.rs:1-20`) has none; `ojas_model::forward_loss` exists.
- **TrainConfig lag [V]:** `ojas_model::TrainConfig` (`trainer.rs:89-127`) has `activations`, `ignore_index`, `chunk`, `git_sha`; `TRAIN_FIELDS` (`ojas-capi/src/train.rs:40-57`, 16 fields) and Go `TrainConfig` (`go/api.go:271-294`) have none. `muon_ns5` is already tracked in `gp-bf16-compute-tier`.
- **Untyped errors [V]:** `ojas-capi/src/session.rs:356,364` return `format!("unknown model id {id}")`; Go tests use `strings.Contains` on messages (34 in api_test.go, 7 in governor_test.go); `go/ffi.go:645` matches "workers still running" in production code. `gp-gpu-runtime-hardening` covers DeviceLost only.
- **No fuzzing of the C payload parser [V]:** only hand-written truncation cases in `wire.rs`; the one fuzz target is Go `FuzzDecodeProfile` (`go/profile_test.go:104`). Mind the mac resource limits: run fuzzing bounded (time and memory capped), not open-ended.

Ask before widening the C ABI's attack surface beyond these ops.

**Added by the second gap audit 2026-10-07** ([V] re-read by the auditor; [A] read by an audit subagent, not re-read):
- **Infallible decode under a heap ceiling [V wire.rs:70, A others]:** `Reader::values` ends in `.collect()` on caller-sized input; the same holds for the other sites in the criterion. This contradicts the contract in ojas-gusset-engine/src/lib.rs:13-15 and go/api.go:588-592 [A]. The abort itself is inferred, not run.
- **One global error slot [V]:** `static LAST_ERROR: Mutex<String>` (session.rs:232). The race with engineInit/engineReset is inferred from ffi.go:416-455 and :522 [A].
- **Panic blast radius [V]:** `dispatch` (engine.rs:107) has no `catch_unwind`; Go's ErrPanic path resets the engine (ffi.go:480-483 [A]), so the per-model E_POISONED path cannot be reached from Go after a panic.
- **Tokenizer behind the model lock [V]:** `session.lock_state()` at tokenize.rs:36 and :52.
- **Module layout [V]:** `replace github.com/bharathvbcr/gusset => ../../../devtools/gusset` (go/go.mod:7).
- Buffer-budget error, disk-full kind, copy counts, Model handle and SetModelRoot items are [A].
