# Unification Plan: ojas as the Training Framework, tessl as the Metal Extension

Status (2026-10-31): a plan, not a landing. No code changed with this document. The eleven owner decisions in section 10 are **OPEN**; nothing here presumes an answer, and no phase that depends on one starts before it is made.

**Update, 2026-11-01: the safetensors half of P8 has landed.** The owner decided that ojas owns safetensors IO and checkpoints. `ojas_io::safetensors` is canonical, tessl's reader is a thin adapter over it with its 950-line parser deleted, and `ojas-qwen35` opens checkpoints through ojas-io instead of through tessl. The decision, the evidence and what it cost are recorded in [`io-ownership.md`](io-ownership.md), paired with `tessl/docs/io-ownership.md`. The rest of P8 — tessl's `json.rs`, `npy.rs` and the per-entry slab sink — has not landed, and `plain.rs` stays: `npy.rs` needs it, and the adapter keeps tessl's payload policy there. The other ten decisions in section 10 remain open.

This page is the repository's copy of the synthesized judgement written for the ojas and tessl unification. It keeps the source plan's labels: **V** verified by a read, grep or DevMap query (with `file:line`), **I** inferred, **U** unverified. The tessl claims are grep and read only, because tessl's DevMap index is stale (generation 1470). ojas's DevMap is fresh (generation 7156) but `degraded: partial`: every walk hit its depth cap, so DevMap counts below are lower bounds. Line counts are `wc -l` unless marked "non-blank". Nothing was built or run on the GPU to write the plan.

## 1. Owner direction

Two statements bind the plan. The source plan does not record when each was made, so the date column is the date the pair was recorded in this repository, and the original dates are unverified.

| Statement | Content | Recorded |
| :--- | :--- | :--- |
| Decision | The Cargo arrow points from the extension to ojas's core, and the Metal backend is tessl's. ojas is the universal framework. | 2026-10-31 |
| Refinement | ojas has more focus on training with Rust (CPU, CUDA, Metal through tessl, wgpu). tessl is Metal and heavy optimization for inference on the Mac, and ojas's Metal extension. | 2026-10-31 |

Two goals follow from them: no duplicate code, and one CPU and GPU memory pool on unified memory.

The refinement came after the three proposals the plan judged, so the plan resolves a fork none of them decided: who owns Metal training (section 4).

## 2. Verdict of the judgement

Three proposals were scored on seven dimensions (1 to 10).

| Dimension | A: tessl absorbs the backend | B: device HAL in ojas-core | C: staged extension |
| :--- | ---: | ---: | ---: |
| Correctness against the code | 7 | 8 | 7 |
| Memory design soundness | 8 | 8 | 8 |
| Dedup completeness | 8 | 4 | 5 |
| Migration safety | 5 | 8 | 7 |
| Respect for constraints | 5 | 9 | 7 |
| Cost | 5 | 8 | 6 |
| Adherence to both owner statements | 7 | 4 | 6 |
| **Total** | **45** | **49** | **46** |

B wins on points. It is the only proposal whose every phase is consistent with the code as it is evolving (kernel lanes are adding `Cmd` variants to ojas-metal now, section 9), it needs no tessl MSRV change, and its claims checked clean. It loses on the two things the owner asked for: the Cargo arrow stays `ojas-metal -> tessl`, and tessl's generic copies (readers, `String` errors, the hand-written training step) stay. The synthesis takes B's HAL in ojas-core and additive phase order, A's dedup targets (readers, scalar formation, the ojas-metal accounting layer) behind one MSRV probe, and C's stop points and sibling-crate mechanism.

### Spot-check ledger

The verifications behind the scores, condensed.

| Claim | Result |
| :--- | :--- |
| `ojas-core` has an empty `[dependencies]` table and `forbid(unsafe_code)` | V: `ojas-core/Cargo.toml:10`, `ojas-core/src/lib.rs:9` |
| tessl exports its kernel directory as `DEP_TESSL_KERNELS` | V: `tessl/build.rs:57` emits `cargo:kernels=`, `tessl/Cargo.toml:14-17` |
| ojas-metal's `build.rs` does not consume it | V: no `DEP_TESSL` in `ojas-metal/build.rs`; it compiles `kernels/{name}` only (`:21, :46`) |
| `MetalBackend::numerics()` is `Fast` | V: `ojas-metal/src/backend.rs:1977-1979` |
| `GpuRuntime` stays `!Send` | V: `tessl/src/runtime.rs:1023-1030`; the compile_fail doctest is attached to `type PendingRecycle` (`runtime.rs:574-586`), not `GpuRuntime`, so tessl `docs/architecture.md:499` is stale on that point |
| CUDA already charges the one `Budget` | V: `ojas-cuda/src/budget.rs:10-17` |
| tessl's manifest has no ojas dependency and its CI has no ojas checkout | V: `tessl/Cargo.toml`; `ci.yml` jobs `fmt`, `kernel-emulator` (ubuntu) and `check` (macos-26) |
| `ojas-core` inherits `rust-version.workspace` (1.97); tessl declares 1.82 | V: `Cargo.toml:32`, `ojas-core/Cargo.toml:6`, `tessl/Cargo.toml:5` |
| tessl is not a Cargo workspace | V: no `[workspace]` table in `tessl/Cargo.toml`; `numcheck/` is the precedent for an unpublished sibling package |
| No consumer pins tessl or ojas by revision | V: `ojas-metal/Cargo.toml:22`, `ojas-qwen35/Cargo.toml:21`, Lappi and gemma-metal use path dependencies |
| `device.rs:5062` pins the OOM strings with `include_str!` of tessl's `runtime.rs` | V |
| `StepProvider` is defined twice | V: `ojas-train/src/lappi/step.rs:267` and `ojas-cuda/src/step.rs:285` |
| Four `GpuRuntime::new` sites | V for three: `ojas-metal/src/device.rs:576`, `ojas-metal/src/gpu.rs:330`, `ojas-qwen35/src/step.rs:399`; the fourth, gemma-metal `kernels.rs:816`, came from a probe (not re-read) |

### Refuted and corrected claims

1. **"I saw no kernels-directory export."** Refuted: `tessl/build.rs:57` emits it. B's `-I ../../tessl/kernels` should be `-I $DEP_TESSL_KERNELS`.
2. **Proposal A: "tessl hand-backward is the fast Apple provider."** Unsupported. No paired Tape-versus-provider throughput row exists under `bench/results/` (checked: `2026-10-15-qwen35-2b-tape-gate`, `2026-10-17-lora-step`, `2026-10-21-lora-stepcost-ab`, `2026-10-21-tape-provider-memory`, all Tape arms). The provider's 2B leg was stopped by the memory guard.
3. **"0.8B LoRA step 3,110 ms" as a provider reference.** That row is the Tape arm (`2026-10-17-lora-step/README.md`, `tape-lora`).
4. **"ojas-qwen35 implements neither StepProvider trait."** True inside the ojas workspace, misleading overall: Lappi's `qd-train-metal/src/provider.rs:297` implements `StepProvider for Qwen35Provider` over `ojas_qwen35` (V). Retiring the tessl step is therefore a Lappi change.
5. **"tessl cross entropy lacks collapse handling."** Refuted by the kernels probe (tessl never uses the collapsing form); `cross_entropy.metal` was not re-read, so this stays I.
6. **`tests/build_artifact_contract.rs:179` "asserts the fast flag on every compile."** Narrower: lines 172-184 are a shell shim that checks one synthetic TensorOps compile for `-fmetal-math-mode=fast` and refuses `-ffp-contract=off` on it. `build.rs:274-279` turns contraction off only for `qwen35_adamw.metal`. "Every compile" stays I.
7. **"Three live `GpuRuntime::new` sites."** Four (above). Two of the four disappear by deletion in P6, not by migration.
8. **"Phase 6 pin bump: yes."** Wrong: no pins exist (above). A Phase 6 move is a coordinated landing, not a bump.
9. **"The C ABI shields internal types."** True for memory, incomplete for training: 13 of the 31 `capi.rs` exports are the Qwen3.5 training step, AdamW and GDN-train (`capi.rs:928-2071`, V), bound by `python/tessl_torch/__init__.py:322-368` (section 4).
10. **`ojas-metal/src/gpu.rs` (3,735 lines) is a third Qwen-shaped training step inside ojas.** DevMap `impact(gpu.rs::Session)` at depth 2 finds only `gpu.rs`'s own tests and one import from `device.rs` (`gate_dbias_threads`, `device.rs:39`). `rg` across the workspace (excluding `backups/`) finds no `TinyState` or `gpu::Session` user outside the file, and `lib.rs:73` exports the module with no external importer found. Both signals are lower bounds. It is a delete candidate (D11).

## 3. Who owns Metal training: ojas

**Decision recorded in the plan: ojas owns Metal training. tessl serves it as the Metal extension: the training kernels and their launchers stay in tessl, and the whole-step Rust orchestration retires.** This is a recommendation pending D5; nothing is removed until the owner answers.

Evidence (V unless marked):

- The 2026-10-09 re-rank already named "ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only" as the Mac LoRA route (`tasks/gp-imported-research-trees.md:66`).
- The real 2B step runs through the Tape within tessl's own bounds against transformers: loss 4.54e-5 relative (bound 1e-4), 149 of 149 gradients within 1e-2 of peak (worst 4.05e-3), the Tape within 2.3e-7 of tessl's own step loss, 17.60 GB device peak (`bench/results/2026-10-15-qwen35-2b-tape-gate/README.md`).
- The tessl provider's 2B leg is 30.13 GB resident at open and stages a whole f32 table to read back any gradient, over the guard (same README). Its `grad_sq_norm` downloads every bank entry (`ojas-model/src/qwen35/bank.rs:314-341`, I).
- LoRA and full-dW steps are measured on the Tape at 0.8B and 2B, T=2,048 and 9,638, min of 5 paired (`2026-10-21-lora-stepcost-ab`, `2026-10-17-lora-step`).

What it means for the code:

- **Stays in tessl (kernels and launchers):** `gdn_train.metal`, `qwen35_bwd.metal`, `qwen35_attn_bwd.metal`, `qwen35_adamw.metal` with `qwen35_adamw_math.h`, `cross_entropy.metal`, `qwen35_train_storage.metal`, and the per-op Rust launchers ojas-metal already imports (`device.rs:25-33`: gdn_train, gated_rms_norm, conv1d_silu and their backwards). tessl's ubuntu `kernel-emulator` CI job keeps gating them.
- **Retires from tessl (generic training orchestration, a second copy):** the whole-step layer in `qwen35_train.rs` (4,725 lines: `PendingStep`, `Qwen35Grads`, the bank, `Supervise`), the host AdamW object and scalar formation in `qwen35_adamw.rs` (1,518 lines, `adamw_scalars` at `:507`), and the step-sequencing part of `qwen35_bwd.rs` (988). The split inside those files is I; the ceiling is 7,231 lines. This line is conditional on D5(a).
- **Retires from ojas:** the tessl-step shell in `ojas-qwen35` (`step.rs`, the `Snapshot` and provider plumbing; the host-neutral `config`, names, groups and validators stay, because `Qwen35TextConfig::tape_spec` feeds the Tape tower, `ojas-qwen35/Cargo.toml:23`), and `ojas-metal/src/gpu.rs` (D11).
- **Retires in Lappi (the owner's call):** `qd-train-metal`'s `Qwen35Provider` repoints to `TapeStepProvider`.
- **Gate before anything is deleted:** one paired, interleaved, min-of-N A/B of `TapeStepProvider` against the tessl provider at 0.8B-Base, T=2,048, micro-batch 1, checkpointing on (both fit under 28 GiB there), with gradient parity between the two recorded. If the provider wins by more than the threshold the owner sets, the answer is not to keep two engines but to promote its fused per-layer backward kernels as `Backend` ops behind the Tape (the "trait decision that has to cover CPU, Metal and wgpu", `ojas-qwen35/src/lib.rs:8-10`), then retire the step.
- **The training step crosses tessl's C ABI.** `capi.rs` exports 13 training symbols (`tessl_qwen35_train_step`, `_step_ragged`, `_forward`, `_backward`, `_discard`, `tessl_qwen35_adamw_init/free/step/set_step_count/step_count`, `tessl_qwen35_grad_sq_norm`, `tessl_gdn_train_forward/backward`; `capi.rs:928-2071`), bound by `python/tessl_torch/__init__.py:322-368` and pinned by `python/tests/test_qwen35.py` and `tests/capi*.rs`. Retiring the orchestration is an ABI-breaking removal (13 to 14) that ends `tessl_torch`'s Qwen3.5 training path. That is D5: (a) retire the training ABI with the step, consistent with "tessl is inference on the Mac"; or (b) keep the C-ABI training product, in which case tessl's step stays as that product's engine and only the ojas-side duplicates retire, with Lappi repointing to the Tape. Under (b) the second copy of training logic in tessl is accepted knowingly, for a named consumer, and the two engines stay parity-gated against each other (`real_2b_step_matches_transformers` on the tessl side, `gpu_tape_2b.rs` on the ojas side).

## 4. Target architecture and ownership

```
                         ojas workspace (universal; Linux + macOS CI)
 ojas-core      Tensor, DType, OjasError, Budget (THE ledger), Backend, *_dims, autocast,
                DeviceBuffer + HAL: AllocError, MappedBuffer/HostLease, Timeline, Provenance tag,
                StepProvider contract (moved here or to ojas-model; see ledger)
 ojas-io/data   safetensors (+ per-entry read, "read into caller slab" sink), json, npy, BPE
 ojas-cpu/simd  reference backend, Numerics::Exact contract       ojas-autograd  Tape
 ojas-model/train/infer   towers, TapeStepProvider, optimizers, trainer loop, light decode/sampling
 ojas-kernels   launch geometry, WGSL, parity harness
 ojas-capi / ojas-gusset-engine / ojas-device
 ---- device extensions: each depends on ojas-core (and ojas-kernels/io); none on another ----
 ojas-cpu (in tree)         ojas-wgpu, ojas-cuda, ojas-hip (in tree; copy-based; host_view = Unsupported)
 ojas-metal (in tree)       Metal op router: impl Backend, Link/Cmd device thread, ojas_* overlay
                            kernels (safe math, status words). Depends on tessl. Linux: stub.
 tessl (sibling repo)       Apple/Metal extension and Mac inference product:
   depends on ojas-core     runtime, queue, pipelines, BufferPool (StorageModeShared), SharedBuf +
   (after the MSRV probe)   Timeline implementing ojas-core's HAL traits, typed AllocError -> OjasError,
                            canonical kernel sources (DEP_TESSL_KERNELS), training kernel launchers,
                            Qwen3.5/EmbeddingGemma 2/BERT inference engines (ICB, q4/q8, KV, decode
                            graphs), C ABI (python, go), gemma-metal/Lappi qd-metal/BINN consumers
```

The arrow is `ojas-metal -> tessl -> ojas-core`. There is no cycle because `ojas-core` has no dependencies and no other ojas crate is reached from tessl. `tessl -> ojas-io` is a second, separately gated step (P8).

| Concern | Owner | Notes |
| :--- | :--- | :--- |
| Types, errors, dtype, bf16 and f16 bit rules | ojas-core | tessl's `_gpu` conversion (hardware-flush model, `tensor.rs:815`) stays in tessl as a named, tested variant; the host copy is deleted only after an all-2^16-input bit-diff |
| Ledger | `ojas_core::Budget` | One process root; children per host pool, tessl pool cache and session; charges the bucket (`allocated_bytes_for`, `runtime.rs:1178`) |
| Allocator and pool | tessl `BufferPool` (`runtime.rs:245`) | One `Arc<Shared>` (pool, deferred queues, timeline, poison) per process; runtimes stay per thread |
| Fences | tessl `Timeline` and `SharedBuf` stamps | Section 5 |
| Device thread, encoding, pipelines, residency | tessl `GpuRuntime` (`!Send`, `runtime.rs:1023-1030`) | ojas-metal's `Link/Cmd` stays as the thread boundary; it shrinks by the copy paths, `Book`, `MemCap` and `classify_alloc` |
| Kernel arithmetic shared by both | tessl headers (`qwen35_adamw_math.h`, `reduce_tree.h`, `attn_tiled.h`) | ojas-metal entry points `#include` them through `DEP_TESSL_KERNELS`; ojas keeps the status-word and degenerate-row contract; two metallibs (fast and safe) stay |
| Training engine | ojas (Tape, `Backend::adamw_step`, bank, `StepProvider`, trainer) | Section 3 |
| Mac inference engines | tessl | Qwen3.5 prefill and decode with ICB and quantization, EmbeddingGemma 2 encode, BERT, KV caches, decode graphs; `ojas-infer` stays a light `DeviceDecoder` over `Backend` (`ojas-infer/src/lib.rs:9-15`), not a tuned Mac engine |
| Checkpoint and weight IO | ojas-io | **Decided and landed for safetensors, 2026-11-01** ([`io-ownership.md`](io-ownership.md)): tessl's parser is deleted and its reader is an adapter. tessl's inference loaders using the slab sink is still to do |
| Tokenizers | ojas-data (BPE, training data); tessl `embedgemma2_tok.rs` (Gemma inference) | Different algorithms serving different products, not a duplicate; a shared rank core would be a one-caller abstraction, so no |
| Qwen3.5 config | ojas-qwen35 `config` (host-neutral) for training; tessl `config_json.rs` for inference | One parser, two refusal lists, after the owner names the owner (D7) |

## 5. Memory design

Premise (V): every tessl pooled buffer is `StorageModeShared` (`tessl/src/runtime.rs:299`). The CPU and GPU already see the same bytes and the software hides it: `HostData` is an owned `Vec` (`ojas-core/src/tensor.rs:159-164`) and `DeviceBuffer::read_bytes` copies (`tensor.rs:19-33`).

**One ledger.** `ojas_core::Budget` is the only accountant. The tessl pool charges `allocated_bytes_for(bytes, kind)` (the bucket) to a `Budget` child before `newBufferWithLength`. A failed charge is `OjasError::CapacityExceeded` through a typed `AllocError`, never a clamp and never a string (`device.rs:515-551` and its `include_str!` pin at `:5062` go). Free-listed bytes are charged to a `pool.cache` child so the 2 GiB cache (`runtime.rs:60`) is visible. On a refused charge the pool runs a fixed ladder: trim the free list, retry once, refuse. The ladder never waits on the GPU inside `try_reserve`; a caller that wants deferred releases drained calls an explicit `reclaim(timeline_value)` first. This changes `Budget` from "live" to "resident" semantics (cache plus recycle lag count): **decision D3, OPEN**. It replaces ojas-metal's `Book`, `freed_since_wait` and `mem_cap` (`device.rs:449-483, 761-822`) and the stale bucket copy at `gpu.rs:214` (deleted with `gpu.rs`).

**One pool per process.** The pool, deferred-release queues, timeline state and poison latch move into an `Arc<Shared>: Send + Sync`. `GpuRuntime` keeps its `!Send` encoding state plus the `Arc`. Buffers stay pinned to the runtime that encodes them (`nn.rs:92-96`). After P6 only two constructors remain in this repo family (`device.rs:576` and gemma-metal `kernels.rs:816`); `gpu.rs:330` and `step.rs:399` go by deletion.

**Same bytes.** `ojas-core` gains, all defaulted to `Unsupported`:

- `AllocError`.
- `trait MappedBuffer: DeviceBuffer { host_read() -> Box<dyn HostLease>; host_write() }`, with `HostLease: Send + Sync` exposing `as_bytes`, `as_f32` and `as_u32`.
- `trait Timeline { submitted, completed, wait }`.

`Payload` gains `Mapped(Arc<dyn MappedBuffer>)`, charged once through `from_device_reserved` (`tensor.rs:848-874`, probe). `Tensor::host_view()` returns a short-scoped guard; `f32_slice` keeps refusing with `Placement`. CPU kernels take the guard through the `validate::f32_values` and `u32_values` funnel (`validate.rs:313-323`, V; about 60 production sites, probe). `validate::Shared` (`kv.rs`, the one `'static` persistent-pool user) needs the `Send` lease and waits for P3b. For outputs, `Scratch<T>` becomes an enum over `Vec` and a mapped allocation at the `into_raw` seam (`budget.rs:484`, probe), opt-in; the default stays `Vec`, so Exact callers are untouched. A write lease resets `finite` to unknown and `compute` to f32, as `host_data_mut` does.

**Fence protocol** (sync probe, pieces re-verified):

- Each `SharedBuf` carries atomic `last_use` and `last_write`, set at bind time to `pending_commit_value()` (`runtime.rs:2188-2196`, the same stamp deferred release uses).
- A CPU read waits for `completed >= last_write`. A CPU write waits for `completed >= last_use` and is exclusive. Fencing is whole-buffer, because Metal makes even non-overlapping unsynchronized writes undefined (`MTLBuffer.rs:15-21`, V).
- A GPU bind against a live host view is refused with an error, never blocked.
- A stamp above `submitted` returns `NotSubmitted`; the caller flushes through the device thread (`commit(false)`) and retries. Nothing waits on an uncommitted value.
- Every fence result checks poison. Waiting goes through the private `lifetime_event`, never the forgeable `handoff_event` (`runtime.rs:392-396, 1278`).
- Role-aware `bind_shared_ro` and `bind_shared_rw` are required, because the binders record only "bound" (`dispatch.rs:302, 316`, probe).
- The one new `unsafe impl Send + Sync` sits in tessl on a `(NonNull<u8>, len, Retained<MTLBuffer>)` wrapper; objc2 marks device, queues and events thread-safe but not `MTLBuffer` (`MTLBuffer.rs:23`, V). `ojas-core` and `ojas-metal` stay `forbid(unsafe_code)`. Asserting it is **decision D6, OPEN**; no lane may add it before then.

**Exact determinism.** Exact is a property of the CPU op's fixed reduction order. A mapped operand changes neither order nor alignment assumptions (SIMD uses unaligned loads). Metal is `Numerics::Fast`, so GPU-written bytes are Fast-tier: the buffer carries a `Provenance { Exact, Fast }` tag beside the existing compute tag (`tensor.rs:109-124`). `CpuBackend` under Exact records and allows by default, and refuses when the caller asks for end-to-end Exact (**D4, OPEN**). Gate: Exact CPU outputs from `Vec` and from a mapped slab are bit-identical at 1, 2 and N threads.

**No silent fallback.** wgpu, CUDA and HIP keep counted copies; `host_read` is `Unsupported`; an op they lack is `Unsupported`. CUDA already charges `Budget`. Whether wgpu's 512 MiB staging (`context.rs:109`, probe) is charged is U and is P9's first check.

**Two blockers before any `SharedBuf` code lands.**

1. ICB replay: `decode_icb.rs` captures binds, so per-bind stamps are wrong on replayed command buffers unless replay re-stamps (U).
2. The scratch spike crate at `/Users/bharath/Code/research/ojas-ug-spike-uma` (834 lines, not a git repository, depends on tessl by path) encodes the four probes this design needs: T1 pointer sharing bit-check, T2 event fence correctness and cost, T3 copy versus lease versus raw-pointer cost at 4, 64 and 512 MiB over 5 interleaved rounds, T4 `bytesNoCopy` over a page-aligned mmap. No results are recorded anywhere in ojas. Running it and committing its rows is P0's first artifact.

**What to promise.** Copies are at most 45.5 of 2,566.5 ms per SAM frame (1.8%; 5.7% with mixed rows; n=1 at load 9-21) and well under 1% of a Qwen LoRA step (cost probe). The win is one ledger, one pool, fewer host copies (E20: -330 and -569 MiB) and less code, not throughput.

## 6. Dedup ledger

| Family | Canonical owner | Deleted | Lines (`wc -l` unless noted) |
| :--- | :--- | :--- | :--- |
| Qwen3.5 training step orchestration | ojas (Tape, `TapeStepProvider`) | `ojas-qwen35` step shell; Lappi `qd-train-metal` provider repointed; and only under D5(a), tessl `qwen35_train.rs` step layer, `qwen35_adamw.rs` host optimizer and `adamw_scalars`, step sequencing in `qwen35_bwd.rs`, the 13 training C-ABI symbols and their Python binding (ABI 14) | most of 3,062 in ojas-qwen35; up to 7,231 in tessl under D5(a) (split inside files is I) |
| Legacy tiny Metal step | none (delete) | `ojas-metal/src/gpu.rs` (keep `gate_dbias_threads`, moved to `device.rs`) and `per_head_gate.metal` if no other caller (U) | -3,735 |
| `StepProvider` trait | One definition below the extensions (ojas-core, or ojas-model if it must stay out of the frozen API) | `ojas-cuda/src/step.rs:285-326`; `ojas-train/src/lappi/step.rs:267` re-exports | about -40, one contract |
| AdamW scalar formation | `ojas-core` (`check_adamw`, `clip_scale`, `next_step`, `AdamWConfig`, `backend.rs:129-262`, plus one `scalars_for_step -> [f32; 8]`) | tessl `qwen35_adamw.rs:507` (with the step), ojas-metal `backend.rs:1403`, ojas-train `lappi/adamw.rs:166-171` | 3 copies to 1 |
| AdamW MSL math | tessl `qwen35_adamw_math.h` (8-float struct with `grad_scale`) | ojas `ojas_adamw_elem` (`ojas_backend.metal:1528-1541`); the `check` and `apply` entry points keep the status gate and `#include` the header; ojas's 7-float struct becomes tessl's 8 | about -14 MSL, one math source |
| Residual add, embedding fwd and bwd, argmax | tessl arithmetic headers; ojas entry points keep status and policy as constants | `ojas_backend.metal:410-425, 805-820, 995-1013, 854-904` | about -100 MSL |
| RMS norm | Tolerance-level merge only (bits differ) | `ojas_backend.metal:556-631` | about -76 MSL, optional |
| Softmax, cross entropy, general attention, reductions, KV write | Two by design | Nothing | 0 (D8) |
| ojas-metal accounting and copy layer | tessl pool and `Budget` | `classify_alloc` and the `include_str!` pin (`device.rs:515-551, 5062`), `Book`, `MemCap` and `freed_since_wait` (`:449-483, 761-822`), the retry loop (`:934-994`), the `to_vec` plus decode readback and `settle(Wait::Upload)` copy routes (`:1787-1855`), `memory.rs` slack | about -300 (I) |
| Allocation error strings | tessl typed `AllocError` at alloc and poison boundaries | tessl's other ~915 `Result<_, String>` signatures stay behind the additive typed API | boundary only |
| safetensors reader | ojas-io | **Landed 2026-11-01** ([`io-ownership.md`](io-ownership.md)): tessl `safetensors.rs` 950 -> 567, a thin adapter with no parser; its Syntax profile became an ojas-io `JsonLimits`; `parser_mutation.rs` re-pointed, with one contract narrowed (header amplification is refused, not merely bounded) | -383 in `safetensors.rs`, -318 net across tessl |
| json and npy readers | ojas-io | Not started. tessl `json.rs` 516, `npy.rs` 417. `plain.rs` 145 stays either way: `npy.rs` needs it, and the adapter keeps tessl's payload policy and `checked_numel` there | about -933, still gated on P5 |
| CUDA format code | ojas-io and ojas-core | `ojas-cuda` `json.rs`, `npy.rs`, `bf16.rs` (646, from proposal B's count, U) | about -646 |
| bf16 and f16 conversions | Both, named | None now; tessl host `tensor.rs:796` only after an exhaustive bit-diff | 0 |
| Tokenizers | ojas-data and tessl, by product | None | 0 |
| Qwen3.5 config double parse | One parser, two refusal lists (D7) | `cross_check` (`config.rs:727`) once the owner is named | small |
| Four `GpuRuntime::new` sites | One `Arc<Shared>` | `gpu.rs:330` and `step.rs:399` by deletion; `device.rs:576` and gemma-metal opt in | 4 to 2 |

Net, as the plan states it: kernel merges are about 200 MSL lines out of 3,610. The real deletions are the training-step duplicates (about 10-11k lines across tessl, ojas-qwen35 and `gpu.rs`), tessl's readers (2,028) and the ojas-metal accounting layer (about 300). Every row that removes a duplicate deletes the old path in the same change that moves its callers, and needs two independent signals (DevMap `devmap_impact` and `rg -uu`) before the delete.

## 7. Phases

Every phase lands alone and reverts by reverting its commit. No consumer pin exists (all path dependencies), so a tessl phase lands in Lappi, gemma-metal and arch02 at once: each is a coordinated landing announced before it merges. tessl edits come from a tessl-rooted session. GPU runs go through the heavy-step queue (the owner). Phases marked (lanes) wait for the in-flight kernel lanes of section 8.

| Phase | Content | Fails before | Parity gate | Benchmark | Repos | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **P0 measure and clean** | Run and record the `ojas-ug-spike-uma` T1-T4 rows. Add the `upload` and `to_host` micro-bench at 1, 16, 66 and 170 MB against a memcpy arm and a GPU-streaming arm (interleaved, min-of-N, load recorded). Drop the `to_host` zero-fill and the `to_vec` plus decode double pass (`tensor.rs:1005-1081`, `device.rs:1848-1854`). Add the f32 `fresh_from` twin. Fix stale text: `typed-storage-plan.md:3-8`, `ojas-metal/build.rs:3`, `ojas_backend.metal:1`, the two UMA-card lines, tessl `architecture.md:499` | Bench shows 7-16x above the floor; a readback byte-equality test | Readback bytes identical (tolerance 0.0 already at `backend_parity.rs:337-368`) | The bench itself | ojas (plus spike rows) | S |
| **P1 tessl primitives** | Typed `AllocError`; `Deferred` queues and timeline state into `Arc<Shared>`; `Timeline`; `SharedBuf` with role-aware `bind_shared_ro` and `bind_shared_rw`; settle ICB re-stamping first. All additive; the `HostMapping` lease is untouched | New tests: reader overlap, writer exclusivity, `NotSubmitted`, poison propagates, GPU write-bind refused against a live reader, ICB replay re-stamps | All tessl kernel tests bit-identical (no kernel change) | `bench_nn_kernels` before and after (two relaxed stores per bind; the `active_m4` lock in `pending_commit_value`) | tessl | M |
| **P2 ojas-core HAL and contract moves** | `AllocError`, `MappedBuffer` and `HostLease`, `Timeline`, `Provenance`, all defaulted to refuse; the harness test lists them like `the_method_list_names_every_trait_method`; `StepProvider` moved to its canonical home, ojas-train and ojas-cuda import it | Trait-list test; CUDA `step_refuses.rs` compiles against the moved trait | None (no behaviour change) | None | ojas | S |
| **P3a one ledger** (lanes) | ojas-metal maps `AllocError` to `CapacityExceeded`; the pool charges buckets to a `Budget` child; release on recycle; free list charged; trim ladder; delete `classify_alloc`, the `include_str!` pin, `Book`, `MemCap`, `freed_since_wait` and the retry loop. Depends on D3 | A test that rewords tessl's OOM text (breaks the classifier today); refusal at exactly the bucket total; charged equals allocated bucket bytes | `adversarial_oom.rs` (463 lines) green | Decode alloc A/B; peak resident bytes | ojas, tessl (small) | M |
| **P3b mapped tensors** (lanes) | `Payload::Mapped`, `host_view`, `MetalBuffer: MappedBuffer` over `SharedBuf` (lease handed back through `Reply` once per tensor); readback and upload via leases; `Scratch` enum opt-in; `Provenance` tag; cache resets. Depends on D4 and D6 | A counter shows zero `read_bytes` copies on a mapped readback; stale `FINITE_YES` after a GPU write caught | Exact CPU outputs bit-identical, `Vec` versus mapped, at 1, 2 and N threads | Against P0: SAM readbacks (9.0 and 11.2 ms) and mask uploads (25.3 ms) against the memcpy floor | ojas | L |
| **P4 one pool** | `GpuRuntime::with_shared(Arc<Shared>)`; `device.rs:576` uses it; gemma-metal opts in; process-global `TriFlags` (`ab_flags.rs`, `decode_icb.rs`) into `Shared`; tests keep a per-test `Shared::new()` | Two runtimes under one `Shared` show one resident total | Existing Metal parity gates | Qwen LoRA step (10-21 rows) and decode A/B | tessl, ojas, gemma-metal | M |
| **P5 the arrow** | Probe: `cargo +1.82 check -p ojas-core` (from a build-allowed session). Pass: `ojas-core` sets its own `rust-version`, tessl adds `ojas-core = { path = "../ojas/ojas-core" }`, tessl implements `ojas_core::{MappedBuffer, Timeline}` directly and returns `OjasError` at alloc and poison boundaries; ojas-metal's adapters delete; tessl CI gains a macOS job with an ojas checkout (mirroring ojas's `TESSL_REF` layout). Fail: D2 (raise tessl to 1.97, or stop at proposal B's arrow) | A tessl test constructing `OjasError::CapacityExceeded` from the pool | All tessl and ojas-metal suites unchanged | None | tessl, ojas; every tessl consumer needs the sibling checkout (coordinated landing) | M |
| **P6 training ownership** | Paired A/B of the Tape against the tessl provider at 0.8B, T=2,048 (section 3). Then delete `ojas-metal/src/gpu.rs` and the `ojas-qwen35` step shell (`config` stays); Lappi repoints (the owner). Under D5(a) also: tessl's step orchestration, host AdamW and the 13 training C-ABI symbols, ABI 13 to 14, with `python/tests/test_qwen35.py` and `tests/capi*.rs` updated. If the provider wins by more than the threshold: promote its fused per-layer ops as `Backend` methods first (the AGENTS.md checklist), then retire | A Tape-versus-provider gradient parity row; `gpu_tape_2b.rs` still green; under D5(a) a capi contract test that the removed symbols are absent | 2B tape gate bounds unchanged | The A/B itself; `2026-10-21-lora-stepcost-ab` re-run | ojas, tessl, Lappi (owner), `python/tessl_torch` under D5(a) | L |
| **P7 kernel single-sourcing** (lanes) | `-I $DEP_TESSL_KERNELS` in `ojas-metal/build.rs`; AdamW header first, then residual add, embedding and argmax; RMS at tolerance; two metallibs stay; ST_OUT kernels stay in the safe lib until `nnan/ninf` is shown not to hide a computed NaN; an entry-name suffix, because `add_overlay` refuses duplicates (`runtime.rs:1771-1799`) | Byte-compare old against new on adversarial inputs per family | Bit-equal for AdamW, add, embedding and argmax; tolerance for RMS; tessl kernel census and IR lint | Kernel suites against current kernels | ojas, tessl (headers) | M |
| **P8 IO** | **Safetensors half landed 2026-11-01** ([`io-ownership.md`](io-ownership.md)): ojas-io absorbed eight bounds tessl's reader had (including a FIFO that hung the reader and an order-dependent shape overflow), tessl's parser is deleted, `ojas-qwen35` is migrated, `parser_mutation.rs` re-pointed. No MSRV ask was needed: tessl already declared `ojas-io` as a path dependency and used it nowhere. Still to do: per-entry read and the "read into caller slab" sink (page-aligned pool slab, `pread` straight in); tessl's inference loaders using it; `json.rs` and `npy.rs`. `plain.rs` is not deleted. Depends on D9 and D10 | `parser_mutation` re-pointed; checkpoint bit-equality | Load bit-equality for SAM f32, Gemma bf16, Qwen bf16 | Load time against today (SAM 3.20 s) | ojas, tessl | M |
| **P9 siblings** | wgpu staging charged to `Budget` (check first); ojas-cuda `json`, `npy` and `bf16` delegate to ojas-io and ojas-core; CUDA-C is unchecked on this Mac | Existing tests | Existing gates | None | ojas | S |

**Stop points.** After P0 (the cheap win quantified). After P4 (one ledger, one pool: the memory goal). After P6 (one training engine). After P8 (no generic copy left in tessl). P7 and P9 can run any time after P2.

## 8. How the in-flight kernel lanes fit

The plan's snapshot: five worktrees sat at `c389342` (the main commit) with uncommitted edits (V, `git status` in each).

- `gp/ug-core`: six new `Backend` methods (`interpolate_bilinear_backward`, `group_norm_backward`, `gelu_tanh_forward` and `_backward`, `seq_mean_forward`, `l2_normalize_forward`) in `ojas-core/src/backend.rs`, plus autocast, shapes, spatial, CPU general attention and tests; 26 files, +1,460 and -118.
- `gp/ug-metal-attn-bwd`, `-encoder-bwd`, `-maxpool` and `-spatial-bwd`: each edits `ojas-metal/{build.rs, src/backend.rs, src/device.rs, src/link.rs}` and adds new `.metal` files (`attn_gen_bwd.metal` with `attn_bwd_tiled.h`, `encoder_bwd.metal`, `max_pool.metal`, `spatial_bwd.metal`), new `src/backend/` and `src/device/` submodules, oracle fixtures and tests.

Other sprint lanes (`gp/ug-docs-ojas`, `-bucket-dedup`, `-alloc-error`, `-const-dedup`, `-adamw-scalars`, `-geometry-dedup`, `-tiny-step`, `-one-runtime`, and `ug/docs` in tessl) were opened after the plan's snapshot, so the plan does not classify them; they are shown in this repository's worktree list. The mapping below is inferred from the lane names alone (their diffs were not read for this document), so treat each row as I:

| Lane (by name) | Nearest plan item | Basis |
| :--- | :--- | :--- |
| `gp/ug-alloc-error` | P1 and P3a: typed `AllocError`, retiring the `classify_alloc` string match | name |
| `gp/ug-bucket-dedup` | Ledger row "ojas-metal accounting layer": one bucket formula (`gpu.rs:214` copy) | name |
| `gp/ug-adamw-scalars` | Ledger row "AdamW scalar formation": three copies to one | name |
| `gp/ug-const-dedup`, `gp/ug-geometry-dedup` | Dedup of shared constants and launch geometry; no row in the ledger above, so the match is loose | name |
| `gp/ug-tiny-step` | The legacy tiny Metal step (`gpu.rs`) and D11 | name |
| `gp/ug-one-runtime` | P4 one pool: fewer `GpuRuntime::new` sites | name |
| `gp/ug-docs-ojas`, `ug/docs` (tessl) | Documentation of the same direction; this page is a sibling of that work | name |

Any of these that deletes code the owner has not yet approved (for example `gpu.rs`, D11) must stop short of the deletion; section 9 lists what stays.

How the kernel lanes fit: they are ojas-contract training kernels (safe math, status words, "tessl has no kernel for the rest", `attn_bwd_tiled.h:1-10`) for EmbeddingGemma 2 and SAM backwards. They belong in ojas-metal's overlay under the ownership rule (tessl owns arithmetic both sides need; ojas owns its training contract), and they need no tessl change. Three consequences:

1. P1 to P6 avoid `link.rs`, `backend.rs` and `device.rs` except for the small accounting regions (`device.rs:449-551, 761-822, 934-994, 1787-1855`). P3a and P3b land after the Metal lanes or rebase onto them. Any structural reshaping of `Cmd` (proposal A's closure owner) or relocation of ojas-metal (proposal C's Phase 6) is off the table until the lanes land, and is the owner's decision afterwards (D1).
2. P7's `-I $DEP_TESSL_KERNELS` should land before the attention-backward lane finalizes `attn_bwd_tiled.h`, so it includes tessl's `attn_tiled.h` and `reduce_tree.h` pieces instead of copying them. Whether it copies today is U; the header says it is "after tessl's `qwen35_attn_bwd.metal`".
3. Under the refinement, ojas's EmbeddingGemma 2 ops are training and evaluation ops over `Backend`; the tuned Mac encode path is tessl's `embedgemma2*` (C ABI, go/engine). Do not grow a second tuned Gemma encode in ojas-metal; a Mac inference caller uses tessl.

## 9. What does not change until the owner answers

This document is the plan only. Until the matching decision in section 10 is made:

- No `unsafe impl Send` or `Sync` is added anywhere (D6).
- No `Budget` semantic change is made; "live" stays the meaning (D3).
- No C-ABI symbol is removed from tessl or from `ojas-capi`, and the tessl C ABI stays at 13 training symbols (D5).
- tessl's `rust-version` stays 1.82 and its CI stays standalone (D2).
- No tessl training code (`qwen35_train.rs`, `qwen35_adamw.rs`, `qwen35_bwd.rs`) is deleted (D5).
- `ojas-metal/src/gpu.rs` and `per_head_gate.metal` stay (D11).

## 10. Open owner decisions

All eleven are **OPEN**. The "plan's lean" column repeats what the source plan recommended; it is not an answer.

| ID | Decision | Plan's lean | Status |
| :--- | :--- | :--- | :--- |
| D1 | Where `impl Backend for MetalBackend` lives. Stay in the ojas workspace as `ojas-metal` (training-side op router, Linux stub, no generic logic), with tessl implementing ojas-core's HAL directly; or, matching the decision's literal wording, a `tessl-ojas` package in the tessl repo (needs a `[workspace]` table there) after the lanes land. The second is a pure move with zero deletion and puts the training backend in the inference repo | Stay in ojas | **OPEN** |
| D2 | MSRV. If `ojas-core` (and later `ojas-io`) does not build at 1.82: raise tessl to 1.97, or stop the arrow at proposal B's shape. Decides whether tessl CI stays standalone | Probe first (P5) | **OPEN** |
| D3 | Budget semantics: resident (cache and recycle lag charged, trim on pressure) versus live | Resident | **OPEN** |
| D4 | Exact default with Fast-provenance operands: record and allow, or refuse | Record and allow | **OPEN** |
| D5 | Training retirement. (a) Retire tessl's Qwen3.5 training step with its 13 C-ABI training symbols and `tessl_torch`'s training path (ABI 14), or (b) keep the C-ABI training product and retire only the ojas-side duplicates. Plus the threshold for the Tape-versus-provider A/B, and the Lappi `qd-train-metal` repoint (a consumer change in another repository) | No lean stated beyond the A/B gate | **OPEN** |
| D6 | Permission to assert `Send + Sync` on tessl's buffer wrapper, and to amend AGENTS.md naming tessl `SharedBuf` as the sanctioned owner (Apple states no explicit `MTLBuffer` thread-safety; objc2's markers and refcounting are the basis, I) | Not stated | **OPEN** |
| D7 | Qwen3.5 config refusal-list owner (training versus inference lists) | Not stated | **OPEN** |
| D8 | Permanent two-kernel families (softmax, cross entropy, attention, KV write, decode kernels) accepted as by design | Accept | **OPEN** |
| D9 | Weight loading: pool slab with `pread` versus whole-file mmap for SAM f32 only (per-tensor no-copy is dead: 0 of 1,623, 1,376 and 632 tensors page-aligned, cost probe) | Pool slab with `pread` | **OPEN** |
| D10 | gemma-metal: stay a tessl consumer with its own crates, or adopt ojas-io and ojas-data (new-dependency rule) | Not stated | **OPEN** |
| D11 | Delete `ojas-metal/src/gpu.rs` and `per_head_gate.metal` (two callerless signals, both lower bounds) | Delete | **OPEN** |

## 11. Gaps and unverified

- DevMap: every walk capped (`walk_incomplete`), 52 files pattern-recovered, Metal call resolution 4.4%, Metal-to-Rust edges invisible; the tessl index is stale, so all tessl blast radius is grep.
- Not measured: fence-wait cost against the copy path (spike T2 and T3 unrun), the `active_m4` lock cost per bind, Tape-versus-provider throughput, a fresh `block_ab` profile.
- Not checked: `cargo +1.82 check -p ojas-core`; ICB replay re-stamping; whether `E0277` is checked on stable; whether `attn_bwd_tiled.h` copies tessl header code; the wgpu staging charge; the inner split of `qwen35_train.rs` and `qwen35_bwd.rs` between launch and orchestration; `per_head_gate.metal` callers beyond `gpu.rs`.
- The advisor review of the source plan's finished document did not run (rate-limited); a first advisor pass shaped its sections on the verdict, ownership and in-flight work.
- CUDA-C edits cannot be checked on this Mac.
- The original dates of the two owner statements in section 1 are not in the source plan.
