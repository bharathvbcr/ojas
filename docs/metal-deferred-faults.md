# Metal deferred-fault contract

Status: **implemented** (round 5, phase B). The user approved Metal adopting wgpu's deferred-fault contract; phase A wrote this contract and phase B implemented it in `ojas-metal`. §9 records what was built, where it departs from or corrects §2–§8, the measurements, and the gaps. Sections 1–8 are kept as written in phase A, with phase B corrections marked **(B)**.

Claims carry a label: **(V)** verified against source or a measurement, **(I)** inferred, **(U)** unverified.

## 1. Why

Every Metal op today ends with one blocking wait and a read of its status words (`ojas-metal/src/device.rs`, `Worker::finish`).
- **The floor.** The wait costs 0.13–0.30 ms even for a one-element op (round-4 control rows, `scratchpad/metal_r4_ab_final.txt`) (V).
- **adamw_full** in the round-2 bench is 0.16× torch, with 170 waits per call, one per tensor (`docs/bench-gpu-vs-torch.md`, "Round 2") (V).
  - Phase A's status-read fix moved its min from 103.4 to 69.9 ms in an interleaved A/B, but the median stayed about 222 ms, set by the 170 waits.
  - That A/B used its own scalars, step 0 and zeroed moments, so it is not a reproduction of the bench row (V, `scratchpad/metal-r5-ab1.txt`).
- **Training step.** One micro-step is about 1,000 trait calls (`docs/framework-design.md` §9) (I), so at K=4 a training step pays about 4,170 waits.

wgpu avoids this by recording ops into a shared encoder and reporting non-finite values at the next sync point (`ojas-wgpu/src/backend.rs:17-41`) (V). Metal adopts the same contract, so batching ops into fewer command buffers becomes legal.

## 2. The contract, mirroring wgpu

Wording follows `ojas-wgpu/src/backend.rs:17-44` and `ojas-wgpu/README.md:75` wherever Metal can mean the same thing.

1. **A fault does not fail its op.** An op whose kernel produces a non-finite value, or reads one, returns `Ok`. The op's status words record it, and the op keeps going.
2. **The next sync point reports it.** The next `Backend::sync`, `Backend::download` (of any tensor) or `Backend::clip_grad_norm` on this backend returns `OjasError::NonFinite { op }`. `op` names the **first op in recording order** on this backend that faulted since the previous report (F11's rule) (V for wgpu, `ojas-wgpu/src/backend.rs:376-390`).
3. **A fault is reported once and never lost.** After it is reported, the next sync point returns `Ok` unless a new fault arrived.
   - Metal is stronger than wgpu here: wgpu may report a fault twice when two threads' syncs overlap (`ojas-wgpu/src/context.rs:18-19`). On Metal, scanning and reporting both run on the single device thread, so each fault is reported to exactly one caller (I; phase B pins it in a test).
4. **The fault belongs to the backend, not the caller.** Threads sharing one `MetalBackend` (or its clones) share one pending fault, reported to whichever thread reaches a sync point first. Callers that need their faults kept apart open one backend each. This is the same rule as wgpu (`ojas-wgpu/src/backend.rs:36-40`).
5. **A caller that must not run past a fault calls `sync`** after the steps it cares about. The trainer already does (`docs/framework-design.md` §3, step 6).
6. **Precedence inside one op is unchanged.** Its status words are judged in the current order: non-finite input, index out of range, then no valid row, then non-finite output (`Worker::finish`). Across ops, the first op in recording order wins.

### 2.1 Sync points

| Call | Waits | Reports a pending fault | Notes |
| :--- | :---: | :---: | :--- |
| `Backend::sync` (new override) | yes | yes | The only call whose sole job is to report. The override is required: `&dyn Backend`, `Arc<MetalBackend>` and generic callers reach it through T6 forwarding (`ojas-core/src/backend.rs:938-940`) (V) |
| `Backend::download` (new override) | yes | yes, **after** the read succeeds | As wgpu (`ojas-wgpu/src/backend.rs:1131-1137`): read first, then report. On a fault the host copy is dropped and `NonFinite` is returned. It keeps reading through `tensor.to_host(self.budget())`, so the readback is still charged to the backend's own budget (round 3's `downloads_are_counted_on_the_backends_own_budget`). Today Metal uses the trait default (`ojas-core/src/backend.rs:426-428`) (V) |
| `Backend::clip_grad_norm` | yes (it must read the norm) | yes, **before** it scales anything | As wgpu (`ojas-wgpu/src/backend.rs:1843-1889`). Its scale pass is then recorded without a wait |
| `Tensor::to_host` / `DeviceBuffer::read_bytes` | yes | **no**: the fault stays pending | A raw read is not a backend call, so it cannot return a typed fault for the backend. It may observe the fault (§4.3) but does not clear it |
| every other op, `adamw_step` and `muon_ns5_step` included | no | no | Recorded only |

**Optimizers defer, and stay all-or-nothing per call.** `adamw_step` and `muon_ns5_step` return `Ok` and record their work.
- **Why defer:** waiting per call is exactly the 170-wait cost being removed.
- **Why it is safe:**
  - The trainer reaches the optimizers only after `clip_grad_norm`, a sync point that has already reported every fault from forward and backward (`docs/framework-design.md` §3, step 4).
  - Each optimizer call still decides on the device whether *its own* values are finite, and writes nothing otherwise (§3).
  - The trainer calls `sync` after the optimizer steps, and a fault there poisons it (§3, step 6).
- **Unchanged behaviour:** this is the same partial-update-across-parameters behaviour Metal has today. A synchronous `NonFinite` from parameter j already leaves parameters 0..j updated, and the trainer poisons (I).

### 2.2 What stays synchronous (immediate `Err`, nothing recorded)

Everything the host can decide without the device:
- **Shape and dtype:** dtype, rank, shape agreement, empty tensors, and the core validators (`linear_ce_dims`, `cached_attention_dims`, `kv_cache_write_dims`, `permute_output_shape`).
- **Placement:** a host tensor, another device's tensor, or another `MetalBackend`'s tensor.
- **Capacity:** `CapacityExceeded` from the budget, and from a device allocation, which happens on the device thread while the op is recorded, before the call returns.
- **Limits:** the head-dim limit (`refuse_unsupported_metal_head_dim`), `u32` indexing limits, `kv_cache_write`'s `at + Tn > Tcap`, and `cached_attention_forward`'s `kv_len` range.
- **Arguments:** `check_adamw` (step counter, config), Muon config, and `clip_grad_norm`'s `max_norm` (non-finite or negative).
- **Uniqueness:** an in-place op whose target is shared is `Shape`.
- **Token ids and the valid-row count (new, as wgpu).** Today these are found on the device by `ojas_check_ids` (`device.rs`, the `ST_RANGE` and `ST_COUNT` words) (V).
  - Under the contract, a U32 upload keeps a host copy of its ids, as wgpu does (`ojas-wgpu/src/backend.rs:9-11`).
  - Embedding ids and cross-entropy / linear-CE targets are then range-checked, and the valid count formed, on the host.
  - So `OutOfRange` and the all-ignored `NonFinite` stay immediate.
  - Every U32 Metal tensor is an upload (no Metal op produces U32), so the copy always exists (V by inspection of `backend.rs`).
- **Poisoned** (§4.4) and **`Unsupported`**.

**Observable precedence change (as on wgpu).** Today "a non-finite logit outranks an out-of-range target" (`tests/cross_entropy.rs:96-107`, `tests/linear_ce.rs` refusals). Under the contract:
- the out-of-range target is refused immediately, and the op is not recorded;
- so the NaN in the logits is never reported by that call. If an earlier op produced the NaN, that op's fault is still pending.

## 3. Outputs of a faulted op, and the in-place ops

**General rule.** This follows the trait (`ojas-core/src/backend.rs:650-657` on `accumulate_grad`, and `Backend::sync`'s doc at `:638-648`) (V).
- Once a sync point reports `NonFinite`, the outputs of the faulted op, and of every op recorded after it that read them, are **invalid**: their contents are unspecified (in practice they hold the NaN or infinity). They are still valid tensors to drop.
- Outputs of ops recorded before the first faulting op are valid.
- Ops recorded after it that never read a faulted value are also valid, but the report does not say which those are.

**In-place ops keep all-or-nothing under batching.** Each decides on the device, in the same command buffer, from its own call's status words only. It never consults the backend-wide pending fault, so an earlier op's fault cannot block it, and its own fault cannot leak into a later call's decision.

| Op | Today (sync) | Under the contract | How |
| :--- | :--- | :--- | :--- |
| `adamw_step` | `NonFinite`; p, m, v bit-identical | `Ok`; p, m, v bit-identical; `NonFinite { "adamw_step" }` at the next sync point | `ojas_adamw_check` sets this call's words; `ojas_adamw_apply` returns early if they are set (`kernels/ojas_backend.metal`, AdamW section). No wait. The kernels are as of round 5 phase A (V by inspection): the gated-apply kernels now read their status as plain `device const uint*` (§4.2), a prerequisite this contract builds on |
| `muon_ns5_step` | as AdamW, for p and momentum | as AdamW | Every stage checks into this call's words; `ojas_copy_if_clean` commits p and momentum only if they are clear (V by inspection) |
| `accumulate_grad` | `NonFinite`; `acc` bit-identical | `Ok`; `acc`'s **values** bit-identical (stronger than the trait's "invalid"); `NonFinite { "accumulate_grad" }` at the next sync point | `ojas_acc_check` then `ojas_acc_apply` on this call's words. A **unique** `acc` keeps its buffer. A **shared** `acc` is replaced by a new buffer when the call returns `Ok` (today it is replaced only on success); on a fault that new buffer holds the unmodified copy, so the values match but the buffer is new (I: the copy precedes the gated apply) |
| `kv_cache_write` | `NonFinite`; cache unchanged | `Ok`; cache unchanged; `NonFinite { "kv_cache_write" }` at the next sync point | `ojas_check_finite` on `src` into this call's words, then `ojas_kv_write` copies only if they are clear (V by inspection). Satisfies the trait's "on any error the cache is unchanged" (`ojas-core/src/backend.rs:726`) |
| `clip_grad_norm` scale | after the norm is read | after the norm is read (a sync point), then recorded | Scaling happens only when the read found no pending fault and a finite norm |

wgpu writes NaN into `acc` and relies on "invalid" (`ojas-wgpu/src/backend.rs:2133-2160`) (V). Metal keeps its stronger guarantee because it costs nothing extra.

## 4. Batching mechanics

### 4.1 Recording and committing

Today each `Cmd` ends with `rt.synchronize()`, which is `commit(true)` in tessl (`tessl/src/runtime.rs:1788-1790`) (V). Under the contract a `Cmd` records its dispatches with tessl's async encode, already on (`device.rs`, `Worker::open`), and returns as soon as they are encoded. The device thread still runs one `Cmd` at a time, so recording order is channel order.

A command buffer is committed at four triggers:
1. **A sync point or a read.** A waited commit (`commit(true)`), then the status scan (§4.2).
2. **An overlap cap**, every `OVERLAP_DISPATCHES` dispatches (initial value 256).
   - This is an unwaited `commit(false)`, so the GPU runs while the host keeps recording.
   - tessl alternates two command allocators, and a commit with both in flight waits for one (`runtime.rs:1361-1380`) (I: read, not exercised). That bounds the work in flight.
3. **A memory cap**, a waited commit with no report, when either of these exceeds its cap:
   - **Bytes allocated since the last waited commit.** Initial cap: min(1 GiB, a quarter of the budget's cap).
     - tessl returns a dropped buffer to its pool, and removes it from residency, only at a waited commit (`drain_cold_recycles`, `runtime.rs:899-935`, called only from `commit(true)` paths at `:1545, :1573, :1638`) (V).
     - Between waited commits, freed device memory is not reusable, so without this cap a whole step's temporaries would stay resident.
   - **The status slab** (§4.2) filling.
4. **tessl's constant arena.** It is reset only by a waited commit, and grows 16 bytes per dispatch to 16 MiB, about 1.05M dispatches (`runtime.rs:1551-1566`) (V, from tessl's own comment). The memory cap and the slab cap both trigger long before that.

A waited commit at a cap scans the status words and **holds** any fault it finds for the next sync point. It does not report it, so a cap commit never changes what a call returns.

**Buffer lifetime is safe without new code.** MTL4 command buffers do not retain their resources (`runtime.rs:1820-1826`) (V). tessl already keeps every dropped `GpuBuffer` alive in pending queues until a waited commit (`tessl/src/tensor.rs:74-90`) (V). So op temporaries dropped at the end of a `Cmd`, and `Cmd::Free`, are safe while their commands are still recorded or in flight (I: follows from the two facts; phase B's stress test exercises it).

### 4.2 Status words and first-fault naming

Today each op allocates a 32-byte status buffer and dispatches `ojas_status_init` (`device.rs`, `Worker::status`) (V). Under the contract:
- **The slab.** The worker owns a **status slab**: shared memory, `SLAB_SLOTS` slots (initial value 4,096) of `ST_WORDS` words each.
  - An op takes the next slot and binds it exactly where it binds its status buffer today, so no kernel changes.
  - The worker appends `(slot, op name, need_count)` to a pending list, in recording order.
- **Reading status inside kernels.** A kernel that reads another dispatch's status words binds them as plain `device const uint*`. A per-thread `atomic_load` of one address cost about 6 ms per 38.6M-element pass (phase A2, V), and the gated-apply kernels read a slot on every element.
- **Clearing.** The host zeroes slots after a waited commit, when the GPU is idle. This removes today's per-op allocation and `ojas_status_init` dispatch (I: saves one dispatch per op).
- **Scan.** After every waited commit the worker reads the slots of the pending list in order. The first one with a set word becomes the **pending fault**, if none is pending already, judged with today's per-op precedence. Then the list and the slab are reset.
  - The host reads shared memory directly after the wait, so there is no copy to fail between seeing a fault and clearing it. wgpu's hold/release pair (`ojas-wgpu/src/context.rs:10-19`) exists for that copy, and Metal does not need it.
- **Report.** A sync point reports the pending fault and clears it (`take_faults` in wgpu terms).
- **Error kinds.** Device-detected faults are only non-finite values (`ST_IN`, `ST_OUT`) once ids are host-checked (§2.2), so every deferred error is `NonFinite { op }`.

### 4.3 A failed wait or read (F12)

- **A failed waited commit** (a timeout, or a commit-feedback error):
  - tessl marks the runtime failed (`encode_failed`, `runtime.rs:1534-1538, :1615-1618`) (V).
  - The scan does not run, the pending list is kept, and the call returns `Backend { "... runtime poisoned ..." }`. capi maps that text to `E_DEVICE_LOST` (`ojas-capi/src/lib.rs:56-79`) (V).
  - A fault already pending is never cleared by a failed wait. Since the tessl runtime cannot recover, every later call reports the device error rather than the older `NonFinite` (I).
- **A raw read** (`Tensor::to_host`) that waits and scans keeps the fault pending for the next backend sync point. A read that fails after the scan has recorded the fault keeps it too. This is F12's rule; on wgpu it is pinned by `a_failed_read_keeps_the_fault_for_the_next_sync` (`ojas-wgpu/src/backend.rs:2455-2473`) (V for wgpu).

### 4.4 `Poisoned` and device loss

- **A device-thread panic** poisons the backend, as today (`device.rs`, `Worker::serve`). Every later call returns `Poisoned`, sync points included. A pending fault is dropped with the poisoned state, because the backend can no longer report anything else (I; phase B pins that nothing hangs, as `a_device_thread_panic_mid_stream_fails_later_calls_cleanly` does today).
- **Device loss and command-buffer errors** surface at the next waited commit (§4.3), not at the op that caused them. That is the same deferral as wgpu, which records a lost device apart from its error queue and reports it on every later check (`ojas-wgpu/src/context.rs:21-22, :853-863`) (V).

### 4.5 Link and Cmd changes

- **`Cmd`.** Every op variant is unchanged.
  - New: `Cmd::Sync` (wait, scan, report).
  - `Cmd::Read` waits and scans but does not report. `MetalBackend::download` sends `Cmd::Read`, then `Cmd::Sync` semantics, i.e. a report with no second wait.
  - `Cmd::ClipNorm` reports before reading the norm.
- **`Link`.** Unchanged. `Link::call` still waits for the device thread to *encode* the op, so encode-time errors (capacity, unknown buffer, dispatch geometry) stay immediate and outputs get ids. Moving to fire-and-forget `post` would need host-assigned ids and is not part of this contract (I).
- **`Reply`.** Unchanged.
- **Host shadows.** `MetalBuffer` gains an `Option<Arc<[u32]>>` host copy for U32 uploads (§2.2).

## 5. Observable changes for ojas-capi and Go

The coordinator makes these edits. Listed only; ojas-metal does not edit them.

**Error kinds that can now arrive from a later call:**
- `E_NONFINITE` from any recorded op now arrives at the next `clip_grad_norm`, `download` or `sync` in the same call, or, if the call returned early, in the **next** call on the same session.
- `E_DEVICE_LOST` ("runtime poisoned") likewise arrives at the next waited commit.
- `E_CAPACITY` stays immediate.
- Out-of-range ids stay immediate (§2.2), so `GenerateGreedy`'s "a prompt id of 2 or more is an out-of-range error" (`go/api.go:162-164`) is unchanged.

**Functions and docs that change:**
1. **`ojas-capi/src/step.rs` `run` (`:207-275`).**
   - CE forward/backward non-finite values now surface at `clip_grad_norm` (`:247-249`). AdamW's surfaces at `download(&loss_t)` (`:268`).
   - Both stay `E_NONFINITE` and the message still names the faulting op. No state changes, since `run` writes nothing into the session (V by reading `:1-14`).
   - **Leak:** `run` returns early at `check()?` (`:236, :241, :245, :257`) and at `?` after recorded ops. A fault recorded before such a return stays pending on the session's backend and is reported by that session's **next** Step or GenerateGreedy.
   - **Required change:** call `backend.sync()` on every exit path after the first recorded op (or at entry, attributing a leftover fault to the previous call), so a fault never crosses calls.
   - Same for `run_part` (`:393-525`, `check()?` at `:444, :449, :453, :492`).
2. **`ojas-capi/src/generate.rs` `greedy` (`:65-104`).** Faults from `embedding_forward` and `linear_forward` surface at `download(&last)` (`:98-101`). There are early returns at `check()?` (`:81, :87, :90, :93`). Same leak and the same `sync()` fix.
3. **`ojas-capi/src/step.rs` module doc (`:8-11`):** "`clip_grad_norm` reads its norm through the backend's own status read" stays true; add that it is also where earlier non-finite values are reported on Metal and wgpu.
4. **`ojas-capi/src/lib.rs` `kind_of` doc (`:56-66`):** unchanged mapping; note that on Metal and wgpu a kind may come from a later call than the one that caused it.
5. **`go/api.go`.**
   - The `DeviceMetal` / `DeviceWgpu` paragraph (`:36-46`) and `Step` (`:134-138`): document that `ErrNonFinite` and `ErrDeviceLost` may report a fault from an earlier op in the same call. With the capi fix above it can never come from an earlier call.
   - The sentinel block (`:59-74`) is unchanged.
6. **`ojas-core/src/backend.rs` `Backend::sync` doc (`:638-645`)** says "CPU, and Metal today" keep the default. It becomes "CPU keeps this default; wgpu and Metal override it". That is the core lane's file.
7. **`docs/pytorch-parity-plan.md:184`** ("CPU and Metal refuse synchronously") and **`docs/bench-gpu-vs-torch.md:235`** ("24 synchronous ops") describe the old contract.

## 6. Tests that pin per-op refusal today, and where each goes

"Re-point" means:
- the op now returns `Ok`;
- the next `sync()` returns `NonFinite { op }` naming the same op;
- a second `sync()` returns `Ok`;
- for in-place ops, the state is still bit-identical after the `sync`.

Host-decided refusals stay as they are.

| Test (file:line) | Pins today | Under the contract |
| :--- | :--- | :--- |
| `tests/backend_contract.rs:81-101` `non_finite_inputs_are_refused_by_every_op_family` (`nf` at `:87`) | every op family refuses a NaN input synchronously | re-point: `Ok`, then `sync` names the op; one backend per family, or a sync between families |
| `tests/backend_contract.rs:103-126` `non_finite_gradient_leaves_adamw_state_untouched` (`:120`) | `NonFinite`; p, m, v untouched | re-point; state bit-identical after `sync` |
| `tests/backend_contract.rs:130-180` `adamw_refusal_at_the_last_element...` (`:151`, `:172`) | `NonFinite { "adamw_step" }`; bit-identical | re-point; then the clean step at the end must return `Ok` **and** the following `sync` must be `Ok` (the earlier fault was reported, not carried) |
| `tests/backend_contract.rs:183-197` `an_update_that_overflows_leaves_adamw_state_untouched` (`:191`) | overflow is `NonFinite` | re-point. **(B)** Unchanged instead: this test's `lr = f32::MAX` overflows the f32 step scalar itself, which `adamw_scalars` refuses on the host before anything is recorded (V: the test passes unmodified). The device-side overflow case is in `adamw_refusal_at_the_last_element...`, which is re-pointed |
| `tests/backend_contract.rs:199-229` `muon_refusal_at_the_last_element...` (`:221`) | `NonFinite { "muon_ns5_step" }` | re-point |
| `tests/backend_contract.rs:231-249` `non_finite_gradient_leaves_muon_state_untouched` (`:245`) | as above | re-point |
| `tests/backend_contract.rs:251-264` `non_finite_gradient_is_not_clipped` (`:258`) | clip refuses a NaN gradient | **unchanged**: clip is a sync point. Add: a NaN produced by an *earlier* op is reported by clip, which scales nothing |
| `tests/backend_contract.rs:284-294` `out_of_range_token_ids_are_refused` (`:289`, `:292`) | device `ST_RANGE` | **unchanged** result (immediate `OutOfRange`), now from the host copy |
| `tests/backend_contract.rs:296-305` `an_all_ignored_batch_is_non_finite` (`:301`, `:303`) | device `ST_COUNT` | **unchanged** (immediate), now from the host count |
| `tests/accumulate_grad.rs:85-121` `a_non_finite_sum_is_refused...` (`:110`) | `NonFinite`; `acc` bit-identical and same buffer | re-point; `acc`'s values bit-identical after `sync`. Unique case: buffer pointer unchanged. Shared case: the pointer **changes** (a new buffer holding the old values), and the other handle is untouched |
| `tests/attention_forward.rs:168-195` (`:181`, `:191`) | `causal_sdpa_forward` refuses NaN and overflowing scores | re-point |
| `tests/attention_backward.rs:114-140` (`:127`, `:138`) | as above, backward | re-point |
| `tests/cross_entropy.rs:67-91` `a_non_finite_logit_anywhere_is_refused_even_in_an_ignored_row` (`:83`, `:88`) | `NonFinite` from forward and backward | re-point (the check stays inside `ojas_ce_fused`) |
| `tests/cross_entropy.rs:96-107` `a_non_finite_logit_outranks_an_out_of_range_target` (`:103`, `:106`) | NaN outranks the bad target | **flips**: the bad target is refused immediately (`OutOfRange`), and nothing is pending afterwards (`sync` is `Ok`) |
| `tests/kv_cache.rs:250-320` `cached_attention_refusals` (`:298`, `:310`; `:258` is a host `OutOfRange`) | `NonFinite` for q, k, v and overflow | re-point `:298`, `:310`; `:258`, Shape, head-dim and Placement are unchanged |
| `tests/kv_cache.rs:367-400` `kv_cache_write_refusals_leave_the_cache_unchanged` (`:388`; `:375` host) | NaN src refused, cache unchanged | re-point `:388` with the cache bit-identical after `sync`; `:375` and the shared-cache `Shape` are unchanged |
| `tests/linear_ce.rs:312-363` `refusals` (`:322` nonfinite helper; `:332` `OutOfRange`) | NaN x or w, overflow, all-ignored, bad target | re-point NaN x and w and overflow. **Flip** "w with a bad target" and "overflow" with `bad_t`: immediate `OutOfRange`. All-ignored stays immediate |
| `tests/norms.rs:100-142` `non_finite_inputs_and_overflowing_squares_are_refused` (`:114, :120, :134, :139`) | `NonFinite { "rms_norm_*" }` | re-point |
| `tests/norms.rs:149-245` `qk_norm_is_the_composition_and_refuses_in_its_order` (`:203`, `:233`, `:240`) | q's fault outranks k's host refusals; budget fallbacks | host refusals of k (shape, placement, budget) become **immediate**, and q's NaN is pending (reported by the next `sync`). Re-point each case to "returns k's host error now; `sync` then names `rms_norm_*`". In `rms_pair`, the k-*validation* fallback has no purpose left (k's host refusal is immediate and q's fault is pending either way, which is the composition's order). Only the *budget* fallback (both sides' scratch at once) remains; phase B removes the other. **(B) Corrected: phase B kept the k-validation fallback. (Round 6) Removed again: with `ojas_core::shapes` adopted, `rms_qk_norm_*_dims` validates both pairs before anything is recorded, so a refused k records nothing, matching wgpu and the shape contract's "a refused call records nothing". The four norms rows now expect nothing pending.** Earlier note: Without it, k's host error returns before q is recorded, so q's NaN is never pending and `sync` is `Ok`, which contradicts this row's own re-point and the "unchanged" row for `qk_norm_fuses_only_when...` (which asserts the fallback). With it, the composition records q and then refuses k, which is exactly this row's re-point (V: both tests pass) |
| `tests/permute.rs:134-150` `non_finite_inputs_are_refused_like_the_cpu_reference` (`:143`, `:147`) | Metal matches CPU's synchronous refusal | re-point: Metal `Ok` then `sync` names `permute`; CPU stays synchronous. This is the round-1 coordinator decision ("same rule as every other op on your backend") |
| `tests/training_step.rs` `metal_training_steps_track_cpu_loss_and_read_back_only_the_loss` | no readback before the loss; +1 readback for the loss | **unchanged**: waits are not readbacks. Add a wait-count assertion (§7) |
| `tests/concurrency.rs` | bits equal to a serial run, no faults | unchanged; extended in §7 |
| `src/backend.rs:1428-1486` `a_device_thread_panic_mid_stream_fails_later_calls_cleanly` | panic poisons; later calls `Poisoned`, no hang | unchanged; add a pending fault before the panic (§7) |
| `src/backend.rs:1575-1600` `qk_norm_fuses_only_when...`, `:1607-1625` `a_kv_write_past_capacity_is_refused_before_any_dispatch` | host-side decisions | unchanged |
| `src/gpu.rs` NonFinite sites (`:295`, `:667`, ...) | the tiny tessl training step, not `MetalBackend` | out of scope for this contract |

## 7. Test plan for the new behaviour (phase B, each written to fail first)

All of these fail today, because today the op itself returns `Err`.

1. **The first faulting op is named.** `silu_forward(NaN)` returns `Ok`, then `mul_forward(y, b)` returns `Ok`, then `sync` returns `NonFinite { "silu_forward" }`, then `sync` returns `Ok`. A later fault in a different op is named after itself (the wgpu shape: `ojas-wgpu/tests/faults.rs`, `the_first_faulting_op_is_named_not_the_lowest_bit`).
2. **A later clean op does not clear it.** Fault, then 50 clean ops, then `sync` still names the first.
3. **Every sync point reports.**
   - `sync`, `download` of an unrelated tensor, and `clip_grad_norm` each report a pending AdamW fault.
   - clip leaves its gradients bit-identical, and `to_host` does **not** clear it.
   - This mirrors `adamw_fault_is_named_at_every_kind_of_sync_point`.
4. **A fault survives a failed read.** A `cfg(test)` hook fails the next `Cmd::Read` after its scan; the read returns `Backend`, and the next `sync` still names the op (F12).
5. **A fault survives a cap commit.**
   - Record a fault, then enough ops to trigger the memory cap and the slab cap. Both waited commits return nothing, and `sync` names the first op.
   - With more than `SLAB_SLOTS` ops before the fault, the name is still right.
6. **In-place all-or-nothing under batching.**
   - `adamw_step`, `muon_ns5_step`, `accumulate_grad` (unique and shared) and `kv_cache_write` with a NaN at the last element return `Ok` and leave their targets bit-identical after `sync`.
   - A later clean call on the same tensors applies normally: its own words decide, not the pending fault.
7. **Host refusals stay immediate.** Shape, dtype, placement, capacity, head dim, `kv_len`, `at + Tn`, out-of-range ids and all-ignored each return `Err` at the call, and nothing is pending (`sync` is `Ok`).
8. **Concurrency.**
   - Six threads on one backend run the round-3 mixed sequence; one thread injects a NaN op.
   - Exactly one `sync` across all threads reports it, and every clean thread's bits equal the serial run.
   - A second backend sees nothing.
9. **Poisoned.** A pending fault, then an injected device panic: later calls, `sync` included, return `Poisoned` within a watchdog, and drop completes.
10. **Waits per training step.**
    - A `cfg(test)` (or doc-hidden) counter of waited commits on `MetalBackend`.
    - A sequence of N recorded ops does 0 waits until `sync`, then exactly 1. This fails today: N waits.
    - `training_step.rs` asserts the bound below.

**Wait-count bench target.** At K=4, B=4, T=1024 on the 124M model:

| | Waits per step, today | Target |
| :--- | :--- | :--- |
| Forward and backward ops | about 4 × 1,000 (one per op) (I: `docs/framework-design.md` §9) | 0 except cap commits |
| `clip_grad_norm` | 2 (norm read and scale) | 1 |
| Optimizer steps | 170 | 0 |
| `sync` | 0 | 1 |
| Loss `download` | 1 | 1 (on an already-drained queue) |
| Total | about 4,170 | **3 + memory-cap commits** (about step bytes / cap; with a 1 GiB cap, a few per step) (I) |

`adamw_full` (170 tensors) should go from 170 waits to 1. Its time should then approach the bandwidth bound of reading p, g, m, v and writing p, m, v: about 3.5 GB, roughly 15 ms at the 240 GB/s CE reached in round 3. Today it is 70–103 ms min and about 222 ms median after phase A's fix (V), and 169–245 ms in earlier rounds (I; to be measured in phase B).

## 8. Open questions for the coordinator

1. **Overlap commits** (`commit(false)` every 256 dispatches) are what let the GPU run while the host records. tessl's two-allocator backpressure is read but not exercised by ojas yet (U). Phase B could start with waited commits only at sync points and caps, then measure overlap separately. **Proposal:** do both, with an interleaved A/B.
2. **Memory-cap value.** A quarter of the budget cap bounds the extra resident memory, but a large budget gives few waits. Is a fixed 1 GiB ceiling acceptable?
3. **capi leak fix** (§5, items 1–2): `sync` at entry, or on every early exit? This is a coordinator decision.

**(B) Answers, as implemented.**
1. Overlap commits are on, every 256 dispatches, checked between commands; a stress test pins them (§9.3).
2. The memory cap is min(1 GiB, budget cap / 4), plus a working-set trigger. tessl exposes `GpuRuntime::current_allocated_bytes` (`MTLDevice::currentAllocatedSize`, `tessl/src/runtime.rs:767-775`) and `memory_info().recommended_working_set` (V), so no objc2 call was needed.
3. capi's `settled` (`ojas-capi/src/lib.rs`) syncs at the end of every device step and generate (the coordinator's change; its tests pass on this build, §9.3).

## 9. Phase B: what was built

### 9.1 Mechanics

All in `ojas-metal/src/device.rs` unless named.
- **Status slab.** One shared buffer of `SLAB_SLOTS = 4096` slots × 8 words, allocated and zeroed at open. `Worker::status(op)` takes the next slot at the start of the op and appends `(slot, op, command, sealed)` to the pending list, in recording order.
  - Removed: the per-op status allocation, `ojas_status_init`, `ojas_check_ids` (kernels and `KERNELS` entries), `finish` and `verdict`.
- **Scan** (`Worker::scan`, after every waited commit, `Worker::settle`).
  - The first pending slot with `ST_IN` or `ST_OUT` set becomes the held fault, if none is held.
  - The host then zeroes every used slot not still pending.
  - A failed wait or mapping returns before the scan, so the pending list is kept.
- **A command that fails** drops its own slots unless they are sealed. `rms_sides` seals q's slot once q is encoded, so a k-side failure still leaves q's fault pending, as in the composition.
- **Commit triggers.**
  - Checked between commands (`Worker::after_command`), in this order:
    1. Bytes allocated since the last waited commit exceed `mem_cap`.
    2. `current_allocated_bytes()` is above 3/4 of `recommended_working_set` **and** this backend has allocated more than 64 MiB since its last waited commit. The device figure counts every allocation on the device, other backends' included (U: from tessl's comment and Apple's `currentAllocatedSize` wording, not measured). Gating on the backend's own count keeps a steady state, or another backend's growth, from making every op wait: above the threshold a backend waits at most once per 64 MiB it allocates. A first version gated on device-wide growth since the last wait; it was replaced because other backends in the same process move that figure.
    3. 256 dispatches since the last commit: an unwaited `commit(false)`.
  - At slot take: a full slab forces a waited commit.
  - In an allocation: one that fails as an exhausted device does gets one waited commit (which recycles dropped buffers) and one retry. That commit does not scan the current command's slots, whose kernels may not be encoded yet (`Scan::BeforeCurrent`).
- **Sync points.**
  - `Cmd::Sync` backs the new `Backend::sync` override (`backend.rs`).
  - `download` calls `tensor.to_host(self.budget())`, a `Cmd::Read` that waits, scans and holds, and then `sync`, which has nothing left to wait for.
  - `Cmd::ClipNorm` records its checks, settles, reports, and only then reads the norm. The scale is recorded.
- **Host ids.**
  - `MetalBuffer::ids` holds the host's copy of every U32 upload.
  - `MetalBackend::ids` range-checks the tensor's window: the first bad position is reported with the same detail text as before.
  - It also counts the valid rows. Cross-entropy and linear CE refuse a count of 0 at the call, and otherwise pass it to `ojas_ce_fused`, `ojas_lce_grad` and `ojas_ce_mean` as a `constant uint`.
- **Uploads.** tessl maps a buffer for the host only after `commit_m4(true)` (`tensor.rs:165-199`, `runtime.rs:572-579`) (V). So an upload made while work is recorded waits. The worker makes that wait itself (`settle` in `Worker::upload`), so it is counted and scanned.
- **Wait counter.** `MetalBackend::waits()` (doc-hidden) counts every waited commit the worker makes. Tessl's own commit counter cross-checks it (`tests/wait_count.rs`).

### 9.2 Departures from §2–§8

- **The k-validation fallback in `rms_pair`:** kept in round 5, removed in round 6 when Metal adopted `ojas_core::shapes` (see §6, the norms row). Only the budget fallback remains.
- **Unchanged: `an_update_that_overflows...`.** It stays an immediate host refusal (§6).
- **Overlap commits fall between commands, not inside one.** One command, for example linear CE at 4096 × 50,304, can encode more than 256 dispatches before the next commit. tessl's own cap of 100,000 dispatches per command buffer (`runtime.rs:1497`) still bounds it (V by reading).
- **An encode error discards earlier work.** It poisons tessl and discards the whole open command buffer (`abort_open_batch`, `runtime.rs:1474-1483`), including ops other calls recorded earlier. Their faults are lost, and every later call reports the runtime error (I; the device error wins, as §4.3 says).
- **`download` is two commands.** If another thread records between them, the `sync` waits again. That is still correct.
- **§7's "a few memory-cap commits per step" was low.** It is about 40 at 124M (§9.3).

### 9.3 Evidence

**Tests.** 149 `ojas-metal` tests pass, run serially and with default threads; there were 130 at the end of phase A (V).
- New `tests/deferred_faults.rs` covers §7 items 1–3 and 5–8 and 10.
- New `tests/wait_count.rs` holds the tessl cross-check.
- New unit tests in `src/backend.rs` cover item 4 (a failed read), item 5 (memory cap), item 9 (`Poisoned`), the overlap stress test, the allocation retry, the working-set growth gate, a failed command's slots, and a fused QK-norm whose k side fails.

**Failing before the change** (`scratchpad/metal-r5b-tests-before.txt`) (V):
- `deferred_faults`: 10 of 10 failed. Most failed at the op itself returning `Err`. The waits test failed on its first wait count. Host refusals failed on the flipped embedding precedence.
- `wait_count` failed at `commits == 0` (left 20, right 0).
- Of the first 6 new unit tests, 5 failed then (failed read, memory cap, working set, allocation retry, `Poisoned`).
- The overlap stress test passed before and after. It is a characterization test: bits equal to a serial run.
- Two unit tests were written after the implementation: `a_call_that_fails_after_recording_leaves_no_fault` and `a_fused_qk_norm_keeps_qs_fault_when_ks_side_fails`. Their only fail-first evidence is that mutants M11 and M12, which remove exactly the code they pin, fail them.
- The working-set test was tightened when its gate changed (tiny ops: 0 waits; 192 MiB: 1 to 3 waits), and M5 was re-run against the new gate (killed).

**Re-pointed** exactly per §6, with the corrections in §9.2. A shared `deferred()` helper (`tests/common/mod.rs`) asserts three things: `Ok`, then `sync` names the op, then a second `sync` is `Ok`. Each in-place test checks its state bit-identical after the `sync`.
- `tests/cross_entropy.rs`'s `a_non_finite_logit_outranks_an_out_of_range_target` is **renamed** `an_out_of_range_target_is_refused_at_once_even_beside_a_non_finite_logit`, because §6 flips what it pins. Same inputs, plus a `sync` that must be `Ok`.
- `tests/training_step.rs` gained the §7 item 10 assertion: exactly 2 waits per step, from `clip_grad_norm` and the loss download. Mutant M15, where every command waits, fails it.

**Mutations: 19 of 19 killed** (`scratchpad/metal-r5b-mutations.txt`) (V). The mutants were:
- last fault wins; report does not clear; slots not zeroed;
- the retry scans the current command; no working-set growth gate; overlap never fires;
- clip, or `download`, does not report;
- id range off by one; ignore index not skipped;
- a failed command keeps its slots; the rms side is not sealed;
- a cap commit reports; a raw read clears the fault; every command waits; a full slab is not settled;
- the adamw apply ignores its status; the CE valid count is off by one; `sync` does not wait.

**Interleaved A/B, one process.** A waits after every op, emulating the old per-op wait with `sync`; B is deferred and syncs once. Times are min / median ms. Run 1 was at load 19–20 and run 2 at load 15–16 (`scratchpad/metal-r5b-ab.txt`, `metal-r5b-ab2.txt`).

| Row | A, run 1 | B, run 1 | A, run 2 | B, run 2 | Waits A / B |
| :--- | ---: | ---: | ---: | ---: | :---: |
| silu, 1 element (per op, 200 ops) | 0.367 / 1.053 | 0.007 / 0.014 | 1.106 / 1.294 | 0.007 / 0.011 | 200 / 1 |
| accumulate_grad 50304 × 768 | 3.215 / 4.114 | 3.269 / 3.820 | 3.083 / 4.591 | 2.987 / 4.560 | 1 / 1 |
| adamw_full (170 tensors) | 183.1 / 246.6 | 22.6 / 26.0 | 250.0 / 273.1 | 22.2 / 23.1 | 170 / 1 |
| block_fwd_bwd | 125.9 / 138.7 | 82.6 / 94.3 | 118.4 / 134.1 | 67.6 / 77.7 | 50 / 1 |

- **adamw_full** now sits near §7's bandwidth estimate of about 15 ms. Phase A measured the old code at a 69.9 ms min and about 222 ms median (I: a different process and load).
- **A is an emulation.** The old code also allocated and initialized a status buffer per op, so A is not a bit-exact reproduction of the old path (I).

**Overlap on vs off** (run 2, both deferred):

| Row | On, min / median ms | Off, min / median ms | Speedup |
| :--- | ---: | ---: | :---: |
| silu, 64 elements, chain of 2,000 | 14.78 / 16.03 | 18.95 / 19.92 | 1.28× |
| adamw_full | 21.91 / 22.59 | 21.86 / 22.79 | 1.00× |
| block_fwd_bwd | 72.91 / 75.94 | 73.38 / 76.70 | 1.01× |

The overlap path was verified, not assumed:
- In `wait_count`, 600 one-element ops give at least 2 tessl commits and 0 waits (V).
- Six threads × 1,000 tiny ops on one backend give bits equal to a serial run with overlap off (V).
- About 80 unwaited commits happen with no wait between them, so both allocators are in flight. tessl's backpressure wait itself cannot be observed from ojas (I).

**Waits per ojas-model training step** (`Trainer::step` on Metal, K = 4; `scratchpad/metal-r5b-trainer-waits.txt`):

| Spec | Backend ops per step | Host uploads | Waits before (I: one per op, + clip scale + loss read) | Waits after (V) | Step time |
| :--- | ---: | ---: | ---: | ---: | ---: |
| tiny (B 2, T 32) | 596 | 28 | about 598 | 25 | 40–49 ms |
| 124M (B 4, T 1024) | 3,356 | 28 | about 3,358 | 71–72 | 4.83–5.04 s |

- **Tiny:** the 25 waits are 2 sync points and 23 upload waits.
- **124M:** 2 sync points, at most 28 upload waits, and the rest (about 41) are memory-cap commits (I: by subtraction; there are no per-trigger counters).

**Other checks.**
- `ojas-model`'s `metal_tiny_five_steps_match_cpu` and `metal_g9_resume_matches_the_straight_run` pass.
- `ojas-capi`: 45 tests pass.
- `cargo build --workspace --release --all-targets` builds.
- `cargo clippy -p ojas-metal --all-targets -D warnings` is clean.
- `examples/metal_bench.rs` now syncs inside each timed run.

### 9.4 Gaps

- **Upload waits are now the largest share** after memory caps: 28 host uploads per trainer step, 7 per micro-step.
  - tessl has no host write that skips the wait for a fresh buffer no recorded work references.
  - The fix belongs in tessl (an unwaited write for a new allocation) or in the trainer (upload all K micro-batches' ids and the step's constants before recording).
- **Memory-cap commits per 124M step (about 41) are inferred, not counted per trigger.** A per-trigger counter would settle it. The cap trades waits for resident memory: at 1 GiB, the step's freed temporaries recycle every 1 GiB of allocation.
- **The capi early-return test runs on wgpu only** (`an_early_return_never_leaks_a_deferred_fault_into_the_next_call`). A Metal twin would pin `settled` on Metal; that file belongs to the coordinator.
- **Stale text outside this lane:**
  - `bench/ojas_rows.rs:21-22` says Metal's `sync` is "the trait default, a no-op". It is now an override that waits.
  - The `Backend::sync` doc in `ojas-core`, and §5's list.
