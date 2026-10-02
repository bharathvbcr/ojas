# Typed Host Storage: Plan

Status (2026-10-01):
- **Step 1 has landed and is verified.** It was approved by the user and gated in a staging copy before being copied into the live tree.
- **Integration run 4:** Rust 1071 passed and 0 failed (`--no-fail-fast`), Go 38/38, and clippy `-D warnings` is clean ([`status.md`](status.md)).
- **Steps 2–4 are not started.**
- **Rollback point:** the snapshot `wip/framework-layer-2026-10-01` (c686bb0), taken before step 1.
- **Not yet run for step 1:** `target-matmul/parity.sh` and the `block_ab.sh` / `sample` profile that the success criterion below calls for.

## Why

A host `Tensor` keeps its elements in a `Vec<u8>` (`ojas-core/src/tensor.rs`, `Payload::Host`). A CPU kernel that needs `&[f32]` (Accelerate, the packed GEMM, attention) first decodes a copy with `to_f32_vec`. Every result is built in a `Scratch<u8>` and encoded. The macOS `sample` profile of one nanolab block, forward and backward at 6 threads, shows:

| Bucket | Samples | Source |
| :--- | ---: | :--- |
| BLAS (Accelerate) | about 3100 | the GEMMs themselves |
| Storage tax | 975 | `to_f32_vec` 263, `memmove` 195, `f32_input_list` 184, `alloc_out` 179, `madvise` 154 |
| `softmax_rows` | 365 | attention |

Counts are top of stack from `target-matmul/block.sample.txt` (2026-10-01; `target-matmul/` is uncommitted scratch, so reproduce with `OJAS_BENCH_MODE=loop OJAS_BENCH_OPS=block OJAS_BENCH_DIRSEL=step OJAS_BENCH_SAMPLE_OUT=<file>` in `bench_ops.rs`, which runs `/usr/bin/sample`). The BLAS row sums the three hottest Accelerate frames. The storage tax is about 20–25% of active block time. That share is measured. What removing it would save is an estimate until it is measured, because some of those copies also warm the cache for the kernel that follows.

Two CPU paths already avoid the copy: `validate::f32_words` reads `&[[u8; 4]]` in place, and `pool::scoped::fill` writes output blocks in place. They do not help an operand that has to be a `&[f32]`, and they cost every kernel a decode at each load.

## What does not change

- `byte_offset`, `view`, `narrow`, `reshape`, `strides` and `storage_len` keep their meaning in bytes. The GPU crates bind buffers at byte offsets (`ojas-metal/src/gpu.rs`, `ojas-wgpu/src/backend.rs`).
- A view never changes dtype (`Tensor::view` copies `self.dtype`). So a storage's element type is fixed when it is created.
- `byte_offset` must already be a multiple of the dtype size (`misaligned_byte_offset_is_refused`). On host storage it maps to element `byte_offset / dtype.size()` with no remainder.
- `to_f32_vec`, `to_u32_vec`, `from_f32`, `from_u32` and `write_f32` keep their signatures. That covers 966 call sites across 12 crates, mostly tests, which do not change.
- `Payload::Device` and `DeviceBuffer` stay as they are.
- `#![forbid(unsafe_code)]` stays in the 15 crates whose `lib.rs` has it today. No new dependency is added: `bytemuck` appears in `Cargo.lock` only through wgpu.

## API shape (ojas-core)

```rust
enum Payload {
    Host(HostData),
    Device(Arc<dyn DeviceBuffer>),
}

enum HostData {
    F32(Vec<f32>),
    U32(Vec<u32>),
    /// Bf16 and F16; the tensor's dtype says which.
    Half(Vec<u16>),
}
```

New on `Tensor`:

| Method | Meaning |
| :--- | :--- |
| `f32_slice(&self) -> Result<&[f32]>` | The contiguous window, borrowed. Error order matches `to_f32_vec` today: dtype, then placement, then layout, then window. |
| `u32_slice(&self) -> Result<&[u32]>` | The same for `U32`. |
| `f32_slice_mut(&mut self) -> Result<&mut [f32]>` | Requires a uniquely owned allocation, as `write_f32` does now. `write_f32` becomes `f32_slice_mut()?.copy_from_slice`. |
| `from_scratch<T: HostElement>(Scratch<T>, shape) -> Result<Tensor>` | Moves the vector and its reservation into the tensor without a copy. The dtype comes from `T`. This replaces `from_scratch(Scratch<u8>, shape, dtype)`. |
| `to_ne_bytes(&self) -> Result<Vec<u8>>` | An encoded copy, for the paths that really need bytes: host-to-device uploads and dumps. |

`HostElement` is a sealed trait implemented for `f32` and `u32`, so `Scratch<T>` (already generic, `ojas-core/src/budget.rs`) stays the only path that allocates and charges.

**`contiguous_bytes` is retired.** Without `unsafe` or a new dependency, a `Vec<f32>` cannot be lent as `&[u8]`. Its production callers move as follows:

| Caller | Move to |
| :--- | :--- |
| `ojas-cpu/src/validate.rs:277`, `pointwise.rs:302`, `accum.rs:40-41`, `layout.rs:38`, `kv.rs:38,192` | `f32_slice` / `u32_slice`; drops the `as_chunks::<4>` decode |
| `ojas-infer/src/kernels.rs:23` (`copy_f32`) | `f32_slice` |
| `ojas-metal/src/backend.rs:605`, `ojas-wgpu/src/backend.rs:1089` (upload) | `to_ne_bytes`; both already copy into an owned buffer |
| Test helpers in `ojas-model`, `ojas-autograd`, `ojas-core/tests/bench_decode.rs` | `to_ne_bytes` or the typed slices |

**`from_le_fill` is retired for the same reason.** It lends `FnOnce(&mut [u8])` over the new tensor's storage (`ojas-core/src/tensor.rs:134`). Its one production caller is checkpoint load, `ojas-model/src/checkpoint.rs:476`, which fills it with `SafeTensors::read_into(name, offset, dst)`. Three replacements:

| Option | Extra memory | Extra pass | Where little-endian is handled |
| :--- | :--- | :--- | :--- |
| Whole-tensor staging `Vec<u8>`, then decode | the tensor's size, charged | one | `ojas-core` |
| Typed fill `FnOnce(&mut [f32])` | none | none in core; the reader decodes | every reader |
| **Chunked reader** `from_le_reader(shape, dtype, budget, read: FnMut(offset, &mut [u8]))` | one bounded chunk (64 KiB, `LE_READ_CHUNK_BYTES`), charged | one, chunk by chunk | `ojas-core` |

The recommendation is the chunked reader. `read_into` already takes an offset (`ojas-io/src/safetensors.rs:235`). The transient stays bounded however large the tensor is. Endianness stays in one place, as it is now (`from_le_fill` swaps on big-endian targets). The cost is one decode pass at checkpoint load, which is not in the profiled hot path.

**The private encode and decode helpers** all live in `ojas-core/src/tensor.rs`, and their callers move with them:

| Helper | Callers | After |
| :--- | :--- | :--- |
| `write_ne_f32` (:809) | `from_f32` (:162), `write_f32` (:208) | `copy_from_slice` into `HostData::F32` |
| `write_ne_u32` (:815) | `from_u32` (:177) | `copy_from_slice` into `HostData::U32` |
| `decode_f32` (:846), `decode_u32` (:855) | `to_f32_vec`, `to_u32_vec`; `ojas-core/tests/bench_decode.rs` measures them | `to_vec` of the typed slice. `bench_decode.rs` is retired or retargeted at the readback decode. The unit test `decoders_refuse_a_ragged_window_with_their_exact_message` retires too: typed storage cannot hold a ragged window. |

Device readback (`to_host`) decodes the `Vec<u8>` from `DeviceBuffer::read_bytes` into `HostData`. For a moment both buffers are alive, twice the tensor size, and only the result is charged; today the bytes move into storage with no second buffer. **Decision (peer, 2026-10-01): charge the transient and leave `DeviceBuffer` unchanged in step 1.** A `read_f32_into` on the trait would widen it for Metal, wgpu, CUDA and HIP inside a change that already cannot be split. Readback is not in the profiled hot path. That trait change comes later, as its own change, and only if a readback benchmark shows the extra pass matters.

## Invariants to keep

Each of these has a test today and must still pass unchanged:

- **Charge before allocating.** The reservation is the last field, so the vector is freed before the charge is released (`Scratch`, `Storage`).
- **The budget is released when the last view drops** (`budget_is_released_when_last_view_drops`).
- **Writes need sole ownership** (`write_requires_unique_owner_and_matching_len`, `ensure_writable_agrees_with_write_and_never_writes`).
- **Placement is reported before layout or ownership** (`device_tensor_refuses_every_host_accessor`, `strided_device_view_reports_placement_not_layout`).
- **Window bounds and overflow** (`randomized_views_match_reference_bounds`, `overflowing_shapes_are_errors_not_panics`).
- **Host-to-host `to_host` is a shared clone, not a readback** (`host_to_host_is_a_shared_clone_and_not_a_readback`).
- **The Exact goldens keep their bits** (`ojas-cpu/tests/exact_golden.rs`). Typed storage changes where the bytes live, not the arithmetic.

## Tests this plan retires or rewrites

Deleting a test needs the user's approval, so every affected test is listed here.

**Retired (the user decides):**

| Test | Why it goes | What still guarantees the property |
| :--- | :--- | :--- |
| `decoders_refuse_a_ragged_window_with_their_exact_message` (`ojas-core/src/tensor.rs:1060`) | It calls the private `decode_f32` and `decode_u32` with byte slices of length 1–13 that are not whole elements. Step 1 deletes both decoders, and typed storage cannot hold a ragged window: a `Vec<f32>` has whole elements only. | The type system. `from_le_reader` replaces the byte entry point and sizes every chunk from the shape. A source that runs short fails inside `read`, which drops the tensor; the test gate covers that. |
| `bench_to_f32_vec_against_push_loop` (`ojas-core/tests/bench_decode.rs:140`, `#[ignore]` timing benchmark) | It times the byte decode inside `to_f32_vec`, and that decode no longer exists: `to_f32_vec` becomes a slice copy. | Nothing is lost on correctness, since it is a benchmark. The alternative to retiring it is to retarget it at the `from_le_reader` decode, which is the one decode left on the host. |

**User decision (2026-10-01):**
- Step 1 is approved, and the coordinator session drives it.
- `decoders_refuse_a_ragged_window_with_their_exact_message` is retired.
- `bench_to_f32_vec_against_push_loop` is retargeted at the `from_le_reader` decode, not deleted.

**Rewritten, with the op name changed:** `tensor_decode_contract.rs:221, 238, 304, 313` pin `op: "Tensor::contiguous_bytes"`. That method is deleted, so the layout and placement errors reached through `to_f32_vec` and `to_u32_vec` now name the method that was called (see "Step 1 API" below). The variant, the detail text and the check order are unchanged; only the `op` string changes.

**Rewritten, with their assertions kept:**
- The four `from_le_fill` tests in `tensor.rs` move to `from_le_reader`, as listed in the test gate.
- `ojas-core/tests/tensor_decode_contract.rs` (9 tests) and `bench_decode.rs::reference_and_to_f32_vec_agree_bit_for_bit` build tensors from raw bytes through `raw_tensor`, which calls `from_scratch(Scratch<u8>)`. `raw_tensor` moves to `from_le_reader`. The bit-exact assertions, including the special f32 patterns and random bytes at many lengths, stay as they are.

## Step 1 API (exact; patch call sites against this)

Every length is a count of **elements**. Byte offsets (`byte_offset`, `view`, `narrow`, `storage_len`) stay in bytes. `storage_len` is the element count times `dtype.size()`.

```rust
/// Sealed. Implemented for f32 (DType::F32) and u32 (DType::U32).
pub trait HostElement: Copy + Default + private::Sealed { const DTYPE: DType; }

impl Tensor {
    pub fn f32_slice(&self) -> Result<&[f32], OjasError>;
    pub fn u32_slice(&self) -> Result<&[u32], OjasError>;
    pub fn f32_slice_mut(&mut self) -> Result<&mut [f32], OjasError>;
    /// `scratch.len()` must equal the shape's element count; dtype is T::DTYPE.
    pub fn from_scratch<T: HostElement>(scratch: Scratch<T>, shape: &[usize]) -> Result<Tensor, OjasError>;
    /// Encode the contiguous window, native-endian, into `dst`, of any dtype.
    /// `dst.len()` must equal the element count times dtype.size(). The caller
    /// allocates `dst`, so the caller decides whether the transient is charged.
    pub fn write_ne_bytes(&self, dst: &mut [u8]) -> Result<(), OjasError>;
    /// The same encoding into a new vector, after the window checks and
    /// before anything is allocated (a broadcast view is refused, not sized).
    /// Not charged; for device uploads that already reserved the device copy.
    pub fn to_ne_bytes(&self) -> Result<Vec<u8>, OjasError>;
    /// Replaces from_le_fill. `read(byte_offset, chunk)` fills `chunk`, at most
    /// LE_READ_CHUNK_BYTES (64 KiB), with the little-endian bytes of the tensor starting at `byte_offset`.
    /// The chunk is charged to `budget` beside the tensor and released on every path.
    pub fn from_le_reader<E: From<OjasError>>(
        shape: &[usize], dtype: DType, budget: &Budget,
        read: impl FnMut(u64, &mut [u8]) -> Result<(), E>,
    ) -> Result<Tensor, E>;
}
```

**Deleted:** `contiguous_bytes`, `from_le_fill`, the byte `from_scratch(Scratch<u8>, shape, dtype)`, and the private `write_ne_*` and `decode_*` helpers.

**Unchanged signatures:** `to_f32_vec`, `to_u32_vec`, `from_f32`, `from_u32`, `write_f32`, `ensure_writable_f32`, `zeros` and `to_host`.

**Error order.** Each error names the method that was called (`"Tensor::f32_slice"`, `"Tensor::to_f32_vec"`, and so on), and its detail text is the same as today.
- **Readers:** `f32_slice`, `u32_slice`, `to_f32_vec`, `to_u32_vec` and `write_ne_bytes`.
  1. Dtype. `write_ne_bytes` takes any dtype, so it skips this check.
  2. Placement.
  3. Layout: `Shape "view is not contiguous"`.
  4. Window: `OutOfRange`.
  5. For `write_ne_bytes` only, `dst` length: `Shape`.
- **`f32_slice_mut`:**
  1. Dtype.
  2. Placement.
  3. Layout.
  4. Window overflow.
  5. Ownership: `Shape "contiguous write requires a uniquely owned allocation"`.
  6. Bounds.
- **`write_f32` and `ensure_writable_f32`:** unchanged. The length check, `Shape "data len {n} != shape product {m}"`, still comes **before** placement. They go through a private `writable_f32(len)`, not through `f32_slice_mut()?.copy_from_slice`, which would panic on a length mismatch and reorder the errors.

**`to_host` on a device tensor:**
- It reserves the typed result and one read piece, `min(window, READBACK_CHUNK_BYTES)` (1 MiB), before reading.
- It reads the window with `read_bytes(offset + done, n)` one piece at a time, using the trait's existing offset and length, and decodes each piece straight into the result.
- It drops the piece's charge before returning.
- The peak charge is the window plus one piece, not twice the window. The first combined gate, which reserved a whole-window transient, failed `ojas-model`'s save-headroom test. That is why the readback is chunked.
- A short read on any piece is `Backend`. The result and the piece are both released, and nothing is counted.
- `device_readbacks` and `Budget::record_readback` still count one call per `to_host`.

**Loader piece (`LE_READ_CHUNK_BYTES`): 64 KiB, not 1 MiB.** It is the only transient a load adds. With 1 MiB, every tensor under 1 MiB got a full second copy, and `ojas-model/tests/resume_memory.rs` (heap-counted) failed. 64 KiB matches the save path's `LE_CHUNK`. A 1.5 GB checkpoint takes about 24k positioned reads.

**Contract changes from this, for the user:**
- **Save headroom.** It was exactly one tensor. It is now one tensor plus one readback piece (at most 1 MiB) on a device backend; the CPU backend does no readback. The test was rewritten as `save_fits_in_one_tensor_plus_one_readback_piece_of_host_headroom`, keeping its slack 0 (fits) and slack 4 (refused) pattern. For the tiny spec, whose embedding is 64 KiB, the piece is the whole embedding.
- **Resume heap test.** The assertion is unchanged: the transient must be under half the embedding. The spec now uses vocab 1024, so the embedding is 256 KiB, four pieces. At 64 KiB the tiny embedding equals one piece, so the test could not tell one piece from a whole second copy. Measured after the change: a 64,892-byte transient against a 262,144-byte embedding.
- **Follow-up that tightens save to one piece of headroom.** A streaming save that reads device bytes piece by piece without building a host tensor. It moves the deferred-fault sync point (`download`), so it is a separate change.

**Call-site rules:**
- `Scratch::<u8>::try_alloc(n_bytes)` followed by `from_scratch(.., DType::F32)` becomes `Scratch::<f32>::try_alloc(n_elements)` followed by `from_scratch(.., shape)`.
- Drop the explicit `* 4` byte factor (for example `pointwise.rs:84`, `backend.rs:619`).
- A reader that decoded `contiguous_bytes()?.as_chunks::<4>()` reads `f32_slice()?` or `u32_slice()?` directly.

**Save path (`checkpoint.rs:312`):** it encodes `f32_slice()?` in `LE_CHUNK` pieces into the existing bounded `le` buffer. It needs no new core API and allocates no whole-tensor transient.

**`compute_run_id` (`trainer.rs:296`):** it hashes `to_ne_bytes()` of each element of `f32_slice()?`. The run id hash is byte-identical to today's.

## Migration order

Every step leaves the whole workspace compiling and the integrated gate green.

1. **ojas-core** (the peer's crate), one change:
   - `HostData`, the typed accessors, generic `from_scratch` and `to_ne_bytes`;
   - `contiguous_bytes`, `from_le_fill` and the private encode helpers deleted, and `from_le_reader` added;
   - checkpoint load (`ojas-model/src/checkpoint.rs`) moved to `from_le_reader`;
   - the call sites in the table above moved in the same change, so no step leaves two storage paths.
   - Already in step 1, as staged on 2026-10-01: in ojas-cpu, `validate::f32_words`/`u32_words` became `f32_values`/`u32_values` (borrowed `&[f32]`/`&[u32]`), `bytes_all_finite` was deleted, and every kernel that read words in place (optim, cross-entropy, embedding, exp, kv, accum, permute) reads typed values. Outputs built in a `Scratch` use `Scratch<f32>`. 9 files, +185 −258.
   - Measured after landing (byte-storage binary against step 1, minimum of 3–5 interleaved rounds, 6 threads, load 23–27). Step 1 alone moves no time: block forward+backward 123.1 against 125.0 ms, cross-entropy, AdamW, clip and decode within noise. The in-place kernels already read the bytes without a copy, so the decode they lose was nearly free. Embedding forward (1.01 to 1.11 ms) and permute (0.130 to 0.143 ms) are about 9% slower. In the `sample` profile the output row copy in `pointwise::embedding_forward` doubled (439 to 809 samples), along with thread start-up and `vm_deallocate`. That looks like fresh output pages being faulted in on each call; it is not yet explained, and step 3 reworks output allocation. Profiles: `target-matmul/emb-pre-ts.sample.txt`, `emb-cur.sample.txt`.
2. **ojas-cpu owned copies:**
   - `validate::f32_in`, `copy_checked` and `f32_input_list` still copy through `to_f32_vec`, because the GEMM, attention and norm paths hand their operands to the pool as owned `Arc<Vec<f32>>`;
   - step 2 lets those paths borrow (scoped tasks, as `pool::scoped` already does), so the copies leave the profile;
   - the heap gate may already have fallen in step 1 for the in-place ops; step 2 is where the GEMM-input copies go.
3. **ojas-cpu outputs:**
   - `alloc_out`, `F32Out` and `scoped::fill` targets become `Scratch<f32>`;
   - kernels write `f32` and hand the vector to `from_scratch` with no encode;
   - this removes the `alloc_out` and `memmove` share.
4. **Cleanup:**
   - delete the byte decode helpers that no longer have callers;
   - before deleting each one, check it with `devmap_dead_symbols` plus `rg -uu`.

The GPU crates change only at their two upload sites, in step 1. Device readback keeps `DeviceBuffer::read_bytes`.

## Test gate

Before step 1 lands:
- **New `ojas-core` unit tests:**
  - typed slices of narrowed and offset views;
  - wrong-dtype accessors return `Dtype` errors;
  - device tensors return `Placement`;
  - `from_scratch` refuses a length or reservation mismatch, and its charge is released;
  - `f32_slice_mut` refuses a shared allocation.
  - `from_le_reader`, carrying over the four `from_le_fill` tests:
    - the buffer the reader sees is zeroed;
    - the tensor is dropped, and its charge released, when `read` fails;
    - an over-budget shape is refused before `read` runs;
    - `U32` works.
  - New `from_le_reader` tests:
    - shapes that are not a multiple of the chunk size;
    - a `read` that fails on a later chunk (a short source) drops the tensor and releases both charges;
    - the chunk's charge is released on success and on error;
    - a checkpoint save and load round trip keeps every bit (the existing `ojas-model` checkpoint tests).
  - `to_host` charges the decode transient: a readback that fits the budget only without the transient is refused.
- **The existing tensor unit tests**, unchanged.

After each step:
- `target-matmul/gate.sh`: the integrated CPU gate, 78 test binaries plus clippy with `-D warnings`.
- `redteam_ops_heap.rs` (64 KiB slack). Peak heap must not rise, and should fall in steps 2 and 3.
- `target-matmul/parity.sh`: every op and the composed block against torch.
- `ojas-infer` three-way parity under both numerics tiers.

After steps 2 and 3, `target-matmul/block_ab.sh`: the block, forward and backward, interleaved against the frozen binary and torch. The plan only counts as successful if the storage bucket shrinks in a new `sample` profile and the block time falls.

## Open questions

1. **Step 1 cannot be split.** Typed storage cannot lend `&[u8]` without `unsafe`, so `contiguous_bytes` cannot survive alongside it, and there is no green state in between. One change touches `ojas-core`, `ojas-cpu`, `ojas-infer`, `ojas-model`, `ojas-metal`, `ojas-wgpu` and about 15 test sites, in a crate this session does not own. It needs a snapshot first, and either the peer making the change or a quiet window agreed with the peer.
2. **Half precision.** Should Bf16 and F16 get host kernels later? Nothing builds a half-precision host tensor today except `Tensor::zeros`, so `HostData::Half` only holds data. If host half kernels come, they use `Vec<u16>` with explicit conversion, the same as now.
