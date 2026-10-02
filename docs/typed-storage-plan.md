# Typed Host Storage: Plan

Status: proposed, not started (2026-10-01). `ojas-core` belongs to the peer session, so step 1 needs its agreement and a fresh snapshot first.

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

Device readback (`to_host`) decodes the `Vec<u8>` from `DeviceBuffer::read_bytes` into `HostData`. For a moment both buffers are alive, twice the tensor size, and only the result is charged; today the bytes move into storage with no second buffer. There are two ways to handle this, and the trait is the peer's call: (a) charge the transient and accept the extra pass, or (b) add `read_f32_into(&self, offset, out: &mut [f32])` to `DeviceBuffer` so each backend decodes straight into the typed vector.

## Invariants to keep

Each of these has a test today and must still pass unchanged:

- **Charge before allocating.** The reservation is the last field, so the vector is freed before the charge is released (`Scratch`, `Storage`).
- **The budget is released when the last view drops** (`budget_is_released_when_last_view_drops`).
- **Writes need sole ownership** (`write_requires_unique_owner_and_matching_len`, `ensure_writable_agrees_with_write_and_never_writes`).
- **Placement is reported before layout or ownership** (`device_tensor_refuses_every_host_accessor`, `strided_device_view_reports_placement_not_layout`).
- **Window bounds and overflow** (`randomized_views_match_reference_bounds`, `overflowing_shapes_are_errors_not_panics`).
- **Host-to-host `to_host` is a shared clone, not a readback** (`host_to_host_is_a_shared_clone_and_not_a_readback`).
- **The Exact goldens keep their bits** (`ojas-cpu/tests/exact_golden.rs`). Typed storage changes where the bytes live, not the arithmetic.

## Migration order

Every step leaves the whole workspace compiling and the integrated gate green.

1. **ojas-core** (the peer's crate), one change:
   - `HostData`, the typed accessors, generic `from_scratch` and `to_ne_bytes`;
   - `contiguous_bytes` and `write_ne_f32` deleted;
   - the call sites in the table above moved in the same change, so no step leaves two storage paths.
2. **ojas-cpu inputs:**
   - `validate::f32_in` and `f32_input_list` return borrowed `&[f32]` instead of decoded copies;
   - `f32_words` and `bytes_all_finite` scan `&[f32]`;
   - this step removes the `to_f32_vec` copies from the profile.
3. **ojas-cpu outputs:**
   - `alloc_out`, `F32Out` and `scoped::fill` targets become `Scratch<f32>`;
   - kernels write `f32` and hand the vector to `from_scratch` with no encode;
   - this removes the `alloc_out` and `memmove` share.
4. **Cleanup:**
   - delete the byte decode helpers that no longer have callers;
   - before deleting each one, check it with `devmap_dead_symbols` plus `rg -uu`.

The GPU crates change only at their two upload sites, in step 1.

## Test gate

Before step 1 lands:
- **New `ojas-core` unit tests:**
  - typed slices of narrowed and offset views;
  - wrong-dtype accessors return `Dtype` errors;
  - device tensors return `Placement`;
  - `from_scratch` refuses a length or reservation mismatch, and its charge is released;
  - `f32_slice_mut` refuses a shared allocation.
- **The existing tensor unit tests**, unchanged.

After each step:
- `target-matmul/gate.sh`: the integrated CPU gate, 78 test binaries plus clippy with `-D warnings`.
- `redteam_ops_heap.rs` (64 KiB slack). Peak heap must not rise, and should fall in steps 2 and 3.
- `target-matmul/parity.sh`: every op and the composed block against torch.
- `ojas-infer` three-way parity under both numerics tiers.

After steps 2 and 3, `target-matmul/block_ab.sh`: the block, forward and backward, interleaved against the frozen binary and torch. The plan only counts as successful if the storage bucket shrinks in a new `sample` profile and the block time falls.

## Open questions

1. **Step 1 cannot be split.** Typed storage cannot lend `&[u8]` without `unsafe`, so `contiguous_bytes` cannot survive alongside it, and there is no green state in between. One change touches `ojas-core`, `ojas-cpu`, `ojas-infer`, `ojas-metal`, `ojas-wgpu` and about 15 test sites, in a crate this session does not own. It needs a snapshot first, and either the peer making the change or a quiet window agreed with the peer.
2. **Readback**: (a) or (b) above.
3. **Half precision.** Should Bf16 and F16 get host kernels later? Nothing builds a half-precision host tensor today except `Tensor::zeros`, so `HostData::Half` only holds data. If host half kernels come, they use `Vec<u16>` with explicit conversion, the same as now.
