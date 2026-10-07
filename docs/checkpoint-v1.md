# Checkpoint v1 Specification

Little-endian binary format for persisting training state, model parameters, and optimizer moments in **ojas**.

The in-memory types are [`CheckpointV1`](file:///Users/bharath/Code/research/ojas/ojas-core/src/checkpoint.rs) and its member fields in `ojas-core`. Serialization and deserialization are implemented in [`ojas-io`](file:///Users/bharath/Code/research/ojas/ojas-io/src/checkpoint.rs) (`write_checkpoint` and `read_checkpoint`).

---

## Binary Layout Overview

```mermaid
flowchart TD
    subgraph FileFrame["Checkpoint v1 Binary File Layout (Little-Endian)"]
        direction TB
        Prefix["1. Prefix (12 Bytes)\n- Magic: 'OJAS0001' (8 bytes ASCII)\n- Version: 1 (u32, 4 bytes)"]
        
        subgraph FixedBody["2. Fixed Header & Hyperparameters"]
            Config["Config Blob\n- len: u64\n- bytes: [len]u8 (JSON hyperparameters)"]
            TokHash["Tokenizer Hash\n- exactly 32 bytes (raw BLAKE3 / SHA-256)"]
            GitSHA["Git Commit SHA\n- exactly 20 bytes (raw SHA-1)"]
            Step["Training Step\n- u64 (checked math)"]
            Cursor["Data Cursor\n- shard: u64\n- token_index: u64"]
            RNG["RNG State Blob\n- len: u64\n- bytes: [len]u8"]
        end

        subgraph ArrayBody["3. Named Tensor Record Sections"]
            Weights["Weights Section (length-prefixed array of NamedBlob)"]
            Muon["Muon Momentum Section (same format)"]
            AdamW1["AdamW First Moment Section (same format)"]
            AdamW2["AdamW Second Moment Section (same format)"]
        end

        Prefix --> Config
        Config --> TokHash
        TokHash --> GitSHA
        GitSHA --> Step
        Step --> Cursor
        Cursor --> RNG
        RNG --> Weights
        Weights --> Muon
        Muon --> AdamW1
        AdamW1 --> AdamW2
    end
```

> [!IMPORTANT]
> **Magic String & Version Verification:** Checkpoint v1 files begin strictly with the 8-byte ASCII string `OJAS0001` followed by little-endian `u32` value `1`. Any file with mismatched magic bytes returns `OjasError::InvalidCheckpoint` immediately—**never attempting guessing or silent migration**.

---

## NamedBlob Record Structure

Each parameter or optimizer tensor is serialized as a self-contained, offset-aware record:

```mermaid
flowchart LR
    subgraph Record["NamedBlob Record Layout"]
        direction TB
        NL["name_length: u64"] --> Name["name: UTF-8 bytes"]
        Name --> DT["dtype_tag: u32\n(0=F32, 1=Bf16, 2=F16, 3=U32)"]
        DT --> RK["rank: u32"]
        RK --> Shape["shape: rank x u64"]
        Shape --> BO["byte_offset: u64"]
        BO --> PL["payload_length: u64"]
        PL --> Payload["payload: [payload_length]u8"]
    end
```

### Record Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `name_length` | `u64` | Byte length of the UTF-8 encoded parameter name. |
| `name` | `[name_length]u8` | Parameter identifier (e.g., `"layers.0.attn.q_proj.weight"`). |
| `dtype_tag` | `u32` | Data type tag: `0` = F32, `1` = Bf16, `2` = F16, `3` = U32. |
| `rank` | `u32` | Number of dimensions. |
| `shape` | `[rank]u64` | Array of dimension extents. |
| `byte_offset` | `u64` | Byte offset of element 0 inside `payload`. |
| `payload_length` | `u64` | Total byte length of the stored payload buffer. |
| `payload` | `[payload_length]u8` | Raw little-endian tensor data. |

> [!WARNING]
> **Offset Preservation:** Readers that ignore `byte_offset` will corrupt tensor alignment. `ojas-io` strictly verifies that `byte_offset + payload_size <= payload_length`.

---

## Prefix Specification

| Byte Offset | Width | Type | Canonical Value |
| :--- | :--- | :--- | :--- |
| `0..8` | 8 bytes | ASCII | `OJAS0001` (`CHECKPOINT_MAGIC`) |
| `8..12` | 4 bytes | `u32` | `1` (`CHECKPOINT_VERSION`) |

---

## Body Ordering

1. **`config`**: Length-prefixed arbitrary bytes storing training hyperparameters and model specification.
2. **`tokenizer_hash`**: Fixed 32 raw bytes (BLAKE3 or SHA-256 hash of the tokenizer vocabulary).
3. **`git_sha`**: Fixed 20 raw bytes storing the repository commit SHA at checkpoint creation.
4. **`step`**: `u64` little-endian integer. Refuses values that could cause overflow (`u64::MAX`).
5. **`data_cursor.shard`**: `u64` epoch of the batch sampler (`ojas_data::BatchSampler`).
6. **`data_cursor.token_index`**: `u64` ordinal of the next window within that epoch (not a token offset). `ojas_model::Trainer::cursor` documents the same meaning.
7. **`rng_state`**: Length-prefixed state buffer for deterministic resumption. The trainer writes the sampler's layout below; the container itself treats it as opaque bytes.
8. **`weights`**: `u64` section byte length, `u64` record count, followed by serialized `NamedBlob` records.
9. **`optimizer.muon_momentum`**: Same framing; stores momentum buffers for 2D matrix parameters.
10. **`optimizer.adamw_first_moment`**: Same framing; stores first moments for 1D parameters.
11. **`optimizer.adamw_second_moment`**: Same framing; stores second moments for 1D parameters.

---

## `rng_state` Layout v1

`ojas_model::Trainer` stores [`ojas_data::SamplerRngState`](file:///Users/bharath/Code/research/ojas/ojas-data/src/sampler.rs) here. The batch sampler draws nothing as it runs. Its order is a keyed permutation of `(seed, epoch, ordinal)`, and `data_cursor` already holds the epoch and ordinal. The rest of its random state is which permutation runs and which key it uses:

| Byte Offset | Width | Type | Field |
| :--- | :--- | :--- | :--- |
| `0..4` | 4 bytes | `u32` | version, `1` (`RNG_STATE_VERSION`) |
| `4..8` | 4 bytes | `u32` | generator, `1` = 4-round Feistel over SplitMix64 (`RNG_GENERATOR_FEISTEL4_SPLITMIX64`) |
| `8..16` | 8 bytes | `u64` | seed, the train config's `data_seed` |

Resume decodes it exactly. A length other than 16, an unknown version or generator, or a seed different from the caller's `data_seed` is refused. An empty `rng_state` is refused too: files saved before this layout wrote one, and they are not migrated. `permutation_is_pinned` (`ojas-data/tests/sampler.rs`) pins window starts against an independent Python oracle (`target-robust/sampler_oracle.py`). A change that moves any of them needs a new generator tag. `g9_forty_steps_equal_twenty_saved_dropped_resumed_and_twenty_more` (`ojas-model/tests/checkpoint.rs`) proves bit for bit on CPU that 20 steps, a save, a resume and 20 more equal 40 uninterrupted steps.

---

## Free Space Before Save

`Trainer::save` computes the exact length of each file it will write: `model.safetensors` and `optim.safetensors` (`ojas_io::safetensors_file_len`), and `state.ojck` (`ojas_io::checkpoint_file_len`). It adds 1 MiB per file for block rounding. That total must fit the space an unprivileged writer can still allocate on the filesystem that stages the directory: `ojas_device::available_disk_bytes` of its parent (`statfs` on macOS, `statvfs` on Linux, `f_bavail`). The previous checkpoint stays until the swap, so all of it is new space. When the total does not fit, or the probe fails, the save is refused before a staging directory exists. Other platforms have no probe, so saves there are refused.

---

## Device Tensors Are Not Streamed in Pieces (Decision)

A device tensor is read back with one whole-tensor `Backend::download`, written in 64 KiB little-endian pieces, and dropped before the next tensor is read. It is not streamed off the device in pieces. The reasons:

- `Backend` has no ranged readback. Adding one changes the shared trait in `ojas-core/src/backend.rs` and every backend implementation (CPU, Metal, wgpu, CUDA, HIP).
- The host cost is already bounded: one tensor plus one 64 KiB piece at a time. `write_tensors` refuses with `CapacityExceeded` before the file is created when the largest device tensor cannot be charged to the backend's budget. `save_fits_in_one_tensor_plus_one_readback_piece_of_host_headroom` and `save_reads_each_tensor_once_and_holds_one_at_a_time` (`ojas-model/tests/checkpoint.rs`) test that bound.

Revisit this when one parameter tensor no longer fits in host memory beside the model. The embedding of a very large vocabulary is the likely first case.

---

## Verification & Safety Guarantees

> [!CAUTION]
> 1. **Truncation Defense:** Checkpoint files truncated before an expected field boundary return `Err(OjasError::TruncatedCheckpoint)` rather than filling missing bytes with zeros.
> 2. **Integer Overflow Protection:** Dimension shapes whose product overflows `u64` or `usize` are caught during header decoding and rejected before memory allocation.
