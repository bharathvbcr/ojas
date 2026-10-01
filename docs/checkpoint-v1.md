# Checkpoint v1 Specification

Little-endian binary format for persisting training state, model parameters, and optimizer moments.

The in-memory types are [`CheckpointV1`](file:///Users/bharath/Code/research/ojas/ojas-core/src/checkpoint.rs) and its member fields in `ojas-core`. Serialization and deserialization are implemented in [`ojas-io`](file:///Users/bharath/Code/research/ojas/ojas-io/src/checkpoint.rs) (`write_checkpoint` and `read_checkpoint`).

---

## Binary Layout Overview

```mermaid
flowchart TD
    subgraph FileFrame["Checkpoint v1 File Structure (Little-Endian)"]
        direction TB
        Prefix["1. Prefix (12 Bytes)\n- Magic: 'OJAS0001' (8 bytes)\n- Version: 1 (u32, 4 bytes)"]
        
        subgraph FixedBody["2. Fixed Header & State Fields"]
            Config["Config Blob\n- len: u64\n- bytes: [len]u8"]
            TokHash["Tokenizer Hash\n- exactly 32 bytes (raw)"]
            GitSHA["Git Commit SHA\n- exactly 20 bytes (raw SHA-1)"]
            Step["Training Step\n- u64"]
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

---

## NamedBlob Record Structure

Each parameter or optimizer tensor is serialized as a self-contained record:

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
| `byte_offset` | `u64` | Byte offset of element 0 inside `payload`. Readers must strictly account for this offset. |
| `payload_length` | `u64` | Total byte length of the stored payload buffer. |
| `payload` | `[payload_length]u8` | Raw little-endian tensor data. |

---

## Prefix Specification

| Byte Offset | Width | Type | Canonical Value |
| :--- | :--- | :--- | :--- |
| `0..8` | 8 bytes | ASCII | `OJAS0001` (`CHECKPOINT_MAGIC`) |
| `8..12` | 4 bytes | `u32` | `1` (`CHECKPOINT_VERSION`) |

Any file presenting an alternative magic string or version number is rejected immediately with `OjasError::InvalidCheckpoint`. Version numbers are never silently clamped or migrated without explicit caller consent.

---

## Body Ordering

1. **`config`**: Length-prefixed arbitrary bytes storing the training hyperparameters and model specification.
2. **`tokenizer_hash`**: Fixed 32 raw bytes (BLAKE3 or SHA-256 hash of the tokenizer vocabulary).
3. **`git_sha`**: Fixed 20 raw bytes storing the repository commit SHA at checkpoint creation.
4. **`step`**: `u64` little-endian integer. Refuses values that could cause overflow (`u64::MAX`).
5. **`data_cursor.shard`**: `u64` dataset shard index.
6. **`data_cursor.token_index`**: `u64` token offset within the current shard.
7. **`rng_state`**: Length-prefixed state buffer for deterministic resumption.
8. **`weights`**: `u64` section byte length, `u64` record count, followed by serialized `NamedBlob` records.
9. **`optimizer.muon_momentum`**: Same framing; stores momentum buffers for 2D matrix parameters.
10. **`optimizer.adamw_first_moment`**: Same framing; stores first moments for 1D parameters.
11. **`optimizer.adamw_second_moment`**: Same framing; stores second moments for 1D parameters.

---

## Verification & Safety Guarantees

* **Offset Integrity:** A reader that ignores `byte_offset` is structurally broken. `ojas-io` validates that `byte_offset + payload_size <= payload_length`.
* **Truncation Defense:** Files truncated before any expected field boundary return `OjasError::TruncatedCheckpoint` rather than filling trailing memory with zero bytes.
* **Shape Overflow Prevention:** Shapes whose product of dimensions overflows `u64` or `usize` are caught during header decoding and rejected.
