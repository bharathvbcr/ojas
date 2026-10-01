# ojas-io

`ojas-io` provides serialization and deserialization routines for **safetensors** model files and binary **Checkpoint v1** containers, hardened against malicious payloads, corrupted headers, and integer overflow attacks.

---

## Formats & Parsing Engine

```mermaid
flowchart TD
    subgraph StorageFormats["Persistent Storage Formats"]
        ST[".safetensors File\n- Strict JSON header\n- Positioned reads, no mmap\n- F32, BF16, F16, I64, U16 dtypes\n- BF16/F16: exact decode, RNE encode"]
        CKPT[".ckpt File (Checkpoint v1)\n- 12-byte prefix (OJAS0001)\n- Weights & optimizer moments\n- Training cursor & RNG state"]
    end

    subgraph SecurityFilters["Hardened Validation Pipeline"]
        JSONParser["json.rs\n- Max recursion depth = 64\n- Rejects duplicate keys"]
        STValidator["safetensors.rs\n- Enforces header length cap (100 MB)\n- Rejects overlapping byte spans\n- Validates shape overflow"]
        CKPTValidator["checkpoint.rs\n- Little-endian validation\n- Validates NamedBlob byte_offsets\n- Rejects truncated frames"]
    end

    subgraph EngineOutput["Core Data Structures"]
        Tensors["ojas-core::Tensor"]
        Checkpoint["ojas-core::CheckpointV1"]
    end

    ST --> STValidator
    STValidator --> JSONParser
    JSONParser --> Tensors
    CKPT --> CKPTValidator
    CKPTValidator --> Checkpoint
```

---

## Binary Checkpoint v1 Record Structure

```mermaid
flowchart LR
    subgraph Frame["Checkpoint v1 Binary Envelope"]
        Prefix["Magic: 'OJAS0001' (8B) + Version: 1 (4B)"]
        Config["Config Blob"]
        Hashes["Tokenizer Hash (32B) + Git SHA (20B)"]
        State["Step (u64) + Cursor (Shard + Tok) + RNG"]
        Sections["Weights, Muon & AdamW NamedBlob Records"]
    end

    Prefix --> Config --> Hashes --> State --> Sections
```

---

## Safety Invariants & Hardening Guarantees

> [!IMPORTANT]
> 1. **Header Length Enforcement:** Safetensors header sizes are capped at 100 MB (`MAX_HEADER_SIZE`) to prevent memory exhaustion from crafted denial-of-service headers.
> 2. **Duplicate Key Rejection:** Any JSON or safetensors header containing duplicate parameter identifiers returns `OjasError::InvalidSafetensors` immediately.
> 3. **Non-Overlapping Range Validation:** Tensor byte spans inside safetensors files must not overlap. Overlapping spans are rejected to eliminate memory alias vulnerabilities.
> 4. **Shape Overflow Detection:** If the product of dimension sizes overflows `usize` or `u64`, decoding halts before memory allocation.
> 5. **JSON Recursion Boundary:** JSON parsing is hard-limited to a maximum recursion depth of 64, preventing stack overflow from deeply nested payloads.
> 6. **Truncation Defense:** Checkpoint files truncated before an expected field boundary return `OjasError::TruncatedCheckpoint`—**never zero-padding missing data**.
