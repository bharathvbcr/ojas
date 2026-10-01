# ojas-io

`ojas-io` provides serialization and deserialization routines for **safetensors** model files and binary **Checkpoint v1** containers.

---

## Formats Handled

```mermaid
flowchart TD
    subgraph InputFiles["File Formats"]
        ST["Safetensors File (.safetensors)\n- F32, I64, U16 dtypes\n- Strict JSON header\n- Direct buffer memory mapping"]
        CP["Checkpoint v1 File (.ckpt)\n- 12-byte header (OJAS0001)\n- Parameter & optimizer states\n- Training cursor & RNG state"]
    end

    subgraph Parser["ojas-io Parser Engine"]
        JSON["json.rs\n- Max depth = 64\n- Rejects duplicate keys\n- Validates strict types"]
        STParser["safetensors.rs\n- Detects shape overflow\n- Validates non-overlapping ranges\n- Enforces header length caps"]
        CPParser["checkpoint.rs\n- Little-endian decoding\n- Validates NamedBlob offsets\n- Bounded memory allocation"]
    end

    subgraph CoreOutputs["Output Structures"]
        Tensors["ojas-core::Tensor"]
        Checkpoint["ojas-core::CheckpointV1"]
    end

    ST --> STParser
    STParser --> JSON
    CP --> CPParser
    STParser --> Tensors
    CPParser --> Checkpoint
```

---

## Safety & Hardening Invariants

1. **Header Length Limits:** Safetensors header sizes are capped to prevent denial-of-service via corrupted header allocations.
2. **Duplicate Key Rejection:** JSON and safetensors readers reject files containing duplicate parameter keys.
3. **No Overlapping Ranges:** Tensor byte spans inside safetensors files must not overlap. Overlapping data ranges return an error.
4. **Shape Overflow Protection:** If the product of dimension sizes exceeds integer limits, decoding is aborted before allocating buffers.
5. **JSON Parser Recursion Cap:** JSON parsing is bounded to a depth of 64 to prevent stack overflow from untrusted nested payloads.
