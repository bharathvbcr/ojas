# ojas-data

`ojas-data` provides token streaming from pre-tokenized binary files, counter-based deterministic RNG, and BPE tokenizer fixtures.

---

## Data Pipeline

```mermaid
flowchart TD
    subgraph DataSources["Dataset Ingestion"]
        FineWeb["FineWeb Binary Files (.bin)\n- 256-byte header with magic\n- u16 token stream"]
        Custom["Headerless Windows"]
    end

    subgraph Loaders["ojas-data Pipeline"]
        Tokens["tokens.rs\n- Validates magic count & offsets\n- Rejects odd-length buffers\n- Windowed batch extraction"]
        RNG["rng.rs\n- Counter-based deterministic RNG\n- Zero global state\n- Bit-identical across threads"]
        BPE["bpe.rs\n- Bijective GPT-2 byte alphabet mapping\n- Fixture encoding/decoding"]
    end

    FineWeb --> Tokens
    Custom --> Tokens
    Tokens --> Batches["Batches: [B, T] u16 / u32 tokens"]
```

---

## Tokenizer Invariant: Byte Identity

`TIKTOKEN_GPT2_BYTE_IDENTITY` is `"verified-20-strings"`. Those twenty strings match tiktoken 0.12.0 `gpt2` `encode_ordinary`. That is not a full-corpus check.
