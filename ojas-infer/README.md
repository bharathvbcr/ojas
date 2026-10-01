# ojas-infer

`ojas-infer` implements an autoregressive inference engine with KV caching, greedy token decoding, and strict logit validation.

---

## Inference Architecture

```mermaid
flowchart TD
    subgraph GenerationFlow["Autoregressive Generation Loop"]
        Prompt["Input Prompt Tokens [u32]"] --> Model["CpuGpt Forward Pass"]
        Cache["KV Cache (Pre-allocated per layer)"] <--> Model
        
        Model --> Logits["Logits Vector [V]"]
        Logits --> Validator{"Logit Validation\n(Are all values finite?)"}
        
        Validator -->|Non-Finite Detected| Err["Return OjasError::NonFinite\n(Never default to token 0!)"]
        Validator -->|Finite| Argmax["argmax_token() / greedy_decode()"]
        
        Argmax --> NextToken["Next Token ID (u32)"]
        NextToken --> CheckDone{"Terminal Token or Length Cap?"}
        
        CheckDone -->|No| Append["Append Token to Prompt"]
        Append --> Model
        CheckDone -->|Yes| Output["Completed Generation"]
    end
```

---

## Key Invariants

1. **Finite Logit Requirement:** Upstream engines have suffered from defects where an all-NaN logit vector silently falls back to token 0. In `ojas-infer`, any non-finite logit produces `OjasError::NonFinite`, failing immediately rather than poisoning inference.
2. **KV Cache Bounds:** Cache slots advance monotonically and refuse writes beyond allocated sequence limits.
3. **Tied Embedding Decoding:** Matrix multiplication utilizes tied embedding weights directly without duplicate allocations.
