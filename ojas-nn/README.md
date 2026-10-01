# ojas-nn

`ojas-nn` is the model assembly crate for full GPT architectures.

---

## Planned Architecture

```mermaid
flowchart TD
    Config["ModelConfig (12 layers, d_model=768, n_head=12, head_dim=64)"] --> GPT["GPT Module"]
    GPT --> Embed["Tied Token Embedding"]
    GPT --> Blocks["12x TransformerBlock"]
    Blocks --> RMSNorm["Pre-RMSNorm (eps=1e-6)"]
    Blocks --> CausalAttn["Causal Attention + Per-Head Gate + Value Residual"]
    Blocks --> SwiGLU["SwiGLU MLP"]
    GPT --> Head["Tied LM Output Head"]
```

*Note: Reference block operators are currently executed directly via [`ojas-cpu`](file:///Users/bharath/Code/research/ojas/ojas-cpu) and [`ojas-metal`](file:///Users/bharath/Code/research/ojas/ojas-metal).*
