# ojas-optim

`ojas-optim` is the standalone optimizer crate for AdamW and Muon NS5.

---

## Dual Optimizer Partitioning

```mermaid
flowchart LR
    Params["Parameters"] --> Split{"Rank >= 2?"}
    Split -->|Yes (Matrices)| Muon["Muon NS5 Optimizer\n(bf16 Newton-Schulz iterate)"]
    Split -->|No (Embeddings & Biases)| AdamW["AdamW Optimizer\n(PyTorch single-tensor order)"]
```

*Note: The reference implementations currently live in [`ojas-cpu::backend::optim`](file:///Users/bharath/Code/research/ojas/ojas-cpu) and are verified by unit tests in `ojas-cpu`.*
