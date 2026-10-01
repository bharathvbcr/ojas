# ojas-metal

`ojas-metal` provides an Apple Silicon GPU training step running natively on macOS with **Metal 4**.

It bridges high-throughput GEMM and normalization kernels from `tessl` with a custom MSL shader for nanolab's per-head output gate.

---

## Metal Execution Pipeline

```mermaid
flowchart TD
    subgraph HostInit["Host Setup (macOS / Metal 4)"]
        Open["gpu::Session::open() (Requires Apple GPU)"]
        Shape["TinyShape: batch <= 2, seq <= 16, d_model=64, n_head=1, head_dim=64, vocab <= 128"]
    end

    subgraph GPUForward["GPU Forward Pass"]
        PreNorm["tessl RMSNorm (f32, eps=1e-6)"]
        QKV["tessl GEMM (Q, K, V Projections)"]
        Attn["tessl flash_attn_rows (Head Dim = 64)"]
        Gate["Custom MSL: per_head_gate.metal\n(Sigmoid gate broadcast + multiply)"]
        SwiGLU["tessl SwiGLU (SiLU + Mul GEMM)"]
        Head["LM Head Projection"]
    end

    subgraph LossBackward["Loss & Backward Pass"]
        CE["tessl Chunked Cross-Entropy\n(Offset-safe dh with 16-element canary padding)"]
        BwdGate["Custom MSL Gate Backward"]
        BwdGEMM["tessl GEMM Backward Gradients"]
    end

    subgraph Optimizer["Optimizer Step"]
        AdamW["tessl fused AdamW on LM Head"]
    end

    HostInit --> GPUForward
    GPUForward --> LossBackward
    LossBackward --> Optimizer
```

---

## Shaders & Kernels

### 1. Per-Head Sigmoid Gate (`kernels/per_head_gate.metal`)
Nanolab default GPT applies a linear gate with bias followed by sigmoid to modulate attention head outputs. `ojas-metal` compiles native MSL shaders:
* `per_head_gate_forward`: Computes $g = \sigma(x W_g + b_g)$ and outputs $y = \text{attn} \odot g$.
* `per_head_gate_backward`: Computes analytic gradients with respect to attention outputs and gate projections.

### 2. Cross-Entropy `dh` Canary Offset Defense
Upstream implementations often assume gradient slices start at buffer offset 0. `ojas-metal` prepends a canary prefix (`DH_PAD = 16`) to buffer allocations and verifies the prefix remains unaltered after cross-entropy writes (`ce_nonzero_dh_offset_keeps_prefix`).

### 3. Strict Head Dimension Policy
Metal flash-attention kernels are structured for $d_{\text{head}} = 64$. Any shape with $d_{\text{head}} > 64$ triggers `OjasError::UnsupportedHeadDim`, preventing silent memory corruption or silent truncation.
