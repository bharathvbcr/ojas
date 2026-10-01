# ojas

**ojas** (vital energy, the essence training spends) is a deterministic, high-performance **Rust deep learning engine and framework** designed as a safe, modern alternative to PyTorch and TensorFlow for systems research, production training, and edge inference.

ojas pairs a compile-time safe Rust engine with an in-process Go API via `gusset`, zero-copy tensor views, reverse-mode automatic differentiation, strict memory budgeting, and native hardware acceleration across Apple Silicon Metal 4, WebGPU (WGSL), NVIDIA CUDA, and AMD ROCm/HIP.

```
Status:          Active Core & Multi-Backend Engine
Verification:    189 Workspace Tests + 11 Go Tests Passed
Host Target:     Apple M5 Pro (Metal 4), macOS 27 (Darwin 27.0.0 arm64)
License:         MIT OR Apache-2.0
```

---

## Why ojas? (A PyTorch & TensorFlow Alternative)

Mainstream frameworks like PyTorch and TensorFlow carry decades of legacy baggage: hidden global state, silent CPU fallbacks that mask hardware misconfigurations, nondeterministic multithreaded reductions, memory fragmentation, and cumbersome multi-language bindings that rely on heavyweight Python runtimes.

ojas is engineered from first principles with radical architectural guarantees:

```mermaid
flowchart TD
    subgraph PyTorchTF["Legacy Frameworks (PyTorch / TensorFlow)"]
        L1["Implicit Global State & Seed Drift"]
        L2["Silent CPU Fallback on Missing GPU\n(Hides deployment bottlenecks)"]
        L3["Dynamic Memory Clamping & OOM Panics"]
        L4["Heavyweight Python-Centric Runtime Stack"]
        L5["Integer Overflow Hazards in Long Training Steps"]
        L6["Complex C++ Codebase with Opaque Abstractions"]
    end

    subgraph ojasPrinciples["ojas Architectural Guarantees"]
        O1["Bit-Identical Determinism across Single-Threaded Runs"]
        O2["Zero Silent Fallbacks: Explicit Hardware Refusal\n(Fail fast, fail loud)"]
        O3["Explicit Budget Governance: try_reserve() Never Clamps"]
        O4["Pure Rust Core with Direct In-Process Go & C-ABI Bindings"]
        O5["Checked Arithmetic Everywhere (overflow-checks = true in Release)"]
        O6["Lean, Modular Crates with #![forbid(unsafe_code)] by Default"]
    end

    L1 -.->|Solved By| O1
    L2 -.->|Solved By| O2
    L3 -.->|Solved By| O3
    L4 -.->|Solved By| O4
    L5 -.->|Solved By| O5
    L6 -.->|Solved By| O6
```

---

## Framework Architecture

ojas is structured as a modular stack of focused, single-responsibility crates:

```mermaid
flowchart TD
    subgraph Applications["User Applications & Service APIs"]
        GoApp["Go Services & Applications\n(github.com/bharathvbcr/ojas/go)"]
        RustApp["Rust Binaries, Services & CLIs"]
        ForeignApp["C / C++ / Embedded Host Applications"]
    end

    subgraph Interop["In-Process Interop & Foreign Function Interface"]
        Gusset["ojas-gusset-engine\n(libgusset.a Worker Pool & Ticket Queue)"]
        CAPI["ojas-capi\n(C-ABI Engine, Session Store, Panic Isolation Boundary)"]
    end

    subgraph HighLevel["Model, Autograd & Inference Layer"]
        Infer["ojas-infer\n(Autoregressive Decoding, KV Cache, Logit Validation)"]
        NN["ojas-nn\n(Neural Network Modules & Parameter Containers)"]
        Autograd["ojas-autograd\n(Dynamic Reverse-Mode Tape & f64 Gradcheck)"]
        Optim["ojas-optim\n(Dual Optimizers: AdamW & Muon NS5)"]
    end

    subgraph CoreEngine["Tensor Substrate & System Types"]
        Core["ojas-core\n(Tensor, Strides, Offset Slicing, Budget, DType, Errors)"]
        Device["ojas-device\n(Device Abstraction, Hardware Probing, Explicit Routing)"]
        IO["ojas-io\n(Strict Safetensors Parser, Binary Checkpoint v1)"]
        Data["ojas-data\n(Dataset Streaming, Token Binary Formats, Deterministic RNG)"]
        Oracle["ojas-oracle\n(IEEE-754 f64 Numerical Reference Fixtures)"]
    end

    subgraph HardwareBackends["Pluggable Hardware Compute Backends"]
        CPU["ojas-cpu\n(Bit-Identical Single-Threaded Reference Kernels)"]
        Metal["ojas-metal\n(Apple Silicon Metal 4 Acceleration via tessl)"]
        WGPU["ojas-wgpu\n(Cross-Platform WGSL via WebGPU)"]
        CUDA["ojas-cuda\n(NVIDIA PTX Kernel Launcher via cudarc)"]
        HIP["ojas-hip\n(AMD ROCm/HIP Kernel Launcher)"]
    end

    GoApp --> Gusset
    Gusset --> CAPI
    ForeignApp --> CAPI
    RustApp --> HighLevel
    CAPI --> HighLevel

    HighLevel --> CoreEngine
    HighLevel --> HardwareBackends
    HardwareBackends --> CoreEngine
    CoreEngine --> HardwareBackends
```

---

## Core Capabilities & Subsystems

### 1. General-Purpose Tensor Engine (`ojas-core`)
* **Zero-Copy Views:** Tensor representations track an explicit `byte_offset`. Sub-views created via `.narrow()` adjust byte offsets without memory re-allocation or buffer copying.
* **Strict Memory Budgets:** Memory consumption is governed by an explicit `Budget`. When an allocation exceeds limits, `try_reserve()` returns `OjasError::CapacityExceeded` immediately rather than swapping or resizing.
* **Type System:** First-class support for `F32`, `Bf16`, `F16`, and `U32` types with explicit promotion and precision rules.

### 2. Reverse-Mode Automatic Differentiation (`ojas-autograd`)
* **Tape-Based Computation Graph:** Dynamic tape recording variable nodes (`Var`) and operational adjoints for forward and reverse sweeps.
* **Numerical Gradcheck Oracle:** Analytical f32 gradients are verified against double-precision (`f64`) central finite differences, ensuring mathematical soundness down to machine tolerances.

```mermaid
flowchart LR
    subgraph ForwardTape["Forward Execution"]
        x["Var x"] --> op1["Op: Linear(x, W)"]
        W["Var W"] --> op1
        op1 --> h["Var h"]
        h --> op2["Op: Activation(h)"]
        op2 --> loss["Var Loss"]
    end

    subgraph BackwardTape["Reverse-Mode Adjoint Execution"]
        dLoss["Seed dLoss = 1.0"] --> adj2["Adjoint: dActivation"]
        adj2 --> adj1["Adjoint: dLinear"]
        adj1 --> dW["Accumulate dW"]
        adj1 --> dx["Accumulate dx"]
    end

    ForwardTape ==> BackwardTape
```

### 3. Native & Portable Hardware Backends
* **Single-Threaded CPU (`ojas-cpu`):** Guaranteed bit-identical mathematical reference implementing matrix multiplication, normalizations, rotary positional embeddings, causal attention, gating, and optimizers.
* **Apple Silicon Metal 4 (`ojas-metal`):** High-throughput execution using Apple's latest Metal API via `tessl` and custom Metal Shading Language (`.metal`) kernels.
* **Portable WebGPU (`ojas-wgpu`):** Cross-platform shaders written in WGSL executing across Vulkan, Metal, and DX12.
* **NVIDIA CUDA & AMD ROCm (`ojas-cuda`, `ojas-hip`):** Native driver and runtime launch pipelines with compile-time feature gates (`--features cuda`, `--features hip`) and explicit refusal when hardware is absent.

```mermaid
flowchart TD
    Dispatch["Tensor Compute Dispatch"] --> BackendCheck{"Selected Device"}
    
    BackendCheck -->|Device::Cpu| CPU["ojas-cpu (Reference Math)"]
    BackendCheck -->|Device::Metal| Metal["ojas-metal (Metal 4 MSL via tessl)"]
    BackendCheck -->|Device::Vulkan| WGPU["ojas-wgpu (Portable WGSL)"]
    BackendCheck -->|Device::Cuda| CUDA["ojas-cuda (PTX via cudarc)"]
    BackendCheck -->|Device::Hip| HIP["ojas-hip (ROCm via hip-runtime-sys)"]
```

### 4. Advanced Optimizer Suite (`ojas-optim`, `ojas-cpu`)
* **Dual Optimizer Architecture:**
  * **Muon NS5:** Orthogonalized matrix optimization using quintic 5th-order Newton-Schulz iterations with bf16 compute for 2D hidden weight matrices.
  * **AdamW:** PyTorch single-tensor order (decay first, bias correction, $\varepsilon = 10^{-8}$ outside the square root) for 1D parameters, embeddings, and normalization scales.
* **Safe Counters:** Step counters utilize checked integer math, refusing to wrap at `u64::MAX`.
* **Safe Clipping:** Gradient norm clipping with denominator regularizer (`CLIP_GRAD_NORM_EPS = 1e-6`) to prevent zero division.

### 5. In-Process Multi-Language Integration (`go/`, `ojas-capi`)
* **Zero-IPC Overhead:** Execute high-performance neural network workloads directly inside Go applications via in-process CGO and `gusset`.
* **Panic Boundary Isolation:** Rust panics are intercepted at the C-ABI boundary via `std::panic::catch_unwind`, cleanly dropping corrupted session state without aborting the host process.

```mermaid
sequenceDiagram
    autonumber
    participant Go as Go Application (go/api.go)
    participant Gusset as Worker Pool (libgusset.a)
    participant CAPI as FFI Engine (ojas-capi)
    participant Kernel as Hardware Kernel (CPU / Metal)

    Go->>Gusset: ojas.Step(sessionID, StepRequest)
    Gusset->>CAPI: dispatch(OP_STEP, payload)
    Note over CAPI: catch_unwind boundary protects host
    CAPI->>Kernel: Forward pass + Loss + Backward + AdamW
    Kernel-->>CAPI: StepStats{Loss, GradNorm, Lr}
    CAPI-->>Gusset: Serialized f32 stats
    Gusset-->>Go: ojas.StepStats
```

### 6. Reference Model Implementations
* **v1 Flagship Model:** A 124M parameter nanolab default GPT (12 layers, 768 width, 12 heads, 64 head dimension, vocab 50,304, SwiGLU, rotary embeddings, QK-norm, causal attention, per-head sigmoid output gate, value residual connection, and tied embeddings).
* **Extensible Architecture:** Designed for modular construction of arbitrary neural network topologies (vision transformers, diffusion backbones, recurrent models, and MoE architectures).

---

## Workspace Crate Catalog

| Crate | Purpose | Key Symbols | Status |
| :--- | :--- | :--- | :---: |
| [`ojas-core`](file:///Users/bharath/Code/research/ojas/ojas-core) | Core tensor engine & invariants | `Tensor`, `Backend`, `Budget`, `DType`, `OjasError`, `CheckpointV1` | **Verified** |
| [`ojas-cpu`](file:///Users/bharath/Code/research/ojas/ojas-cpu) | Deterministic reference compute | `CpuBackend`, exact mathematical kernels, single-thread loop | **Verified** (18 tests) |
| [`ojas-metal`](file:///Users/bharath/Code/research/ojas/ojas-metal) | Apple Silicon Metal 4 backend | `tiny_train_step`, `per_head_gate.metal`, causal attention | **Verified** (10 tests) |
| [`ojas-autograd`](file:///Users/bharath/Code/research/ojas/ojas-autograd) | Reverse-mode tape & gradcheck | `Tape`, `Var`, `central_diff`, f64 finite difference validation | **Verified** (5 tests) |
| [`ojas-io`](file:///Users/bharath/Code/research/ojas/ojas-io) | Formats & serialization | Safetensors parser (F32/I64/U16), Checkpoint v1 binary codec | **Verified** (12 tests) |
| [`ojas-data`](file:///Users/bharath/Code/research/ojas/ojas-data) | Dataset ingestion & tokenization | Binary token streams, Fineweb format, deterministic RNG | **Verified** (5 tests) |
| [`ojas-infer`](file:///Users/bharath/Code/research/ojas/ojas-infer) | Autoregressive inference engine | `CpuGpt`, KV cache, greedy decoding, logit validation | **Verified** (4 tests) |
| [`ojas-capi`](file:///Users/bharath/Code/research/ojas/ojas-capi) | C-ABI dispatch & foreign host API | `dispatch`, `install_engine`, session store, panic boundary | **Verified** (10 tests) |
| [`ojas-gusset-engine`](file:///Users/bharath/Code/research/ojas/ojas-gusset-engine) | Go link archive | Umbrella staticlib `libgusset.a` for Go CGO integration | **Verified** |
| [`go/`](file:///Users/bharath/Code/research/ojas/go) | Go client SDK | Package `ojas`: `Load`, `Step`, `GenerateGreedy`, `Close` | **Verified** (7 tests) |
| [`ojas-device`](file:///Users/bharath/Code/research/ojas/ojas-device) | Hardware discovery | `Device` probe, explicit backend routing without fallbacks | **Verified** (3 tests) |
| [`ojas-wgpu`](file:///Users/bharath/Code/research/ojas/ojas-wgpu) | Portable WGSL shaders | WGSL compute pipeline, Metal/Vulkan HAL execution | **Verified** (7 tests) |
| [`ojas-oracle`](file:///Users/bharath/Code/research/ojas/ojas-oracle) | Mathematical fixtures | IEEE-754 f64 oracle data for exact op verification | **Verified** (2 tests) |
| [`ojas-cuda`](file:///Users/bharath/Code/research/ojas/ojas-cuda) | NVIDIA CUDA backend | Optional feature `cuda`; explicit device validation | **Verified** (2 tests) |
| [`ojas-hip`](file:///Users/bharath/Code/research/ojas/ojas-hip) | AMD ROCm/HIP backend | Optional feature `hip`; explicit device validation | **Verified** (2 tests) |
| [`ojas-nn`](file:///Users/bharath/Code/research/ojas/ojas-nn) | High-level neural modules | Module abstractions for neural network layers | Scaffold |
| [`ojas-optim`](file:///Users/bharath/Code/research/ojas/ojas-optim) | Standalone optimizers | Optimizer interfaces (reference math lives in `ojas-cpu`) | Scaffold |
| [`ojas-engine`](file:///Users/bharath/Code/research/ojas/ojas-engine) | Standalone daemon | Engine daemon scaffolding | Scaffold |

---

## Verification & Test Commands

### 1. Workspace Test Suite
```bash
cargo test --workspace --release -- --test-threads=1
```
*Verification:* **189 passed; 0 failed; 0 ignored** across all Rust crates.

### 2. In-Process Go Client Tests
```bash
cargo build -p ojas-gusset-engine
cd go && PKG_CONFIG_PATH="$PWD" go test -tags gusset_pkgconfig -v -count=1
```
*Verification:* **11 passed; 0 failed; 0 skipped** (`TestPathEscapeAndMissingFile`, `TestDoubleFree`, `TestSessionCap`, `TestCloseEmpty`, `TestStepLossAndShape`, `TestStepRejectsMixedLogitAndTokenFields`, `TestConcurrentLoadStepGenerateFreeOneHandle`, `TestConcurrentSessionStress`, `TestMalformedInputsAreErrors`, `TestGenerateNaN`, `TestPoisonDropsSession`).

---

## Documentation Directory

* [`docs/architecture.md`](file:///Users/bharath/Code/research/ojas/docs/architecture.md) — Comprehensive architectural specification and memory layout.
* [`docs/status.md`](file:///Users/bharath/Code/research/ojas/docs/status.md) — Verification run logs, test matrices, and machine profile.
* [`docs/op-coverage.md`](file:///Users/bharath/Code/research/ojas/docs/op-coverage.md) — Mathematical specification of core operators and reference models.
* [`docs/checkpoint-v1.md`](file:///Users/bharath/Code/research/ojas/docs/checkpoint-v1.md) — Binary checkpoint specification and framing.
* [`docs/dtype-policy.md`](file:///Users/bharath/Code/research/ojas/docs/dtype-policy.md) — Data types, precision policies, and epsilon invariants.
* [`docs/backends.md`](file:///Users/bharath/Code/research/ojas/docs/backends.md) — Hardware support matrix, adopted libraries, and backend dispatch.
* [`docs/audit.md`](file:///Users/bharath/Code/research/ojas/docs/audit.md) — Defect analysis of upstream engines and ojas defensive countermeasures.
* [`docs/baseline.md`](file:///Users/bharath/Code/research/ojas/docs/baseline.md) — Benchmarks, toolchain baselines, and Apple Accelerate BLAS measurements.

---

## License

Dual-licensed under either **MIT** ([`LICENSE-MIT`](file:///Users/bharath/Code/research/ojas/LICENSE-MIT)) or **Apache-2.0** ([`LICENSE-APACHE`](file:///Users/bharath/Code/research/ojas/LICENSE-APACHE)) at your option.
