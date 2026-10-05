# ojas-core

`ojas-core` is the foundational crate of the **ojas** stack. It defines core tensor abstractions, error invariants, memory budgeting, data types, and checkpoint schema structures.

It depends exclusively on the Rust standard library (`#![forbid(unsafe_code)]`).

---

## Core Abstractions

```mermaid
classDiagram
    class Tensor {
        +Arc~Vec~u8~~ bytes
        +usize byte_offset
        +Vec~usize~ shape
        +Vec~usize~ strides
        +DType dtype
        +narrow(dim, start, len) Result~Tensor~
        +as_slice~T~() Result~&[T]~
        +numel() usize
        +from_device(buf, shape, dtype) Tensor
        +to_host(backend) Result~Tensor~
        +is_device() bool
    }

    class Budget {
        +u64 cap_bytes
        +AtomicU64 live_bytes
        +AtomicU64 peak_bytes
        +try_reserve(bytes) Result~Reservation~
        +check_room(bytes) Result~()~
        +peak_bytes() u64
        +reset_peak()
        +child(cap_bytes) Budget
        +live_bytes() Result~u64~
    }

    class Reservation {
        -usize bytes
        -Arc~Budget~ budget
        +drop()
    }

    class DType {
        <<enumeration>>
        F32
        Bf16
        F16
        U32
        +size_bytes() usize
    }

    class OjasError {
        <<enumeration>>
        CapacityExceeded
        UnsupportedHeadDim
        NonFinite
        OutOfRange
        ShapeMismatch
        DeviceMismatch
        TruncatedCheckpoint
    }

    Tensor --> DType
    Budget ..> Reservation : creates
    Tensor ..> OjasError : returns on invalid view
```

---

## Zero-Copy Tensor View Model

Tensors track memory via an explicit `byte_offset` into an underlying `Arc<Vec<u8>>` or a backend's device buffer handle. Slicing with `narrow()` creates a new view without copying bytes:

```mermaid
flowchart TD
    subgraph Storage["Physical Allocation Buffer (Arc<Vec<u8>>)"]
        Padding["Leading Offset [0 .. byte_offset)"]
        Active["Active Tensor Elements [byte_offset .. byte_offset + active_bytes)"]
        Remaining["Unused Capacity"]
    end

    subgraph ViewA["Tensor View A [Batch=2, Seq=16, Dim=64]"]
        OffsetA["byte_offset = 0"]
        ShapeA["shape = [2, 16, 64]"]
    end

    subgraph ViewB["Tensor View B: narrow(1, 4, 8)"]
        OffsetB["byte_offset = 4 * 64 * 4 = 1024 bytes"]
        ShapeB["shape = [2, 8, 64]"]
    end

    ViewA --> Active
    ViewB --> Active
```

> [!NOTE]
> Every offset calculation accounts strictly for `dtype.size_bytes()`. View bounds are checked before pointer arithmetic to guarantee safe memory access.

---

## Memory Budgeting & Capacity Control

All tensor allocations in `ojas` must secure permits from an explicit `Budget`. Memory is never silently overcommitted or clamped:

```mermaid
stateDiagram-v2
    [*] --> Initialized: Budget#58;#58;new(max_bytes)
    
    Initialized --> Active: try_reserve(req) [req <= remaining]
    Active --> Active: try_reserve(req) [req <= remaining]
    Active --> Initialized: drop(Reservation) [releases bytes]
    
    Initialized --> Rejected: try_reserve(req) [req > remaining]
    Active --> Rejected: try_reserve(req) [req > remaining]
    
    Rejected --> [*]: Returns Err(OjasError#58;#58;CapacityExceeded)
```

> [!IMPORTANT]
> When an allocation request exceeds remaining budget capacity, `try_reserve()` returns `Err(OjasError::CapacityExceeded)` immediately. The engine **refuses to silently swap, clamp shapes, or resize limits**.
> 
> * **Preflight Room Check:** `check_room(bytes)` verifies whether `bytes` more would fit without reserving or charging anything. High-level callers use this to fail fast before initiating multi-step operations.
> * **Peak High-Water Mark:** `peak_bytes()` tracks the highest live allocation recorded since budget creation or the last reset. `reset_peak()` restarts tracking at the current live bytes, allowing callers (like `Trainer::step`) to measure exact memory usage per phase.
> * **Hierarchical Child Budgets:** `child(cap_bytes)` creates scoped sub-budgets that charge parent budgets hierarchically, guaranteeing multi-session bounds within a single global ceiling.
> 
> *Note on infallible allocations:* Standard Rust allocations like `vec!`, `format!`, and OS thread spawning (`thread::spawn`) operate outside the software budget and can still abort if the host operating system exhausts physical memory.

---

## Compute Backend Contract (`Backend`)

The `Backend` trait defines the uniform operator interface implemented by compute engines (`CpuBackend`, `MetalBackend`, `WgpuBackend`):
* Forward and backward linear transformations, activations (GELU, SiLU, ReLU), and causal multi-head attention.
* Fused head cross-entropy computation (`linear_cross_entropy_mean`) and key-value cache operations.
* In-place optimizer steps: `adamw_step` and `muon_ns5_step`.
* **Optimizer Scratch Sizing:** `optimizer_scratch_bytes(kind, rows, cols)` reports the exact or bounding scratch memory needed for an optimizer step. This permits training routines to preflight memory before taking steps, preventing out-of-memory errors from poisoning parameter states.

---

## Device Tensors & Readback Tracking

`Tensor::from_device` encapsulates device-resident memory handles managed by GPU backends (`MetalBackend`, `WgpuBackend`).

```mermaid
flowchart LR
    HostMem["Host Tensor (Arc<Vec<u8>>)"] -->|"Backend::upload()"| DeviceMem["Device Tensor (DeviceBuffer)"]
    DeviceMem -->|"Tensor::to_host()"| HostCopy["Host Tensor Copy"]
    DeviceMem -.->|"device_readbacks() counter increments"| Counter["Readback Audit Counter"]
    HostCopy --> Counter
```

> [!CAUTION]
> Host accessors such as `.to_f32_vec()` or `.as_slice::<f32>()` **refuse device tensors** rather than triggering an implicit, unmetered readback across PCIe / unified memory buses. Callers must explicitly call `.to_host(backend)`, which increments the counted readback metric.

---

## Immutable Numerical Invariants

1. **Epsilon Regularization Constants:**
   * `RMS_NORM_EPS = 1e-6`: Denominator variance offset in normalization.
   * `ADAMW_EPS = 1e-8`: Second moment offset added **outside** the square root.
   * `CLIP_GRAD_NORM_EPS = 1e-6`: Denominator safety offset for global gradient norm clipping.
   *(These constants represent distinct physical dimensions and are never interchanged).*

2. **Step Counter Arithmetic:**
   `next_step(current)` increments training step counters using `checked_add`, returning `OjasError::OutOfRange` at `u64::MAX` rather than silently wrapping to zero.

3. **No Silent Clamping:**
   Dimensions exceeding kernel capabilities (e.g. $d_{\text{head}} >$ `METAL_MAX_HEAD_DIM` = 256 on Metal and wgpu) yield `OjasError::UnsupportedHeadDim` loud and early.

4. **Cached Finiteness:**
   `Tensor::all_finite_cached(scan)` runs a backend's NaN/infinity scan at most once per storage. A finite result for the whole allocation answers for every view of it until the storage is next written, and every mutable host access resets it. A smaller window is scanned and records nothing. The scan is trusted, so it must check every element with `f32_all_finite`; debug builds re-check a `true` for the whole allocation and panic if it was wrong.
