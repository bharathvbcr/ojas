# ojas-cuda

`ojas-cuda` provides an optional NVIDIA CUDA GPU driver and kernel execution probe for **ojas** via `cudarc`.

It is feature-gated under `--features cuda` and is disabled in the default workspace build.

---

## Execution Model & Compilation Gate

```mermaid
flowchart TD
    Init["CudaDevice::open()"] --> CheckFeature{"Is 'cuda' feature flag enabled?"}
    
    CheckFeature -->|"No (Default Build)"| NotCompiled["Return Err(DeviceError::NotCompiled)\n(Strict refusal; never CPU fallback)"]
    CheckFeature -->|Yes| CudaContext["Initialize CudaContext(0) via cudarc"]
    
    CudaContext --> Kernel["Compile PTX Kernel via NVRTC:\naffine_f32 (y = x * scale + bias)"]
    Kernel --> Stream["Stream Launch & Host-Device Memory Transfers"]
    Stream --> Result["Result Vector (f32)"]
```

---

## Architectural Role & Boundaries

> [!NOTE]
> `ojas-cuda` does **not** implement the `ojas_core::Backend` trait and is not reachable from the Go C-ABI runtime (`ojas-capi`). It serves as an isolated affine kernel execution probe for systems testing on NVIDIA hardware.

---

## Defensive Countermeasures & Safety Guarantees

> [!IMPORTANT]
> 1. **Zero Silent Fallback:** Attempting to invoke CUDA operations in a build without `--features cuda` immediately returns `Err(DeviceError::NotCompiled)`. The engine **never silently redirects GPU workloads to host CPU execution**.
> 2. **Device Mismatch Defense:** Passing a mismatched device handle (such as `Device::Cpu`) to `CudaDevice::affine_f32` raises `Err(DeviceError::DeviceMismatch)` loud and early.
