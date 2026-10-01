# ojas-cuda

`ojas-cuda` provides NVIDIA CUDA GPU execution via `cudarc`.

It is feature-gated under `--features cuda` and is disabled in the default build.

---

## Execution Model

```mermaid
flowchart TD
    Open["CudaDevice::open()"] --> CheckFeature{"Is 'cuda' feature enabled?"}
    
    CheckFeature -->|No (Default Build)| NotCompiled["Return DeviceError::NotCompiled\n(Strict refusal; never CPU fallback)"]
    CheckFeature -->|Yes| CudaContext["Initialize CudaContext(0) via cudarc"]
    
    CudaContext --> Kernel["NVRTC PTX Compilation:\naffine_f32 (y = x * scale + bias)"]
    Kernel --> Stream["Stream Launch & Memory Transfers (HtoD / DtoH)"]
    Stream --> Result["Result Vector (f32)"]
```

---

## Safety & Invariants

1. **No Silent Fallback:** Attempting to use CUDA in a build without the `cuda` feature flag returns `DeviceError::NotCompiled`. It will never silently run operations on the host CPU.
2. **Device Mismatch Check:** Passing `Device::Cpu` to `CudaDevice::affine_f32` immediately raises `DeviceError::DeviceMismatch`.
