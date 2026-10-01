# ojas-hip

`ojas-hip` provides AMD ROCm/HIP GPU execution via `hip-runtime-sys`.

It is feature-gated under `--features hip` and is disabled in the default build.

---

## Execution Model

```mermaid
flowchart TD
    Open["HipDevice::open()"] --> CheckFeature{"Is 'hip' feature enabled?"}
    
    CheckFeature -->|No (Default Build)| NotCompiled["Return DeviceError::NotCompiled\n(Strict refusal; never CPU fallback)"]
    CheckFeature -->|Yes| HipContext["Initialize hipInit() & hipGetDeviceCount()"]
    
    HipContext --> Kernel["hipMalloc / hipMemcpy / hipFree"]
    Kernel --> Stream["Launch HIP affine_f32 kernel"]
    Stream --> Result["Result Vector (f32)"]
```

---

## Safety & Invariants

1. **No Silent Fallback:** Attempting to use HIP in a build without the `hip` feature flag returns `DeviceError::NotCompiled`. It will never silently run operations on the host CPU.
2. **Device Mismatch Check:** Passing `Device::Cpu` to `HipDevice::affine_f32` immediately raises `DeviceError::DeviceMismatch`.
