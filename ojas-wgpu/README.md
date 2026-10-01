# ojas-wgpu

`ojas-wgpu` implements portable compute shaders using **WebGPU (WGSL)** via the `wgpu` runtime.

---

## Compute Pipeline

```mermaid
flowchart LR
    Host["Host Buffer (f32 slice)"] --> Instance["wgpu::Instance / Adapter Request"]
    Instance --> HAL["Hardware Abstraction Layer\n(Metal on macOS, Vulkan on Linux)"]
    HAL --> Shader["WGSL Compute Shader\ny = x * scale + bias"]
    Shader --> Output["Result Buffer (f32)"]
```

---

## Verification & Device Details

On this machine (Apple M5 Pro):
```
ojas-wgpu adapter: Apple M5 Pro vendor=Apple (wgpu vendor field 0) hal=Metal
```

* **NaN Propagation:** Passing `f32::NAN` through affine scaling properly produces NaNs without pipeline stalling.
* **Empty Buffer Safety:** Zero-length input buffers return `Ok` with `dispatched == false` without allocating compute passes.
* **Size Validation:** Buffers exceeding `u32::MAX` return `DeviceError::NoDevice`.
