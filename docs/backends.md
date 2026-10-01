# Hardware Backends & Device Architecture

Survey date: 2026-10-01.

This machine is an **Apple M5 Pro** (`system_profiler`: Metal 4, 20 GPU cores, vendor Apple `0x106b`). Neither `nvcc` nor `/opt/rocm` is present.

---

## Device Routing Pipeline

ojas strictly forbids silent fallbacks. If a GPU execution is requested and the hardware or compile-time feature is absent, execution fails immediately with an explicit error:

```mermaid
flowchart TD
    Request["Caller Requests Device::X"] --> Match{"Device Variant"}

    Match -->|Device::Cpu| CPU["ojas-cpu\nSingle-Threaded Reference"]
    
    Match -->|Device::Metal| MetalCheck{"macOS & Metal 4 Available?"}
    MetalCheck -->|Yes| MetalPath["ojas-metal\ntessl + per_head_gate.metal"]
    MetalCheck -->|No| MetalFail["Return DeviceError::NoDevice\n(Never CPU fallback!)"]

    Match -->|Device::Vulkan| WgpuCheck{"wgpu Adapter Available?"}
    WgpuCheck -->|Yes| WgpuPath["ojas-wgpu\nPortable WGSL Compute Pipeline"]
    WgpuCheck -->|No| WgpuFail["Return DeviceError::NoDevice\n(Never CPU fallback!)"]

    Match -->|Device::Cuda| CudaCheck{"Built with --features cuda?"}
    CudaCheck -->|Yes| CudaPath["ojas-cuda\nPTX Launch via cudarc"]
    CudaCheck -->|No| CudaFail["Return DeviceError::NotCompiled\n(Never CPU fallback!)"]

    Match -->|Device::Hip| HipCheck{"Built with --features hip?"}
    HipCheck -->|Yes| HipPath["ojas-hip\nKernel Launch via hip-runtime-sys"]
    HipCheck -->|No| HipFail["Return DeviceError::NotCompiled\n(Never CPU fallback!)"]
```

---

## What Ran on This Machine

`cargo test -p ojas-device -p ojas-wgpu -p ojas-cuda -p ojas-hip` passed 14 unit tests (3 + 7 + 2 + 2):

```
ojas-wgpu adapter: Apple M5 Pro vendor=Apple (wgpu vendor field 0) hal=Metal
```

* **wgpu Adapter:** Verified name `Apple M5 Pro`, HAL `Metal`.
* **Zero Dispatch on Empty:** Empty inputs return `Ok` with `dispatched == false` without executing command encoders.
* **NaN Propagation:** Passing `f32::NAN` through affine scale properly propagates NaNs.
* **Overflow Protection:** Buffers exceeding `u32::MAX` byte length return `DeviceError::NoDevice`.
* **Strict Rejection:** Passing `Device::Cuda` to the wgpu pipeline returns `DeviceError::DeviceMismatch`, not a fallback.

---

## Package Adoption Decisions

```mermaid
flowchart LR
    subgraph Adopted["Adopted Frameworks"]
        WGPU["wgpu (30.0.1)\nPortable compute launcher"]
        Pollster["pollster (1.0.1)\nBlocks async adapter acquisition"]
        Cudarc["cudarc (0.19.10)\nOptional feature 'cuda'"]
        HipSys["hip-runtime-sys (0.1.2)\nOptional feature 'hip'"]
        Tessl["tessl (0.2.0)\nApple Metal 4 training kernels"]
    end

    subgraph Rejected["Rejected Frameworks"]
        Burn["Burn (~2,900 ms/step vs 56.6 ms/step)"]
        Candle["Candle (Avoid heavy generic dependency)"]
        Cubecl["CubeCL (Indirect compiler dependency)"]
        Cust["Cust (Superseded by cudarc)"]
        Gorgonia["Go GPU modules (Keep Gusset C-ABI boundary)"]
    end
```

### Adopted

| Package | Version | License | Role | Decision | Rationale |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `wgpu` | `30.0.1` | MIT OR Apache-2.0 | Portable compute launcher | **Adopt** | Executes portable WGSL across Vulkan, Metal, and DX12. |
| `pollster` | `1.0.1` | MIT OR Apache-2.0 | Future blocker | **Adopt** | Minimal future executor for `request_adapter()`. |
| `cudarc` | `0.19.10` | MIT OR Apache-2.0 | CUDA driver & NVRTC bindings | **Adopt (Feature)** | Modern safe CUDA bindings. Feature `cuda` off by default. |
| `hip-runtime-sys` | `0.1.2` | MIT OR Apache-2.0 | AMD HIP runtime bindings | **Adopt (Feature)** | Direct C-ABI bindings for ROCm/HIP. Feature `hip` off by default. |
| `tessl` | `0.2.0` | MIT OR Apache-2.0 | Apple Metal 4 kernels | **Keep** | Native Apple Silicon GEMM and normalization. |

### Rejected

| Package | Version | Reason for Rejection |
| :--- | :--- | :--- |
| `burn` | `0.21.0` | Heavy runtime; measured ~2,900–3,400 ms/step vs 56.6 ms/step native. |
| `candle-core` | `0.11.0` | Avoid broad general runtime dependency on performance-critical paths. |
| `cubecl` | `0.10.0` | JIT kernel compiler layer; ojas relies on static native Metal & WGSL shaders. |
| `cust` | `0.3.2` | Unmaintained legacy rust-cuda bindings; superseded by `cudarc`. |
| Go GPU modules | — | Go GPU engines (`gorgonia`, `gocu`) add GC interference. Gusset IPC cleanly isolates compute. |

---

## CUDA and HIP Verification Status

| Check | Command | Verified Result |
| :--- | :--- | :--- |
| **CUDA Default** | `cargo test -p ojas-cuda` | **Passed:** `tests::default_build_reports_not_compiled` |
| **CUDA Bindings** | `cargo check -p ojas-cuda --features cuda` | **Passed:** Typecheck only (dynamic loading; `nvcc` absent) |
| **CUDA Execution** | Kernel launch | **Skipped:** No NVIDIA GPU present |
| **HIP Default** | `cargo test -p ojas-hip` | **Passed:** `tests::default_build_reports_not_compiled` |
| **HIP Bindings** | `cargo check -p ojas-hip --features hip` | **Skipped:** Build stopped in `hip-runtime-sys` (no `/opt/rocm`) |
| **HIP Execution** | Kernel launch | **Skipped:** No AMD GPU present |
