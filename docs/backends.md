# Hardware Backends & Device Architecture

Survey date: 2026-10-01.

This machine is an **Apple M5 Pro** (`system_profiler`: Metal 4, 20 GPU cores, vendor Apple `0x106b`). Neither `nvcc` nor `/opt/rocm` is present.

---

## Who Selects the Device

There is no device router in Rust. `ojas-device` defines the `Device` kinds (`Cpu`, `Metal`, `Cuda`, `Hip`, `Vulkan`), `require_kind` (returns `DeviceMismatch` when two kinds differ), `probe()` (host CPU only; GPU probes live in the crates that link those runtimes), and memory planning (`ResourcePlan`, `ResourcePolicy`). It does not open a GPU and does not choose a backend.

A caller chooses a backend by constructing it in Rust (`CpuBackend`, `MetalBackend`, `WgpuBackend`), or through the Go `LoadOn(ctx, device, threads, path)` call, which `ojas-capi` turns into a session on one backend at load time:

```mermaid
flowchart TD
    LoadOn["Go LoadOn(ctx, device, threads, path)"] --> Kind{"device"}

    Kind -->|DeviceCPU| CPU["CpuBackend (1 thread)"]
    Kind -->|"DeviceCPUParallel (threads 1..=256; 0 or >256 refused)"| CPUP["CpuBackend, threads"]
    Kind -->|DeviceMetal| Metal{"Metal device opens?"}
    Kind -->|DeviceWgpu| Wgpu{"wgpu adapter opens?"}

    Metal -->|Yes| MetalB["MetalBackend session\nStep / GenerateGreedy on device"]
    Metal -->|No| MetalErr["Load returns explicit device error\n(NO silent CPU session)"]
    Wgpu -->|Yes| WgpuB["WgpuBackend session\nStep / GenerateGreedy on device"]
    Wgpu -->|No| WgpuErr["Load returns explicit wgpu error\n(NO silent CPU session)"]
```

> [!IMPORTANT]
> **Zero Silent Fallback Policy:** CUDA and HIP have no `LoadOn` selector and do not implement `Backend`. If Metal or wgpu cannot open their respective hardware contexts, `LoadOn` returns an immediate error—**it will never silently substitute a CPU session**.

On a GPU session, `Step` reads back only the loss (4 bytes) and `GenerateGreedy` only the last-row logits it needs (8 bytes). `Generate` with caller-supplied logits takes the argmax on the host.

Device tensors: `Tensor::from_device` wraps a backend buffer, `to_host` copies it back and is counted by `device_readbacks()`, and `device_buffer_mut` requires sole ownership. The default `Backend::upload` returns `Unsupported` for a host tensor on a non-CPU backend, so a backend that has not implemented upload does not silently compute on the host.

---

## Memory Residency & Zero-Copy Lifecycle

```mermaid
flowchart LR
    subgraph HostRAM["Host RAM"]
        HostTen["Host Tensor (Arc<Vec<u8>>)"]
        Audit["device_readbacks() Audit Counter"]
    end

    subgraph GPUMem["GPU Unified / VRAM"]
        DeviceTen["Device Tensor (DeviceBuffer)"]
    end

    HostTen -->|Backend::upload() [Explicit]| DeviceTen
    DeviceTen -->|Tensor::to_host() [Counted Transfer]| HostTen
    DeviceTen -.->|Triggered on download| Audit
```

> [!CAUTION]
> Host tensor accessors (`to_f32_vec`, `as_slice`) refuse device-resident tensors immediately rather than reading them back implicitly across the bus. This prevents unmetered PCIe/memory bus bottlenecks from corrupting latency profiles.

---

## Backend Status (2026-10-01, Apple M5 Pro)

| Backend | Implements `Backend` | Numerics | Tests | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `CpuBackend` (`ojas-cpu`) | Yes | `Fast` default; `Exact` opt-in | 89 passed, 3 ignored in release workspace run | Exact: packed GEMM, persistent thread pool, bit-identical to golden digests at threads 1, 2, 3, 7, 16, 18. Fast on macOS: $\ge 2^{21}$ multiply-adds dispatch to Accelerate `cblas_sgemm`; smaller stay on `tile_fast` |
| `ojas-simd` | No (kernels for `ojas-cpu`) | — | 24 passed | NEON ~110 GFLOP/s single thread; Accelerate ~1.4–1.9 TFLOP/s on $256^3$, $512 \times 768 \times 768$, $2048^3$ |
| `MetalBackend` (`ojas-metal`) | Yes, every op, device-resident | `Fast` | 66 passed (`cargo test -p ojas-metal --release -- --test-threads=1`) | Head dim above 64 refused. Per-op fixed cost ~1.4 ms, measured under load |
| `WgpuBackend` (`ojas-wgpu`) | Yes, device-resident | `Fast`, 1e-4 relative tolerance | Release: kernels 8, unit 20, contract 3, parity 19, residency 3; bench 1 ignored | Muon returns `Unsupported`. GEMM race resolved with whole-`vec4` stores |
| `ojas-kernels` | No (pure WGSL & geometry) | — | 8 passed | Geometry calculations (`gemm_grid`, `attention_tiles`), `math.wgsl`, parity harness |
| `ojas-cuda` | No | — | 8 passed (feature off) | One affine kernel behind `--features cuda` |
| `ojas-hip` | No | — | 8 passed (feature off) | Copy probe behind `--features hip`; no kernel |

> [!WARNING]
> Both `MetalBackend` and `WgpuBackend` enforce $d_{\text{head}} \le 64$ for attention operations. Passing $d_{\text{head}} > 64$ raises `OjasError::UnsupportedHeadDim` loud and early.

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
| Go GPU modules | — | Go GPU engines (`gorgonia`, `gocu`) add GC interference. The in-process gusset C-ABI boundary keeps compute in Rust. |

---

## CUDA and HIP Verification Status

**Reported** from an earlier session; not re-run on 2026-10-01.

| Check | Command | Reported Result |
| :--- | :--- | :--- |
| **CUDA Default** | `cargo test -p ojas-cuda` | **Passed:** `tests::default_build_reports_not_compiled` |
| **CUDA Bindings** | `cargo check -p ojas-cuda --features cuda` | **Passed:** Typecheck only (dynamic loading; `nvcc` absent) |
| **CUDA Execution** | Kernel launch | **Skipped:** No NVIDIA GPU present |
| **HIP Default** | `cargo test -p ojas-hip` | **Passed:** `tests::default_build_reports_not_compiled` |
| **HIP Bindings** | `cargo check -p ojas-hip --features hip` | **Skipped:** Build stopped in `hip-runtime-sys` (no `/opt/rocm`) |
| **HIP Execution** | Copy probe (there is no HIP kernel) | **Skipped:** `--features hip` was not built; no `/opt/rocm` and no AMD GPU |
