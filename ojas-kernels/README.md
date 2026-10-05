# ojas-kernels

`ojas-kernels` provides hardware-agnostic kernel source strings, workgroup execution grid geometry, shader assembly utilities, and mathematical parity harnesses for GPU backends across the **ojas** stack.

It depends exclusively on `ojas-core` (`#![forbid(unsafe_code)]`) and does not link directly to Metal, WebGPU, CUDA, or HIP driver runtimes.

> [!NOTE]
> By decoupling kernel text and grid geometry calculations from backend driver types, `ojas-kernels` enables out-of-process runners, cross-compilers, and lightweight unit testing of GPU dispatch logic without requiring a physical GPU adapter.

---

## Architectural Role

```mermaid
flowchart TD
    subgraph Consumers["GPU Compute Backends"]
        WGPU["ojas-wgpu (WgpuBackend)"]
        Metal["ojas-metal (MetalBackend)"]
        CUDA["ojas-cuda (CUDA PTX)"]
        HIP["ojas-hip (ROCm / HIP)"]
    end

    subgraph KernelSubstrate["ojas-kernels (Pure Geometry & Shaders)"]
        Geo["geometry.rs\n- gemm_grid(M, N)\n- grid_1d(N, tile_size)\n- attention_tiles(head_dim, shared_bytes)\n- cached_attention_splits(rows, kv_len)\n- fold_grid(batch, rows)"]
        Src["source.rs & wgsl/\n- wgsl_module(id)\n- affine_wgsl(), affine_cuda()"]
        Harness["harness.rs\n- linear_close()\n- max_abs()\n- splitmix_f32()"]
    end

    subgraph Core["Base Engine"]
        OjasCore["ojas-core (Tensors, DType, Budget)"]
    end

    WGPU --> KernelSubstrate
    Metal --> KernelSubstrate
    CUDA --> KernelSubstrate
    HIP --> KernelSubstrate
    KernelSubstrate --> OjasCore
```

---

## Geometry & Workgroup Tiling

Execution grid dimensions are computed dynamically based on hardware limits and operator topologies:

```mermaid
flowchart LR
    subgraph MatrixInput["Matrix C[M, N]"]
        Dims["M rows, N cols"]
    end

    subgraph Tiling["gemm_grid(M, N)"]
        TileX["x_blocks = ceil(N / tile)"]
        TileY["y_blocks = ceil(M / tile)"]
        TileZ["z_blocks = 1"]
    end

    subgraph GPUGrid["Workgroup Dispatch Grid"]
        Grid["Grid(x_blocks, y_blocks, z_blocks)\ntile = gemm_tile(M, N): 128 when M, N >= 128, else 64"]
    end

    Dims --> Tiling
    Tiling --> GPUGrid
```

### Key Geometric Invariants
* **`ATTENTION_MAX_HEAD_DIM = 256`:** The tiled attention template is instantiated for padded head widths 16, 32, 64, 128 and 256; a wider head is refused with `OjasError::UnsupportedHeadDim`. `attention_tiles` picks the forward row/key blocks and the backward block that fit the device's workgroup storage (every width fits 16 KiB), with `ATTENTION_PARTS = 4` lanes per row.
* **`GEMM_TILE = 64`, `GEMM_BIG_TILE = 128`:** `gemm_tile` picks the 128x128 register-blocked tile (8x8 outputs per lane, double-buffered k steps) when both output sides reach 128, else the 64x64 tile. Both sum k in ascending order. Words 7-10 place A, B and C at element offsets and make C accumulate; an accumulating call starts its registers from C, so a product split along k rounds as one call would.
* **Checked Arithmetic:** `cover_1d` and `grid_1d` compute ceiling block counts using safe checked division: `(len + block - 1) / block`, asserting that block sizes are non-zero.

---

## Shader Catalog & Dispatch Modules

```mermaid
flowchart TD
    subgraph Shaders["Shader Distribution Pipeline"]
        direction TB
        Common["common.wgsl\nParameter words, fault word, first-fault record"]
        AffineWGSL["affine.wgsl\nPointwise linear scaling: y = x * scale + bias"]
        GemmWGSL["gemm.wgsl\n64x64 and 128x128 register-blocked tiled matmul"]
        AttnWGSL["attention.wgsl\nTiled causal flash attention: online-softmax forward; dQ and dK/dV backward passes without atomics"]
        CachedWGSL["cached_attention.wgsl\nSplit-key attention against a time-major KV cache, merged in a fixed order"]
        CudaKernels["CUDA PTX Kernels\nAffine and GEMM source templates"]
    end

    subgraph Assembly["wgsl_module(id)"]
        Header["Inject common math declarations"]
        Body["Concatenate operator kernel text"]
        Validation["Return compiled WgslModule ready for naga validation"]
    end

    Common --> Assembly
    AffineWGSL --> Assembly
    GemmWGSL --> Assembly
    AttnWGSL --> Assembly
    CachedWGSL --> Assembly
```

---

## Parity Verification Harness (`harness.rs`)

`linear_close` runs `linear_forward` on two backends from the same deterministic inputs (`splitmix_f32`), moving data with each backend's own `upload` and `download`, so a device backend runs its kernel. `ojas-wgpu/tests/parity.rs` calls it against `ojas-cpu`.

> [!IMPORTANT]
> `max_abs` returns `f64::INFINITY` when either side holds a non-finite value, so no tolerance accepts a NaN or infinite output.
