# ojas Status & Verification Matrix

Recorded: 2026-10-01, Apple M5 Pro, macOS 27. 

> [!IMPORTANT]
> **Verification Protocol Standards:**
> * **Verified:** A claim is verified when this session executed the command directly and recorded its exact output.
> * **Reported:** An earlier lane or audit session recorded the outcome and this session did not re-execute it.
> * **Inferred:** Logically deduced from code structure or invariant dependencies.

---

## Latest run: after the framework round (verified, 2026-10-01 18:52–18:55)

Each crate ran `cargo test -p <crate> --release -- --test-threads=1` on the uncommitted tree (`target-baseline/run_integration2.sh`, logs in `target-baseline/logs-integration-2/`). The script first ran `cargo build --workspace --release --all-targets`, which exited 0. Load average was 12–17. Afterwards, the Metal cached-attention CPU-parity test (G4) was un-ignored and passes (`ojas-metal --test kv_cache`: 8 passed).

| crate | previous run (below) | this run |
| :--- | :--- | :--- |
| ojas-core | 55, 1 ignored | 93, 2 ignored |
| ojas-cpu | 106, 5 ignored | 189, 10 ignored |
| ojas-simd | 20 | 20 |
| ojas-autograd | 33 | 67 |
| ojas-io | 42 | 66 |
| ojas-data | 26 | 26 |
| ojas-infer | 31 | 35 |
| ojas-oracle | 6 | 6 |
| ojas-device | 18 | 18 |
| ojas-kernels | 9 | 11 |
| ojas-capi | 43 | 44 |
| ojas-gusset-engine | 0 | 0 (its tests are the Go suite) |
| ojas-metal | 95 | 129, 1 ignored (now 130, 0 ignored, after G4 was un-ignored) |
| ojas-wgpu | 66, 1 ignored | 115, 3 ignored |
| ojas-qwen35 (new member) | — | 23, 8 ignored (GPU parity; the Lappi session ran 7 of them: 6 pass, `gpu_real_2b` faults at its gradient read-back) |
| ojas-cuda, ojas-hip (features off) | 8 + 8 | 8 + 8 |
| **total** | **566 passed, 7 ignored** | **858 passed, 0 failed, 26 ignored** |

**What changed:**
- **Framework trait methods T1–T6:** `sync`, `accumulate_grad`, the fused `linear_cross_entropy_mean` with a two-dimensional `CeChunk`, `cached_attention_forward`, `kv_cache_write`, and `Backend` for `&B` and `Arc<B>`. Each is native on CPU, Metal and wgpu.
- **Tape (A1–A3):** `backward_seeded`, `take_grad` and a fused head.
- **Shape contract:** `ojas_core::shapes` ([`shape-contract.md`](shape-contract.md)).
- **ojas-io:** streaming safetensors writing and `replace_dir_with`.
- **Metal:** faster AdamW, RMSNorm and CE kernels.
- **wgpu:** tiled FlashAttention-2 and a 128-tile GEMM.
- **Bugs fixed, each with a test the lane reports failed before the fix.** The tests pass in this run; I did not re-run the pre-fix code:
  - wgpu `Backend::sync` was not overridden, so deferred faults were dropped;
  - wgpu device loss was not named;
  - wgpu `clip_grad_norm` could return another caller's norm under concurrency;
  - the Tape dropped its seed on device backends;
  - the safetensors writer could produce headers its own reader refused;
  - a CPU pool worker held a finished batch's buffers past `Pool::run`.

## Previous run: after the parity and hardening lanes (verified, 2026-10-01)

Each crate ran `cargo test -p <crate> --release -- --test-threads=1` on the uncommitted tree (`target-baseline/run_integration.sh`, logs in `target-baseline/logs-integration/`). ojas-metal was re-run with `--no-fail-fast` after the tiny-step head-dim fix. Load average was 16–36. The plan and findings are in [`pytorch-parity-plan.md`](pytorch-parity-plan.md).

| crate | before (per crate, same method) | after |
| :--- | :--- | :--- |
| ojas-core | 37 | 55 passed, 1 ignored |
| ojas-cpu | 89, 3 ignored | 106, 5 ignored |
| ojas-simd | 20 | 20 |
| ojas-autograd | 28 | 33 |
| ojas-io | 36 | 42 |
| ojas-data | 21 | 26 |
| ojas-infer | 14 | 31 |
| ojas-oracle | 6 | 6 |
| ojas-device | 18 | 18 |
| ojas-kernels | 8 | 9 |
| ojas-capi | 41 | 43 |
| ojas-gusset-engine | 0 | 0 |
| ojas-metal | 66 | 95 |
| ojas-wgpu | 45, 1 ignored | 66, 1 ignored |
| ojas-cuda, ojas-hip (features off) | 8 + 8 | 8 + 8 |
| **total** | **445** | **566 passed, 0 failed, 7 ignored** |

**Re-run after the `CpuBackend` default became `Numerics::Fast`** (verified, 17:30; another session made the change with the user's approval). The non-GPU crates passed: 401, 0 failed, 10 ignored. The changed counts include that session's new tests:

| crate | count |
| :--- | :--- |
| ojas-core | 63, 2 ignored |
| ojas-cpu | 133, 8 ignored |
| ojas-autograd | 36 |
| ojas-infer | 32 |

The rest are unchanged. Logs are in `target-baseline/logs-postflip/`. The ojas-capi, ojas-metal and ojas-wgpu re-runs after the flip belong to that session and are not recorded here.

**GPU against torch MPS:** see [`bench-gpu-vs-torch.md`](bench-gpu-vs-torch.md). ojas is slower on most rows: a nanolab block forward + backward is 0.37× torch on Metal and 0.18× on wgpu. It is faster on Metal attention backward (1.24–1.38×) and `clip_grad_norm` (4.6–6.4×). Those runs were under 84–100% external GPU load, so only direction is verified.

The per-crate total differs from the workspace run below because of feature unification. `ojas-simd` runs 20 tests alone and 24 under `--workspace`, where `ojas-cpu` turns on its `accelerate` feature. 445 + 4 = 449.

Go: 27 passed. The infer lane ran it against a `libgusset` built into `target-lane-infer`. Some ojas-core tests come from a concurrent session's decode-path work (`ojas-core/tests/tensor_decode_contract.rs`, `bench_decode.rs`).

**What changed:**
- `Backend::permute` on CPU, Metal, wgpu and the `Tape`, so multi-head attention trains through autograd (gradchecked H=3 block).
- Metal causal attention is a tiled TensorOps forward plus a FlashAttention-2 backward at D ≤ 128 (`METAL_MAX_HEAD_DIM` = 128).
- Muon NS5 on wgpu.
- CPU optimizer steps are all-or-nothing.
- CPU input copies are charged to `Budget`.
- wgpu names the first faulting op, and a fault survives a failed read.
- NaN-safe parity comparators.
- `ojas-infer` runs the nanolab block (QK-norm, RoPE, value residual, gate, GQA) with sampling.
- BF16/F16 safetensors.
- A seeded, resumable batch sampler.
- Typed C-ABI error kinds.

The sections below are the earlier record from the same day. Where they disagree with this table (counts, Metal head dim 64, wgpu Muon unsupported, no sampling), this table is current.

## Workspace Test Suite Results (earlier the same day)

On 2026-10-01, `cargo test --workspace --release -- --test-threads=1` finished in 57s: **449 passed, 0 failed, 4 ignored**. Doc-tests ran and contained no tests. The 4 ignored are `ojas-cpu` benches (2), one exact-golden case (1), and the wgpu bench (1). CUDA and HIP suites are the default build: features `cuda` and `hip` were not enabled, and no device kernel ran.

```mermaid
flowchart TD
    subgraph SuiteSummary["Workspace Test Execution (2026-10-01)"]
        Total["449 Passed | 0 Failed | 4 Ignored"]
    end

    subgraph Backends["Hardware Compute Backends"]
        CPU["ojas-cpu: 89 passed, 3 ignored"]
        SIMD["ojas-simd: 24 passed"]
        Metal["ojas-metal: 66 passed"]
        WGPU["ojas-wgpu: 45 passed, 1 ignored"]
        Kernels["ojas-kernels: 8 passed"]
    end

    subgraph CoreAndIO["Core & Subsystems"]
        Core["ojas-core: 37 passed"]
        IO["ojas-io: 36 passed"]
        Autograd["ojas-autograd: 28 passed"]
        Data["ojas-data: 21 passed"]
        Infer["ojas-infer: 14 passed"]
        Device["ojas-device: 18 passed"]
        Oracle["ojas-oracle: 6 passed"]
    end

    subgraph FFIAndProbes["FFI & Probes"]
        CAPI["ojas-capi: 41 passed"]
        CUDA["ojas-cuda: 8 passed (feature off)"]
        HIP["ojas-hip: 8 passed (feature off)"]
    end

    SuiteSummary --> Backends
    SuiteSummary --> CoreAndIO
    SuiteSummary --> FFIAndProbes
```

| Crate | Passed | Failed | Ignored | Notes |
| :--- | ---: | ---: | ---: | :--- |
| [`ojas-core`](file:///Users/bharath/Code/research/ojas/ojas-core) | 37 | 0 | 0 | Invariants, byte-offsets, memory budget |
| [`ojas-cpu`](file:///Users/bharath/Code/research/ojas/ojas-cpu) | 89 | 0 | 3 | Exact & Fast packed GEMM, persistent thread pool |
| [`ojas-simd`](file:///Users/bharath/Code/research/ojas/ojas-simd) | 24 | 0 | 0 | NEON, AVX2, Accelerate BLAS |
| [`ojas-metal`](file:///Users/bharath/Code/research/ojas/ojas-metal) | 66 | 0 | 0 | Device-resident Metal 4 training step |
| [`ojas-wgpu`](file:///Users/bharath/Code/research/ojas/ojas-wgpu) | 45 | 0 | 1 | Portable WGSL compute, whole-vec4 stores |
| [`ojas-kernels`](file:///Users/bharath/Code/research/ojas/ojas-kernels) | 8 | 0 | 0 | Grid geometry, math shaders, parity harness |
| [`ojas-autograd`](file:///Users/bharath/Code/research/ojas/ojas-autograd) | 28 | 0 | 0 | Dynamic tape, device tape, f64 gradcheck |
| [`ojas-io`](file:///Users/bharath/Code/research/ojas/ojas-io) | 36 | 0 | 0 | Safetensors, Checkpoint v1 |
| [`ojas-data`](file:///Users/bharath/Code/research/ojas/ojas-data) | 21 | 0 | 0 | Token streaming, counter-based RNG |
| [`ojas-infer`](file:///Users/bharath/Code/research/ojas/ojas-infer) | 14 | 0 | 0 | KV cache, greedy decode, logit validator |
| [`ojas-capi`](file:///Users/bharath/Code/research/ojas/ojas-capi) | 41 | 0 | 0 | C-ABI engine, session store, panic boundary |
| [`ojas-gusset-engine`](file:///Users/bharath/Code/research/ojas/ojas-gusset-engine) | 0 | 0 | 0 | Umbrella staticlib (libgusset.a) |
| [`ojas-device`](file:///Users/bharath/Code/research/ojas/ojas-device) | 18 | 0 | 0 | Host CPU probe, ResourcePolicy |
| [`ojas-oracle`](file:///Users/bharath/Code/research/ojas/ojas-oracle) | 6 | 0 | 0 | IEEE-754 f64 analytical fixtures |
| [`ojas-cuda`](file:///Users/bharath/Code/research/ojas/ojas-cuda) | 8 | 0 | 0 | One affine kernel behind feature cuda |
| [`ojas-hip`](file:///Users/bharath/Code/research/ojas/ojas-hip) | 8 | 0 | 0 | Copy probe behind feature hip |
| **Total Workspace** | **449** | **0** | **4** | **All passing** |

---

## In-Process Go Client Test Suite

**Verified.** `cargo build -p ojas-gusset-engine` then:
```bash
cd go && PKG_CONFIG_PATH="$PWD" go test -a -tags gusset_pkgconfig -count=1 -timeout 15m ./...
```
Exited 0 with **27 tests passed** (total runtime 3.417s with `-a`).

> [!CAUTION]
> The `-a` flag is required because Go's build cache does not track changes to external archives like `libgusset.a`. `go/gusset.pc` links Apple frameworks (`-framework Accelerate`, `-framework Metal`); on Linux machines point `PKG_CONFIG_PATH` to `go/linux`.

---

## Component Implementation Status

```mermaid
flowchart LR
    subgraph Backends["Hardware Compute Backends"]
        direction TB
        b1["ojas-cpu CpuBackend (Exact, or Fast with Accelerate)"]
        b2["ojas-metal MetalBackend (every op, device-resident, Fast)"]
        b3["ojas-wgpu WgpuBackend (device-resident WGSL, Fast)"]
    end

    subgraph Host["Host-Side Subsystems"]
        direction TB
        h1["ojas-core (Tensor host & device, Budget, Numerics)"]
        h2["ojas-autograd Tape (Reverse-mode AD, device tape)"]
        h3["ojas-infer (KV cache, greedy decode, CPU)"]
        h4["ojas-capi + go/ (In-process Go & C-ABI bridge)"]
        h5["ojas-io, ojas-data, ojas-oracle"]
    end

    subgraph Probes["Hardware Probes (Not Backends)"]
        direction TB
        p1["ojas-cuda (one affine kernel, feature cuda)"]
        p2["ojas-hip (copy probe, feature hip)"]
        p3["ojas-device (kinds, host probe, ResourcePolicy)"]
    end

    Host --> Backends
```

---

## Defect Countermeasures & Hardening Invariants

```mermaid
flowchart TD
    subgraph InvariantChecks["Defensive Hardening Guarantees"]
        C1["CE dh offset preservation: 16 canary padding elements (DH_PAD)"]
        C2["AdamW step counter overflow: next_step checked_add refuses u64::MAX"]
        C3["All-ignored cross-entropy: Zero valid targets returns Err(NonFinite)"]
        C4["AdamW non-finite store: Aborts before touching moment buffers"]
        C5["Metal head dim limit: Dimensions > 64 produce UnsupportedHeadDim"]
        C6["Safetensors parser: Capped 100MB header, max depth 64, no overlaps"]
    end
```

* **CE `dh` offset preservation (verified):** `ojas-metal` reserves 16 canary padding elements (`DH_PAD`) before the `dh` slice; verified by `ce_nonzero_dh_offset_keeps_prefix` (`ojas-metal/src/gpu.rs`).
* **AdamW step counter overflow (verified):** `next_step` uses `checked_add` and refuses `u64::MAX`; verified by `step_count_at_u64_max_is_refused` (`ojas-metal/src/gpu.rs`).
* **All-ignored cross entropy (verified):** Mean loss over zero valid targets returns `OjasError::NonFinite`; verified by `all_ignored_cross_entropy_is_nonfinite_not_a_zero_loss` (`ojas-cpu/tests/redteam.rs`).
* **AdamW non-finite store protection (verified):** Non-finite updates are rejected before moments change; verified by `adamw_refuses_a_nonfinite_f32_store_without_touching_moments` (`ojas-cpu/tests/redteam.rs`).
