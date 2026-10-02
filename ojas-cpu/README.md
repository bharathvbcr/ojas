# ojas-cpu

`ojas-cpu` provides the high-performance CPU backend and reference implementation for **ojas**, executing the full **nanolab default GPT** training step, autoregressive inference, and autograd gradient evaluation.

It supports both the bit-identical **Exact tier** and the hardware-accelerated **Fast tier** (via `ojas-simd` and Apple Accelerate).

---

## Architectural Topology & Execution Modes

```mermaid
flowchart TD
    subgraph Client["Execution Request"]
        Op["Matrix Multiplication / Transformer Step"]
    end

    subgraph BackendConfig["CpuBackend Configuration"]
        Config{"Numerics Mode"}
        ExactMode["Numerics::Exact (opt-in reference)\n- Packed GEMM\n- Persistent thread pool (1..=256 threads)\n- No FMA, ascending-k reductions\n- Bit-identical across all thread counts"]
        FastMode["Numerics::Fast (Default)\n- FMA enabled\n- On macOS, products >= 2^13 multiply-adds route to Accelerate BLAS (2^21 elsewhere, to sgemm_tile)\n- Smaller products run on tile_fast SIMD"]
    end

    subgraph ThreadPool["Persistent Worker Pool (pool.rs)"]
        Pool["Scoped Thread Pool\n- Avoids repeated thread spawn latency\n- Chunked work splitting across CPU cores"]
    end

    subgraph Kernels["Compute Operators"]
        GEMM["Packed GEMM / BLAS"]
        RMS["RMSNorm (eps = 1e-6)"]
        RoPE["Half-Split RoPE"]
        SDPA["Causal SDPA (Arbitrary head dimension d)"]
        SwiGLU["SwiGLU MLP"]
        DualOpt["Dual Optimizers (Muon NS5 + AdamW)"]
    end

    Op --> Config
    Config -->|Exact| ExactMode
    Config -->|Fast| FastMode
    ExactMode --> Pool
    FastMode --> Pool
    Pool --> Kernels
```

---

## Transformer Layer & Optimizer Flow

```mermaid
flowchart TD
    subgraph LayerFlow["Transformer Forward Pass"]
        X["Input Tensors [B, T, D]"] --> Norm1["RMSNorm (eps = 1e-6)"]
        Norm1 --> Proj["Q, K, V Projections (Packed GEMM)"]
        Proj --> RoPE["Half-Split RoPE (rotates -x2, x1)"]
        RoPE --> QKN["RMS QK-Norm"]
        QKN --> SDPA["Causal SDPA (Causal mask, scale = 1/sqrt(d))"]
        Norm1 --> Gate["Linear Gate: sigmoid(x W_g + b_g)"]
        SDPA --> GatedAttn["Per-Head Gated Attention = SDPA * Gate"]
        Gate --> GatedAttn
        GatedAttn --> VR["Value Residual Blend: lerp(Attn, V0, sigmoid(vr_lambda))"]
        VR --> OutProj["Output Projection"]
        OutProj --> Res1["Residual Add (+)"]
        X --> Res1
        Res1 --> Norm2["Post-RMSNorm"]
        Norm2 --> SwiGLU["SwiGLU MLP: (x W_g) * silu(x W_up) W_down"]
        SwiGLU --> Res2["Residual Add (+)"]
        Res1 --> Res2
    end

    subgraph BackwardOptim["Loss, Backward & Dual Optimizers"]
        Res2 --> CE["Chunked Cross-Entropy (Mean over valid tokens)"]
        CE --> Clip["Gradient Clipping (CLIP_GRAD_NORM_EPS = 1e-6)"]
        Clip --> OptSplit{"Parameter Rank"}
        OptSplit -->|Rank >= 2 (2D Weights)| Muon["Muon NS5 Optimizer (bf16 Newton-Schulz)"]
        OptSplit -->|Rank < 2 (1D / Embeds)| AdamW["AdamW (decay first, eps = 1e-8 outside sqrt)"]
    end

    LayerFlow --> BackwardOptim
```

---

## Exact vs. Fast Numerics Contract

> [!NOTE]
> * **Exact Tier (`Numerics::Exact`, opt-in via `.with_numerics(Numerics::Exact)`):** The reference contract. Uses an ascending-$k$ packed GEMM without hardware FMA. Successive executions with identical inputs produce **bit-identical floating point digests** across 1, 2, 3, 7, 16, and 18 thread configurations.
> * **Fast Tier (`Numerics::Fast`, the default since 2026-10-01):** Leverages SIMD FMA instructions via `ojas-simd`. On macOS, products with at least $2^{13}$ multiply-adds (`FAST_WHOLE_CALL_MACS`) dispatch to Apple Accelerate `cblas_sgemm` as one call; elsewhere the cutoff is $2^{21}$ and the call goes to `ojas-simd`.

---

## Defensive Countermeasures & Safety Guarantees

> [!IMPORTANT]
> 1. **Zero-Loss Defect Defense:** Cross-entropy evaluated over sequences containing zero valid tokens (e.g. all targets match `ignore_index`) immediately returns `Err(OjasError::NonFinite)` rather than falsely returning $0.0$ loss.
> 2. **AdamW Moment Protection:** If parameter updates evaluate to non-finite values (NaN / Inf), the optimizer aborts before touching internal moment buffers, preventing persistent corruption.
> 3. **Arbitrary Head Dimension:** Unlike Metal kernels that enforce $d_{\text{head}} \le 64$, `ojas-cpu` supports arbitrary head dimensions (e.g., $d=65$ verified in test suites).
> 4. **No Allocation Leaks:** The persistent worker pool reuses memory across training steps, preventing thread spawn overhead and memory fragmentation.
> 5. **AdamW and Gradient Clip Charge Nothing (since 2026-10-01):** `adamw_step` updates the parameter and both moments in place in two passes (check every element finite, then store), and `clip_grad_norm` scales each gradient where it is after the norm pass. Neither uses a buffer or charges the `Budget`, so both succeed on an exhausted budget; before, each charged one buffer (the parameter's length, or the largest gradient's) and refused with `CapacityExceeded` when it did not fit. Both are still all or nothing: a non-finite value, a shared or device target, or a bad layout is refused before anything is written, and the results are bit-identical to the buffered versions. Gated by `adamw_charges_nothing` and `clip_charges_nothing` (`tests/heavy_ops.rs`) and the heap gate in `tests/redteam_ops_heap.rs`.
> 6. **Operands Are Read in Place, Never Copied (since 2026-10-01):** every op reads its tensor inputs where they are: a borrowed slice on the calling thread (the GEMM core's `Mat`, and `scoped` threads where a pass splits), or a clone of the tensor that the pool's tasks share (`validate.rs` `Shared`, one more owner of the storage, freed before the op returns). Inputs are therefore never charged: an op's budget peak is its output plus the scratch its kernel documents, so ops succeed in less room than before, when each operand was copied and the copy charged (`tests/budget_inputs.rs` pins each op's exact peak; `tests/redteam_linear_budget.rs` bounds linear at outputs plus GEMM scratch). Mul, add, the value-residual blend and (since 2026-10-02) causal SDPA forward and backward write straight into their output tensors: SDPA tasks fill disjoint slices of the outputs on scoped threads instead of returning per-task results that were joined and then copied, so its outputs are no longer held twice and its uncharged per-task parts are gone (`tests/budget_inputs.rs` pins the forward at its output plus one task's 24 floats; bits match the earlier build on the blocked and the per-row paths). The NaN refusal is unchanged: every operand is checked at entry, before anything is charged.
> 7. **Each Storage Is Scanned for NaN Once Until It Is Written (since 2026-10-02):** the entry scan goes through `Tensor::all_finite_cached` (ojas-core), which records on the storage that a scan of the whole allocation found every value finite. Later ops on that tensor, its clones and its views skip the scan; a smaller view over a storage not known finite still scans its own window. Every host write (`f32_slice_mut`, `write_f32`) needs sole ownership and resets the record, so a NaN written in place after a successful op is refused by the next one (`tests/redteam.rs` `a_nan_written_in_place_after_a_successful_op_is_refused_by_the_next`; the core test `a_nan_written_after_a_finite_result_is_caught_on_the_next_call` fails if the reset is removed). Outputs built by `fill_out` (mul forward and backward, add forward, the value-residual blend forward, SDPA forward and backward) are recorded finite as they are made; outputs on the copy path (`alloc_out`: linear, silu, rms, rope, gate, cross-entropy, among others) and outputs finite by construction (embedding, permute, the CE gradient, the add backward copies) start unknown and are scanned once by their first consumer, as every operand was before. The finiteness test itself has one owner, `ojas_core::f32_all_finite`. Interleaved A/B on `bench_ops` (6 threads, 9 rounds, min / median, under a load of about 30): embedding forward 0.08 / 0.09 of before (its 50304×768 table was rescanned every call), embedding backward 0.39 / 0.47, mul forward 0.69 / 0.64, mul backward 0.79 / 0.72; linear was within the noise of that load. `bench_ops` reuses its inputs, so these numbers are for operands already recorded finite. An op's first touch of an unknown operand now scans each operand in turn (each split into 1M-value pieces on the pool) where it used to scan all of them in one parallel pass, so an op with several fresh operands under 1M values each scans them serially. The `block` case, whose intermediate activations are fresh every step, measures the net: block step 0.95 of before by min and 0.98 by median, block forward 0.98 / 0.88 (7 interleaved rounds against the pre-flag tree, load 20–28): no regression, within noise of a small gain.
