# PyTorch parity plan

Recorded 2026-10-01 on an Apple M5 Pro running macOS 27. Each claim carries a label:

- **verified**: a command was run, or a line was read, in this session (file:line given).
- **inferred**: reasoned from evidence.
- **unverified**: taken from a document and not re-checked.

The research behind this plan came from six read-only agents plus a per-crate test baseline. Their findings are summarized here, and the load-bearing ones were spot-checked against source.

## 0. Baseline of the working tree before any edit (verified)

`target-baseline/run_baseline.sh` ran `cargo test -p <crate> --release -- --test-threads=1` on the uncommitted tree as it stood.

| crate | passed | failed | ignored |
| :--- | ---: | ---: | ---: |
| ojas-core | 37 | 0 | 0 |
| ojas-cpu | 89 | 0 | 3 |
| ojas-simd | 20 | 0 | 0 |
| ojas-autograd | 28 | 0 | 0 |
| ojas-io | 36 | 0 | 0 |
| ojas-data | 21 | 0 | 0 |
| ojas-infer | 14 | 0 | 0 |
| ojas-oracle | 6 | 0 | 0 |
| ojas-device | 18 | 0 | 0 |
| ojas-kernels | 8 | 0 | 0 |
| ojas-capi | 41 | 0 | 0 |
| ojas-gusset-engine | 0 | 0 | 0 |
| ojas-metal | 66 | 0 | 0 |
| ojas-wgpu | 45 | 0 | 1 |
| **total** | **429** | **0** | **4** |

`docs/status.md` records 24 for `ojas-simd`. That count is from a `--workspace` run, where feature unification enables `accelerate`; run per crate, as here, it is 20. Both are correct.

## 1. The PyTorch features that training and inference actually need

The ranking comes from the call sites in the user's own code (nanolab, the Rust_MLKit sprint trainer, Lappi `qd_train`, tessl_torch). Upstream modded-nanogpt is excluded.

| # | PyTorch feature | Used by | ojas today |
| :--- | :--- | :--- | :--- |
| 1 | Linear / GEMM | all | Has: no-bias `x @ W^T` on CPU, Metal and wgpu |
| 2 | Embedding, tied to LM head | all | Has |
| 3 | RMSNorm (also used as QK-norm) | nanolab, sprint | Has |
| 4 | Causal SDPA with GQA, sliding window, head_dim 64/128/256 | all | Partial: causal only, grouped-query on CPU, Metal and wgpu, Tq = Tk. Metal and wgpu reach D ≤ 256, and Metal is tiled (flash-style) both ways. Sliding window attention is missing (tracked in Task `gp-sliding-window-attention`). Head dim 256 GQA is scoped in Task `gp-head-dim-256-gqa` |
| 5 | RoPE | all | Has. `Backend::permute` moves `[B,T,H,D]` ↔ `[B,H,T,D]` (F1 closed) |
| 6 | Fused or chunked linear + cross-entropy | nanolab, Lappi | Has: `Backend::linear_cross_entropy_mean` tiles rows/vocab without materializing full logits on CPU, Metal and wgpu |
| 7 | Activations: SiLU/SwiGLU, GELU, ReLU², sigmoid | nanolab | Has: SiLU, GELU, ReLU, Sigmoid, Tanh in `ojas-cpu/src/pointwise.rs` |
| 8 | AdamW (fused, fp32 master) | all | Has, one tensor per call |
| 9 | Muon NS5 in bf16, batched | nanolab, sprint | f32 on CPU, Metal and wgpu; not bf16; one matrix per call (bf16 NS5 tracked in Task `gp-muon-bf16-ns5`) |
| 10 | Hand-written LR schedules (cosine, WSD) | all; `lr_scheduler` has 0 sites | Has: `CosineSchedule`, `WsdSchedule`, `LrSchedule` in `ojas-cpu/src/schedule.rs` |
| 11 | `clip_grad_norm_` | all | Has |
| 12 | bf16 autocast / bf16 weights | sprint, Lappi | Optional `Autocast` region, off by default. On CPU, Metal and wgpu it rounds matmul-class f32 operands and activation outputs to bf16. Storage, norms, embeddings, the loss and both optimizers stay f32. Checkpoints stay f32; there is no bf16 weight dtype (bf16 compute tier scoped in Task `gp-bf16-compute-tier`) |
| 13 | Activation checkpointing | nanolab, Lappi | Missing (scoped in Task `gp-activation-checkpointing`) |
| 14 | `torch.save` / safetensors | all | Has: Checkpoint v1, streaming `SafeTensorsWriter`, `replace_dir_with`, `Trainer` checkpoint save and resume |
| 15 | Token-bin data loading (memmap u16/u32) | all; DataLoader has 0 sites | `TokenBin` plus a seeded, epoch-shuffled, resumable `BatchSampler` (`DataCursor`). Supports `u16` and `u32` streams (headerless and FineWeb), so vocabularies > 65536 (Qwen, LLaMA 3) are supported (tracked in Task `gp-tokenbin-u32`) |
| 16 | KV cache, temperature/top-k sampling | nanolab | Has: CPU `forward_token`, `Backend::kv_cache_write`, wgpu KV cache, `ojas-infer` greedy and temperature/top-k/top-p sampler |
| 17 | `torch.cuda/mps` synchronize, memory stats, seeding | about 330 sites | Has: `Backend::sync`, `Budget::peak_bytes()`, `Budget::reset_peak()`, `ResourcePlan`, `ojas-device` profiling, Go `SYSTEM_PROFILE` (opcode 16) |
| 18 | Distributed collectives | sprint only | Missing; not needed on one device |
| 19 | GDN chunk rule + causal conv1d (Qwen3.5) | Lappi | Missing in ojas. Exists Metal-only in tessl (hybrid layer primitives scoped in Task `gp-hybrid-layer-primitives`) |

**The framework layer is landed (verified).**
- `ojas-model` defines the nanolab GPT once (spec, `state_dict` names, order-independent init, the block over `Graph`) with `Trainer<B>` and checkpoint save/resume.
- The Go/C API (`ojas-capi`) drives real weights: `LoadModel`, `OpenTrainer`, `TrainStep`, `SaveCheckpoint`, `Resume`, `GenerateIDs`, `SystemProfile`, `SetMemoryCeiling`.
- Metal and wgpu maintain device-resident activations and report deferred faults at sync.

## 2. Which packages are pure Rust/Go and cross-platform

| Package | Language | Cross-platform today | Notes |
| :--- | :--- | :--- | :--- |
| ojas-core, -cpu, -simd, -autograd, -io, -data, -infer, -device, -kernels, -oracle | Rust (`forbid(unsafe)` except simd/device) | Linux / macOS / Windows (inferred; never built off this Mac) | Accelerate is a macOS-only feature |
| ojas-wgpu | Rust + WGSL | Vulkan / DX12 / Metal (inferred); run only on the Metal HAL | The portable GPU backend |
| ojas-metal, tessl | Rust + MSL | Apple silicon only | ojas-metal builds as a stub elsewhere |
| ojas-capi, ojas-gusset-engine, go/, gusset | Rust + Go/cgo | Unix only. gusset refuses Windows with `compile_error!` | |
| ojas-cuda, ojas-hip | Rust | Probes only, never executed | |
| Rust_MLKit gemma-metal, metal-native, Lappi qd-metal | Rust + MSL | Apple only. They hard-code absolute tessl paths | |
| Rust_MLKit `crates/tessl` | — | — | Dead vendored copy. Nothing depends on it, and it has drifted |

**Duplication to converge.** The canonical owner is listed first in each row.

| Capability | Owner, then duplicates |
| :--- | :--- |
| safetensors | ojas-io, then tessl, qd-metal, qd-export |
| Muon | ojas-cpu/ojas-metal, then metal-native (the most complete GPU version) and burn-port |
| Sampling / KV designs | ojas-infer, then gemma-metal |

## 3. Where PyTorch is the bottleneck: evidence from the user's own records

| Rank | Where | Evidence | Applies to ojas target |
| :--- | :--- | :--- | :--- |
| 1 | GDN chunked delta rule, pure-torch fallback on MPS | About 64% of the Qwen3.5-2B forward at T=1k (`Lappi-decision/HANDOFF/mac-gpu-lane-2026-09-29.md:114-120`) | Lappi only |
| 2 | Attention backward: masked SDPA on CUDA; long-context grad on MPS | 53% of GPU time (`Lappi AUDIT/det-attention-backward-scoping-2026-10-01.md:13-15`). MPS crashes at INT_MAX for T=34,522 (`HANDOFF/realft-2026-09-20.md:774-800`) | All |
| 3 | Dispatch-bound small models on MPS | MPS 3.8× slower than CPU on a 172K-parameter model (`HANDOFF/trainer-2026-09-20.md:71`) | All |
| 4 | Full-vocabulary LM head + CE | 16–19% of the forward. bf16 cuts CE from 4042 to 1114 ms in tessl | All |
| 5 | bf16 correctness | MPS bf16 logits off by up to 0.83. bf16 AdamW `exp_avg_sq` freezes | All |
| 6 | MPS allocator / missing peak-memory API | Cache reached 30 GiB with 16 GiB live | All |
| — | GEMM | **Not** a torch bottleneck: tessl bf16 vs MPS geomean is 1.03× | — |

**Where ojas stood against torch MPS before this session's lanes.** These were the first optimization targets.
- Metal attention backward at B4 H8 T2048 D64: 306 ms for ojas against 60 ms for torch. That measurement was single-run, under load, and not like-for-like.
- Metal per-op fixed cost was quoted as about 1.4 ms. The Metal lane later measured the device wait's floor at 0.19–0.23 ms; the 1.4 ms is a second mode it attributes to GPU contention.
- Linear forward 2048³: 3.86 ms against 2.81 ms.

**After the lanes:** see §8 and [`bench-gpu-vs-torch.md`](bench-gpu-vs-torch.md). The new tiled attention backward measures 16.5–19.4 ms against torch's 25–26 ms at that shape. ojas was faster in 15 of 15 paired rounds.

**Torch MPS profile of one nanolab-default step on this machine (verified, under heavy load).**

The run imported the real `nanolab.model.GPT` and `build_optimizers` (123.7M parameters, 12 layers, d768, 12×64, V50304). nanolab itself never selects MPS (`nanolab/utils.py:29`). Other processes held 50–68% of the GPU throughout, at load average 25–88, so the shares below are the result and the absolute times are pessimistic. Every step was fp32 eager with B=4 and T=1024: B=8 fp32 thrashes memory, while bf16 fits at B=8. The step median was 1531 ms, about 2675 tok/s.

| Share of fp32 step | Component | Why torch is slow there |
| ---: | :--- | :--- |
| 24.6% | Causal SDPA forward and backward (12×) | With grad, MPS runs `_scaled_dot_product_attention_math` and stores the full T×T scores (480 MiB at B8). Under bf16 autocast, q and k arrive as fp32, so the share rises to 37.9% |
| 25.1% | Linear layers (768→768 ×48, 768→2048 ×24, 2048→768 ×12) | GEMM is near peak. Fusing QKV saves about 30% of the q/k/v share |
| 13.8% | Tied lm_head + cross-entropy | The full `[B·T, 50304]` logits are materialized |
| 11.2% | Muon NS5 | 1.69 TFLOP of bf16 at about 8 TFLOP/s, against 20–25 for large GEMM: batched 768-size GEMM efficiency |
| 9.5% | `clip_grad_norm_` | Up to torch 2.14, `linalg.vector_norm` on MPS ran each full reduction on one threadgroup: 12× slower than `square().sum().sqrt()` in fp32 and 25× in bf16 (2.14.1, 2026-10-05). torch 2.15 fixes it (pytorch/pytorch#198611), which cuts this row from 137 ms to 8.5 ms. `foreach` is still refused on MPS |
| ~20% | RMSNorm, QK-norm, RoPE, SwiGLU, gate, value residual | `rms_norm` is fused only for fp32 with no grad. Training splits it into 5 ops |
| 1.0% | AdamW | A per-parameter Python loop, because MPS has no foreach path |

**Decode** at B=1 (prompt 128, generate 128, KV cache) runs about 9 ms/token fp32, or 111 tok/s. Each token issues 783 aten ops, so dispatch overhead is the limit, not compute (inferred). No CPU fallback warnings appeared.

The scripts are in `target-baseline/research/mps_profile/` (`mps_profile.py`, `run_all.sh`, plus probes). The benchmark lane moves the torch side into the repo.

**Targets this sets for ojas, in order:**
1. Flash-style attention forward and backward with no T×T materialization.
2. A one-pass gradient norm (about 3 ms where torch takes 150).
3. Fused chunked linear + CE.
4. Fused elementwise chains.
5. Batched Muon GEMMs.
6. A decode path with few dispatches per token.

GEMM itself is not a target.

## 4. Audit findings: DevMap plus source reads, new beyond docs/audit*.md

| id | sev | where | defect | status |
| :--- | :--- | :--- | :--- | :--- |
| F1 | high | `ojas-core/src/backend.rs`, `ojas-autograd/src/tape.rs` | No transpose/permute op. RoPE `[B,T,H,D]` cannot reach SDPA `[B,H,T,D]`, so multi-head training is impossible through `Tape` | verified |
| F2 | med | `ojas-cpu/src/backend.rs:608-610,636-638,669-670` | AdamW, Muon and clip write outputs one at a time with no up-front ownership check. A partial failure leaves the parameter moved and the moments stale | verified |
| F3 | med | `ojas-wgpu/src/backend.rs:17-27,1672-1749` | Non-finite input returns `Ok` and is reported later. AdamW rolls back silently. This breaks the trait contract and CPU/Metal parity | verified by the auditor |
| F4 | med | `ojas-metal/src/gpu.rs:1586`, `ojas-kernels/src/harness.rs:17-23`, `ojas-wgpu/src/backend.rs:1774` | `fold(0, max)` ignores NaN, so an all-NaN kernel output passes parity | verified |
| F5 | med | `.github/workflows/test.yml:79-83` vs `ojas-wgpu/src/context.rs:189-194` | CI installs lavapipe, but the context refuses CPU adapters | inferred |
| F6 | med | `ojas-autograd/tests/wgpu_tape.rs:125-147` | With no adapter the test prints SKIP and passes | verified by the auditor |
| F7 | med | `ojas-cpu/src/validate.rs:34` | Input host copies are not charged to `Budget`. Peak is about 1.5× the reported amount on CE | verified by the auditor |
| F8, F9 | med | docs | bf16 Muon and tessl kernels are claimed "Verified" but are not what runs. Chunked CE is claimed for the trait, but only the tiny step has it | verified by the auditor |
| F10 | low | `ojas-capi/src/lib.rs:32-56` | Error kind is picked by substring over text that includes user paths, so `busy_model.safetensors` maps to `ErrBusy` | verified by the auditor |
| F11, F12 | low | `ojas-wgpu` | Fault names the lowest bit, not the first op. The fault word can be lost on a poll error | verified / inferred |
| F13, F15 | low | `ojas-kernels/src/harness.rs`, `ojas-wgpu/src/host.rs` | Unwired parity harness and host-slice API. Two signals each: no callers in `rg -uu`, 0 DevMap callers | verified by the auditor |
| F14 | low | `pow_u64`, `clip_scale`, `check_adam` ×3 | Copied per backend; should be one owner in ojas-core | verified by the auditor |
| F16–F18 | low | stale docs | `metal_head_dim_policy`, the greedy extra forward, the scoped-threads comment | verified by the auditor |
| F19 | low | `.gitignore` | `target-*` lanes are not ignored, and DevMap indexes build output | verified |

### Resolution status (2026-10-01, after the lanes)

**Provenance.**
- "Fail → pass" evidence (the test failed on the pre-fix code and passes after) comes from the lane reports. That evidence is not verified independently.
- The coordinator's integration run (`target-baseline/logs-integration/`) verified only the final state: every named test below passes in it.
- Go 27 is lane-reported, against a debug `libgusset`.

| id | resolution | gating test(s) |
| :--- | :--- | :--- |
| F1 | **Closed.** `Backend::permute` on CPU/Metal/wgpu; `Tape::permute` with backward | `ojas-autograd/tests/multihead.rs` (H=3 block gradcheck; it fails when permute is replaced by reshape), `*/tests/permute.rs` on each backend |
| F2 | **Closed.** Every target is checked with `Tensor::ensure_writable_f32` before the first write | `ojas-cpu/tests/optim_atomic.rs` (4) |
| F3 | **Kept by design.** Deferral avoids a sync per op. The contract is now documented in the `ojas-wgpu/src/backend.rs` module docs: the next `sync`, `download` or `clip_grad_norm` returns `NonFinite` naming the first faulting op, and the optimizers write nothing on a non-finite call. A trainer must `sync` before advancing its step counter | `ojas-wgpu/tests/faults.rs` (characterization; it passed before the change too) |
| F4 | **Closed.** Comparators return ∞ on any non-finite value | `ojas-kernels` `non_finite_values_never_pass_parity`, `ojas-metal` `max_abs_does_not_hide_non_finite_outputs` |
| F5 | **Fixed, unverified on Linux.** A CPU adapter is allowed only with `OJAS_WGPU_ALLOW_CPU_ADAPTER=1`, which CI sets | `context::tests::a_cpu_adapter_needs_the_explicit_opt_in`. Linux CI has never run |
| F6 | **Closed.** A missing adapter fails unless `OJAS_ALLOW_NO_GPU=1` | `a_missing_adapter_fails_unless_explicitly_allowed` |
| F7 | **Closed.** Operands are read in place and never copied or charged (since 2026-10-02; `ojas-cpu/tests/budget_inputs.rs` pins each op's exact peak at output plus its own scratch), and since 2026-10-01 the optimizer temporaries: Muon charges its temporaries, and AdamW updates in place with no buffer and charges nothing, so it succeeds on an exhausted budget | `ojas-cpu/tests/budget_inputs.rs`; `ojas-cpu/tests/redteam_ops_heap.rs` (heap peak ≤ charged peak + 64 KiB for `adamw_step` and both `muon_ns5_step` shapes; AdamW peaks under 1 KiB at a 1 MiB parameter); `adamw_charges_nothing`. The capi planner `part_limit_bytes` was corrected too |
| F8, F9 | **Docs corrected** (`op-coverage.md`, README). **F8 measured by the oracle lane:** nanolab's stock bf16 NS5 and an f32 NS5 differ by up to 1.8e-4 nats over steps 1–5 and 1.1e-3 over 40 steps, so parity runs patch nanolab to f32 NS5 | `ojas-oracle/fixtures/tiny/trace_ns5_{f32,bf16}.safetensors` |
| F10 | **Closed in Rust.** Error kinds come from the typed variant. **Open in Go:** `go/ffi.go:534` `inBandSentinel` still matches by substring | `a_user_path_never_selects_an_error_kind`, `error_kinds_follow_the_ojas_error_variant` |
| F11 | **Closed.** The first faulting op is recorded | `the_first_faulting_op_is_named_not_the_lowest_bit` |
| F12 | **Closed.** A failed read keeps the fault; device-lost is never dropped | `a_failed_read_keeps_the_fault_for_the_next_sync`, `device_lost_is_reported_even_past_the_error_cap` |
| F13, F15 | **Closed.** The host-slice API, `math.wgsl`, `gemm_cuda`, `reduce_sum_wgsl` and `ParityOp` were deleted, about 1,090 lines removed, each with two signals of no use. `linear_close` was fixed and wired in | `parity.rs::shared_parity_harness_runs_the_device_kernel` |
| F14 | **Closed.** One owner in ojas-core: `pow_u64`, `clip_scale`, `check_adamw`. The CPU, Metal, wgpu and tiny-step copies were deleted | `ojas-core` `check_adamw_refuses_before_any_update`, `tiny_adamw_refuses_the_configs_the_trait_refuses` |
| F16 | **Closed.** `metal_head_dim_policy` was removed; the test calls the core rule | `ojas-cpu/tests/ops.rs::metal_policy_refuses_above_its_limit_and_cpu_does_not_truncate` |
| F17 | **Closed.** No forward after the last emitted token | `greedy_decode_fits_prompt_plus_n_minus_one_slots` |
| F18 | **Closed** (comments) | — |
| F19 | **Closed.** `/target-*/` is ignored | — |

**Open residue, recorded rather than fixed:**
- **Permute on an empty tensor:** CPU accepts a zero-length axis, while Metal refuses it with `Shape`. This is reachable only through a zero-extent view.
- **Permute with NaN:** wgpu moves the bits and raises the `permute` fault bit, so the error is deferred. CPU and Metal refuse synchronously. Each follows its backend's non-finite contract.
- **Metal i32 plane cap:** `t·d > i32::MAX` returns `Unsupported`. That is tested by a construction check, not at real size.
- **No start-up check that MPP pipelines accept 128 threads per threadgroup.** That would need objc2-metal. It works on the M5 Pro.
- **Linux builds and links, but has never run.** The Lappi session's L-cuda lane cross-checked and linked 67 test executables for `aarch64-unknown-linux-gnu` against the live tree at 22:35Z, including `ojas-cuda --features cuda`. That is reported in [`cuda-backend-scoping.md`](cuda-backend-scoping.md) §1.5 and was not re-run here. No binary has run on Linux. Windows has never been built, and gusset refuses it.
- **CUDA backend scoping** is in [`cuda-backend-scoping.md`](cuda-backend-scoping.md) and scoped for implementation in Task `gp-cuda-backend-provider`. It recommends a whole-step Qwen3.5 provider first, with a general `CudaBackend` as the destination, and lists six asks for the user. It also notes that `ojas-cuda`'s `cuda-13040` binding pin mismatches the GH200's CUDA 12.8, so a missing-symbol call would panic.
- **`commit_resize` (`ojas-cuda/src/lib.rs:148`) is dead in the `cuda` build.** Its test covers a helper the real resize path never calls (L-cuda finding; `cargo check --features cuda` warns).
- **ojas-core has no typed device-lost variant.** `kind_of` still reads backend detail text.
- **Readback-count race (found by the ojas-cpu session).** `device_readbacks()` is process-wide, so "no readback" assertions fail under parallel `cargo test` when sibling tests download (wgpu `muon.rs` 3, `permute.rs` 2). The CI gates pass `--test-threads=1` and are unaffected. **Class fix landed in ojas-core:** `Budget::device_readbacks()` counts per budget tree, from any thread (`budget_readbacks_count_only_their_own_tree_across_threads`). **Pending, in slot (d):** move the GPU tests (wgpu muon/permute/residency/parity/contract, autograd device and wgpu tapes, capi `readbacks_during`, metal training_step) onto it, and verify them under default threads.
- **`CpuBackend` now defaults to `Numerics::Fast`.** A concurrent session made this change with the user's approval. Under Fast, `ojas-infer`'s "cached decode equals full forward" is bit-identical only while every GEMM is below 2^21 multiply-adds; above that the full forward goes to Accelerate. **Since resolved** by the ojas-cpu session (lane-reported; ojas-infer 35 passed):
- `CpuGpt::with_numerics` sets the numerics. Under Exact, cached decode equals the full forward bit for bit, now asserted.
- Under Fast the gap is 3.5e-7.
- One test covers the 2^21-MAC Accelerate cutoff.
- **Attention LSE (tracked in Task `gp-attention-lse-return`):** a forward that emits LSE would remove the backward's stats pass, about 10–25% of the backward time (Metal lane estimate). That needs a trait change.
  - wgpu round 5 timed each backward dispatch: `attn_bwd_prep` reruns the forward, 11.5 of 52.1 ms at [4,12,1024,64] and 30.3 of 141.4 ms at [4,8,2048,64]. LSE from the forward would save about 21% of the wgpu backward (inferred).
  - Measured and reverted: register-blocked score dots (1.4–1.6× slower), dropping dkv's Q reload (≤ 7% of dkv), and a shared P/dS tile (no gain).
- **wgpu `Queue::drop` blocks without bound inside wgpu (wgpu-core 30.0.1 `queue.rs:275-289`; Metal `waitUntilCompleted`).** ojas now drops the queue on a helper thread and waits at most `DROP_WAIT` (2 s). On timeout the queue, its device and one parked thread leak until the GPU finishes. `tests/drop.rs` was still waiting at 1.0 s before the fix and returns at 0.31 s after.
- **capi demo paths over- or under-charge (found by the ojas-cpu budget lane).** Both are scheduled for removal in framework items 11–12, which replace these demo paths.
  - `generate.rs` greedy still holds its own embedding-output reservation, though every backend now charges outputs itself. That is P bytes double-charged during `embedding_forward`. It is safe (fails closed) and below the call's later peak at `linear_forward`.
  - `step.rs` `run_part` returns the gradient as an uncharged `Vec` (within the 3Y bound).
- **Process-wide memory ceiling restored (capi, after item 12).** Item 12's per-session budgets had no shared cap, so 64 sessions could account 64 GiB. Every session budget is now a child of one root `Budget` (`ojas-capi/src/session.rs` `session_budget`; default 1 GiB). A budget above the ceiling is `E_CAPACITY` at load. The ceiling is raised only by opcode 15 / Go `SetMemoryCeiling`, which refuses 0 and refuses while any model is open. Tests: `sessions_together_cannot_pass_the_process_ceiling`, `the_ceiling_changes_only_with_no_model_open`, `load_refuses_a_session_budget_above_the_process_ceiling`, and Go `TestMemoryCeilingRoundTripsAndRefusesWhileAModelIsOpen`.
- **wgpu round 2 (lane-reported timings at load 16–20; tests re-verified):**
  - **SDPA:** tiled FlashAttention-2 in WGSL, 4.7–6.6× over the scalar kernel. It is still about 8× behind torch MPS on forward at [4,12,1024,64]: 11.7 ms vs about 1.4 ms.
  - **GEMM:** a 128×128 register-blocked tile, 1.26–1.56× faster, at about 2.6–3.2 TFLOP/s vs torch's 5.2.
  - **Two real bugs fixed:**
    - `Backend::sync` was not overridden, so trait callers dropped deferred faults. It is now the one owner, at `backend.rs:1035`.
    - A lost device was not named after `destroy()`. `WgpuContext::failure` now polls for up to 5 s for the lost callback, and only on the error path. **Behaviour change:** a failing wait or map can take up to 5 s longer to return.
- **Symlinks were followed on aarch64, arm and powerpc Linux and on the BSDs (fixed, io round 4; security).**
  - **The bug:** ojas-capi and ojas-model each hard-coded `0x20000` as `O_NOFOLLOW` for every Linux target. That is the x86 value; on aarch64, arm and powerpc it is a different flag, so a symlinked model or checkpoint file was opened through the link. The BSDs got no flag at all.
  - **The fix:** a single `ojas_io::open_nofollow`, which does three things:
    - lstat-refuses symlinks and non-regular files before opening;
    - opens with the per-target flag, and refuses any target outside its table;
    - checks that the opened file's (dev, ino) matches the lstat.
  - **Test:** `the_flag_makes_the_kernel_refuse_a_symlink` checks for ELOOP from the kernel. It is verified on macOS aarch64; the Linux values come from libc source and were compile-checked, not run.
  - **Security impact:** this only narrows access; nothing that was refused before is now accepted.
- **A capi call that returned early leaked a deferred fault into the session's next call (fixed).** Found while reviewing the Metal deferred-fault contract.
  - **Failure (verified on a live wgpu session):** a step cancelled after the cross-entropy forward had recorded a NaN returned `E_CANCELLED`. The session's next, clean step then failed with `E_NONFINITE: step: cross_entropy_mean_forward`.
  - **Fix:** `ojas_capi::settled` drains `Backend::sync` at the end of every device step and generate call. A deferred fault outranks the later error, because it was recorded first.
  - **Test:** `an_early_return_never_leaks_a_deferred_fault_into_the_next_call` failed before the fix and passes after; `ojas-capi` passes 45 of 45 serially.
- **wgpu matrix units need a user decision (wgpu round 4).**
  - **What exists (verified from the registry source):** wgpu 30 exposes `Features::EXPERIMENTAL_COOPERATIVE_MATRIX`. On this M5 Pro, Metal reports 8×8×8 configurations for f32, f16, and f16 inputs with an f32 accumulator.
  - **The blocker:** the feature is gated behind `unsafe ExperimentalFeatures::enabled()`, and `ojas-wgpu` is `#![forbid(unsafe_code)]`.
  - **Design:** a GEMM and SDPA design exists in the wgpu lane's round-4 report, but nothing is implemented.
  - **Subgroups, measured and dropped:** a subgroup SDPA path was 0.98–1.02× of the portable kernel and was removed. A barrier-ceiling probe found under 10% possible in the CE and norm reductions.
- **wgpu fault word is full (wgpu round 3).** All 32 op bits are in use; the last four went to `accumulate_grad`, `linear_cross_entropy_mean`, `cached_attention_forward` and `kv_cache_write`. A 33rd fault-reporting op needs a second word or a 64-bit word first.
- **wgpu `clip_grad_norm` returned another caller's norm under concurrency (fixed, wgpu round 3).** Job scratch went back to the pool before the read. The status buffer is now held until after the read. `clip_norms_stay_correct_while_other_threads_submit` failed 5/5 before the fix and passes 5/5 after.
- **Fused linear CE tolerance (Metal round 4).** A chunk that splits the problem changes the order the gradients are summed in. Against an f64 reference, Metal's fused result is as close as, or closer than, the unfused composition, but it differs from the composition by up to 4.8e-6 of the largest entry. The gate compares both against f64: the fused error may be at most 1.25 × the composition's error + 1e-7, and the fused result must be within 1e-5 normwise of the composition. When the chunk covers the whole problem, the result is bitwise equal to the composition. wgpu's fused result is within 2.35e-7 of the composition, after its GEMM accumulate was changed to start from C.
- **kv-cache limits on wgpu.** `cached_attention_forward` refuses Tq·H > 65535 rather than folding the grid. `kv_cache_write` refuses a uniquely owned cache view with a non-zero byte offset, which is stricter than the trait requires.
- **io:** the single-file `replace_with` still replaces a symlink as a link (documented). Only `replace_dir_with` refuses symlinks. The `ojas-io/README.md` names `MAX_HEADER_SIZE` and `OjasError::InvalidSafetensors`, which predate this work and are wrong.
- **autograd:** ojas-autograd has no ojas-metal dev-dependency, so G6 (seeded backward) is verified on CPU, the test doubles and wgpu, but not on Metal. Adding one changes Cargo.lock. Test helpers (`Resident`, `Rng`, `lin`, `ce`) are duplicated across `tests/common/mod.rs`, `device_tape.rs` and `gradcheck_random.rs`.
- **Metal deferred faults landed (Metal round 5).** Approved by the user; contract in [`metal-deferred-faults.md`](metal-deferred-faults.md).
  - **Waits per training step:** about 3,358 before, 71–72 after, for an `ojas-model` 124M step with K=4. The before figure is inferred as one wait per op; the after figure is verified. Most of the remaining waits are 28 host uploads and about 41 memory-cap commits.
  - **Interleaved A/B** (lane-reported):
    - `adamw_full`: 183→22.6 ms min;
    - `block_fwd_bwd`: 125.9→82.6 ms min;
    - one tiny op: 0.37→0.007 ms.
  - **Tests:** ojas-metal 149/149 and ojas-capi 46/46, both verified, including a Metal early-return leak test that fails without `settled`.
  - **Remaining:** upload waits. tessl has no unwaited host write for a fresh buffer.
- **Metal per-op wait was the floor before round 5 (Metal round 3).** AdamW over 170 tensors is about 1 ms per tensor of synchronous waits. Removing them means adopting wgpu's deferred-fault contract, which needs user sign-off because it changes C/Go error timing.

**DevMap gaps recorded during the audit:**
- The index went stale mid-session.
- `dead_symbols` returned `walk_incomplete` with every row at 0.40.
- `clones` missed the renamed near-copies in F14.
- Metal/WGSL kernels are launched by string name, so host-to-kernel edges do not exist; C++/Metal net resolution is 36/1000.

## 5. Work queue

The order follows the user's sequence: correctness and framework gaps that block training a real model first, then the measured ojas-vs-torch gaps, then breadth.

**Lanes share the checkout.** Each owns its crates and uses its own `CARGO_TARGET_DIR=target-<lane>`. Edits to `ojas-core/src/backend.rs`, `Cargo.toml` and `Cargo.lock` are serialized through the coordinator.

| Lane | Owns | Work |
| :--- | :--- | :--- |
| L1 layout | ojas-core trait, ojas-autograd, plus the matching op in cpu/metal/wgpu | F1: `permute`/`transpose` + `contiguous` with backward on all three backends and on `Tape`; a multi-head gradcheck. Hoist the F14 helpers into ojas-core |
| L2 cpu-harden | ojas-cpu | F2: validate every output target before the first write. F7: charge input copies |
| L3 wgpu-harden | ojas-wgpu, ojas-kernels, CI | F3: decide sync vs deferred non-finite and make the contract say which. F4 (wgpu and kernels). F5/F6. F11/F12. Muon on wgpu (f32 NS5 to match CPU/Metal) |
| L4 metal-perf | ojas-metal | F4 (metal). Attention backward from 306 ms toward torch's 60 ms. Per-op fixed cost. head_dim 128. **Done:** tiled forward and backward, D ≤ 128 (§8) |
| L5 infer/io/data | ojas-infer, ojas-io, ojas-data | Sampling (temperature/top-k/top-p, seeded). RoPE with position offset in the infer model. BF16/F16 safetensors decode. Batch sampler over `TokenBin` |
| L6 bench | `ojas-metal/benches`, `ojas-wgpu/tests/bench*`, `scripts/` | Paired, interleaved A/B harness: ojas Metal and wgpu against torch MPS at nanolab shapes, per kernel plus full step, min-of-N, using tessl's `paired_cross_runtime.py` protocol. Runs serialized after the other lanes, so timings are not taken under GPU contention |

The next phase is the framework layer: a multi-layer, multi-head GPT on `Tape` with device-resident optimizer state, wired to checkpointing and to a Go `Load`/`Step`/`Generate` that uses real weights. It depends on L1.

## 6. Done-gate

1. Every crate is green in release with `--test-threads=1`, and the total is at least the 429 baseline. No test is deleted or weakened.
2. Every fix ships with a test that fails on the pre-fix code.
3. Each F-finding is closed with a test, or recorded here with the reason it stays open.
4. Each benchmarked kernel reports an ojas-vs-torch-MPS ratio from the paired protocol, with spread. A row with spread above 10% is flagged rather than quoted.
5. `docs/status.md` counts are replaced with this run's counts.

## 8. GPU benchmark verdict against torch MPS (2026-10-01)

The full tables, method and raw data are in [`bench-gpu-vs-torch.md`](bench-gpu-vs-torch.md) and `bench/results/2026-10-01/`. The harness is `bench/run_paired.sh`.

The paired protocol ran 5 rounds × 3 runs, alternating which runtime went first, and every row passed parity against torch. **But the GPU was 84–100% busy with unrelated work** (ollama `llama-server`, Chrome), so the 10% spread rule flagged 83 of 88 rows in run A and 88 of 88 in runs B and C. Done-gate item 4 is therefore **not met**:
- **Direction** is verified only where all 15 per-round ratios agree.
- **Magnitudes** below are run A medians, with the 15-round range as the uncertainty.
- A quiet-GPU re-run is required before any ratio is quoted as a number.

In this table, a ratio is torch time / ojas time, so below 1 means ojas is slower.

| | Metal | wgpu |
| :--- | :--- | :--- |
| **One nanolab block, forward + backward** | **0.37×** [0.34–0.60], slower 15/15 | **0.18×** [0.15–0.21], slower 15/15 |
| Linear (4 shapes) | 0.46–0.78× | 0.29–0.38×, slower 15/15 |
| SDPA forward | 0.50–0.61× | 0.01–0.02× |
| SDPA backward, T1024 and T2048 | **1.24× and 1.38×, faster 15/15** | 0.04× |
| Cross-entropy forward / backward | 0.97× / 0.35× | **2.79×** / 0.53× |
| `clip_grad_norm`, 123.7M params | **4.58× faster** against torch 2.13; about 0.45× against the 2.15 nightly (unpaired) | **6.36× faster** against torch 2.13; about 0.44× against the 2.15 nightly |
| AdamW, full parameter set | 0.09× | 0.70× |
| Muon against torch fp32 / bf16 NS5 | 0.71–0.77× / 0.44–0.45× | 0.41–0.44× / 0.25–0.27× |
| RoPE forward / backward | mixed | **3.67× / 4.73× faster** |

**Worst rows and their causes.** The bench lane read these from source.

1. **wgpu SDPA.** One query row per lane with scalar dot products (`attention.wgsl`). The fix is tractable: port the tiled design that sped up Metal.
2. **Metal AdamW.** Per tensor: 4 input checks, 3 copies out, 3 output checks, 2 device waits, 3 copies back. The fix is tractable: fused multi-tensor step, fewer copies.
3. **Metal QK-norm weight gradient.** `ojas_rms_bwd_w` runs 64 threads over 49,152 rows in series. The fix is tractable: a two-stage reduction, as wgpu does.
4. **Metal per-op cost.** A status buffer, finite-check passes and a device wait on every op; the block issues 53 synchronous ops. Fixing this is a contract decision: defer faults to `sync` as wgpu does (`framework-design.md` §9).
5. **wgpu GEMM.** Scalar vec4 FMA tiles with no matrix units, about 1.6 TFLOP/s against torch's 5.2. WGSL has no portable cooperative-matrix path today, so this is largely a ceiling of the portable backend rather than a bug (inferred).

`residual_add_bwd` scores 0.01× only because the API returns two gradient copies where torch returns the input tensor. It is an artefact of the API, not a slow kernel.

## 7. Framework layer (completed)

The design is in [`framework-design.md`](framework-design.md).
- `ojas-model` crate implemented: nanolab GPT architecture written once (spec, init, Graph block, Eval, and `Trainer<B>`).
- Five `Backend` additions landed across CPU, Metal, and wgpu: `sync`, `accumulate_grad`, fused chunked `linear_cross_entropy_mean`, `cached_attention_forward` and `kv_cache_write`, plus blanket impls for `&B` and `Arc<B>`.
- `Tape` extensions landed.
- Device-resident `Trainer<B>` with checkpoint directory save/resume verified.
- In-process Go API implemented via `ojas-capi`: `LoadModel`/`NewModel`, `OpenTrainer`, `TrainStep`, `SaveCheckpoint`, `Resume`, GPT-2 tokenizer, and `GenerateIDs`.
- Follow-up prioritized work has been converted into active GitPulse task briefs in `tasks/` (`gp-bf16-compute-tier`, `gp-head-dim-256-gqa`, `gp-tokenbin-u32`, `gp-cuda-backend-provider`, `gp-attention-lse-return`, `gp-activation-checkpointing`, `gp-muon-bf16-ns5`, `gp-hybrid-layer-primitives`, `gp-sliding-window-attention`).
