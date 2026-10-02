# L-cuda: scoping an ojas CUDA backend for Lappi on the GH200 (design only, 2026-10-01)

> **Provenance:** this is a copy of `Lappi-decision/AUDIT/ojas-training-2026-10-01/cuda-backend-scoping.md` (Lappi main `226f74f`), written by the Lappi session's L-cuda lane and placed here beside [`framework-design.md`](framework-design.md) at its request. The ojas coordinator read it in full. Nothing in it has been implemented, and its asks (§6.3) await the user. Relative paths below are relative to the Lappi repo unless prefixed `ojas/`.

Lane L-cuda. Design and scoping only: no kernel was written, no GPU was used, nothing was run on the
box, ojas and tessl were not edited. The user decided on 2026-10-01 to start scoping the ojas CUDA
backend now, as CPU-only work. Fable had advised deferring it (`fable-advice.md` Q2, human ask 6).
This document is that scoping. The lead relays it to the ojas coordinator, which wants it beside
`ojas/docs/framework-design.md`.

**Labels.** **[V]** means I read the line or ran the command (file:line or the command is given).
**[R]** means reported by a document or a test I did not run. **[I]** means inferred, with the
reasoning shown. **[U]** means unverified.

**Paths.** `ojas/` is `/Users/bharath/Code/research/ojas`, `tessl/` is
`/Users/bharath/Code/research/tessl`, `cudarc/` is
`~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.19.10`. Other paths are relative to
the Lappi repo.

**Navigation record.**
- DevMap: ojas index gen 2114 and tessl gen 873, both fresh [V `devmap_status`]. I used `devmap_skeleton` for `ojas-core/src/backend.rs` and `tessl/src/qwen35_train.rs`. Rust call resolution is low: net 338‰ on ojas and 318‰ on tessl, and skeleton signatures were "not extracted". So the inventory below comes from file reads and `rg`, not graph edges (`GAP-L-CUDA-INVENTORY-FROM-READS-2026-10-01`).
- GitPulse on ojas: 1 worktree, 103 unstaged and 97 untracked files, 0 overlapping (one worktree, so same-worktree collisions are invisible) [V `gitpulse_insights`].
- GitPulse on this Lappi worktree: `REPOSITORY_TRUST_REQUIRED` on every facet (`GAP-L-CUDA-GITPULSE-UNTRUSTED-2026-10-01`).
- No `ListAgents` tool was exposed to this lane (`GAP-L-CUDA-NO-LISTAGENTS-2026-10-01`). The live ojas coordinator is known only from the lead's message.

---

## 0. Summary

**Recommendation.** Build **(B) first**: a whole-step Qwen3.5 training provider on CUDA, the CUDA
counterpart of the Metal `ojas-qwen35` provider (Metal via tessl) that lane L-ojas-qwen35 is writing.
It mirrors tessl's step seam: `train_forward` → `PendingStep::hidden` → `train_backward_into` into a
bank, then `grad_sq_norm` and in-place AdamW.

Make **(A)**, a general ojas CUDA `Backend` over the existing 25 trait ops plus T1–T6, the
**destination**. The two converge in three places:
- one CUDA runtime crate;
- one CUDA-C kernel source module, which (A)'s trait methods later call;
- one step-provider trait, which the trainer is generic over.

(A) alone cannot run a single Qwen3.5 step, even with T1–T6. About ten Qwen ops are missing from the
trait (§2.A.3), and each needs a trait edit, which the coordinator owns.

**Fastest to "Lappi trained on the GH200" (quick rows, rungs a–d):** (B), on the correctness path:
- tessl's sequential GDN ported as is;
- cuBLAS for GEMM if the human allows its cudarc feature, otherwise a hand-written `mma.sync` GEMM;
- a deterministic FA2-style attention at D=256 with GQA.

That is **~35–45 specialist engineer-days, ~2.5–3 weeks wall across 4–5 lanes** [I, §3 table]. It
lands at a step speed that is **~5–10× slower than the PyTorch campaign** [I, §3.2 arithmetic],
because tessl's GDN is a token-sequential scan.

**Campaign speed** (parity with PyTorch's 15–26k positions/s) additionally needs:
- fla's chunked GDN as hand-written CUDA (the largest single item);
- the GEMM and attention performance tiers.

That adds **~25–35 days**, so **60–80 days** to a CUDA path that could replace PyTorch for 3-seed
campaigns [I]. Fable's 30–50 days (`fable-advice.md:36`) roughly covers the correctness path only.

**Top risks** (§7):
1. GDN speed, and the chunked GDN backward on Hopper, where fla has already shipped a wrong-gradient bug (#640).
2. The toolchain and ABI: cudarc's binding pin, a missing-symbol panic, and the fact that no ojas CUDA kernel has ever run on a real GPU.
3. There are no gaps in the GH200 queue. Every debug cycle is an inserted item on a box with no Rust toolchain.
4. Placement and ownership: the ojas tree is live, and rule 6 places "kernel work" in tessl.
5. cuBLAS bf16-in/f32-out is unverified.

**Human asks** (§6.3): placement and rule-6 scope; enabling cudarc's `cuda` feature with the binding
pin moved to `cuda-12080`; the cuBLAS feature (yes or no); GH200 minutes (insert into the post-F
queue, or wait until after item 10); ojas file ownership with the coordinator; the speed target
(correctness path or campaign speed).

---

## 1. What exists

### 1.1 `ojas-cuda`: a probe, not a backend

- **Feature gate.** `cuda = ["dep:cudarc"]`, off by default [V `ojas/ojas-cuda/Cargo.toml:10-13`].
- **cudarc pin.** `cudarc = "=0.19.10"` with `default-features = false` and features `std, driver, nvrtc, fallback-dynamic-loading, cuda-13040` [V `Cargo.toml:21-29`]. cuBLAS is deliberately off (the comment on line 20). cudarc is already in ojas's `Cargo.lock` [V: the `--locked --offline` cross-check in §1.5 compiled `cudarc v0.19.10` from it; also `fable-advice.md:36`].
- **What it does.**
  - `CudaDevice::open()` returns `NotCompiled` without the feature [V `ojas-cuda/src/lib.rs:33-37`].
  - With the feature it probes `libcuda`/`libnvrtc` by dlopen, because cudarc panics otherwise (`:86-101`). It opens `CudaContext::new(0)` and launches one affine kernel `y = x*scale + bias` as a self-test (`:45-58`).
  - The kernel is compiled by **NVRTC at run time with `fmad: Some(false)`** (`:231-236`), so ojas already chose NVRTC and FMA-off for its only CUDA kernel.
  - It waits on an event with a bounded 30 s poll (`:332-337`, `:181-197`) and maps `CUDA_ERROR_OUT_OF_MEMORY` (2) to `Capacity` (`:129-141`).
- **Not a `Backend`.** It "does **not** implement the `ojas_core::Backend` trait and is not reachable from the Go C-ABI runtime" [V `ojas-cuda/README.md:28`, `ojas/README.md:24,199`].
- **Never run on a GPU.** ojas's own verification table records "CUDA Execution — Skipped: No NVIDIA GPU present". The bindings were type-checked only [R `ojas/docs/backends.md:122-131`; "not re-run on 2026-10-01"].
- **Side finding for the coordinator (no edit made).** `commit_resize` (`lib.rs:148`) is dead in the `cuda` build: `cargo check --features cuda` warns "function `commit_resize` is never used" [V, command in §1.5]. The production resize at `lib.rs:287-300` does not call it, so `failed_resize_keeps_the_previous_buffers` (`:368-398`) tests a helper the code never runs.
  - The production path still computes all three new buffers with `?` before assigning any of them, so it appears safe [I].
  - The test is tautological as a guard on the real path.

### 1.2 `ojas-hip`: a copy probe

- `hip = ["dep:hip-runtime-sys"]` (`=0.1.2`), off by default [V `ojas/ojas-hip/Cargo.toml:13,20`]. It does "not implement the `ojas_core::Backend` trait and carries no compute kernels" [V `ojas-hip/README.md:28`]. Its `--features hip` build stopped in `hip-runtime-sys` with no `/opt/rocm` [R `backends.md:132`].
- **Relevance here:** `ojas-kernels/src/source.rs:1-2` says the CUDA-C source "is written so a future hiprtc path can reuse it unchanged" [V]. Any inline PTX (`mma.sync`, `wgmma`, `cp.async.bulk`/TMA) in the kernels below breaks that intent. See the GEMM tiers in §3.1.

### 1.3 The `Backend` trait, and T1–T6

- **The trait** is `ojas-core/src/backend.rs:380-636` [V]. It has 25 ops beyond `id`/`budget`/`numerics`/`upload`/`download`:
  - `permute`;
  - embedding fwd/bwd, linear fwd/bwd, `rms_norm` fwd/bwd;
  - `rope_half_split` fwd/bwd, `rms_qk_norm` fwd/bwd, `causal_sdpa` fwd/bwd;
  - `per_head_sigmoid_gate` fwd/bwd, `value_residual_blend` fwd/bwd;
  - silu, mul and residual_add, each fwd/bwd;
  - `cross_entropy_mean` fwd/bwd;
  - `clip_grad_norm`, `adamw_step`, `muon_ns5_step`.
- **The trait is nanolab-shaped:**
  - linear weights are `[out,in]` with no bias;
  - RoPE is the half-split over the full head;
  - the per-head gate is `[n_head, d_model]`;
  - CE is `[rows × vocab]`;
  - `METAL_MAX_HEAD_DIM = 128` (`:47-54`, `:353-379`).
- **`BackendId::Cuda` already exists** (`:13-19`) [V].
- **Numerics contract** (`:21-34`, `:386-393`) [V]. `Exact` means ascending-index f32 reductions, no FMA, bits independent of thread count. `Fast` may fuse or reorder, but "the bits still must not depend on the thread count". A GPU backend that is not ascending-index-without-FMA "must override this to return `Numerics::Fast`".
  - A CUDA backend is `Fast` by this definition [I].
  - Its geometry must not depend on the device's SM count. cuBLAS's own reproducibility guarantee is only "on GPUs with the same architecture and the same number of SMs" [R cuBLAS 13.4 docs, "Results reproducibility"]. Ported to ojas's contract, that is a cross-device caveat a hand-written fixed-geometry GEMM does not have.
- **Residency needs no core edit.** `DeviceBuffer` (`ojas-core/src/tensor.rs:19-33`: `backend`, `byte_len`, `read_bytes`, `as_any`) and `Tensor::from_device` (`tensor.rs:245`) are how Metal (`ojas-metal/src/backend.rs:58`) and wgpu (`ojas-wgpu/src/context.rs:1124`) hold device memory [V]. A `CudaBuffer: DeviceBuffer` in `ojas-cuda` fits as is. `DType` already has `Bf16` (`ojas-core/src/dtype.rs:5-10`) [V].
- **T1–T6** are proposals, not code. `rg` finds no `fn sync`, `accumulate_grad`, `linear_cross_entropy_mean`, `cached_attention_forward`, `kv_cache_write` or blanket `impl<B: Backend` in `ojas-core/src` (exit 1) [V]. As written at `ojas/docs/framework-design.md:192-221` [V]:
  - **T1** `sync()`, default `Ok(())`;
  - **T2** `accumulate_grad(acc, grad)`;
  - **T3** `linear_cross_entropy_mean(input [N,d], weight [V,d], targets, ignore_index, chunk_rows, want_grad)`;
  - **T4** `cached_attention_forward` (GQA folded in);
  - **T5** `kv_cache_write`;
  - **T6** blanket impls for `&B` and `Arc<B>`.
  - T3–T5 default to `Unsupported`.
  - Gates G1–G5 (`:217-221`).
  - The framework layer itself is "proposal, not implemented … awaiting a go/no-go" (`:1-15`).

### 1.4 `ojas-kernels`: CUDA sources

The **only** CUDA-C text in ojas is `affine_cuda()` [V `ojas/ojas-kernels/src/source.rs:33-48`].
Everything else is WGSL:
- 11 modules, 1,263 lines [V `wc -l ojas-kernels/src/wgsl/*.wgsl`];
- attention is capped at `ATTENTION_MAX_HEAD_DIM = 128` (`geometry.rs:146`);
- `GEMM_TILE = 64` (`geometry.rs:127`).

The crate is device-agnostic: geometry, source text and a parity harness (`max_abs`, `splitmix_f32`,
`linear_close`) [V `lib.rs:1-18`, `harness.rs`]. That makes it the natural home for a `cuda/` source
module both designs share (§2.C).

### 1.5 Can ojas build for aarch64 Linux today? Builds and links; never run on Linux

ojas's own docs say "**Linux and Windows have never been built**" [V
`ojas/docs/pytorch-parity-plan.md:187`; also `:73` "inferred; never built off this Mac"]. The cfgs
are clean:
- macOS-only code sits behind `cfg(target_os = "macos")`: Accelerate in `ojas-cpu`, `ojas-metal` (`lib.rs:28,48`), `ojas-device/src/host.rs:54`, and parts of `ojas-capi`;
- each has a Linux or `not(macos)` arm where needed (`ojas-device/src/host.rs:59`, `ojas-capi/src/load.rs:121,269`, `owner.rs:32`);
- `ojas-cpu/Cargo.toml:18` has a `cfg(not(target_os = "macos"))` dependency table [V `rg`].

I measured it today, 2026-10-01T22:35:08Z, against the **live working tree** (103 unstaged and 97
untracked files; not a commit). I used `--locked --offline` so nothing in ojas was written, and the
target dir was in this session's scratchpad:

| Command (run on the Mac, target dir in scratchpad) | Result |
| --- | --- |
| `cargo check --locked --offline --manifest-path ojas/Cargo.toml --target aarch64-unknown-linux-gnu -p ojas-core -p ojas-kernels -p ojas-device -p ojas-cuda --features ojas-cuda/cuda` | **Finished**; one warning: `commit_resize` never used [V] |
| `cargo check … --target aarch64-unknown-linux-gnu --workspace --all-targets` (default features) | **Finished, no warnings or errors** [V] |
| `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" cargo test --locked --offline --no-run … --workspace` | **67 test executables linked** for aarch64 Linux against the box's glibc 2.39 sysroot [V] |
| same, `--release -p ojas-cuda --features ojas-cuda/cuda` | ELF 64-bit aarch64 PIE, `NEEDED libgcc_s.so.1, libc.so.6` only (`llvm-objdump -p`), sha256 `040df619…5c5` [V]. libcuda and libnvrtc are dlopened at run time, so the binary links with **no CUDA on the Mac** |

**Answer:** ojas **builds and cross-links** for `aarch64-unknown-linux-gnu` today, including
`ojas-cuda --features cuda`. It has **never been run on Linux**. One attempt to run the CPU test
binaries in a local arm64 container was skipped: the podman machine was stopped [V `podman machine
inspect` → `stopped`], and starting a shared VM was out of scope
(`GAP-L-CUDA-OJAS-LINUX-LINKED-NOT-RUN-2026-10-01`). "Linked" is not "works".

### 1.6 The target box

The lead's facts (read 2026-10-01; not re-read here) [R]:
- GH200 480GB, sm_90, driver 580.105.08, 97,871 MiB HBM visible;
- aarch64, Ubuntu 24.04.4;
- CUDA toolkit 12.8 (`nvcc` V12.8.93 at `/usr/local/cuda`);
- no Rust toolchain;
- the PyTorch campaign holds it for about 40 h.

From the repo:
- **Python environment.** The campaign runs in a uv venv installed from `stack/train.lock`, not an image [V `HANDOFF/gh200-2026-09-20.md:17-27`: "`uv pip install -r stack/train.lock` into a 3.12.14 venv"]. That lock pins `torch==2.10.0+cu128`, `flash-linear-attention==0.5.2`, `fla-core==0.5.2`, `causal-conv1d==1.7.0`, `triton==3.7.1` [V `stack/train.lock:12,42,44,174,188`].
- **cuBLAS and NVRTC in the venv.** The lock also pins `nvidia-cublas-cu12==12.8.4.1` and `nvidia-cuda-nvrtc-cu12==12.8.93` [V `stack/train.lock:91,98`]. So `libcublas.so.12` and `libnvrtc.so.12` are **probably present inside the venv's `site-packages/nvidia/*/lib`** [I from the lock plus the "no failures" install]. Their presence under the host's `/usr/local/cuda/lib64` is **[U]**, and the box was not read (`GAP-L-CUDA-BOX-LIBS-UNVERIFIED-2026-10-01`).
- **Driver API level.** The 580 driver series exposes the CUDA 13.0 driver API [I]. `nvidia-smi`'s "CUDA Version" line was not read [U].
- **The ojas binding pin is a mismatch.** cudarc's `cuda-13040` binding set (`ojas-cuda/Cargo.toml:28`) does not match the box's 12.8 libraries [V].
  - cudarc still finds them: its Linux name list includes `lib{name}.so.{major}`, `.so.12`, `.so.11` and `.so.1` regardless of the binding set [V `cudarc/src/lib.rs:204-243`, lines 234-240].
  - Symbols load lazily on first call, and a missing one **panics**: `panic!("Missing symbol {name}: {e}")` [V `cudarc/src/nvrtc/sys/mod.rs:11-13`; the per-call `OnceLock` pattern at `:139` onward].
  - So calling any 13.x-only entry point against 12.8 libraries aborts the process instead of returning an error [I].
  - **Fix:** `cuda-12080` [V the feature exists, `cudarc/Cargo.toml` `[features]`]. It is a one-line change in a coordinator-owned file (ask 2).
- **Price.** $2.29/h [V `HANDOFF/post-f-queue-2026-10-01.md:235`: "capped 32,400 s = $20.61"].

### 1.7 tessl's training step: what (B) mirrors

- **Scope and numerics.** One unpadded sequence at a time. f32 weights, activations and gradients, with GEMMs either exact f32 or **bf16 operands with f32 accumulation** (`GemmOperands`). Every reduction in a fixed order, so "a step's gradients are the same bits on every run". Each layer is rebuilt from its saved residual input before its backward [V `tessl/src/qwen35_train.rs:1-34`; `gemm.rs:996-1042`].
- **The seam.**
  - `train_forward(ids, operands, Supervise) → PendingStep` (`:534-695`);
  - `PendingStep::hidden(positions, out)`, for a loss outside tessl such as the span head (`:302-340`);
  - `train_backward_into(p, dh: Option<(&[u32], &Tensor)>, bank, accumulate)` (`:697-717`);
  - `train_step_into` (`:473-483`);
  - `Supervise::{Causal, Rows{positions, targets, scale}}` (`:245-263`);
  - `AdamW::grad_sq_norm` (fixed-order f32 on GPU, rows summed in f64) [V `qwen35_adamw.rs:438-441`];
  - in-place torch-semantics AdamW with per-entry weight decay [V `qwen35_adamw.rs:1-45`].
- **No step-provider trait exists yet.** Lappi's `crates/qd-train` is a doc-only skeleton [V `crates/qd-train/src/lib.rs:1-7`], and `ojas/ojas-qwen35` is not on disk yet [V `ls` → no such directory]. The trait is Fable's proposal (`fable-advice.md:11`).
- **Every kernel the step launches** [V `qwen35_train.rs:600-695`, `:740-832`, `:946-1455`]:
  - **Forward:** `embed_rows`; `rms_norm` (Qwen `1+w`); GEMM `nn`; `conv1d_silu`; `copy_cols`; `gdn_gates`; `gdn_train_forward`; `gated_rms_norm`; `attn_qk_norm_rope`; `attn_train_forward` (+LSE); `attn_output_gate`; `swiglu`; residual (GEMM + add); `cross_entropy_rows`; `scatter_add_rows`; `ce_gather_rows_f32`.
  - **Backward:** GEMM `tn`/`nt`; `swiglu_bwd`; `residual_add`; `rms_norm_bwd`; `gated_rms_norm_bwd`; `gdn_train_backward`; `gdn_gates_bwd`; `conv1d_silu_bwd`; `attn_gate_bwd`; `attn_train_backward`; `attn_qk_norm_rope_bwd`; `embed_rows_bwd`; `deliver` (copy/add); `zero_f32`.
  - **Optimizer:** `qwen35_adamw`; `grad_sq_norm`.
- **The bench's 12 timed ops** at T=2048, bf16 operands, M5 Pro on battery [V `tessl/bench/results/qwen35_train_step_bf16_m5pro.txt:19-36`]:

  | Op | ms |
  | --- | ---: |
  | gdn_train fwd | 9.0 |
  | gdn_train bwd | 37.1 |
  | attn_train fwd | 7.3 |
  | attn_train bwd | 25.2 |
  | CE + grads | 1,114 |
  | rms_norm_bwd | 0.93 |
  | gated_rms_norm_bwd | 1.18 |
  | swiglu_bwd | 1.07 |
  | conv1d_silu_bwd | 6.7 |
  | attn_qk_norm_rope_bwd | 1.4 |
  | gdn_gates_bwd | 0.75 |
  | embed_rows_bwd | 0.43 |

  The whole step is 5.94 s (345 tok/s). That full-vocab CE is the causal-LM bench. Lappi's `Supervise::Rows` scores a handful of rows per sequence, so CE is negligible for Lappi [I].
- **Shapes** [V `crates/qd-export/tests/fixtures/qwen35_2b_base_config.json`]:
  - 24 layers, 18 `linear_attention` + 6 `full_attention` (every 4th);
  - hidden 2048, MLP 6144;
  - attention 8 query heads : 2 KV heads at head_dim **256**, with an output gate;
  - GDN 16×128 key and 16×128 value heads, conv 4;
  - partial rotary 0.25, `mrope_section [11,11,10]` interleaved;
  - vocab 248,320, tied.
- **Parameters** [I, computed from the config]:
  - GEMM weights per GDN layer: 2048×8224 + 2048² = 21.0 M;
  - per attention layer: 2048×5120 + 2048² = 14.7 M;
  - MLP: 3×2048×6144 = 37.7 M per layer;
  - tower: **1.373 B**; embedding: **0.509 B**; total **≈1.88 B**;
  - f32 copy: 7.5 GB;
  - f32 weights + grads + two AdamW moments: **≈30 GB** of the GH200's ~95 GiB.

---

## 2. Two designs

### 2.A A general ojas CUDA `Backend`, against the existing trait plus T1–T6

#### 2.A.1 What it is

`CudaBackend` in `ojas-cuda`, behind `--features cuda`:
- `impl Backend` for the 25 ops in §1.3 plus T1–T5 (T6 is the blanket impl in core and needs only gate G5);
- `CudaBuffer: DeviceBuffer` for residency (§1.3, no core edit);
- `upload`/`download` overridden as Metal and wgpu do;
- `numerics()` returns `Fast` (§1.3);
- one stream, stream-ordered execution, errors surfacing at T1 `sync`.

This answers `framework-design.md`'s "hardest open question" (`:254-258`) for CUDA: CUDA adopts
wgpu's deferred-fault contract natively, so the ~1,000 trait calls per micro-step do not each pay a
blocking round trip [I].

#### 2.A.2 T1–T6 on CUDA

| Add | CUDA implementation | Gate (from `framework-design.md:217-221`) |
| --- | --- | --- |
| T1 `sync` | `stream.synchronize()` plus a device fault word, as wgpu's `fault.wgsl` does. A non-finite flag written by kernels is read here | G1: a deferred NaN surfaces at `sync` |
| T2 `accumulate_grad` | Elementwise `acc += grad`, one thread per element (exact: one rounding), NaN leaves `acc` unchanged via a pre-check reduction | G2 |
| T3 `linear_cross_entropy_mean` | Chunked logits with an online log-sum-exp: the same algorithm as tessl's `cross_entropy_rows` (§3 K10). **Design input:** T3 chunks over **rows** (`chunk_rows·V·4`, `:147-148`); at Qwen's V=248,320 a 1,024-row chunk is 1.02 GB of scratch. tessl chunks over **vocab** (8,192 columns), bounding scratch at `rows×8192` [V `qwen35_train.rs:56`, `cross_entropy.rs:20-24`]. Suggest T3 take a vocab chunk too | G3 (d) device vs CPU at N=4096, V=50304, 1e-4 |
| T4 `cached_attention_forward`, T5 `kv_cache_write` | The decode path. Reuses the attention kernel's forward with `kv_len ≥ Tq` and GQA | G4 |
| T6 | none (core) | G5 |

#### 2.A.3 What is still missing for Qwen3.5 after T1–T6

Each of these is a **new trait method**, so a coordinator edit [V by comparison of §1.3 against
§1.7]:
- **Q1** GDN chunked fwd/bwd at transformers' seam (`g`, `beta` given; l2norm in kernel), **`published`** rule.
- **Q2** `causal_conv1d_silu` fwd/bwd (depthwise, K=4).
- **Q3** causal attention with **GQA** and **D up to 256** returning **LSE**. The trait's `causal_sdpa` is MHA; T4 has GQA only for cached decode.
- **Q4** Qwen RMSNorm `x·rsqrt(ms+eps)·(1+w)`. The trait's is `·w`.
- **Q5** gated RMSNorm: a per-head norm times `silu(z)`.
- **Q6** an elementwise attention output gate `o·sigmoid(gate)` (`[T, Hq·D]`). The trait has a per-head `[T,H]` gate.
- **Q7** partial RoPE (`rotary_dim = 0.25·256 = 64`), with MRoPE collapsing for text-only input. The interleaved-MRoPE collapse is still inferred (`GAP-OJAS-ADVICE-MROPE-COLLAPSE-INFERRED-2026-10-01`).
- **Q8** GDN gates `g = −exp(A_log)·softplus(a + dt_bias)`, `beta = sigmoid(b)`, fwd/bwd [I from `qwen35_train.rs:1049-1060,1311-1326`; the formula is transformers', not re-read].
- **Q9** a **bf16-operand / f32-accumulate** numerics tier on `linear_*`. The trait is f32 only [R `fable-advice.md:9,27`].
- **Q10** rows-only CE at a large vocab. T3 with `ignore_index` can express rows-only, but see its chunking note above.

#### 2.A.4 Size

The existing GPU backends are the reference:
- `ojas-metal` is about 6.4k Rust lines (`backend.rs` 1,198, `device.rs` 1,800, `gpu.rs` 3,407) plus 1,370 Metal shader lines;
- `ojas-wgpu` is about 4.0k Rust lines (`backend.rs` 2,115, `context.rs` 1,215, `lib.rs` 654) plus 1,263 WGSL lines [V `wc -l`].

A CUDA `Backend` at the same scope is **15–25 specialist days** for the nanolab op set [I].
Q1–Q10 add **~25–30** more, because they are the same kernels as (B)'s K2–K8. And they are gated on
trait governance. So **(A) for Lappi is 40–55 days and blocked on coordinator trait edits before the
first Qwen op runs** [I].

### 2.B A whole-step Qwen3.5 provider on CUDA: the CUDA counterpart of `ojas-qwen35`

#### 2.B.1 What it is

- **The provider.** A `Qwen35Cuda` model, the CUDA twin of the Metal `ojas-qwen35` provider, which holds a tessl `Qwen35Model`. It loads the 2B safetensors to f32 device masters and exposes the **same step-provider trait** as the Metal arm.
- **The trait.** Proposed shape, mirroring tessl's verified API (§1.7). The trait's owner is L-ojas-qwen35 and the coordinator, not this lane:

  ```rust
  trait StepProvider {
      type Pending; type Bank;
      fn train_forward(&self, ids: &[u32], ops: GemmOperands, sup: Supervise<'_>) -> Result<Self::Pending, Error>;
      fn hidden(&self, p: &Self::Pending, positions: &[u32], out: &mut [f32]) -> Result<(), Error>;  // span head input
      fn train_backward_into(&self, p: Self::Pending, dh: Option<(&[u32], &[f32])>,
                             bank: &Self::Bank, accumulate: bool) -> Result<(), Error>;
      fn grad_sq_norm(&self, bank: &Self::Bank) -> Result<f64, Error>;
      fn adamw_step(&mut self, bank: &Self::Bank, hyper: AdamWHyper, wd: &[f32], lr_scale: &[f32]) -> Result<(), Error>;
  }
  ```

  The `lr_scale` vector is Fable's row-10 addition (`fable-advice.md:28`). The trainer (Lappi `qd-train` now, `ojas-train` later; Fable ask 2) is generic over it. The Metal arm and the CUDA arm are interchangeable to the trainer, and parity between them becomes one test.
- **Internals.** A port of `qwen35_train.rs`'s host orchestration (1,747 lines of Metal-bound host code [V `wc -l`]) onto a CUDA runtime, plus the kernel list in §3. The semantics are kept exactly:
  - one sequence at a time into a bank;
  - per-layer checkpoint and rebuild;
  - `Supervise::Rows` with distinct positions;
  - `dh` added at the final norm's output;
  - unpadded sequences, which are Lappi's `train_attention_mask="none"` numerics, as on Metal (`fable-advice.md:21`).

#### 2.B.2 Why it is faster to a trained Lappi

It needs **zero trait edits**, **zero autograd** (hand-written backward, as tessl's), and it is the
seam `qd-train` calls anyway. It also has a hardware-independent golden ready (§5.2).

### 2.C Recommendation and convergence

- **Soonest to Lappi trained on the GH200:** (B). It is the only design whose first runnable artifact is a Qwen3.5 step.
- **Long-term for ojas:** (A). ojas's identity is "any model on any `Backend` through the Tape" (`framework-design.md` §3); a Qwen-only provider does not serve nanolab, decode or the Go API.
- **How they converge.** Build (B) so that nothing in it has to be thrown away:
  1. **One runtime.** `ojas-cuda` grows a `CudaRuntime`: context, one stream, a budgeted allocator, an NVRTC module cache keyed by (source hash, options, NVRTC version), and `CudaBuffer: DeviceBuffer`. (B) uses it now; (A)'s `CudaBackend` wraps the same runtime.
  2. **One kernel source module.** `ojas-kernels/src/cuda/*.cu`, included as strings (as `affine_cuda()` and the WGSL modules are). Each kernel is a function over device pointers and dims, named after its seam (`gdn_chunk_fwd_published`, `attn_gqa_lse_fwd`, …). (B)'s orchestration calls them now. (A)'s Q1–Q10 trait methods, once the coordinator adds them, are thin wrappers over the **same** functions, so there is no second implementation.
  3. **One trainer.** It is generic over the step-provider trait. Later, an (A)-based `Tape` model of Qwen3.5 implements that trait too, and the (B) provider becomes the fused fast path behind it. tessl's hand-written step relates to ojas's proposed `Trainer<B>` the same way (`fable-advice.md:11`).
  4. **One parity suite.** The same fixtures check Metal-(B), CUDA-(B) and, later, CUDA-(A).

---

## 3. The kernel port list

Conventions for every row:
- **Masters** are f32. **Two numerics tiers** mirror `GemmOperands`:
  - `ExactF32`: f32 operands and accumulate. On sm_90 that means CUDA-core FFMA, **not** tensor cores, because TF32 rounds operands to 10 mantissa bits.
  - `Bf16`: operands rounded to bf16 (RNE) at the GEMM/MMA boundary, f32 accumulate, f32 out.
  - Everything that is not a GEMM or MMA stays f32, as on Metal [V `cross_entropy.rs:31-36`].
- **Determinism rule:** every output element written by exactly one thread block; float reductions in a fixed-shape tree; no float atomics; **no launch geometry derived from SM count** (§1.3). Repeat-run bit equality is a test in every row.
- **Rule 9:** every GDN fixture name says `published`.
- **The box oracle** is the campaign venv (§1.6): torch 2.10.0+cu128, fla 0.5.2 on triton 3.7.1, causal-conv1d 1.7.0.

**Days** are specialist engineer-days for a hand-written, tested kernel, including its parity test.
They are estimates [I], calibrated on tessl's Metal versions (§1.7) and ojas's backends (§2.A.4).

| # | Kernel | Algorithm | Numerics, determinism | Parity test (on the box unless noted) | Days |
| --- | --- | --- | --- | --- | ---: |
| K0 | Runtime and plumbing: `CudaRuntime`, `CudaBuffer`, NVRTC cache, f32↔bf16 cast, `copy_cols`, `deliver` (copy / `+=`), `zero`, `scatter_add_rows` (distinct rows), `ce_gather_rows` | elementwise, one thread per element; scatter is ownership-by-row (positions are distinct, `qwen35_train.rs:585-593`) | exact; one rounding per add | host reference, bitwise; repeat-run equality; `require_libraries` probe before any cudarc call (`ojas-cuda/src/lib.rs:86-101`) | 4 |
| K1 | GEMM `nn`/`tn`/`nt` (§3.1) | tiled MMA | `ExactF32` (FFMA, fixed k-order) and `Bf16` (bf16 MMA, f32 accumulate, f32 C); no split-K, or split-K with a fixed-order second pass | vs f64 host on odd shapes (`130×70×260`, as tessl's `gemm.rs` tests); vs `torch.matmul` (fp32, `allow_tf32=False`) at 2B shapes; Bf16 vs torch on bf16-rounded inputs ≤2^-8 rel (`qwen35_train.rs:247`) | 2 / 5 / 12 (tier) |
| K2 | GDN train fwd/bwd, **published**, transformers' seam (§3.2) | (i) tessl's token-sequential scan with a state checkpoint every 64 tokens; or (ii) fla's chunked WY/UT form, chunk 64 | (i) fixed order by construction (per-slice partials + `gdn_train_bwd_finish`, `gdn_train.metal:21-27`); (ii) mma per chunk, sequential inter-chunk state, fixed-order dq/dk/dg reductions | f64 host reference (port of `tessl/tests/common/gdn_train.rs`) at T=1/63/64/65/130 (`tests/gdn_train.rs:1-15`), ≤1e-4 of max (`:241`); vs `fla.ops.gated_delta_rule.chunk_gated_delta_rule(…, use_qk_l2norm_in_kernel=True)` at 2B heads, T=2048/8192 | (i) 5; (ii) 14–18 |
| K3 | `gdn_gates` fwd/bwd | `g = −exp(A_log)·softplus(a+dt_bias)`, `beta = sigmoid(b)` per (t, head); bwd reduces over t for `A_log`, `dt_bias` | f32; fixed-order per-head reduction (tessl's `gates_part`, `qwen35_train.rs:1323`) | host f64; vs transformers' `Qwen3_5GatedDeltaNet` gate code under torch autograd | 1 |
| K4 | `conv1d_silu` fwd/bwd | causal depthwise conv, K=4, zero initial state, then SiLU; bwd: dx by a reversed stencil, dW by a per-channel reduction over t | f32; fixed-order dW (`conv_part`) | host f64; vs `causal_conv1d_fn` (causal-conv1d 1.7.0) and the torch reference | 1.5 |
| K5 | `attn_train` fwd (+LSE) / bwd at **D=256, GQA 4:1** | fwd: FA2 tiled online softmax, writes `lse [B,Hq,T]`. bwd: FA2 split into a **dK/dV kernel per key block** (looping over the 4 query heads of its KV group in fixed order) and a **dQ kernel per query block**, recomputing P from LSE, so each gradient is written once with no atomics, as tessl's (`attn_train.rs:11-17`) | `ExactF32` via FFMA; `Bf16` via bf16 `mma.sync` m16n8k16; f32 softmax and LSE. D=256 bf16 tiles: 64×256×2 B = 32 KiB per Q/K/V/dO tile, against 227 KiB of shared memory per block on sm_90 [I], so the bwd needs Br=Bc=64 or 32 | vs f64 host at T=1/31/32/33/130; vs `F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)` fwd + autograd at 2B shapes, ≤1e-4 rel (`attn_train.rs:124`); LSE vs `torch.logsumexp` | 8 |
| K6 | `attn_qk_norm_rope` fwd/bwd + `attn_output_gate` fwd/bwd | per-head RMSNorm (`1+w`) on q and k, partial half-split RoPE over the first 64 of 256 dims (θ = 1e7), with q/gate de-interleaving from `q_proj`; the gate is `o·sigmoid(g)` | f32; fixed-order reduction for the norm weights' gradient (`part`) | host f64; vs transformers' `Qwen3_5Attention` pieces under autograd, with the snapshot's `config.json` loaded (the MRoPE collapse falsifier) | 2 |
| K7 | `rms_norm` fwd/bwd (`1+w`), `gated_rms_norm` fwd/bwd | row reduction over hidden (2048) or v_dim (128); bwd: dx per row, dW reduced over rows in fixed order | f32; `eps = 1e-6` | host f64; vs transformers' `Qwen3_5RMSNorm` and `Qwen3_5RMSNormGated` | 2 |
| K8 | `swiglu` fwd/bwd, `residual_add` | elementwise | exact f32 | host bitwise | 0.5 |
| K9 | `embed_rows` fwd/bwd | gather; bwd adds `dresid` rows into `[V,H]` by token id. **Deterministic:** sort the ids once on the host, then one block per distinct id sums its rows in position order | f32 | host f64; repeated-id case; vs `nn.Embedding` autograd | 1 |
| K10 | `cross_entropy_rows` | gather supervised rows; walk vocab in 8,192-column chunks with an online LSE; the second walk forms `dlogits = (softmax − onehot)·scale`, then `dh += dlogits·W_c`, `dW_c = dlogitsᵀ·h` via K1 | GEMMs on `GemmOperands`; softmax exp/log in precise f32 (`cross_entropy.rs:31-36`); no `[rows,V]` matrix | host f64 (`tests/cross_entropy.rs:185`: 1e-5 abs + 1e-5 rel); Bf16 ≤2^-7 (`:258`); vs `F.cross_entropy` at V=248,320 on chosen rows | 2.5 |
| K11 | AdamW in place + `grad_sq_norm` | torch single-tensor AdamW order (decoupled decay, `lerp` m, v, bias correction outside the sqrt; f64 host scalars) with per-entry wd **and** `lr_scale`; grad norm = per-block f32 partials in fixed order, summed in f64 on the host | matches `ojas-core/src/backend.rs:273-286`, which is tessl's order | vs `torch.optim.AdamW` two-group oracle over 3 steps ≤1e-6 (`fable-advice.md:28`); grad norm ≤1e-5 rel (`tessl/tests/qwen35_adamw.rs:88`) | 1.5 |
| S | Step orchestration (B) | port `train_forward`/`backward`/bank/`hidden`/`train_backward_into`; safetensors bf16 → f32 masters; workspace sizing; non-finite kill | — | tessl's tiny fixture (§5.2): loss ≤1e-5 rel, grads ≤1e-4 of each parameter's max (`tests/qwen35_train.rs:199-216`, `:343`) | 6 |
| H | Harness and shipping | cross-build script, sha256 manifest, box run script (`flock gpu.lock`, `timeout`, cost line), `.npy` mismatch dumps instead of printf, fixture shipping | — | runs green on the Mac with the feature off | 3 |

**Totals** [I]:

| Path | Kernel choices | Days |
| --- | --- | --- |
| Correctness | K0 4 + K1 cuBLAS 2 + K2(i) 5 + K3 1 + K4 1.5 + K5 8 + K6 2 + K7 2 + K8 0.5 + K9 1 + K10 2.5 + K11 1.5 + S 6 + H 3 | **40** |
| Correctness, no cuBLAS | K1 `mma.sync` instead | **43** |
| Campaign speed | + K2(ii) 14–18, + K1 WGMMA tier 12 (only if the profile says GEMM dominates), + K5 tuning 3–5 | **+29–35 → 69–78** |

### 3.1 GEMM: cuBLAS through cudarc, or hand-written

**What cuBLAS costs in dependencies.**
- cuBLAS is **not a new crate**. It is cudarc's `cublas` feature (`cublas = ["driver"]` [V `cudarc/Cargo.toml` features]) on a dependency ojas already has.
- It dlopens `libcublas.so.12` at run time, which the box venv very likely has (§1.6).
- The bf16-in / **f32-out** call tessl's semantics need is **not** in cudarc's safe API: `Gemm<half::bf16>` writes **bf16 C** [V `cudarc/src/cublas/safe/gemm.rs:150-180`, `CUDA_R_16BF` for C at `:176`], and uses `half` (`f16 = ["dep:half"]`), which would be a new crate.
- The public `cudarc::cublas::result::gemm_ex` [V `cudarc/src/cublas/result.rs:375`] takes raw `cudaDataType_t`s. So `A,B = CUDA_R_16BF, C = CUDA_R_32F, CUBLAS_COMPUTE_32F` is reachable with **no `half`**, using u16 storage.
- That cuBLAS supports this combination is **[U]**: the docs fetch did not return the GemmEx type table (`GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01`). The falsifier is the M0 smoke probe (§6.1).

**Determinism.**
- cuBLAS is bitwise reproducible run to run "on GPUs with the same architecture and the same number of SMs". That guarantee does not hold with multiple active streams unless each stream has its own workspace (`cublasSetWorkspace`) or handle. `cublasSetAtomicsMode` voids it for the routines that use atomics [R cuBLAS 13.4 docs, "Results reproducibility"; applicability to 12.8.4.1 is [I]].
- The design rule: one handle, one stream, explicit workspace, atomics mode never set, plus a repeat-run bit-equality test. The cuBLAS library version, SM count and driver go in every ledger row.

**Three tiers, not two:**

| Tier | Path | Days | Notes |
| --- | --- | ---: | --- |
| 1 | cuBLAS `gemm_ex` | 2 | Expected near the best achievable at these shapes (M = 8,192 rows, N/K ∈ {2048, 5120, 6144, 8224}) [I]. Also serves as the **oracle** for tiers 2–3 on the box. **The user prefers focused kernels over heavy dependencies:** cuBLAS here is a feature flag on an existing crate and a library already on the box, not a new crate, but it is a closed-source dependency in the numerics, and its algorithm choice can change with the library version. Ask 3 |
| 2 | hand-written `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with `ldmatrix` and `cp.async` (sm_80-class instructions, which run on sm_90) | 5 | Fixed tile geometry, no split-K, so bits are independent of SM count, which fits ojas's `Fast` contract better than cuBLAS. A well-tuned version reaches roughly half of Hopper's dense bf16 peak [I]. Inline PTX breaks the hiprtc reuse in `source.rs:1-2` |
| 3 | WGMMA + TMA, warp-specialized producer/consumer, `sm_90a` | 12 | Needs NVRTC `-arch=sm_90a`. NVRTC lists `sm_90a`/`compute_90a` [R NVRTC 13.4 docs; 12.8 support is [I]], and cudarc's `CompileOptions` has `arch: Option<&'static str>` [V `cudarc/src/nvrtc/safe.rs:231-242`]. Build this only if a profile shows GEMM dominating **after** GDN is chunked. With tessl's GDN, GDN dominates (§3.2) |

**GEMM's share of the step** [I]. Per 8K sequence, GEMM FLOPs are (6 + 2 for the recompute) ×
1.373 B × 8,192 ≈ **9.0e13**. That is ~0.15 s at 600 TFLOPS (tier 1), ~0.3 s at 300 (tier 2), and
~2 s in `ExactF32` FFMA at ~45 TFLOPS. So GEMM is not the first bottleneck on any tier.

**Recommendation:** tier 1 for the correctness path *if* ask 3 says yes, else tier 2. Tier 2 replaces
tier 1 when it is within 10–15% on the step benchmark (or at once, if the human values dropping
cuBLAS). Tier 3 is last and profile-gated.

### 3.2 GDN: port fla's chunked algorithm, or port tessl's Metal kernel

**Same operator, two algorithms.**
- Both are the **published** rule (decayed read `u = v_t − (exp(g_t)S_{t−1})ᵀk̂`) at transformers' seam [V `tessl/src/gdn_train.rs:1-27`; `gdn_train.metal:1-11`; the published-rule definition in `AUDIT/gdn-reference-and-contracts.md` §2.2].
- tessl's kernel is a **token-sequential scan**: one 128-thread group per (16-column value slice, batch×head), thread *i* owns state row *i*, a checkpoint every 64 tokens, and the backward recomputes each chunk from its checkpoint [V `gdn_train.metal:13-31`, `gdn_train.rs:39-44`].
- fla's is the **chunked WY/UT** form (chunk 64): intra-chunk work as dense matmuls, inter-chunk state passed sequentially over T/64 chunks [R `AUDIT/gdn-reference-and-contracts.md` §3; fla source not re-read].

**The arithmetic that decides it** [I, each factor cited]:
- **Measured on the M5 Pro:** gdn_train fwd 9.04 ms and bwd 37.11 ms per layer at T=2048, bf16, on battery [V bench `:23-24`]. The bench header says plugged-in is about 2× faster [V bench `:5-9`].
- **Per training step:** forward + rebuild + backward = 2 × 9.04 + 37.11 = **55.2 ms per layer per 2,048 tokens**.
- **At T=8,192:** the scan is linear in T, so **~221 ms per layer**, **× 18 GDN layers ≈ 4.0 s per 8K sequence** (≈2.0 s plugged in).
- **On H100:** the scan is latency-bound (one barrier-separated update per token, 128 blocks for 16 heads × 8 slices), so per-token latency on H100 is of the same order as on the M5 [I]. Batching the 4 sequences of a step into one launch (B=4, 512 blocks, all resident) could hide up to ~4× of that [I].
- **The PyTorch baseline on the same box:** shape B (4 × ≤8,441 tokens) runs a **whole step** in 1.93 s on default kernels and 1.20–1.33 s with no mask and flash [V `AUDIT/det-attention-backward-scoping-2026-10-01.md:14,44-46`]. That is ≈0.3–0.5 s per 8K sequence for **everything**. J1 ran at 18.5k pos/s [V `HANDOFF/gh200-phase4-2026-10-01.md:26`].
- **So:** a CUDA port of tessl's scan makes GDN alone **~1–3×** PyTorch's entire step even with B=4 batching (≈0.5–1 s per sequence), and **~4–13×** without it. The whole CUDA step lands ~5–10× slower than PyTorch [I].
- **Rung d is still affordable at that speed** (§5.3). A 3-seed campaign is not.

**Trade-off.**

| | (i) Port tessl's scan | (ii) Port fla's chunked form |
| --- | --- | --- |
| Correctness risk | Low: a 348-line kernel [V `wc -l gdn_train.metal`] with an f64 reference and T-edge tests already written in tessl | High: the WY backward (`chunk_bwd_dqkwg`, dh recurrence, WY inverse backward) is exactly where fla shipped **wrong gradients on Hopper** under triton < 3.7.1 (#640), caught only by fla's version guard [V `HANDOFF/gh200-2026-09-20.md:44-60`] |
| Determinism | By construction | Must be designed in: fla's own kernels make no bitwise claim [U] |
| Speed | ~5–10× slower than PyTorch per step [I above] | Competitive with fla [I] |
| Numerics | Bitwise-equivalent algorithm to Metal, so Metal-vs-CUDA parity is tight | Differs from Metal and torch-fallback rounding, so parity is bounded, not tight |
| Days | 5 | 14–18 |

**Recommendation:**
- **(i) first.** It carries rungs a–c and a reduced rung d. It stays forever as the on-device reference oracle for (ii).
- **(ii) second,** only on the human's "campaign speed" answer (ask 6). It is checked against (i) on device and against fla on the box at T ∈ {64, 65, 2048, 8192}.
- Every fixture name carries `published` (rule 9). nanolab's default `rule="repo"` is the wrong golden.

---

## 4. Toolchain

### 4.1 NVRTC at run time, or nvcc-built PTX/cubin

- **NVRTC at run time** is what `ojas-cuda` does today (§1.1).
  - Kernel sources are strings in the crate, compiled on first use and cached by (source hash, options, NVRTC version).
  - It needs **nothing on the Mac**, and the binary links with only `libc`/`libgcc_s` [V §1.5].
  - On the box it needs `libnvrtc.so.12`, very likely in the venv (§1.6, [I]).
  - Options: `--fmad=false` for `ExactF32` kernels (NVRTC's default is `true` [R NVRTC 13.4 docs]); `-arch=sm_90` (CUBIN directly; NVRTC gives no CUBIN for a virtual arch [R]); `sm_90a` for tier-3 WGMMA.
  - **Cost:** the NVRTC version on the box becomes part of the numerics. Record `nvrtcVersion` with the driver and cuBLAS versions in every row.
- **nvcc-built cubin.**
  - CUDA has no macOS toolkit, so nvcc cannot run on the Mac host [I; NVIDIA's last macOS toolkit was 10.2, not re-checked].
  - Two options:
    - (a) nvcc 12.8 **on the box** [R lead], which needs box time and shell access;
    - (b) nvcc inside a **linux/arm64 CUDA devel container on the Mac's podman**. This is compile-only, no GPU: `stack/Containerfile:18` already uses `nvidia/cuda:12.8.1-devel-ubuntu24.04` (an arm64 variant of that tag is [U]).
  - The cubin is then embedded with `include_bytes!` and loaded via the driver's `load_module`. That pins the compiler into the artifact and removes NVRTC from the box's runtime.
- **Recommendation:** NVRTC at run time for development (zero toolchain, the existing ojas pattern). Then a build-time embed path (b) for the promoted kernels, with a test that the embedded cubin's sha matches a rebuild from source.

### 4.2 Cross-building for the box from the Mac

What is needed, all verified to work today for ojas (§1.5):
- rustup's `aarch64-unknown-linux-gnu` target, which is installed [V `rustup target list --installed`];
- the box-glibc sysroot and linker driver at `/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh` (Apple clang driving rustup's `ld.lld` against the box's glibc 2.39) [V the file];
- the two environment variables from `HANDOFF/post-f-queue-2026-10-01.md:147-151` [V].

**No CUDA SDK is needed on the Mac**: cudarc dlopens at run time. Change the binding pin to
`cuda-12080` first (ask 2).

### 4.3 Running tests on the box without a Rust toolchain

Ship test binaries as `qd-post-f-rules` was shipped [V `post-f-queue:139-161`]:
1. Build with `cargo test --release --no-run --target aarch64-unknown-linux-gnu -p <crate> --features cuda`.
2. Copy the `deps/<crate>-<hash>` executables and the fixture directory.
3. Pin each by sha256 in a run script, which refuses another binary.
4. On the box: `flock /home/ubuntu/queue/gpu.lock timeout <cap> ./<test-bin> --ignored --test-threads=1`.
   - GPU tests are `#[ignore]` so the Mac run (feature off) stays green.
   - Add `LD_LIBRARY_PATH` with the venv's `site-packages/nvidia/{cublas,cuda_nvrtc}/lib` [I paths] or `/usr/local/cuda/lib64` [U].
   - Output is `--format json`-style lines into a ledger row with the binary's sha256.
5. A failing comparison writes the mismatching tensors as `.npy` for offline diagnosis on the Mac. That saves box round trips, since no CUDA bug can be reproduced off the box.

---

## 5. Determinism and the parity ladder on CUDA

### 5.1 The determinism contract

- The kernel rules in §3: one writer per element, fixed-shape reduction trees, no float atomics, no SM-count geometry, one stream.
- For cuBLAS: one handle, explicit workspace, no atomics mode.
- Proof in every rung: **two runs in two processes give bit-identical loss digests and gradient sha256s**.
- The row records (GPU name, SM count, driver, NVRTC, cuBLAS, binary sha256), because bits are only promised within that tuple.
- Bitwise equality with Metal is **not** a goal; Metal-vs-CUDA is a bounded comparison like the torch ones (`fable-advice.md:27`).

### 5.2 A hardware-independent golden is already committed

`tessl/tests/fixtures/qwen35_train/` holds:
- a transformers-autograd step on a tiny Qwen3.5: hidden 64, MLP 128, 2 layers (GDN + attention), 2:1 heads at **D=256**, GDN 1×128 / 1×128, vocab 64;
- `config.json`, `model.safetensors`, `ids.npy`, `loss.npy` and one `grad.*.npy` per parameter [V `ls`, config read];
- generated by `tessl/tools/qwen35_ref/make_train_fixture.py` [V `rg -l`].

Its shapes are the ones the D=256 / DK=128 kernels compile for. A CUDA provider checks against it **on
the box with no torch call**, at tessl's written bounds: loss ≤1e-5 rel, every gradient ≤1e-4 of its
parameter's max [V `tessl/tests/qwen35_train.rs:199-216,343`]; Bf16 ≤2^-8 / 2^-5 (`:247`). This
fixture is outside tessl's crate, so it can be copied into the CUDA crate's test fixtures without
touching tessl.

### 5.3 The CUDA ladder, mirroring Fable's rungs a–d

All rows are `quick=True` with a `quick_reason` (rule 8). Tolerances are written in the test before
the first run. Costs are at $2.29/h; minutes are estimates [I] including the binary copy and NVRTC
compiles.

| Rung | What runs on the box | Pass criterion (same bounds as Fable's) | GPU min | $ |
| --- | --- | --- | ---: | ---: |
| **0** toolchain smoke | `ojas-cuda` affine probe (`open()`); NVRTC `sm_90` + `sm_90a` compiles; `cublasGemmEx` bf16×bf16→f32 status and a 64³ result vs host; device attributes (SM count, smem/block); library versions | every call returns `Ok`; the GEMM ≤2^-8 rel; versions recorded | 2 | 0.08 |
| **a** op level | §3 K0–K11 tests vs f64 host references and tessl's fixture (no torch), then vs torch/fla at 2B shapes in the campaign venv (fla `chunk_gated_delta_rule` with `use_qk_l2norm_in_kernel=True`, SDPA with `enable_gqa`, `F.cross_entropy`, causal-conv1d), with GDN fixtures named `published`; repeat-run bit equality | tessl's bounds per op (§3); Bf16 tiers ≤2^-8 / 2^-5 | 15–25 | 0.6–1.0 |
| **b** tiny tower | the Rust trainer over the CUDA provider, 20 steps, on L-oracle's rung-b fixture (`tools/qd_train_oracle_tiny.py`, `fable-advice.md:80`). The torch CPU oracle numbers are made on the Mac; no torch run on the box. Also Metal-(B) vs CUDA-(B) on identical inputs | ExactF32 vs torch fp32: loss ≤1e-5 rel for steps 0–5, ≤1e-4 to step 20; final weights ≤1e-5 of max; two CUDA runs bit-identical | 3–5 | 0.1–0.2 |
| **c** real 2B | one train step on one real v4 batch vs torch fp32 in the venv (the torch run itself uses the GPU); ≥24 val prompts' letter log-softmax and span scores vs the PyTorch scorer | loss ≤2^-7, grads ≤2^-4 (`qwen35_train.rs:862-868`); ≤0.10 nats on the 17 letters, ≥95% argmax | 20–40 | 0.8–1.5 |
| **d** real fine-tune | F's recipe on v4 shards, seed 0, **reduced schedule** (10% of steps) through `qd-train` on CUDA; export, then the campaign scorer. PyTorch's reduced arm is ≈$0.30 (`fable-advice.md:57`; J1 4,615 steps in 4,040 s, $2.57 [V `gh200-phase4:26`]). The CUDA arm at 5–10× slower is ≈40–80 min, plus scoring [U] | Fable's rung-d criteria unchanged (`fable-advice.md:57`) | 60–120 | 2.3–4.6 |

**Which rung fits a queue gap: none.** There are no gaps.
- Every post-F item waits on `.done` markers and then takes `gpu.lock`. No waiter holds the lock while waiting, and the chain moves on from one item's exit trap to the next [V `post-f-queue-2026-10-01.md:182-185,203-206,230-234`]. A gap is the seconds between one item releasing the lock and the next taking it. Fable reached the same reading (`fable-advice.md:63-65`).
- So any CUDA test is an **inserted item**: a waiter that takes `gpu.lock` between two items and delays the rest of the chain by its own duration. Its price is that duration × $2.29/h if the box is released when the queue ends [I].
- **Rung 0 + rung a** together are one ~20–30 min item, **≈$0.8–1.1**. They are the only rungs short enough to insert without materially moving the campaign.
- **b–d** belong after item 10.
- Every rung is a single-GPU job well under $20, so rule 4 does not require a yes. But Fable's Q4 decision was **zero GH200 time for ojas until the queue ends**. This lane does not override it; inserting rung 0+a is ask 4.

---

## 6. Milestones, lanes, first commands, human asks

### 6.1 Milestones

| M | Content | Exit | Depends on |
| --- | --- | --- | --- |
| M0 | `CudaRuntime` + `CudaBuffer` + NVRTC cache + the rung-0 smoke binary; binding pin at `cuda-12080` | rung 0 green on the box | asks 1, 2, 4, 5 |
| M1 | K0, K1 (tier 1 or 2), K8, K11 + the tiny-fixture loader | rung a for these ops | M0 |
| M2 | K2(i), K3, K4, K6, K7, K9, K10 | rung a for every op | M0 |
| M3 | K5 attention | rung a for attention | M0 |
| M4 | S orchestration + the step-provider trait impl; the Metal-(B) vs CUDA-(B) test | rung b | M1–M3, the trait from L-ojas-qwen35 |
| M5 | real 2B load, rung c; then `qd-train` on CUDA, rung d | rungs c, d | M4, the queue ended (or ask 4) |
| M6 (optional) | K2(ii) fla-chunked GDN; GEMM tier 2→3; attention tuning | step within 1.5× of PyTorch on shape B | ask 6 |

### 6.2 Lanes and first commands

All lanes run from a worktree of whichever repository ask 1 picks; the default below is ojas. **No
lane writes to ojas until the coordinator has named its files (ask 5).** Every lane's first command
is `git status --short` in its own worktree, refusing any dirty file it would touch, followed by
DevMap (`devmap_explore` on the symbol it extends, absolute `repo_path`) and `gitpulse_insights`. In
the commands below, `<WT>` is the lane's own ojas worktree root, and `$SYSROOT_ENV` is the two
`CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_*` variables from §1.5.

| Lane | Owns (proposed) | First command after the status check |
| --- | --- | --- |
| LC-runtime | `ojas-cuda/src/{runtime,buffer,nvrtc_cache}.rs`; rung-0 smoke binary | `cargo test --locked -p ojas-cuda`. Mac baseline with the feature off: ojas reports 8 passed [R `backends.md:71`]. Then `$SYSROOT_ENV cargo test --locked --release --no-run --target aarch64-unknown-linux-gnu -p ojas-cuda --features cuda --target-dir /Users/bharath/qd-campaign/target-aarch64-linux-ojas-cuda` |
| LC-gemm | `ojas-kernels/src/cuda/gemm*.cu`, `ojas-cuda/tests/gemm.rs` | write the failing `#[ignore]` test `gemm_bf16_operands_f32_out_matches_f64_host` (odd shapes 130×70×260 plus 2B shapes), then the same `--no-run` cross-build |
| LC-gdn | `ojas-kernels/src/cuda/gdn_train_published.cu`, `ojas-cuda/tests/gdn_published.rs` | port `tessl/tests/common/gdn_train.rs`'s f64 reference into `ojas-cuda/tests/common/gdn_published_ref.rs` (read-only copy, tessl untouched) and assert it on the host for T=1/63/64/65/130 before any kernel exists: `cargo test --locked -p ojas-cuda --test gdn_published` |
| LC-attn | `ojas-kernels/src/cuda/attn_gqa_lse.cu`, `ojas-cuda/tests/attn.rs` | host f64 causal-GQA reference with LSE and its finite-difference self-test first: `cargo test --locked -p ojas-cuda --test attn` |
| LC-small | K3, K4, K6, K7, K9, K10 sources + tests | host references first, one test file per kernel, each failing until its `.cu` lands |
| LC-oracle | Lappi `tools/cuda_opseam_oracle.py` (oracle only, run on the box later, in the campaign venv) | write it to dump fla/SDPA/CE/causal-conv1d op-seam fixtures at 2B shapes with `published` in every GDN file name; dry-run it on the Mac CPU at tiny shapes: `PYTHONPATH=python /Users/bharath/.venvs/ml/bin/python tools/cuda_opseam_oracle.py --tiny --out /Users/bharath/qd-campaign/cuda-opseam-tiny` (fla's Triton path is CUDA-only, so the Mac dry run covers the torch-reference arms only) |
| LC-step (after M1–M3) | the CUDA arm of `ojas-qwen35` (or its sibling crate; ask 1) | `cargo test -p ojas-qwen35 --features cuda --no-run --target aarch64-unknown-linux-gnu` against the trait L-ojas-qwen35 publishes |

**First box command (the lead, after asks 2, 4 and 5).** Copy the rung-0 binary and its sha256, then:

```
flock /home/ubuntu/queue/gpu.lock timeout 300 /home/ubuntu/bin/ojas-cuda-smoke --ignored --test-threads=1
```

The output goes to `ledger/gh200-ojas-cuda-rung0-<date>.jsonl` with `quick=True`.

### 6.3 The exact human asks

1. **Placement and rule-6 scope.** "CUDA kernels and the CUDA Qwen3.5 provider live in **ojas**: `ojas-cuda` grows the runtime, `ojas-kernels/src/cuda/` holds the sources, and the provider is the `cuda` arm of `ojas-qwen35` (or a sibling `ojas-qwen35-cuda`). Rule 6 ('all kernel work is on canonical tessl') is read as covering Metal kernels, not CUDA. tessl is `objc2_metal`-bound [V `gdn_train.rs:31`, `attn_train.rs:24`], and a CUDA runtime inside tessl would be a second device stack there. Yes, or put CUDA in tessl?" **Recommend: ojas**, because the user's goal is "trained with ojas".
2. **Enable cudarc.** "Build `ojas-cuda` with `--features cuda` for the box, and change its binding pin from `cuda-13040` to `cuda-12080` to match the box's CUDA 12.8. One line in `ojas-cuda/Cargo.toml`, which the coordinator owns; cudarc 0.19.10 is already locked." Without the pin change, a 13.x-only symbol call panics at run time (§1.6).
3. **cuBLAS.** "Turn on cudarc's `cublas` feature: no new crate, and `libcublas.so.12` is already in the box venv [I]. It is the correctness-path GEMM and the oracle for a hand-written `mma.sync` GEMM that replaces it when it is within 10–15%. Or no cuBLAS at all, which costs +3 days and slows rung a." The `half` crate is **not** requested (u16 storage + `result::gemm_ex`). Recommend: yes, for the correctness path and as an oracle.
4. **GH200 time.** "Insert one ~20–30 min item (rung 0 + rung a, ≈$0.8–1.1) into the post-F queue between items, or hold everything until after item 10? Rungs b–d (≈$3–6 total [I]) after the queue." Rule 4 does not require a yes for these. Fable's Q4 said zero, so this is the human's call.
5. **ojas file ownership.** "The coordinator grants L-cuda lanes `ojas-cuda/**`, a new `ojas-kernels/src/cuda/**`, and one `members` line if a new crate is chosen. L-cuda does not touch `ojas-core/src/backend.rs`; (A)'s Q1–Q10 trait additions are a later coordinator proposal. Name the owner of the step-provider trait (L-ojas-qwen35?). The coordinator also asked for this document in `ojas/docs`; the lead relays it, and L-cuda did not edit ojas."
6. **Speed target.** "Is the goal (a) the quick parity ladder a–d (~40 days, CUDA step ~5–10× slower than PyTorch) or (b) a CUDA path that can replace PyTorch for 3-seed campaigns (+29–35 days: fla-chunked GDN, the GEMM and attention tiers)?" Recommend (a) now, and (b) after rung d's numbers exist.

---

## 7. Top risks

1. **GDN.**
   - The correct-first scan is too slow for campaigns (§3.2 arithmetic, [I]; `GAP-L-CUDA-GDN-SPEED-INFERRED-2026-10-01`).
   - The fast chunked backward is where fla already returned wrong gradients on Hopper (#640) [V `gh200-2026-09-20.md:44-60`].
   - Mitigation: (i) is kept as the on-device oracle for (ii); T-edge tests at 63/64/65; fixtures named `published`.
2. **Never run on a GPU.**
   - No ojas CUDA kernel has executed on any NVIDIA device [R `backends.md:130`].
   - The binding pin mismatches the box (`cuda-13040` vs 12.8), and a missing symbol panics [V cudarc].
   - Library presence on the box is inferred from the lock [I/U].
   - Mitigation: rung 0 before anything else.
3. **Iteration loop.**
   - CUDA bugs reproduce only on the box. The box has no Rust toolchain, no queue gaps, and is shared with a 40-h campaign.
   - Mitigation: host references for every kernel before its `.cu`; `.npy` mismatch dumps; one test binary per milestone; NVRTC so no rebuild is needed for option changes [I].
4. **Ownership.**
   - The ojas tree is live (103 unstaged and 97 untracked files [V GitPulse]).
   - The trait is coordinator-owned, rule 6's literal text points kernel work at tessl, and `ojas-qwen35` and its trait do not exist yet [V].
   - Mitigation: asks 1 and 5 before any write.
5. **cuBLAS.**
   - bf16→f32 `GemmEx` is unverified [U].
   - cudarc's safe API is bf16-out only [V].
   - Reproducibility is per (architecture, SM count, library version) [R].
   - Mitigation: rung 0 probe; tier 2 as a drop-in.
6. **Attention at D=256.**
   - FA2 backward tiles at D=256 sit near sm_90's shared-memory limit, so tiles are small [I].
   - Deterministic dQ needs the two-kernel split, which recomputes P [V tessl's design].
   - Correctness risk is moderate; speed is fine, since attention is ~0.04 s per 8K sequence at 200 TFLOPS [I: ≈7.4e12 FLOPs per sequence over 6 layers with recompute].
7. **Parity ceiling.**
   - The CUDA provider follows tessl's numerics: f32 masters, bf16 GEMM operands, f32 everything else, unpadded.
   - F's PyTorch keeps bf16 activations and pads with a mask.
   - Bitwise parity with F is unreachable by design. Bounds are Fable's (`fable-advice.md:27,57`).
8. **Linux ojas is linked, not run** [V §1.5]. A Linux-only runtime failure (dlopen paths, `/proc` probes in `ojas-device/src/host.rs:59`) is untested.

---

## 8. Gap records (appended to `gaps.jsonl` with `qd_train.gaps.append_gap`; listed here in case the commit conflicts)

- `GAP-L-CUDA-NO-LISTAGENTS-2026-10-01`: no `ListAgents` tool was exposed. The live ojas coordinator and the L-ojas-qwen35 lane are known only from the lead's message, and same-worktree peers in ojas are invisible to GitPulse's collisions facet.
- `GAP-L-CUDA-GITPULSE-UNTRUSTED-2026-10-01`: `gitpulse_insights` on the L-cuda Lappi worktree returned `REPOSITORY_TRUST_REQUIRED` on every facet. Its worktrees, collisions and agents are unknown.
- `GAP-L-CUDA-INVENTORY-FROM-READS-2026-10-01`: DevMap answered `status` and `skeleton` only. Rust net resolution was 338‰ (ojas) and 318‰ (tessl), with signatures not extracted. The kernel inventory and the trait comparison come from file reads and `rg`, not graph edges.
- `GAP-L-CUDA-BOX-LIBS-UNVERIFIED-2026-10-01`: the presence of `libcublas.so.12` and `libnvrtc.so.12` on the GH200 (venv `nvidia-*` wheels, host `/usr/local/cuda/lib64`) is inferred from `stack/train.lock`, not read. The driver's CUDA API version (580 series → 13.0) is inferred. Rung 0 is the falsifier.
- `GAP-L-CUDA-OJAS-LINUX-LINKED-NOT-RUN-2026-10-01`: ojas cross-checks and cross-links for `aarch64-unknown-linux-gnu` (67 test executables, live tree at 2026-10-01T22:35Z, not a commit), but no binary has run on Linux. The local arm64 podman machine was stopped.
- `GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01`: cuBLAS `cublasGemmEx` with A/B `CUDA_R_16BF`, C `CUDA_R_32F` and `CUBLAS_COMPUTE_32F` was not confirmed from docs (the fetch did not return the type table) and cudarc's safe API writes bf16 C. Rung 0's probe is the falsifier.
- `GAP-L-CUDA-GDN-SPEED-INFERRED-2026-10-01`: the CUDA speed of tessl's sequential GDN scan is extrapolated from the M5 Pro bench (55.2 ms per layer per 2,048 tokens on battery), not measured on sm_90. The 5–10× step-slowdown estimate rests on it.
