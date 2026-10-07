# ojas-cuda

NVIDIA CUDA for ojas, through `cudarc` (0.19.10, `cuda-12080` bindings, libraries loaded at run
time). Three things live here:

- `CudaDevice`: the device probe (open ordinal 0, run one affine kernel before `Ok`).
- `CudaBackend`: the `ojas_core::Backend` implementation. `upload`, `download` and `sync` work;
  every compute op returns `OjasError::Unsupported` naming the op. A sticky driver failure
  (an illegal address, a failed launch, a device-side assert, …) is
  `OjasError::DeviceLost`, so a caller stops instead of retrying on a dead context.
- `Qwen35Step`: the Qwen3.5 whole-step training provider, design (B) in
  [`docs/cuda-backend-scoping.md`](../docs/cuda-backend-scoping.md), the CUDA counterpart of
  `ojas-qwen35`. Every compute method refuses with `Unsupported` until its kernels are wired and
  have run on a device. The kernels it will run are here already, each tested against a float64
  host reference (table below).

Everything that touches a device is behind the `cuda` feature, off by default. Without it the
crate builds on any host and its host-side tests run there.

**Status (2026-10-06).** Milestones M0 (K0, K1) and M1 (K8, K11, the tiny-fixture loader), K2(i)
(GDN at the published rule), and L-cuda-small's K3, K4, K6, K7, K9, K10. On 2026-10-05 the GH200
ran `rung0` (174/174 checks pass) and every device-test binary (`--ignored --test-threads=1`,
exit 0), all cross-built from this code when it lived in the standalone sibling crate it was
merged from. The merged crate has not yet run on a device.

## What is here

| Piece | Where | Tested on the Mac |
| --- | --- | --- |
| `CudaRuntime`: library probe before any cudarc call, sm_90 check, one stream, cuBLAS handle (32 MiB workspace, default math, atomics never set) | `src/runtime.rs` | refusal path only (`tests/runtime_refusal.rs`) |
| `CudaBuffer<T>`: typed, length-checked, held against a bounded `AllocBudget`; `CudaDeviceBuffer` backs an `ojas_core::Tensor` | `src/buffer.rs`, `src/budget.rs` | budget logic |
| NVRTC compile cache: key = module, FNV-1a of the source, length, options (architecture and `--fmad` included), NVRTC version; bounded in entries and bytes, LRU | `src/nvrtc_cache.rs` | yes |
| K0: f32↔bf16 (RNE), `copy_cols`, `deliver` (copy / `+=`), `zero`, `scatter_add_rows` (distinct rows), `ce_gather_rows` (f32 / bf16) | `src/kernels.rs`, `src/k0.rs`, `src/k0_plan.rs` | plans and host references |
| K1: GEMM `nn`/`tn`/`nt`: ExactF32 (fixed-k FFMA kernel) and Bf16 (cuBLAS `GemmEx` bf16→f32, or the FFMA kernel rounding on load) | `src/gemm.rs`, `src/gemm_plan.rs` | cuBLAS row-major mapping, host references |
| The crate's one `exp`, `log`, softplus, sigmoid, SiLU and SiLU' (device prelude plus bit-identical host emulation) | `src/k8_act.rs` | ulp bounds (release tier), special values |
| K8: SwiGLU forward (f32 or bf16 out) and backward (one or two output buffers), exact-f32 residual add, over `ColWindow`s | `src/k8_plan.rs`, `src/k8_kernels.rs`, `src/k8.rs` | plans, bitwise host references |
| K11: AdamW in place (torch single-tensor order, per-entry lr scale, decay and step count, inactive entries untouched) and the squared gradient norm (per-chunk f32 partials, f64 sum) | `src/k11_host.rs`, `src/k11_kernels.rs`, `src/k11.rs` | the f32 emulation against the decay-sensitive torch golden (≤ 1e-6, all 7 mutations ≥ 100x) and the adamw_f float64 golden |
| K2(i): the GDN training scan at the published rule, forward and backward | `src/gdn_kernels.rs`, `src/gdn_plan.rs`, `src/gdn.rs`, `src/gdn_host.rs` | the bitwise host mirror against the float64 reference; the reference against tessl's in process |
| K3 gates, K4 conv1d, K6 q/k norm + RoPE, K7 RMSNorm, K9 embed, K10 cross-entropy rows, and their shared reductions | `src/small_common.rs` and one `*.rs` / `*_cuda.rs` pair each | host mirrors and goldens |
| Tiny-fixture loader: tessl's `qwen35_train` (config through `ojas_io`'s JSON reader, BF16 safetensors, `.npy` gradients), into device buffers | `src/tiny_fixture_published.rs` | parse, tamper and symlink refusal |
| The report, the watchdog and the end of `main`, shared by both rung binaries | `src/report_cli.rs` | unit tests, and the watchdog end to end (`tests/watchdog_exit.rs`) |
| Rung-0 and rung-(a) binaries | `src/bin/rung0.rs`, `src/bin/runga.rs` | arguments, refusal path |

Host references (`src/host_ref.rs`) are bitwise for K0 and for the FFMA GEMM, and float64 for GEMM
tolerances. The float64 references live in `tests/reference/`.

## Tests

| Files | What | Where they run |
| --- | --- | --- |
| `src/**` unit tests, `tests/reference_*.rs`, `tests/fixture_pins.rs`, `tests/fmt_boundary.rs`, `tests/step_refuses.rs`, `tests/watchdog_exit.rs` | host-side | everywhere; CI's linux job (`cargo test --workspace --release`) |
| `tests/runtime_refusal.rs` | the runtime refuses by name without the libraries | with `--features cuda`, on a host without the driver |
| `tests/device_*.rs` | every kernel against its host reference on the device, `#[ignore]`d | an sm_90 GPU, `--ignored --test-threads=1`; CI builds them (`cargo test -p ojas-cuda --features cuda --no-run`) |

Two tiers run only in release builds (`#[cfg_attr(debug_assertions, ignore)]`), at full
density: the k8_act ulp sweeps (~226M bit patterns, ~16 s in debug) and the K2 reference against
tessl's at Qwen3.5-2B heads (B=1, T=200, H=16, Dv=128). `cargo test --release` runs them.

`tests/reference_gdn_published_vs_tessl.rs` also reads tessl's live
`tests/common/gdn_train.rs` from the sibling checkout and fails, never passes, when it is absent.

## Numerics

- **ExactF32 GEMM.**
  - One CUDA-core FFMA kernel: `acc = fmaf(a, b, acc)` over ascending k, from +0.0, for every
    output. No tensor cores (TF32) and no split-K.
  - Its bits depend only on the inputs. They equal `host_ref::gemm_ffma_f32`, which uses Rust's
    fused `f32::mul_add`.
- **Bf16 GEMM.**
  - Operands are rounded to bf16 by the K0 cast, then `cublasGemmEx` runs with
    `CUDA_R_16BF` A/B, a `CUDA_R_32F` C and `CUBLAS_COMPUTE_32F`.
  - cudarc's safe `Gemm<bf16>` writes a bf16 C, so it is not used.
  - The fallback is the FFMA kernel rounding its operands on load: the same tier, bit-identical
    to the host emulation.
- **cuBLAS determinism scope.**
  - NVIDIA promises bitwise-identical results within one toolkit version only "on GPUs with the
    same architecture and the same number of SMs". The promise does not hold across streams or
    with atomics allowed (cuBLAS docs, "Results reproducibility").
  - This crate uses one stream, one handle, a fixed workspace and atomics never allowed.
  - So the Bf16 cuBLAS tier repeats run to run on one GH200, but a different SM count or cuBLAS
    version may change its bits. Each rung report records the SM count and the driver, NVRTC and
    cuBLAS versions.
  - The FFMA tier has no such scope.
- **NaN** (Fable's ruling, 2026-10-02). No kernel canonicalises a NaN: an input NaN passes
  through, and a NaN an operation makes has the hardware's bits (PTX `0x7fffffff`, aarch64
  `0x7fc00000`, x86 `0xffc00000`). Bitwise host-device claims are for finite values; the
  comparison harness (`check::diff_bits_f32`, `diff_bits_bf16`) matches any NaN with any NaN; and
  a non-finite loss or gradient stops the step before any moment moves.
- **NVRTC options.**
  - `--gpu-architecture=compute_90`: PTX, which the driver JITs for sm_90. cudarc 0.19.10 exposes
    no CUBIN getter.
  - `--fmad=false`: plain `*`/`+` become `.rn` instructions, which nothing may contract.
  - `--ftz=false --prec-div=true --prec-sqrt=true`.
- **Grids and reductions.** Every grid is a function of the shape, never of the SM count. Every
  output element has one writer, and there are no atomics. Every float reduction has a fixed
  tree, defined once in `small_common` (the warp butterfly, the block sums, GDN's four-warp
  column sum) with a bit-exact host mirror.

## Building on the Mac

```bash
cargo test -p ojas-cuda
```

```bash
cargo test -p ojas-cuda --features cuda
```

The first runs the host tests with no CUDA anywhere. The second links cudarc: the
library-refusal test runs, and the device tests compile and are ignored.

Cross-build for the GH200 (aarch64, glibc 2.39 sysroot), the recipe in
`docs/cuda-backend-scoping.md` §1.5. No CUDA SDK is needed: cudarc `dlopen`s the libraries at run
time.

```bash
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" cargo build --offline --release --target aarch64-unknown-linux-gnu -p ojas-cuda --features cuda --bin rung0 --bin runga --target-dir /Users/bharath/qd-campaign/target-aarch64-linux-ojas-cuda
```

```bash
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc" cargo test --offline --release --no-run --target aarch64-unknown-linux-gnu -p ojas-cuda --features cuda --target-dir /Users/bharath/qd-campaign/target-aarch64-linux-ojas-cuda
```

The binaries are under `aarch64-unknown-linux-gnu/release/` in that target dir, and the device
tests under its `deps/`. A binary's sha256 depends on the checkout path, which it embeds: pin the
sha256 of the file you copy, not one rebuilt elsewhere.

## Libraries on the box

- cudarc opens bare sonames (`libcublas.so.12`, `libnvrtc.so.12`, `libcuda.so.1`, …) with
  `dlopen` and **panics** if none loads.
- glibc reads `LD_LIBRARY_PATH` once, when the process starts. So:
  - the venv's library directories must be on it in the launching command;
  - the binary cannot add them itself.
- `CudaRuntime::open` probes `libcuda`, `libnvrtc` and `libcublas` first. It refuses by name,
  exits 2, and lists the names searched and any candidate file that exists but did not load.
- Both directories are needed:
  - `libcublas.so.12` loads `libcublasLt.so.12` from `nvidia/cublas/lib`;
  - NVRTC loads `libnvrtc-builtins.so.12.8` from `nvidia/cuda_nvrtc/lib`.
- `/usr/local/cuda/lib64` on the box holds neither (lead's read-only check, 2026-10-01).

## Rung 0 on the box

A single-GPU job of about 5 minutes.

**1. On the Mac: copy the binary** (`KEY=~/.ssh/bharath_m5_macbook_pro.pem`,
`BOX=ubuntu@192.222.51.246`, `T` the cross-build target dir above):

    shasum -a 256 $T/aarch64-unknown-linux-gnu/release/rung0
    ssh -i $KEY $BOX 'mkdir -p /home/ubuntu/bin /home/ubuntu/ojas-cuda'
    scp -i $KEY $T/aarch64-unknown-linux-gnu/release/rung0 $BOX:/home/ubuntu/bin/ojas-cuda-rung0

**2. On the box: one command.** Check the sha256 against the HANDOFF's value first, with
`sha256sum /home/ubuntu/bin/ojas-cuda-rung0`. Then:

    flock /home/ubuntu/queue/gpu.lock timeout 300 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 /home/ubuntu/bin/ojas-cuda-rung0 --out /home/ubuntu/ojas-cuda/rung0-$(date -u +%Y%m%dT%H%M%SZ); echo "rung0 exit $?"

- `flock` waits for the GPU lock. `timeout 300` is the outer cap.
- The binary enforces its own cap: 280 s by default, `--cap-secs` at most 300. When the cap
  fires, it writes what finished and exits 3.
- `NVIDIA_TF32_OVERRIDE=0` turns TF32 off library-wide. It is recorded in the report. The Bf16
  tier's bf16 operands are unaffected.

**Exit codes** (both rung binaries):

| Exit | Meaning |
| ---: | --- |
| 0 | Every check ran and passed |
| 1 | A check failed or panicked |
| 2 | Refused before any check: arguments, a missing library, a non-sm_90 device, or an existing report |
| 3 | The wall-clock cap expired |
| 124 | The outer `timeout` fired |

**The report** is `<out>/rung0-report.json`, one JSON object of kind `ojas-cuda.rung0` with
`quick: true`, holding:
- the device: name, compute capability, SM count, driver, NVRTC and cuBLAS versions, and the
  cuBLAS math and atomics modes read back;
- the environment;
- the libraries actually mapped (from `/proc/self/maps`);
- the NVRTC cache counts;
- every check with its status (`pass`, `fail`, `panicked`, `not_run`), its detail and its numbers.

A panic inside cudarc, such as a missing symbol, is caught per check and recorded as `panicked`.
It is never recorded as `pass`.

**What rung 0 checks** (`smoke::m0_phases`, which `runga` also runs):
- all kernel modules compile through NVRTC, plus one module at `compute_90a`;
- every K0 kernel equals its host reference bitwise, and a second run equals the first;
- the cuBLAS `GemmEx` bf16→f32 probe at 64³, within `2^-8·max|ref|` of float64, with its
  `cublas_status` recorded either way;
- K1 on tessl's ragged shapes, in every layout, for FFMA ExactF32, FFMA bf16 and cuBLAS bf16:
  - against float64 within tessl's bounds, `1e-4 + 1e-7·k` and `2e-3` (`tessl/src/gemm.rs:2286`);
  - the FFMA engines also bitwise against the host emulation;
- 5-run bit-equality per engine.

## Runga on the box (rung (a): M0 + M1 + K2(i))

`runga` runs every device check of M0 and M1 and K2(i)'s in one process, with rung 0's watchdog,
cap and exit codes. It writes one report, `<out>/runga-report.json` (kind `ojas-cuda.runga`). It
is a single-GPU job of a few minutes. Build it with the cross-build recipe above and pin the
sha256 of the file you copy.

    scp -i $KEY $T/aarch64-unknown-linux-gnu/release/runga $BOX:/home/ubuntu/bin/ojas-cuda-runga

On the box, check `sha256sum /home/ubuntu/bin/ojas-cuda-runga`, then:

    flock /home/ubuntu/queue/gpu.lock timeout 300 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 /home/ubuntu/bin/ojas-cuda-runga --out /home/ubuntu/ojas-cuda/runga-$(date -u +%Y%m%dT%H%M%SZ); echo "runga exit $?"

**What it checks**, in order:
1. the device, as rung 0 opens it, under a 4 GiB budget;
2. M0, as rung 0 runs it (`smoke::m0_phases`);
3. K8: SwiGLU forward (f32, bf16) and backward (shared and separate outputs) and the residual
   add, bit for bit against the host from sentinel-filled buffers, plus the activation sweep
   over ~1M inputs, bit for bit (any NaN matching any NaN);
4. K11 in three tiers:
   - the device trajectory against the f32 emulation, bit for bit, and a repeat;
   - on the decay-sensitive golden, the device within the pre-registered 1e-6 of torch's fp32
     masters, and torch's per-entry step counts;
   - on the adamw_f float64 golden, the sanity bounds;
5. the tiny-fixture loader: upload, read back, bit for bit;
6. K3, K4, K6, K7, K9 and K10 (`small_smoke`'s hooks), one phase each, at smoke scale;
7. K2(i) at the published rule (`gdn_smoke::gdn_published_checks`);
8. `runtime.no_leaked_buffers`: only the cuBLAS workspace is still reserved at the end.

Then, **report-only and never a check**: `extra.gdn_published_timing`.
- It times GDN at 4×8192, H=16, Dv=128, on its own runtime, opened after the first is dropped,
  with `--gdn-timing-budget-gib` (default 24, at least 16).
- It says `not_run` and why when `--no-gdn-timing` is passed, fewer than 60 s of the cap are
  left, the device has less memory than the budget, or an allocation is refused.
- The header's `golden_pins` names the sha256 of every golden compiled into the binary.

## Device tests on the box

From the `--no-run` cross-build above, copy each `deps/device_*-<hash>` binary and run it on the
box:

    flock /home/ubuntu/queue/gpu.lock timeout 600 env LD_LIBRARY_PATH=<cublas lib>:<nvrtc lib> ./device_k0-<hash> --ignored --test-threads=1 --nocapture

The binaries are `device_k0`, `device_k1` (tessl's 25-run determinism soak for every engine and
layout), `device_k8`, `device_k11`, `device_k3_gates_published`, `device_k4_conv1d`,
`device_k6_qk_norm_rope`, `device_k7_rmsnorm`, `device_k9_embed`, `device_k10_ce_rows`,
`device_gdn_published` and `device_gdn_published_mirror`. Each embeds its goldens with
`include_bytes!` and checks them against their pinned sha256, so the box needs no checkout.

`device_gdn_published` (K2(i)) needs `timeout 900`. Its timing test is report-only: it prints one
`GDN_PUBLISHED_TIMING {json}` line per shape (4×2048 and 4×8192, H=16, Dv=128) and needs about
15 GiB free on the device. Its rows go to the ledger as `quick`.
