# ojas-qwen35-cuda

The Qwen3.5 whole-step training provider on NVIDIA CUDA: design (B) in Lappi's
`AUDIT/ojas-training-2026-10-01/cuda-backend-scoping.md`. It is not an ojas
`Backend`. It is a standalone sibling crate with its own `[workspace]`; the ojas
coordinator merges it into `ojas-cuda` and `ojas-kernels` later.

**Status (2026-10-01): milestone M0 (K0, K1) and M1 (K8, K11, the tiny-fixture
loader), plus K2(i) (L-cuda-gdn).**
- Built and host-tested on the Mac.
- Cross-built for the GH200.
- **No device code has run on a GPU.** Every device test is `#[ignore]` and is
  NOT RUN until `rung0` and `runga` run on the box.

## What is here

| Piece | Where | Tested on the Mac |
| --- | --- | --- |
| `CudaRuntime`: library probe before any cudarc call, sm_90 check, one stream, cuBLAS handle (32 MiB workspace, default math, atomics never set) | `src/runtime.rs` | refusal path only (`tests/runtime_refusal.rs`) |
| `CudaBuffer<T>`: typed, length-checked, held against a bounded `AllocBudget` | `src/buffer.rs`, `src/budget.rs` | budget logic |
| NVRTC compile cache: key = module, FNV-1a of the source, length, options (architecture and `--fmad` included), NVRTC version; bounded in entries and bytes, LRU | `src/nvrtc_cache.rs` | yes |
| K0: f32↔bf16 (RNE), `copy_cols`, `deliver` (copy / `+=`), `zero`, `scatter_add_rows` (distinct rows), `ce_gather_rows` (f32 / bf16) | `src/kernels.rs`, `src/k0.rs`, `src/k0_plan.rs` | plans and host references |
| K1: GEMM `nn`/`tn`/`nt`: ExactF32 (fixed-k FFMA kernel) and Bf16 (cuBLAS `GemmEx` bf16→f32, or the FFMA kernel rounding on load) | `src/gemm.rs`, `src/gemm_plan.rs` | cuBLAS row-major mapping, host references |
| Rung-0 smoke binary | `src/bin/rung0.rs` | arguments, report file, watchdog, refusal |
| The crate's one `exp`, `log`, softplus, sigmoid, SiLU and SiLU' (device prelude plus bit-identical host emulation) | `src/k8_act.rs` | ulp bounds, special values |
| K8: SwiGLU forward (f32 or bf16 out) and backward (one or two output buffers), exact-f32 residual add, over `ColWindow`s | `src/k8_plan.rs`, `src/k8_kernels.rs`, `src/k8.rs` | plans, bitwise host references |
| K11: AdamW in place (torch single-tensor order, per-entry lr scale, decay and step count, inactive entries untouched) and the squared gradient norm (per-chunk f32 partials, f64 sum) | `src/k11_host.rs`, `src/k11_kernels.rs`, `src/k11.rs` | the f32 emulation against L-oracle's decay-sensitive torch golden (≤ 1e-6, all 7 mutations ≥ 100x) and the adamw_f float64 golden |
| K11's goldens, embedded with their pins | `src/k11_golden.rs` | parse, F's-builder check |
| Tiny-fixture loader: tessl's `qwen35_train` (config through `ojas_io`'s JSON reader, BF16 safetensors, `.npy` gradients), into device buffers | `src/tiny_fixture_published.rs` | parse, tamper and symlink refusal |
| Report and watchdog shared by the rung binaries; `runga`'s arguments | `src/report_cli.rs` | watchdog (4 tests ported from `rung0`), arguments |
| Rung (a) binary: every M0 and M1 check plus K2(i)'s, one report | `src/bin/runga.rs` | refusal path on the Mac |

Host references (`src/host_ref.rs`) are bitwise for K0 and for the FFMA GEMM, and float64 for
GEMM tolerances. L-cuda-oracle's float64 references live in `tests/reference/`, which that lane
owns.

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
  - Whether cuBLAS 12.8 accepts this combination is **unverified**
    (`GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01`). Rung 0 is the falsifier.
  - The fallback is the FFMA kernel rounding its operands on load: the same tier, bit-identical
    to the host emulation.
- **cuBLAS determinism scope.**
  - NVIDIA promises bitwise-identical results within one toolkit version only "on GPUs with the
    same architecture and the same number of SMs". The promise does not hold across streams or
    with atomics allowed (cuBLAS docs, "Results reproducibility").
  - This crate uses one stream, one handle, a fixed workspace and atomics never allowed.
  - So the Bf16 cuBLAS tier repeats run to run on one GH200, but a different SM count or cuBLAS
    version may change its bits. Each rung-0 report records the SM count and the driver, NVRTC
    and cuBLAS versions.
  - The FFMA tier has no such scope.
- **NVRTC options.**
  - `--gpu-architecture=compute_90`: PTX, which the driver JITs for sm_90. cudarc 0.19.10 exposes
    no CUBIN getter.
  - `--fmad=false`: plain `*`/`+` become `.rn` instructions, which nothing may contract.
  - `--ftz=false --prec-div=true --prec-sqrt=true`.
- **Grids.** Every grid is a function of the shape, never of the SM count. Every output element
  has one writer, and there are no atomics.

## Building on the Mac

```
# Host tests (no CUDA anywhere):
cargo test --offline --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml
# With cudarc linked. The library-refusal test runs here; device tests compile and are ignored:
cargo test --offline --features cuda --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml
```

Cross-build for the GH200 (aarch64, glibc 2.39 sysroot). This is the recipe in
`cuda-backend-scoping.md` §1.5. No CUDA SDK is needed: cudarc `dlopen`s the libraries at run time.

```
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc"
T=/Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda
cargo build --offline --release --target aarch64-unknown-linux-gnu --features cuda --bin rung0 \
  --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml --target-dir $T
cargo test --offline --release --no-run --target aarch64-unknown-linux-gnu --features cuda \
  --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml --target-dir $T
```

The binary is `$T/aarch64-unknown-linux-gnu/release/rung0`. Its sha256 depends on the checkout
path, which the binary embeds. Pin the sha256 of the file you copy, not one rebuilt elsewhere.

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

Rung 0 runs after post-F item 10, run by the lead. It is a single-GPU job of about 5 minutes,
under rule 4's $20, so it needs no yes. Fable's Q4 still holds it until the queue ends.

**1. On the Mac: copy the binary.**

    KEY=~/.ssh/bharath_m5_macbook_pro.pem; BOX=ubuntu@192.222.51.246
    shasum -a 256 /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda/aarch64-unknown-linux-gnu/release/rung0
    ssh -i $KEY $BOX 'mkdir -p /home/ubuntu/bin /home/ubuntu/ojas-cuda'
    scp -i $KEY /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda/aarch64-unknown-linux-gnu/release/rung0 $BOX:/home/ubuntu/bin/ojas-qwen35-cuda-rung0

**2. On the box: one command.** Check the sha256 against the HANDOFF's value first, with
`sha256sum /home/ubuntu/bin/ojas-qwen35-cuda-rung0`. Then:

    flock /home/ubuntu/queue/gpu.lock timeout 300 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 /home/ubuntu/bin/ojas-qwen35-cuda-rung0 --out /home/ubuntu/ojas-cuda/rung0-$(date -u +%Y%m%dT%H%M%SZ); echo "rung0 exit $?"

- `flock` waits for the GPU lock. `timeout 300` is the outer cap.
- The binary enforces its own cap: 280 s by default, `--cap-secs` at most 300. When the cap
  fires, it writes what finished and exits 3.
- `NVIDIA_TF32_OVERRIDE=0` turns TF32 off library-wide. It is recorded in the report. The Bf16
  tier's bf16 operands are unaffected.

**Exit codes:**

| Exit | Meaning |
| ---: | --- |
| 0 | Every check ran and passed |
| 1 | A check failed or panicked |
| 2 | Refused before any check: arguments, a missing library, a non-sm_90 device, or an existing report |
| 3 | The wall-clock cap expired |
| 124 | The outer `timeout` fired |

**The report** is `<out>/rung0-report.json`. It is one JSON object with `quick: true`, holding:
- the device: name, compute capability, SM count, driver, NVRTC and cuBLAS versions, and the
  cuBLAS math and atomics modes read back;
- the environment;
- the libraries actually mapped (from `/proc/self/maps`);
- the NVRTC cache counts;
- every check with its status (`pass`, `fail`, `panicked`, `not_run`), its detail and its numbers.

A panic inside cudarc, such as a missing symbol, is caught per check and recorded as `panicked`.
It is never recorded as `pass`.

**What rung 0 checks:**
- all kernel modules compile through NVRTC, plus one module at `compute_90a`;
- every K0 kernel equals its host reference bitwise, and a second run equals the first;
- the cuBLAS `GemmEx` bf16→f32 probe at 64³, within `2^-8·max|ref|` of float64. This is the
  gap's falsifier: its `cublas_status` is recorded either way;
- K1 on tessl's ragged shapes, in every layout, for FFMA ExactF32, FFMA bf16 and cuBLAS bf16:
  - against float64 within tessl's bounds, `1e-4 + 1e-7·k` and `2e-3` (`tessl/src/gemm.rs:2286`);
  - the FFMA engines also bitwise against the host emulation;
- 5-run bit-equality per engine.

## Runga on the box (rung (a): M0 + M1 + K2(i))

`runga` runs every device check of M0 and M1 and K2(i)'s in one process, with rung 0's watchdog,
cap and exit codes (0 / 1 / 2 / 3 as in the table above). It writes one report,
`<out>/runga-report.json`. It is a single-GPU job of a few minutes, under rule 4's $20. Build
it with the cross-build recipe above, adding `--bin runga`, and pin the sha256 of the file you
copy against the HANDOFF (`HANDOFF/ojas-l-cuda-m1-2026-10-01.md`).

    KEY=~/.ssh/bharath_m5_macbook_pro.pem; BOX=ubuntu@192.222.51.246
    scp -i $KEY <target-dir>/aarch64-unknown-linux-gnu/release/runga $BOX:/home/ubuntu/bin/ojas-qwen35-cuda-runga

On the box, check `sha256sum /home/ubuntu/bin/ojas-qwen35-cuda-runga`, then:

    flock /home/ubuntu/queue/gpu.lock timeout 300 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib NVIDIA_TF32_OVERRIDE=0 /home/ubuntu/bin/ojas-qwen35-cuda-runga --out /home/ubuntu/ojas-cuda/runga-$(date -u +%Y%m%dT%H%M%SZ); echo "runga exit $?"

**What it checks**, in order:
1. the device, as rung 0 opens it, under a 4 GiB budget;
2. M0: NVRTC, K0, the cuBLAS probe, K1 and K1 determinism, as rung 0 runs them
   (`smoke::m0_phases`);
3. K8: SwiGLU forward (f32, bf16) and backward (shared and separate outputs) and the residual
   add, bit for bit against the host from sentinel-filled buffers, plus the activation sweep:
   the crate's `exp`, `exp_nonpos`, `log`, softplus, sigmoid, SiLU and SiLU' over ~1M inputs,
   bit for bit;
4. K11 in three tiers:
   - the device trajectory against the f32 emulation, bit for bit, and a repeat. This covers
     every step's parameters, the final moments, every step's squared-norm partials and the
     per-entry step counts;
   - on L-oracle's decay-sensitive golden, the device within the pre-registered 1e-6 of torch's
     fp32 masters, and torch's per-entry step counts;
   - on the adamw_f float64 golden, the sanity bounds. At F's lr 1e-5 a decay slip is under any
     f32 bound, so this tier cannot judge decay;
5. the tiny-fixture loader: upload, read back, bit for bit;
6. L-cuda-small's K3, K4, K6, K7, K9 and K10 (`small_smoke`'s hooks), one phase each. These
   are smoke-scale: K10's V = 248,320 case runs at hidden 16, so W is 15.9 MB. Do not compare
   it with the full-scale (hidden 2048) memory figure of about 10–14 GB per call;
7. K2(i) at the published rule (`gdn_smoke::gdn_published_checks`);
8. `runtime.no_leaked_buffers`: only the cuBLAS workspace is still reserved at the end.

Then, **report-only and never a check**: `extra.gdn_published_timing`.
- It times GDN at 4×8192, H=16, Dv=128, on its own runtime, opened after the first is dropped,
  with `--gdn-timing-budget-gib` (default 24, at least 16).
- It says `not_run` and why when:
  - `--no-gdn-timing` is passed;
  - fewer than 60 s of the cap are left;
  - the device has less memory than the budget;
  - an allocation is refused.
- The header's `golden_pins` names the sha256 of every golden compiled into the binary.

## Device tests on the box

```
cargo test --offline --release --no-run --target aarch64-unknown-linux-gnu --features cuda ...   # on the Mac
# copy target/.../deps/device_k0-<hash> and device_k1-<hash>, then on the box:
flock /home/ubuntu/queue/gpu.lock timeout 600 env LD_LIBRARY_PATH=<cublas lib>:<nvrtc lib> ./device_k0-<hash> --ignored --test-threads=1
flock /home/ubuntu/queue/gpu.lock timeout 600 env LD_LIBRARY_PATH=<cublas lib>:<nvrtc lib> ./device_k1-<hash> --ignored --test-threads=1
```

`device_k1` includes tessl's 25-run determinism soak for every engine and layout.

`device_k8` and `device_k11` (L-cuda-M1) run the same way, from the same `--no-run` build:

    flock /home/ubuntu/queue/gpu.lock timeout 600 env LD_LIBRARY_PATH=<cublas lib>:<nvrtc lib> ./device_k8-<hash> --ignored --test-threads=1 --nocapture
    flock /home/ubuntu/queue/gpu.lock timeout 600 env LD_LIBRARY_PATH=<cublas lib>:<nvrtc lib> ./device_k11-<hash> --ignored --test-threads=1 --nocapture

- `device_k8` covers `runga`'s K8 checks, plus five repeats of every case.
- `device_k11` covers `runga`'s K11 checks, plus:
  - the decay-sensitive run's `K11_DEVICE_DECAY` line;
  - five device runs of the synthetic banks, each against the emulation;
  - the device against the f64 reference of `tests/reference/adamw.rs` on adamw_f's inputs.

L-cuda-small's device tests run the same way, from the same build:
- `device_k3_gates_published`
- `device_k4_conv1d`
- `device_k6_qk_norm_rope`
- `device_k7_rmsnorm`
- `device_k9_embed`
- `device_k10_ce_rows`

Each embeds its goldens with `include_bytes!` and checks them against the oracle manifest's
sha256, so the box needs no checkout. Their bounds and the pinned artifacts are in that lane's
HANDOFF:

    flock /home/ubuntu/queue/gpu.lock timeout 600 env LD_LIBRARY_PATH=<cublas lib>:<nvrtc lib> ./device_k3_gates_published-<hash> --ignored --test-threads=1

### `device_gdn_published` (K2(i), L-cuda-gdn)

The GDN training scan at the published rule: 10 `#[ignore]` tests. Their bounds and evidence are
in Lappi's `HANDOFF/ojas-l-cuda-gdn-2026-10-01.md`. That HANDOFF pins the artifact:
`device_gdn_published-81ac041bb9f4a1f2`, sha256 `f0e88668a89f5665043c1f71b0faf480c7b69edd207e6a55952c4a5735cbefb0`
(Lappi commit 7817895 on L-cuda-gdn's branch), cross-built in its own target dir,
`/Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-gdn`. A later rebuild changes
the sha256: pin the file you copy, against the newest HANDOFF.

On the Mac:

    KEY=~/.ssh/bharath_m5_macbook_pro.pem; BOX=ubuntu@192.222.51.246
    scp -i $KEY /Users/bharath/qd-campaign/target-aarch64-linux-ojas-qwen35-cuda-gdn/aarch64-unknown-linux-gnu/release/deps/device_gdn_published-81ac041bb9f4a1f2 $BOX:/home/ubuntu/bin/ojas-gdn-published-device

On the box, check `sha256sum /home/ubuntu/bin/ojas-gdn-published-device` against the pin, then:

    flock /home/ubuntu/queue/gpu.lock timeout 900 env LD_LIBRARY_PATH=/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/ubuntu/qd-venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib /home/ubuntu/bin/ojas-gdn-published-device --ignored --test-threads=1 --nocapture 2>&1 | tee /home/ubuntu/ojas-cuda/gdn-published-$(date -u +%Y%m%dT%H%M%SZ).log

- The timing test is report-only. It prints one `GDN_PUBLISHED_TIMING {json}` line per shape
  (4×2048 and 4×8192, H=16, Dv=128) and needs about 15 GiB free on the device.
- Its rows go to the ledger as `quick` (rule 8).
