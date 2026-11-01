# AGENTS.md

Guidance for coding agents (and people) changing ojas. The README says what
ojas is; this file says how to change it without breaking what it promises.

## Layout

ojas is a Cargo workspace (`Cargo.toml` lists the members) plus a Go module
in `go/`.

| Crate | Role |
|---|---|
| `ojas-core` | The API: `Tensor`, `OjasError`, `Budget`, the `Backend` trait, the shape validators (`*_dims`), autocast, the checkpoint schema |
| `ojas-cpu` | CPU reference backend; the `Numerics::Exact` contract lives here |
| `ojas-simd` | Fast-tier f32 GEMM micro-kernels (NEON, AVX2+FMA, optional Accelerate) |
| `ojas-autograd` | The reverse-mode `Tape` and the gradcheck helpers (`central_diff`, `gradients_match`) |
| `ojas-metal` | The Metal extension adapter over tessl: it maps `Backend` ops onto tessl's runtime, allocator and kernels (GPU code is macOS-only; a stub elsewhere). New Mac inference engines belong in tessl, not here |
| `ojas-wgpu` | Portable GPU backend (WGSL through wgpu) |
| `ojas-cuda`, `ojas-hip` | CUDA and HIP probes; their device features are off unless requested |
| `ojas-device` | Explicit device selection, never a silent CPU fallback |
| `ojas-kernels` | Shared launch geometry, kernel sources and a backend parity harness |
| `ojas-oracle` | Reference fixtures (f64 rows, torch goldens) and the parity gates |
| `ojas-io`, `ojas-data` | Safetensors and checkpoint IO; token loaders and the BPE tokenizer |
| `ojas-model`, `ojas-train`, `ojas-infer` | The nanolab GPT, its trainer and recipes, and CPU decoding |
| `ojas-capi`, `ojas-gusset-engine` | The in-process C/Go surface; the static library cgo links |
| `ojas-qwen35` | Host-neutral Qwen3.5 route: config reader, tensor names, optimizer groups, state-directory format (no step; Metal training is the Tape via `TapeStepProvider`) |

Direction & Modular Extensions: ojas is the universal, training-first framework,
and tessl and sparsl are its canonical modular extensions:
- `tessl`: dedicated Apple Silicon Metal 4 extension and Mac inference engine (unified memory allocator, MPP TensorOps, and Metal compute kernels for `ojas-metal`).
- `sparsl`: dedicated Sparse Computing extension (deterministic SpMV, CSR/CSC matrix representations, affine prefix scans, event-driven simulation, and sparse recompute masks).
See `docs/modular-extensions.md` and `docs/architecture.md` for full rationale and contracts.

Sibling checkouts:
- `../tessl` (github.com/bharathvbcr/tessl): Apple Silicon Metal 4 extension (path dependency for `ojas-metal`).
- `../../devtools/gusset` (github.com/bharathvbcr/gusset): in-process FFI bridge (path dependency for `ojas-gusset-engine`).
- `../sparsl` (github.com/bharathvbcr/sparsl): Sparse computing extension.
`.github/workflows/test.yml` reproduces that layout.

## Invariants a change must keep

- **No silent fallback.** A GPU backend that cannot run an op returns
  `OjasError::Unsupported` (or another explicit error). It never computes on
  the host behind the caller's back.
- **Exact means bit-identical.** Under `Numerics::Exact`, every reduction
  runs in a fixed, documented order, and the bits do not depend on the
  thread count, the tiling or the machine. Parallel passes split whole rows
  and never split a sum. Write gathers, not scatters: no atomics, and no two
  threads writing one value.
- **Budgets refuse, they never clamp.** Outputs and scratch are charged to
  the `Budget` before they are allocated. Going over is
  `OjasError::CapacityExceeded`.
- **Validate at the boundary.** Shapes and dtypes are checked by the pure
  `ojas_core::*_dims` validators before a backend reserves or dispatches, so
  every backend refuses a malformed call with the same variant and op name.
  The CPU backend refuses non-finite operands with `OjasError::NonFinite`
  instead of propagating them; new ops follow it.
- **No generic logic in an extension; no duplicate kernels.** Extensions provide
  dedicated compute substrates (`tessl` for Metal 4, `sparsl` for sparse
  computing). Shape and dtype validation, autocast, budget accounting, optimizer
  scalar formation, the `Tape` and the training loop live once, in ojas-core and
  the crates above it. A backend extension (`ojas-metal`, `ojas-wgpu`, `ojas-cuda`,
  `ojas-hip`) only maps a `Backend` op onto its device. Do not add an ojas
  `.metal` kernel where tessl already has one: call tessl's. Where both repos
  already carry the same op, converging them is a numerics decision (math mode,
  reduction order, fault word) that needs its own parity gate. Mac inference
  engines (prefill/decode, ICB, quantization, KV caches) are tessl's product;
  ojas does sampling and eval only.
- **Safe Rust.** Most library crates set `#![forbid(unsafe_code)]` in
  `lib.rs`. `ojas-cuda` and `ojas-hip` lift it only under their device
  features. `ojas-simd`, `ojas-device` and `ojas-gusset-engine` (the C ABI)
  contain `unsafe` blocks. Keep new code safe, and keep `unsafe` inside
  those crates. Release builds keep `overflow-checks = true`, so size
  arithmetic should be checked and refuse, not panic.
  The one sanctioned exception outside these crates is in the sibling tessl
  checkout, not in this workspace: tessl's `SharedBuf` (`src/shared_buf.rs`)
  carries a single audited `unsafe impl Send + Sync` over a pooled
  `StorageModeShared` Metal buffer (host pointer, length, retained
  `MTLBuffer`), so CPU threads can read and write the bytes a Metal kernel
  uses without a copy. It is guarded by per-buffer fences on tessl's commit
  timeline (`last_use` / `last_write` stamps set at bind time, whole-buffer
  leases, poison checked on every fence). ojas code reaches shared memory only
  through tessl's safe `SharedBuf` API; `ojas-core` and `ojas-metal` stay
  `#![forbid(unsafe_code)]` and must not add their own `unsafe impl Send` or
  `Sync` for a device buffer.

## Adding a `Backend` method

1. Add it to the `Backend` trait in `ojas-core/src/backend.rs`, with a
   default that refuses with `OjasError::Unsupported`.
2. Add it to `with_backend_methods!` in the same file. Every forwarding
   wrapper is generated from that list. The `the_method_list_names_every_trait_method`
   test fails if the list and the trait disagree.
3. Add it to the test harness lists in that file (`every_call_reaches`, and
   `default_framework_ops_refuse_and_sync_succeeds` if the default refuses),
   and to the call table in `ojas-model/tests/reach/mod.rs`.
4. Give `ojas-core/src/autocast.rs` an override that states its bf16 policy
   (rounded operands, f32-kept gradients, or passed through).
5. Put the shape validator in `ojas-core` and the CPU implementation in
   `ojas-cpu`. Check it against an independent f64 reference or a torch
   fixture, and check that the thread count does not change a bit.
6. If the op has a gradient, add the `Tape` record and op in
   `ojas-autograd/src/tape.rs`, plus a test that the tape's gradients equal
   the backward calls chained by hand.

## Commands

These are what CI runs (`.github/workflows/test.yml`).
`./scripts/ci_local.sh` runs the same gates locally, except the MSRV
toolchain build (CI's `msrv` job) and the HIP compile check.

```sh
./scripts/fmt_check.sh            # rustfmt over the ojas packages only (--fix to apply)
./scripts/check_tracked_targets.sh
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release -- --test-threads=1
cargo deny check                  # deny.toml: licenses, advisories, bans, sources
go run scripts/sitegen/main.go -check            # after editing any docs/*.md or crate README
go run scripts/benchreport/main.go plots -check  # after adding bench/results
```

- Don't use `cargo fmt --all`: it also formats the tessl and gusset path
  dependencies.
- The site under `site/` is generated from the markdown. Edit the markdown,
  then run `go run scripts/sitegen/main.go` to regenerate it; never edit the
  HTML by hand.
- Test counts quoted in `README.md` and `docs/status.md` come from
  `docs/data/test-runs.json`. Update the data, not the prose.
- The MSRV is `rust-version` in `Cargo.toml` (gusset's floor). CI's `msrv`
  job builds every target with that toolchain.

## Claims and evidence

- A performance claim needs a committed benchmark under `bench/results/`
  that shows it, against the strongest baseline (for example torch MPS or
  Accelerate), with the noise across repeated runs reported.
- A correctness claim needs a test that fails without the change.
- Docs follow the code. When they disagree, fix the doc.
