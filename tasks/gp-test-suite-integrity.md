---
id: "gp-test-suite-integrity"
title: "Tests that cannot fail in CI: skip helpers swallow any device-open error, Go wgpu tests pass with no adapter, wgpu parity tolerance is absolute for small tensors, gradchecks 10^4x loose, weak tests behind done briefs"
status: ready
priority: 1
severity: high
type: test
owner: "unassigned"
due: "none"
labels:
  - "tests"
  - "ci"
  - "metal"
  - "wgpu"
  - "go"
  - "tolerances"
repositories:
  - "ojas"
planned_files:
  - "ojas-model/tests/"
  - "ojas-capi/src/tests.rs"
  - "ojas-capi/src/profile.rs"
  - "ojas-wgpu/tests/common/mod.rs"
  - "ojas-autograd/tests/gradcheck.rs"
  - "ojas-autograd/tests/multihead.rs"
  - "go/api_test.go"
  - "go/model_test.go"
  - "go/profile_test.go"
  - ".github/workflows/test.yml"
acceptance_criteria:
  - "Skip helpers skip only on the specific 'no device' condition (named error kind or message, as Go's 'requires Metal 4'), and fail on every other open error; OJAS_ALLOW_NO_GPU is read one way everywhere (one helper, owned by gp-test-support-dedup)"
  - "Go wgpu tests and ojas-capi/src/profile.rs:466-473 either run on a real adapter in CI (lavapipe installed, OJAS_WGPU_ALLOW_CPU_ADAPTER set) or report SKIP, never PASS, when no adapter opens; the Rust no-adapter cross-check is applied to Go"
  - "The wgpu parity tolerance is relative to the reference magnitude as its doc states (no 1.0 floor), with an explicit absolute floor only where a test needs one; any test that newly fails is triaged as a real bug or a documented tolerance"
  - "Gradcheck tolerances are tightened to what f64 references support (order 1e-5 relative or a stated derivation); per-op tolerances across CPU/Metal/wgpu are listed in one table with the reason for each difference"
  - "Independent tests back the weak done-claims: saved-LSE backward against an independent recompute (the pre-6a0a926 kernel or an f64 reference); the per-group LR proof runs in the Metal protocol; a BPE pre-token cache test (hit and miss give identical ids); an Exact end-to-end run at threads {1, 3, default} compares bits"
  - "A Metal test protocol runs ojas-metal's 228 tests, the 15 guarded Metal tests and the relevant #[ignore] tests on a Mac per release and records the commit and counts in docs/status.md (shared with gp-ci-and-repo-hygiene's protocol criterion)"
  - "Wall-clock bounds in tests are removed or made generous with the reason; the fixture-bless switch requires an explicit value; stress.rs:590 asserts a value, not only finiteness"
---

# Task brief v1

## Title
Tests that cannot fail in CI: skip helpers swallow any device-open error, Go wgpu tests pass with no adapter, wgpu parity tolerance is absolute for small tensors, gradchecks 10^4x loose, weak tests behind done briefs

Task: gp-test-suite-integrity
Type: test
Status: ready
Priority: 1 (High)
Severity: high
Owner: unassigned
Due: none
Labels: tests, ci, metal, wgpu, go, tolerances

## Repositories
- ojas

## Description
Filed by the second audit (2026-10-09), from a read of every skip helper and skip call site, the tolerance helpers and test.yml, plus a heuristic scan of all 1,924 Rust #[test] functions [A]. The core problem: CI has run Rust tests only on macOS since Linux went red, and there:
- 15 Metal tests outside ojas-metal skip;
- ojas-metal's 228 tests are only built;
- 89 #[ignore] tests run in no job;
- the hip feature is never built;
- ojas-cuda/tests/runtime_refusal.rs (macos + cuda) is never even compiled.

Findings:
- **Skips swallow any open error:** Rust skip helpers (ojas-model device.rs:64, gqa_tape.rs:253, metal_waits.rs:75, metal_hybrid_tape.rs:114, qwen35_metal_tape.rs:30; ojas-capi tests.rs:2058, profile.rs:437, owner.rs:130; ojas-infer device.rs:272) skip on any MetalBackend::new / WgpuBackend::open error when OJAS_ALLOW_NO_GPU is set, and the macOS job sets it job-wide. Go narrows its skip to 'requires Metal 4' (api_test.go:880).
- **Go wgpu tests pass without a device:** TestWgpuSessionMatchesCPU passes on any 'wgpu:' load error (go/api_test.go:845-860), and model_test.go:108-112 and profile_test.go:278-282 skip. The Go jobs install no Vulkan driver, so on Linux these are no-ops. ojas-capi/src/profile.rs:466-473 skips on any 'wgpu:' error with no env check.
- **Tolerance helpers:**
  - wgpu parity tolerance floors its scale at 1.0 (ojas-wgpu/tests/common/mod.rs:97, fold(1.0, max)) against its own doc at :10 ('relative to the largest reference magnitude'). It is an absolute 1e-4 for small tensors across about 93 call sites; muon.rs's own relative() shows the intended semantics.
  - Gradchecks use RTOL 2e-2 / ATOL 2e-3 against f64 references (ojas-autograd/tests/gradcheck.rs:9-10, multihead.rs:15-16; scale max(|got|,|expect|), lib.rs:79), so a 1-2% gradient bug passes.
  - The same op is tolerated differently per backend with no stated reason (Metal pointwise 1e-6/1e-5 per element, backend_parity.rs:18-19; Metal attention backward rel 1e-3, attention_forward.rs:141; wgpu one max-scaled 1e-4).
- **Weak tests behind done briefs:**
  - The 'bit-for-bit' saved-LSE test compares the backward with itself (gp-attention-kernels).
  - The per-group LR proof is #[ignore] and macOS-only (gp-autograd-and-model-primitives).
  - The Metal device-loss test runs only locally (gp-gpu-runtime-hardening).
  - No test targets the BPE pre-token cache (gp-tokenizer-throughput-and-decode).
  - 'Exact is bit-identical across thread counts' has no end-to-end test at threads {1, 3, default}, and the full exp_exact sweep is #[ignore] (exp_exact.rs:190).
- **Smaller [A]:**
  - Wall-clock bounds on lavapipe (ojas-wgpu/tests/drop.rs:27,62; bpe.rs:1171).
  - OJAS_BLESS_FIXTURE set to any value rewrites a fixture (ojas-capi/src/tests.rs:96).
  - ojas-cpu/tests/stress.rs:590 asserts only finiteness.
  - About 46 Go assertions match error text against 27 errors.Is checks; typed sentinels are owned by gp-capi-go-surface.

Second audit, 2026-10-09 at d431949: a falsification pass over the first audit, plus read-only audits of GPU kernel source, unsafe/FFI/concurrency, numerics/parsers and test-suite integrity. Labels: [V] re-read by the writer of this pass; [A] read by an audit subagent at d431949, not re-read; [C] command output; [U] unverified (needs a run). No cargo, GPU or device run was made in this audit.

### Mac-training re-rank (2026-10-09)
The user's current focus is LoRA fine-tuning on the Mac (Metal), run by sibling session ojas-7c. Its recommended path, provisional until its user decides, is ojas-model's Tape on ojas-metal, with the tessl provider (ojas-qwen35) getting small fixes only. 16-bit (bf16) LoRA is the critical path; 4/8-bit QLoRA is gated P2, per Unsloth's Qwen3.5 guidance and Lappi AUDIT/training-audit-2026-10-06.md:469.

- Kept P1: Metal tests never run in CI, and the fine-tune runs on Metal.

### Gate run (2026-10-09, clean mirror of d431949)
Run on a `git archive` mirror of HEAD under mac_heavy.sh [C]. Toolchain: rustc 1.99.0, clippy 0.1.99, rustfmt 1.10.0, go 1.27.2. Siblings built: tessl 56b62c05 (clean); gusset 159d352a (31 dirty entries, all agent/editor config, none under crates/ or internal/). Logs are in this session's scratchpad gates/logs/.

| Gate | Result |
|---|---|
| cargo fmt --all --check (and the 18-package -p list) | FAIL, 31 hunks, all in ojas files |
| clippy --workspace --all-targets -D warnings | PASS |
| cargo test --workspace --release, --test-threads=2, OJAS_ALLOW_NO_GPU unset | PASS: 1825 passed, 0 failed, 41 ignored, 211 suites; Metal and wgpu ran on the real GPU, no SKIP lines |
| rustdoc with -D warnings | FAIL, 53 errors in 12 crates |
| cargo +1.97 check --workspace | PASS (default features, no --all-targets) |
| CUDA clippy + --no-run with --features cuda; ojas-hip check | PASS |
| Go: engine build, go vet, go test -tags gusset_pkgconfig, gofmt -l | PASS |
| sitegen -check (clean clone) | FAIL: docs/bench-plots.md has no category |

- Context: on this Mac with a real GPU, all 1825 tests pass, including the Metal suites that CI only builds. The gaps in this brief are about what CI exercises and what a test can catch, not about tests failing today. The 41 ignored tests are the non-cuda-feature ones; the 48 CUDA device tests are behind --features cuda and never listed.

## Acceptance criteria
- [ ] Skip helpers skip only on the specific 'no device' condition (named error kind or message, as Go's 'requires Metal 4'), and fail on every other open error; OJAS_ALLOW_NO_GPU is read one way everywhere (one helper, owned by gp-test-support-dedup)
- [ ] Go wgpu tests and ojas-capi/src/profile.rs:466-473 either run on a real adapter in CI (lavapipe installed, OJAS_WGPU_ALLOW_CPU_ADAPTER set) or report SKIP, never PASS, when no adapter opens; the Rust no-adapter cross-check is applied to Go
- [ ] The wgpu parity tolerance is relative to the reference magnitude as its doc states (no 1.0 floor), with an explicit absolute floor only where a test needs one; any test that newly fails is triaged as a real bug or a documented tolerance
- [ ] Gradcheck tolerances are tightened to what f64 references support (order 1e-5 relative or a stated derivation); per-op tolerances across CPU/Metal/wgpu are listed in one table with the reason for each difference
- [ ] Independent tests back the weak done-claims: saved-LSE backward against an independent recompute (the pre-6a0a926 kernel or an f64 reference); the per-group LR proof runs in the Metal protocol; a BPE pre-token cache test (hit and miss give identical ids); an Exact end-to-end run at threads {1, 3, default} compares bits
- [ ] A Metal test protocol runs ojas-metal's 228 tests, the 15 guarded Metal tests and the relevant #[ignore] tests on a Mac per release and records the commit and counts in docs/status.md (shared with gp-ci-and-repo-hygiene's protocol criterion)
- [ ] Wall-clock bounds in tests are removed or made generous with the reason; the fixture-bless switch requires an explicit value; stress.rs:590 asserts a value, not only finiteness

## Planned files
- ojas-model/tests/
- ojas-capi/src/tests.rs
- ojas-capi/src/profile.rs
- ojas-wgpu/tests/common/mod.rs
- ojas-autograd/tests/gradcheck.rs
- ojas-autograd/tests/multihead.rs
- go/api_test.go
- go/model_test.go
- go/profile_test.go
- .github/workflows/test.yml
