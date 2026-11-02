# Accelerated Engineering Workflows

This document establishes the canonical development, verification, and testing workflows for `ojas`. These workflows replace monolithic, serialized verification passes with layered, fast-feedback execution loops.

---

## Workflow Overview

```
+---------------------------------------------------------------------------------------+
|                                ACCELERATED WORKFLOW SUITE                             |
+---------------------------------------------------------------------------------------+
|  [WF1: Inner Dev Loop]       -> Fast edit/compile check (< 15s)                       |
|  [WF2: Tiered Testing]       -> Tier 1 (Shapes) -> Tier 2 (CPU) -> Tier 3 (Autograd)  |
|                                 -> Tier 4 (Metal) -> Tier 5 (Workspace Release)      |
|  [WF3: Parity Verification]  -> Isolated op check against PyTorch fixtures (Tolerances)|
|  [WF4: Metal Execution]      -> Heavy GPU lock, serialized execution, leak isolation  |
|  [WF5: Worktree & Sync]      -> Sibling paths, worktree cleanup, collision safety     |
+---------------------------------------------------------------------------------------+
```

---

## WF1: Fast Inner Development Loop (< 15s)

**Objective**: Provide sub-15 second syntax, clippy, and unit test verification for individual crates during op and feature development.

### Execution Script
Use `./scripts/dev_fast_check.sh`:

```bash
# Check ojas-core (default)
./scripts/dev_fast_check.sh

# Check a specific crate
./scripts/dev_fast_check.sh ojas-cpu

# Check multiple crates
./scripts/dev_fast_check.sh ojas-core ojas-cpu

# Fast syntax and typecheck only
./scripts/dev_fast_check.sh --check-only ojas-metal

# Run across all workspace packages
./scripts/dev_fast_check.sh --all
```

### Stages
1. **Syntax & Typecheck**: `cargo check -p <crate>`
2. **Surgical Clippy**: `cargo clippy -p <crate> -- -D warnings`
3. **In-crate Unit Tests**: `cargo test -p <crate> --lib -- --nocapture`

---

## WF2: Tiered Testing Pipeline

**Objective**: Structured gate execution from cheapest to most expensive, failing fast before running heavy regression suites.

| Tier | Scope | Target Crate | Budget / Time | Execution Command |
|---|---|---|---|---|
| **Tier 1** | Pure shape/dims validators & core traits | `ojas-core` | < 10s | `cargo test -p ojas-core --lib` |
| **Tier 2** | CPU reference backend & bit-exact numerics | `ojas-cpu` | < 30s | `cargo test -p ojas-cpu --lib --test exact_golden` |
| **Tier 3** | Autograd reverse-mode Tape & gradchecks | `ojas-autograd` | budget unverified (see note) | `cargo test -p ojas-autograd --test gradcheck --test gradcheck_random --test tape_grad_shape` |
| **Tier 4** | Metal backend unit tests (Metal hardware) | `ojas-metal` | < 60s | `cargo test -p ojas-metal --lib -- --test-threads=1` |
| **Tier 5** | Release Workspace Smoke (Pre-commit) | Full Workspace | < 3m | `cargo test --workspace --release --exclude ojas-metal -- --test-threads=4` |

> **On the Tier 3 row (corrected 2026-11-01).** This row used to read
> `cargo test -p ojas-autograd --test tape_parity`. **There is no `tape_parity` test target** — the
> command failed for anyone who ran it. Four signals: `ojas-autograd/tests/` contains no
> `tape_parity*` file; `rg -uu 'tape_parity'` across the repo matched only this line;
> `devmap_search` returns 0 items on a fresh index; and Cargo itself does not list it —
> `cargo metadata --no-deps` reports 28 test targets for `ojas-autograd`, among them `gradcheck`,
> `gradcheck_random` and `tape_grad_shape`, and **no** `tape_parity`. That last check is what
> verifies the replacement command resolves.
>
> The suite itself was **not run** to produce this correction, so the old `< 45s` budget is
> unverified and has been removed rather than copied forward: put a measured number here the first
> time someone runs it on a quiet machine. Note the cost is not only test time — `ojas-autograd`
> carries a macOS-only `ojas-metal` dev-dependency, so a cold run of this row builds `ojas-metal`
> and `tessl` too, which is why it belongs in the heavy-step queue rather than an inner loop.
> Also note this row **cannot be run from an ojas git worktree**: `ojas-metal/Cargo.toml:22` has
> `tessl = { path = "../../tessl" }`, which does not resolve from `.claude/worktrees/<lane>/`, so
> cargo fails there on every toolchain. Run it from the main checkout.

### Escalation Rule
Never run Tier $N+1$ until Tier $N$ passes cleanly. When modifying a core trait in `ojas-core`, always verify Tier 1 before running GPU or autograd tests.

---

## WF3: PyTorch Parity Verification Loop

**Objective**: Verify bitwise numerical correctness of isolated ops against PyTorch reference fixtures without spinning up end-to-end models.

### Fixture Locations

#### 1. Required Activation Fixtures (MANDATORY for `parity_check.sh`)
- **SAM 3.1 Backbone Activation Fixtures**: `SpikeTrack/fixtures/backbone/visem_22_f0.safetensors` (~3.2 GB)
  - *Status*: Verified present on disk.
  - *Role*: Contains pre-recorded input activations and ground-truth output activations from PyTorch oracle across ViT trunk and neck layers (`trunk.blocks.*`, `neck.convs.*`). Required for isolated op parity testing.
- **Pixel Decoder GroupNorm Fixtures**: `SpikeTrack/fixtures/groupnorm/pixel_decoder.safetensors` (~203 MB)
  - *Status*: Verified present on disk.
  - *Role*: Reference input/output activations for GroupNorm layer verification.

#### 2. Model Weights Checkpoint (OPTIONAL for op parity; required for M3+ full-model inference)
- **SAM 3.1 Multiplex Weights**: `SpikeTrack/models/sam3.1/sam3.1_multiplex.safetensors` (~3.5 GB)
  - *Status*: Optional for `parity_check.sh`.
  - *Note*: `parity_check.sh` checks for this path optionally (`# Resolve Weights (optional)`). Its absence does not impede isolated op validation. Full model weights will be staged during Milestone 3 full-model integration.

### Execution Script
Use `./scripts/parity_check.sh`:

```bash
# Verify all CPU parity fixtures
./scripts/parity_check.sh --backend cpu

# Verify a single op on CPU
./scripts/parity_check.sh --backend cpu --op gelu_erf

# Verify Metal parity on Apple Silicon
./scripts/parity_check.sh --backend metal --op attention

# Verify both CPU and Metal backends
./scripts/parity_check.sh --backend all
```

### Strict Numerical Tolerances
Tolerances are enforced bit-for-bit by `ojas-oracle/src/sam_fixtures.rs`:

| Op Family | Max Abs Error ($\text{tol}_{\text{abs}}$) | Max Rel Error ($\text{tol}_{\text{rel}}$) | Min Cosine Similarity ($\text{tol}_{\text{cosine}}$) |
|---|---|---|---|
| **RoPE Table / Cis** | `1e-6` | `1e-3` | `1.0 - 1e-9` |
| **RoPE Interleaved** | `1e-5` | `1e-3` | `1.0 - 1e-9` |
| **Exact GELU (erf)** | `1e-5` | `1e-3` | `1.0 - 1e-9` |
| **LayerNorm (Affine / 2D)** | `1e-4` | `1e-3` | `1.0 - 1e-9` |
| **GroupNorm** | `1e-4` | `1e-3` | `1.0 - 1e-9` |
| **Attention (SDPA)** | `1e-4` | `1e-2` | `1.0 - 1e-8` |

### Verification Formula
All three conditions must hold:
1. $\max |y_{\text{got}} - y_{\text{want}}| \le \text{tol}_{\text{abs}}$
2. $\frac{\max |y_{\text{got}} - y_{\text{want}}|}{\max(|y_{\text{want}}|, 10^{-3} \cdot \max |y_{\text{want}}|)} \le \text{tol}_{\text{rel}}$
3. $\frac{y_{\text{got}} \cdot y_{\text{want}}}{\|y_{\text{got}}\| \|y_{\text{want}}\|} \ge \text{tol}_{\text{cosine}}$

---

## WF4: Metal Test Execution Workflow (Deterministic & Non-Contended)

**Objective**: Prevent GPU resource contention, flaky allocations, and test failures caused by concurrent processes sharing the Metal device.

### Invariants
1. **Exclusive GPU Lock (`mac_heavy.sh`)**:
   Heavy GPU test suites and benchmarks must run under the machine-wide heavy-job lock (`mac_heavy.sh`). The script acquires an atomic lock directory (`mac-heavy.lock.d`), enforces a process memory footprint cap, and exports `MAC_HEAVY_LABEL`.
   ```bash
   # Invocation under the lock
   /Users/bharath/Code/research/Lappi-decision/tools/mac_heavy.sh ojas-metal-test \
     cargo test -p ojas-metal --test sam_fixtures_metal -- --ignored --nocapture --test-threads=1
   ```
2. **Thread Serialization (`--test-threads=1`)**:
   `ojas-metal` tests reading the process-wide device allocation gauge or shared Metal buffer pool fail under concurrent test threads. Always pass `--test-threads=1` when executing Metal tests.
3. **No Silent GPU Fallback**:
   On machines without Metal hardware (e.g. Linux CI or remote nodes), export `OJAS_ALLOW_NO_GPU=1` to allow clean skips. On Apple Silicon, ensure `OJAS_ALLOW_NO_GPU=0` (or unset) to enforce fail-closed execution.
4. **Budget Isolation**:
   Every test must instantiate `MetalBackend::new(Budget::new(bytes))` with a dedicated budget ledger. Test assertions must verify that allocated bytes match charged bytes without leaks.

---

## WF5: Cross-Workspace Synchronization & Worktree Management

**Objective**: Safely coordinate edits across sibling checkouts (`ojas`, `tessl`, `SpikeTrack`) without breaking path dependencies or causing branch collisions.

### Dependency Flow
```
SpikeTrack  ──>  ojas (ojas-core, ojas-cpu, ojas-metal, ojas-io)  ──>  tessl
     │                                                                  ▲
     └──────────────────────────────────────────────────────────────────┘
```

### Safety Rules
1. **Pre-flight Collision Detection**:
   Before modifying shared interface contracts (`ojas-core/src/backend.rs`, `tessl/src/runtime.rs`), run `gitpulse_collision_risk` across all three repositories to detect in-flight edits in other worktrees.
2. **Sibling Path Hygiene**:
   Ensure `SpikeTrack/Cargo.toml` points to clean, active checkouts (`../ojas/` rather than detached temporary worktrees).
3. **Worktree Pruning**:
   Keep worktree count bounded. Avoid dirty uncommitted files across inactive worktrees.
