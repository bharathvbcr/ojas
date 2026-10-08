# Developer guide — gemma-metal + tessl

For agents continuing real-weight load / parity / product tok/s. Reality of the tree.

## Diagnostic logs

Structured failure breadcrumbs go to **stderr** with prefixes like
`[gemma-metal:weights]`, `[gemma-metal:gpu]`, `[gemma-metal:bench]`,
`[gemma-metal:infer]`.

| Knob | Behavior |
|------|----------|
| *(unset)* / `GEMMA_METAL_LOG=1` | **ON** (default) — load / upload / decode / MTP stages + errors |
| `GEMMA_METAL_LOG=0` | Silence stage diag lines |
| *(unset)* / `GEMMA_METAL_INFER_LOG=1` | **ON** for bins — line-level ▶/◀ for every host inference op |
| `GEMMA_METAL_INFER_LOG=0` | Silence line-level inference log (needed for clean tok/s) |
| `cfg!(test)` + unset `INFER_LOG` | Infer log **OFF** (tests stay quiet unless env set) |
| `GEMMA_METAL_TRACE=1\|json\|sync` | Per-op decode µs rollup (separate; see `trace.rs`) |

Each line includes `Elapsed=` seconds since process start. Useful markers:

- `▶` / `✔` / `◀` — stage or op start / success / op end (with µs)
- `⚠ STALL` — CPU wait on GPU (`synchronize`) — potential pipeline stalls
- `ERROR` — failure with context (`Display`/`Debug`)
- `open shard [i/n]` / `tensor …` — weight load progress
- `step pos=…` / `layer[i]` / `gemv_*` / `fa_swa` / `fa_global` — forward ops
- `RSS_*=… MiB` — host RSS via `ps` (bench)

### Dump a full E4B / 31B decode step

Line-level log is **on by default** for `bench` / `serve`. Capture one token’s full
op stream:

```bash
cd gemma-metal

# E4B — full infer stream to a file (INFER_LOG defaults ON):
cargo run --release --bin bench -- --e4b 2>bench/results/infer_e4b.log

# 31B via serve preload path (same env knobs):
GEMMA_METAL_INFER_LOG=1 cargo run --release --bin serve -- --preset 31b 2>bench/results/infer_31b.log

# Product tok/s only (silence both diag families):
GEMMA_METAL_LOG=0 GEMMA_METAL_INFER_LOG=0 cargo run --release --bin bench -- --e4b

# Plus TRACE rollup after each token:
GEMMA_METAL_TRACE=1 cargo run --release --bin bench -- --e4b --trace 2>bench/results/infer_trace_e4b.log
```

One decode token looks like (abridged):

```text
[gemma-metal:infer] Elapsed=12.340s ▶ decode_step | pos=7 seed=Some(128) head=true …
[gemma-metal:infer] Elapsed=12.340s ▶ embed | pos=7 hidden=2560 bytes≈10.0 KiB hot=true
[gemma-metal:infer] Elapsed=12.341s ◀ embed done 412µs
[gemma-metal:infer] Elapsed=12.341s ▶ layer[0] | pos=7 type=SlidingAttention producer=true …
[gemma-metal:infer] Elapsed=12.341s ▶ rms_input | layer=0 …
[gemma-metal:infer] Elapsed=12.341s ◀ rms_input done 38µs
[gemma-metal:infer] Elapsed=12.341s ▶ gemv_q | layer=0 [2560x2560] bytes≈…
… fa_swa / gemv_o / residual_attn / ple_* / gemv_gate|up|down …
[gemma-metal:infer] Elapsed=12.355s ◀ layer[0] done 14012µs
… layers 1..N-1 …
[gemma-metal:infer] Elapsed=12.480s ▶ final_norm | pos=7 …
[gemma-metal:infer] Elapsed=12.481s ▶ gemv_lm_head | [262144x2560] …
[gemma-metal:infer] Elapsed=12.490s ▶ softcap_argmax | …
[gemma-metal:infer] Elapsed=12.491s ⚠ STALL decode_step pos=7 before tok readback
[gemma-metal:infer] Elapsed=12.505s · sample/argmax result next=42 pos=7
[gemma-metal:infer] Elapsed=12.505s ◀ decode_step done 165012µs
```

No in-tree download helper — Hub pulls use `hf download …` (resume via HF cache).
Diag `cache` / `weights` lines report snapshot paths, shard counts, and byte sizes so
incomplete 31B pulls show up before decode.

### Benchmark artifact writes

Normal `cargo test` runs never replace tracked `bench/results/*_latest.json`
files. The microbenchmark tests still exercise their measurement and
serialization paths, and timestamped diagnostics may be preserved, but moving
a diagnostic into the `latest` slot is an explicit action:

```bash
GEMMA_METAL_UPDATE_LATEST_ARTIFACTS=1 cargo test --release <exact-test-name>
```

Those test-produced records carry `evidence_status: "diagnostic_only"`; the
environment flag is not a performance-publication gate and does not add
revision, load, stability, or artifact-hash provenance. Do not cite them as
current benchmark evidence without a separate gated run.

Previously checked-in latest-slot records retained without a fresh gated run
carry `evidence_status: "historical_unverified"`; `_latest` names the continuity
slot, not the currency or publication status of its contents.

## Build / test

```bash
cd gemma-metal
cargo test
cargo run --release --bin bench
cargo run --release --bin bench -- --e4b
cargo run --release --bin serve -- --port 8787 --preset e4b   # or 31b
# Offline / no Metal toolchain: both prebuilt artifacts must be explicit.
TESSL_SKIP_AOT=1 TESSL_PREBUILT_METALLIB=/absolute/path/to/tessl.metallib \
  GEMMA_METAL_SKIP_AOT=1 GEMMA_METAL_PREBUILT_METALLIB=/absolute/path/to/gemma.metallib \
  cargo test
```

Each prebuilt variable must name an existing absolute regular file. The build
scripts canonicalize and track those files; neither falls back to an ignored
`default.metallib` in a source directory.

Normal AOT builds resolve the shared `gelu.h` through Tessl's
`DEP_TESSL_KERNELS` metadata and fail if either the metadata or header is
missing.

Needs macOS 26+ Metal 4 for GPU tests. Overlay: `kernels/*.metal` → `GEMMA_METAL_METALLIB`.

**Test count:** ~65 `#[test]`s (config/quant/PLE/KV/kernels/forward/gpu_model/mtp/parity/…).

## Module map (`gemma-metal/src/`)

| Module | Status | Role |
|--------|--------|------|
| `config` | **Real** | HF JSON → E4B/31B/assistant; presets |
| `quant` | **Real** | Affine Q4/Q8 (+ MLX Q4) |
| `ple` | **Real** | Per-layer split, 4 GiB checks |
| `kv` | **Real** | Producer/consumer; host helpers + GPU session path |
| `weights` | **Real** | Host `HostWeightBanks` from HF dir (PLE Hot skipped on real E4B speed path) |
| `kernels` | **Real** | Tiled GEMV/FA/RMS/PLE/MLP/softcap/KV-store + Hot upload |
| `forward` | **Real** | Host synthetic prefill + parity hooks |
| `gpu_model` | **Real** | Packed async Hot decode; GPU KV; MTP smoke |
| `mtp` | **Synthetic E2E** | Draft/verify in decode loop; no real assistant weights |
| `parity` | **Partial** | Synthetic/JSON; no HF/MLX logits yet |
| `bin/bench` | **Real** | Phase 4 tok/s + microbenches |
| `bin/serve` | **Stub generate** | `--preset e4b\|31b` metadata + OpenAI HTTP |

Version: `gemma-metal-0.1.0-phase6`.

## Kernel inventory extras (Phase 4+)

| Entry | Notes |
|-------|-------|
| `gemv_q4` / `gemv_q4_mlx` | Vectorized 1-thread/row decode GEMV (+ `*_tiled` legacy) |
| `kv_store_timestep` / `kv_ring_densify` | GPU-resident KV |
| `rms_norm_f32` | Residual / hidden RMS |
| `add_inplace_f32` | From tessl utils (residual add) |
| Hot `upload_quant_hot` | Q4/Q8 → `BufferKind::Hot` residency |

### Remaining gaps

1. E4B still ~4.8× below mlx (~15.9 vs ~76); lm_head / GEMV BW + packing.
2. HF/MLX logit parity.
3. MTP cross-KV draft forward + measured backbone accept.
4. 31B custom Hot (HF 4bit shards incomplete; Ollama documented).
5. Optional PLE Hot if it does not tank tok/s.

## Non-negotiables

1. Metal 4-only encode; Hot residency + packed binder.
2. Honest lane — no `w188` until parity green.
3. No M=1 TensorOps decode — dedicated GEMV.
4. Do not touch metal-native training.
5. PLE split — no buffer >4 GiB.
