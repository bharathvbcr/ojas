# gemma-metal

Full-custom **Rust + Metal 4** inference for **Gemma 4**. Prove on **E4B** (PLE + KV-share), then scale to **31B** (no PLE; global `k_eq_v`).

**Out of v1:** vision/audio encoders; HF A10G challenge submissions.

## Goal

| Step | Model | Why |
|------|-------|-----|
| Prove | `google/gemma-4-E4B-it` | Hardest correctness: PLE, KV-share, dual head dims 256/512 |
| Scale | Gemma 4 31B | Different deltas — not a scaled E4B |
| Beat | Local Ollama `gemma4:31b-mlx` ≈ 11–12 tok/s on M5 Pro 64GB | Ship gate ≥15 tok/s Q4 decode |

Product lane is **honest INT4 (+ MTP later)**, not challenge frontier PPL-burn.

## Relation to MLSystemsLab training

| Crate | Role |
|-------|------|
| [`../../tessl/`](../../tessl/) | Shared Metal 4 encode, residency, packed binder, TensorOps/simdgroup **GEMM** (prefill), util ops, MTLTensor prep |
| **`gemma-metal`** (this crate) | Gemma graph: Q4 banks, PLE split, dual KV, decode **GEMV**, dual FA, tokenizer, benches |
| `arch_02_value_resid/metal-native/` | **Training** stays here — do not train in gemma-metal / tessl |

Reuse the runtime substrate (~encode/GEMM). Do **not** drag bwd/Muon/XSA/VE or fork metal-native.

## Layout

```
../MLSystemsLab/research/parameter-golf/
  crates/tessl/   # Phase 0b extract
  gemma-metal/            # this product crate
    src/                  # config … gpu_model, mtp, kernels, thermal, …
    src/bin/bench.rs      # Phase 4 speed harness
    src/bin/serve.rs      # Phase 6 OpenAI-compatible stub
    kernels/              # Gemma-only overlay: PLE + persistent-interp (general kernels live in tessl)
    bench/                # Phase 0 multi-runtime harness → docs/gates.md
    docs/
    DECISIONS.md
```

## Status (tree as of 2026-07-13)

| Phase | Plan | In tree |
|-------|------|---------|
| **0** Baselines + gates | Measure LiteRT/MLX/Ollama | **Done** — measured rows in [`docs/gates.md`](docs/gates.md) |
| **0b** Extract `tessl` | Encode + GEMM + MTLTensor hooks | **Done** |
| **1** Scaffold | Q4 banks, PLE split, dual KV, tokenizer | **Done** |
| **2** E4B hot kernels | Real GEMV / dual FA / PLE / MLP | **Done** |
| **3** Parity | HF/MLX logits | **Partial** — synthetic host forward + JSON stubs; **no** real-weight HF/MLX logit parity |
| **4** E4B speed | Hot banks + decode tok/s | **Measured** — real E4B MLX Q4 **~15.9 tok/s** / TTFT ~230 ms (was ~13.9 / 265 ms); vectorized GEMV + GPU embed; **still below** gate ≥48–60 (vs mlx ~76). PLE Hot skipped. |
| **5** MTP | Draft / verify / centroids | **Real assistant weights loaded** (`google/gemma-4-E4B-it-assistant`); draft~13 ms/4 toks; cross-KV stand-in; backbone accept **unmeasured** |
| **6** 31B + serve | Config deltas + HTTP | **Hot path wired**; HF 31B 4bit pull **blocked** (rate limit, ~129 MB/18 GB); Ollama ~12.3 tok/s documented; `serve --preset 31b` tries Hot when shards land |

Crate version tag: `gemma-metal-0.1.0-phase6` (`src/lib.rs`).

```bash
cargo run --release --bin bench -- --e4b   # real E4B from HF cache
cargo run --release --bin serve -- --preset 31b
```

## Docs

| Doc | Contents |
|-----|----------|
| [`docs/architecture.md`](docs/architecture.md) | E4B vs 31B, dual FA, GEMV vs GEMM, PLE 4GB, KV layout, MTP |
| [`docs/gates.md`](docs/gates.md) | Living Phase-0/4 numbers + locked honest targets |
| [`docs/dev.md`](docs/dev.md) | Build/test, module map, kernel inventory |
| [`docs/research.md`](docs/research.md) | Concise source links |
| [`DECISIONS.md`](DECISIONS.md) | Extract vs rewire, lanes, PLE/FA/Core AI |

## Quick commands

```bash
cd gemma-metal && cargo test
cargo run --release --bin bench
cargo run --release --bin bench -- --e4b
cargo run --release --bin serve -- --port 8787 --preset e4b

# Phase 0 baselines
cd bench && python3 bench.py probe
```

See [`docs/dev.md`](docs/dev.md) and [`bench/README.md`](bench/README.md).
