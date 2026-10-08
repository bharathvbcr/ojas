# MLX baseline (Phase I) — measurement instrument

Python [MLX](https://github.com/ml-explore/mlx) port of **sota-toy shapes** for
arch_02, used to benchmark metal-native step time and (optionally) dynamics on
Apple Silicon. **Not a migration target** — exotic arch_02 modules are omitted
for a 1-day throughput reference.

## Shapes (match `TrainConfig::sota_toy` / metal-native)

| Knob | Value |
|------|-------|
| layers / dim / heads / kv | 4 / 128 / 4 / 2 |
| mlp | 384 |
| T / B / tok/step | 256 / 16 / 4096 |
| vocab | 1024 |
| Muon momentum | 0.92 → 0.95 over 1500 steps |
| Muon / AdamW LRs | 0.025 / embed 0.035 / scalar 0.025 |
| WD / clip | 0.04 / 0.3 |

Optimizer stack: `mlx.optimizers.Muon` + `MultiOptimizer` (AdamW for
`tok_emb`, AdamW catch-all for RMSNorm scales) + `clip_grad_norm`, with
`mx.compile` around the train step.

## Parity gaps vs full arch_02

This is a **minimal Transformer LM** (GQA + RoPE on first 8 dims + SiLU MLP +
tied embed head). Deliberately **not** ported:

- bigram hash embedding / smear gate
- value embedding (VE) / value residual / `vr_lambda`
- XSA, `resid_mix`, U-net skip weights
- logit softcap
- EMA / sliding BPB eval harness

Use metal-native or `train_gpt_sprint_native.py` for full-arch BPB. This
baseline answers: *what step time does a mature Metal stack get at these
shapes with Muon + compile?*

Every number here is on the unfused SDPA backward (GEMM attention), the only
Metal path in mlx 0.31.2 — same caveat called out in the throughput plan. MLX
`main` added a fused SDPA VJP on Metal in ml-explore/mlx#4563 (commit 83b976e,
2026-09-29), but no release contains it yet: v0.32.3, also published 2026-09-29,
predates that commit (checked 2026-10-08). Rerun the step-time numbers on the
first release that ships it before comparing them with fused-backward stacks.

## Requirements

```bash
# mlx 0.31+ (Muon + MultiOptimizer). Apple Silicon + macOS with Metal.
pip install mlx numpy
# or: uv pip install mlx numpy
```

Verified locally against **mlx 0.31.2**. Sandboxed/headless sessions without a
Metal device will fail at import/runtime (`No Metal device available`).

## Run

```bash
cd arch02/mlx-baseline

# Throughput bench (synthetic tokens, ~15 timed steps after warmup)
python train.py --bench --bench-steps 15 --warmup 3

# Short training smoke (synthetic)
python train.py --iters 20 --log-every 5

# Optional FineWeb shards (uint16 SP1024 bins)
python train.py --iters 50 --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024
```

Compare `--bench` ms/step and tok/s against metal-native:

```bash
cd ../metal-native
cargo run --release --bin train -- --bench --bench-steps 15 --tok-mult 1 \
  --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \
  --out out/bench_b16
```

## Layout

| File | Role |
|------|------|
| `model.py` | `MiniTransformer` + `SotaToyConfig` |
| `train.py` | MultiOptimizer step, `--bench` / smoke CLI |
| `README.md` | this file |
