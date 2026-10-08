#!/usr/bin/env python3
"""MLX sota-toy baseline — throughput bench + short training smoke.

Measurement instrument for metal-native (Phase I), not a migration target.
Uses mlx.optimizers.Muon + MultiOptimizer (AdamW for embeddings / scalars),
clip_grad_norm, and mx.compile around the step.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

from model import MiniTransformer, SotaToyConfig

# Match burn-port / metal-native TrainConfig::sota_toy
MATRIX_LR = 0.025
TIED_EMBED_LR = 0.035
SCALAR_LR = 0.025
WEIGHT_DECAY = 0.04
ADAM_BETAS = [0.9, 0.95]
ADAM_EPS = 1e-8
MUON_MOM_START = 0.92
MUON_MOM_END = 0.95
MUON_MOM_WARMUP = 1500
GRAD_CLIP = 0.3


def muon_momentum(step: int) -> float:
    frac = min(step / MUON_MOM_WARMUP, 1.0)
    return (1.0 - frac) * MUON_MOM_START + frac * MUON_MOM_END


def build_optimizer(model: MiniTransformer) -> opt.MultiOptimizer:
    """Muon on 2D bank/linear weights; AdamW on embeddings; AdamW on scalars."""
    muon = opt.Muon(
        learning_rate=MATRIX_LR,
        momentum=MUON_MOM_START,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
        ns_steps=5,
    )
    adam_embed = opt.AdamW(
        learning_rate=TIED_EMBED_LR,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        bias_correction=True,
    )
    adam_scalar = opt.AdamW(
        learning_rate=SCALAR_LR,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        bias_correction=True,
    )

    def is_embed(path: str, _g) -> bool:
        return "tok_emb" in path

    def is_matrix(path: str, g) -> bool:
        # 2D+ non-embedding weights → Muon (attn/MLP linears)
        return (getattr(g, "ndim", 0) >= 2) and ("tok_emb" not in path)

    # Filters: first match wins; final catch-all is AdamW (RMSNorm scales, etc.)
    return opt.MultiOptimizer(
        [adam_embed, muon, adam_scalar],
        [is_embed, is_matrix],
    )


def set_muon_momentum(optimizer: opt.MultiOptimizer, mom: float) -> None:
    for o in optimizer.optimizers:
        if isinstance(o, opt.Muon):
            o.momentum = mom


def loss_fn(model: MiniTransformer, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    # Flatten CE over tokens
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_y = y.reshape(-1)
    return mx.mean(nn.losses.cross_entropy(flat_logits, flat_y))


def make_step(model: MiniTransformer, optimizer: opt.MultiOptimizer):
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    # Capture model + optim state so compiled updates are pure (MLX compile docs).
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x: mx.array, y: mx.array):
        loss, grads = loss_and_grad(model, x, y)
        grads, _norm = opt.clip_grad_norm(grads, GRAD_CLIP)
        optimizer.update(model, grads)
        return loss

    return step


def synthetic_batch(cfg: SotaToyConfig, rng: np.random.Generator) -> tuple[mx.array, mx.array]:
    ids = rng.integers(0, cfg.vocab_size, size=(cfg.batch, cfg.seq_len + 1), dtype=np.int32)
    x = mx.array(ids[:, :-1])
    y = mx.array(ids[:, 1:])
    return x, y


def load_fineweb_tokens(data_dir: str, split: str = "train", max_tokens: int = 0) -> np.ndarray:
    pattern = os.path.join(data_dir, f"fineweb_{split}_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no shards matching {pattern}")
    chunks = []
    total = 0
    for path in files:
        with open(path, "rb") as f:
            header = f.read(256 * 4)  # 256 int32 header (nanogpt-style) if present
            # FineWeb PG bins: often raw uint16 stream; detect via file size
            f.seek(0)
            raw = f.read()
        # Prefer uint16 token stream (arch_02 FineWeb SP1024)
        if len(raw) % 2 == 0:
            arr = np.frombuffer(raw, dtype=np.uint16)
        else:
            arr = np.frombuffer(raw[1024:], dtype=np.uint16)  # skip 1KB header guess
        chunks.append(arr)
        total += arr.size
        if max_tokens and total >= max_tokens:
            break
    tokens = np.concatenate(chunks)
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def batch_from_stream(
    tokens: np.ndarray, cfg: SotaToyConfig, cursor: int
) -> tuple[mx.array, mx.array, int]:
    need = cfg.batch * cfg.seq_len + 1
    if cursor + need > len(tokens):
        cursor = 0
    flat = tokens[cursor : cursor + need].astype(np.int32)
    cursor += cfg.batch * cfg.seq_len
    seq = flat[: cfg.batch * cfg.seq_len + 1]
    # reshape overlapping windows along the stream
    xs, ys = [], []
    for b in range(cfg.batch):
        s = b * cfg.seq_len
        xs.append(seq[s : s + cfg.seq_len])
        ys.append(seq[s + 1 : s + 1 + cfg.seq_len])
    x = mx.array(np.stack(xs))
    y = mx.array(np.stack(ys))
    return x, y, cursor


def count_params(model: MiniTransformer) -> int:
    from mlx.utils import tree_flatten

    return sum(p.size for _, p in tree_flatten(model.parameters()))


def run_bench(args: argparse.Namespace) -> None:
    cfg = SotaToyConfig(batch=args.batch, seq_len=args.seq_len)
    mx.random.seed(args.seed)
    model = MiniTransformer(cfg)
    mx.eval(model.parameters())
    optimizer = build_optimizer(model)
    step = make_step(model, optimizer)
    rng = np.random.default_rng(args.seed)

    # Warmup
    for i in range(args.warmup):
        set_muon_momentum(optimizer, muon_momentum(i))
        x, y = synthetic_batch(cfg, rng)
        loss = step(x, y)
        mx.eval(loss, model.parameters(), optimizer.state)

    times = []
    for i in range(args.bench_steps):
        set_muon_momentum(optimizer, muon_momentum(args.warmup + i))
        x, y = synthetic_batch(cfg, rng)
        t0 = time.perf_counter()
        loss = step(x, y)
        mx.eval(loss, model.parameters(), optimizer.state)
        times.append((time.perf_counter() - t0) * 1e3)

    avg = sum(times) / len(times)
    tps = cfg.tokens_per_step / (avg / 1e3)
    nparams = count_params(model)
    print(
        f"BENCH mlx-baseline | B={cfg.batch} T={cfg.seq_len} "
        f"tok/step={cfg.tokens_per_step} params={nparams} | "
        f"{avg:.1f} ms/step | {tps:.0f} tok/s | "
        f"loss={float(loss):.4f}"
    )
    print(f"  per-step ms: {[f'{t:.1f}' for t in times]}")


def run_smoke(args: argparse.Namespace) -> None:
    cfg = SotaToyConfig(batch=args.batch, seq_len=args.seq_len)
    mx.random.seed(args.seed)
    model = MiniTransformer(cfg)
    mx.eval(model.parameters())
    optimizer = build_optimizer(model)
    step = make_step(model, optimizer)

    cursor = 0
    tokens = None
    rng = np.random.default_rng(args.seed)
    if args.data_dir:
        tokens = load_fineweb_tokens(args.data_dir, "train", max_tokens=args.max_tokens or 1_000_000)
        print(f"loaded {len(tokens)} train tokens from {args.data_dir}")

    t0 = time.perf_counter()
    for i in range(args.iters):
        set_muon_momentum(optimizer, muon_momentum(i))
        if tokens is not None:
            x, y, cursor = batch_from_stream(tokens, cfg, cursor)
        else:
            x, y = synthetic_batch(cfg, rng)
        loss = step(x, y)
        mx.eval(loss, model.parameters(), optimizer.state)
        if (i + 1) % args.log_every == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"step {i:4d} | loss={float(loss):.4f} | "
                f"mom={muon_momentum(i):.3f} | {elapsed:.1f}s"
            )
    print(f"SMOKE done | {args.iters} steps | params={count_params(model)}")


def main():
    p = argparse.ArgumentParser(description="MLX arch_02 sota-toy baseline")
    p.add_argument("--bench", action="store_true", help="throughput timing (synthetic tokens)")
    p.add_argument("--bench-steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20, help="smoke training steps")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="FineWeb SP1024 shard dir (optional; else synthetic)",
    )
    p.add_argument("--max-tokens", type=int, default=0)
    args = p.parse_args()

    print(f"mlx device={mx.default_device()} | metal={mx.metal.is_available()}")
    if args.bench:
        run_bench(args)
    else:
        run_smoke(args)


if __name__ == "__main__":
    main()
