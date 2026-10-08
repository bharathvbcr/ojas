"""Minimal sota-toy Transformer LM for MLX throughput / dynamics reference.

Shapes match metal-native / burn-port `sota_toy`:
  4 layers, dim 128, heads 4, kv 2, mlp 384, T=256, V=1024, B=16.

Parity gaps vs full arch_02 (documented in README):
  no bigram hash, VE, XSA, value-residual, smear, resid_mix, U-net skips,
  or logit softcap. Attention is MLX's unfused SDPA path (no fused Metal bwd).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SotaToyConfig:
    vocab_size: int = 1024
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    model_dim: int = 128
    mlp_dim: int = 384
    seq_len: int = 256
    batch: int = 16
    rope_base: float = 10_000.0
    rope_dims: int = 8
    rms_eps: float = 1e-5
    tied_embed_init_std: float = 0.005

    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads

    @property
    def tokens_per_step(self) -> int:
        return self.batch * self.seq_len


def _rope_freqs(head_dim: int, rope_dims: int, base: float, t: int) -> mx.array:
    """Cos/sin for positions [0, T), applied to the first `rope_dims` of each head."""
    half = rope_dims // 2
    inv = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) / half))
    pos = mx.arange(t, dtype=mx.float32)
    freqs = mx.outer(pos, inv)  # [T, half]
    # pad to head_dim/2 so we can apply to full head (extra dims get identity)
    full_half = head_dim // 2
    if half < full_half:
        pad = mx.zeros((t, full_half - half), dtype=mx.float32)
        freqs = mx.concatenate([freqs, pad], axis=-1)
    return mx.cos(freqs), mx.sin(freqs)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: [B, T, H, D]; cos/sin: [T, D/2]."""
    b, t, h, d = x.shape
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos[:, None, :]  # [T, 1, D/2]
    sin = sin[:, None, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    # interleave
    out = mx.stack([out1, out2], axis=-1).reshape(b, t, h, d)
    return out


class GqaAttn(nn.Module):
    def __init__(self, cfg: SotaToyConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.model_dim
        h, hkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.wq = nn.Linear(c, h * d, bias=False)
        self.wk = nn.Linear(c, hkv * d, bias=False)
        self.wv = nn.Linear(c, hkv * d, bias=False)
        self.wo = nn.Linear(h * d, c, bias=False)
        self.scale = d**-0.5

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        cfg = self.cfg
        b, t, _ = x.shape
        h, hkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        q = self.wq(x).reshape(b, t, h, d)
        k = self.wk(x).reshape(b, t, hkv, d)
        v = self.wv(x).reshape(b, t, hkv, d)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # Expand KV for GQA
        rep = h // hkv
        k = mx.repeat(k, rep, axis=2)
        v = mx.repeat(v, rep, axis=2)
        # [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        # Causal scores — unfused (MLX Metal has no fused SDPA bwd)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = mx.triu(mx.full((t, t), -1e9, dtype=scores.dtype), k=1)
        scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        y = weights @ v  # [B, H, T, D]
        y = y.transpose(0, 2, 1, 3).reshape(b, t, h * d)
        return self.wo(y)


class Mlp(nn.Module):
    def __init__(self, cfg: SotaToyConfig):
        super().__init__()
        self.up = nn.Linear(cfg.model_dim, cfg.mlp_dim, bias=False)
        self.down = nn.Linear(cfg.mlp_dim, cfg.model_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: SotaToyConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rms_eps)
        self.attn = GqaAttn(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.model_dim, eps=cfg.rms_eps)
        self.mlp = Mlp(cfg)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, cfg: SotaToyConfig | None = None):
        super().__init__()
        self.cfg = cfg or SotaToyConfig()
        c = self.cfg
        self.tok_emb = nn.Embedding(c.vocab_size, c.model_dim)
        self.blocks = [Block(c) for _ in range(c.num_layers)]
        self.final_norm = nn.RMSNorm(c.model_dim, eps=c.rms_eps)
        # Tied head: project with tok_emb.weight.T via explicit matmul in forward.
        self._init_weights()

    def _init_weights(self):
        std = self.cfg.tied_embed_init_std
        self.tok_emb.weight = mx.random.normal(self.tok_emb.weight.shape) * std
        for block in self.blocks:
            for lin in (block.attn.wq, block.attn.wk, block.attn.wv, block.attn.wo,
                        block.mlp.up, block.mlp.down):
                fan_in = lin.weight.shape[1]
                lin.weight = mx.random.normal(lin.weight.shape) * (1.0 / math.sqrt(fan_in))

    def __call__(self, input_ids: mx.array) -> mx.array:
        """input_ids: [B, T] int32 → logits [B, T, V]."""
        cfg = self.cfg
        b, t = input_ids.shape
        cos, sin = _rope_freqs(cfg.head_dim, cfg.rope_dims, cfg.rope_base, t)
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        # Tied embedding head: [B,T,C] @ [C,V]
        return x @ self.tok_emb.weight.T
