#!/usr/bin/env python3
"""Cross-stack bisect: load metal-native dump_step2000 weights and continue on MPS/CPU.

Verdict:
  - gnorm explodes again → weights already poisoned before step 2000
  - gnorm stays healthy → fault is metal-native post-2000 dynamics

Usage:
  .venv/bin/python scripts/bisect_continue_mps.py \\
    --weights out/sota_f32_bisect_seed1337/dump_step2000/weights \\
    --data-dir ../../../parameter-golf/data/datasets/fineweb10B_sp1024 \\
    --iters 200 --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_shard(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype=np.uint16, offset=1024).copy()


def bigram_hash(toks: torch.Tensor, vocab: int = 512) -> torch.Tensor:
    """toks [B,T] int64 → bigram idx [B,T]."""
    b, t = toks.shape
    out = torch.empty_like(toks)
    out[:, 0] = vocab - 1
    if t > 1:
        t0 = toks[:, :-1].to(torch.int32)
        t1 = toks[:, 1:].to(torch.int32)
        out[:, 1:] = ((36313 * t1) ^ (27191 * t0)).remainder(vocab - 1).to(toks.dtype)
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, n_kv, rope_dims, qk_gain_init, value_residual):
        super().__init__()
        self.n_head, self.n_kv, self.head_dim = n_head, n_kv, dim // n_head
        self.rope_dims = rope_dims
        self.q_gain = nn.Parameter(torch.full((n_head,), qk_gain_init))
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x, q_w, k_w, v_w, out_w, v0=None, ve=None):
        b, t, c = x.shape
        h, hkv, d = self.n_head, self.n_kv, self.head_dim
        q = F.linear(x, q_w).view(b, t, h, d)
        k = F.linear(x, k_w).view(b, t, hkv, d)
        v = F.linear(x, v_w)
        if ve is not None:
            v = v + ve
        v = v.view(b, t, hkv, d)
        raw = v
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (d,))
        k = F.rms_norm(k, (d,))
        # partial RoPE (half-split)
        rd = self.rope_dims
        if rd > 0:
            half = rd // 2
            inv = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
            pos = torch.arange(t, device=x.device, dtype=torch.float32)
            ang = pos[:, None] * inv[None, :]
            cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]

            def rope(z):
                z1, z2 = z[..., :half], z[..., half:rd]
                rest = z[..., rd:]
                y1 = z1 * cos + z2 * sin
                y2 = z1 * (-sin) + z2 * cos
                return torch.cat([y1, y2, rest], dim=-1)

            q, k = rope(q), rope(k)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        # expand KV
        rep = h // hkv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)  # [B,H,T,D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return F.linear(y, out_w), raw


class Block(nn.Module):
    def __init__(self, dim, n_head, n_kv, mlp_dim, rope_dims, layer_idx, xsa, value_residual):
        super().__init__()
        self.attn = CausalSelfAttention(dim, n_head, n_kv, rope_dims, 1.5, value_residual)
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))))
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.ln = 1.0 / math.sqrt(layer_idx + 1)
        self.xsa = xsa
        self.layer_idx = layer_idx

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v0=None, ve=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_in = F.rms_norm(x_in, (x_in.size(-1),)) * self.ln
        attn_out, raw_v = self.attn(attn_in, q_w, k_w, v_w, out_w, v0=v0, ve=ve)
        if self.xsa:
            # paper XSA on mixed v (approx: skip for bisect speed — use attn_out as-is
            # Full XSA would need GQA-aware proj; omit for dynamics probe.
            pass
        x_mid = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        mlp_in = F.rms_norm(x_mid, (x_mid.size(-1),)) * self.ln
        h = F.linear(mlp_in, up_w)
        h = F.leaky_relu(h, 0.5).square()
        mlp_out = F.linear(h, down_w)
        return x_mid + self.mlp_scale.to(dtype=x_mid.dtype)[None, None, :] * mlp_out, raw_v


class SotaToy(nn.Module):
    def __init__(self):
        super().__init__()
        c, L, h, hkv = 128, 4, 4, 2
        mlp, v, bgv, bgd, ved = 384, 1024, 512, 48, 24
        kvd = hkv * (c // h)
        self.c, self.L, self.kvd = c, L, kvd
        self.bgv = bgv
        self.tok_emb = nn.Embedding(v, c)
        self.bigram_emb = nn.Embedding(bgv, bgd)
        self.bigram_proj = nn.Linear(bgd, c, bias=False)
        self.bigram_scale = nn.Parameter(torch.tensor(0.05))
        self.smear_gate = nn.Parameter(torch.zeros(c))
        self.ve_emb = nn.Embedding(v, ved)
        self.ve_proj = nn.Linear(ved, kvd, bias=False)
        self.ve_scale = nn.Parameter(torch.tensor(0.1))
        self.ve_layer_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(2)]
        )
        self.skip_weights = nn.Parameter(torch.ones(2, c))
        self.qo_bank = nn.Parameter(torch.empty(2 * L, c, c))
        self.kv_bank = nn.Parameter(torch.empty(2 * L, kvd, c))
        self.mlp_up = nn.Parameter(torch.empty(L, mlp, c))
        self.mlp_down = nn.Parameter(torch.empty(L, c, mlp))
        self.blocks = nn.ModuleList(
            [
                Block(c, h, hkv, mlp, 8, i, xsa=(i >= 2), value_residual=True)
                for i in range(L)
            ]
        )
        self.logit_softcap = 30.0
        self.ve_layers = [2, 3]

    def load_npy_tree(self, root: Path):
        def npy(p):
            return torch.from_numpy(np.load(p)).float()

        self.tok_emb.weight.data.copy_(npy(root / "tok_emb/weight.npy"))
        self.bigram_emb.weight.data.copy_(npy(root / "bigram/embed/weight.npy"))
        self.bigram_proj.weight.data.copy_(npy(root / "bigram/proj/weight.npy"))
        self.bigram_scale.data.copy_(npy(root / "bigram/scale.npy").reshape(()))
        self.smear_gate.data.copy_(npy(root / "smear/gate.npy"))
        self.ve_emb.weight.data.copy_(npy(root / "ve_shared/embed/weight.npy"))
        self.ve_proj.weight.data.copy_(npy(root / "ve_shared/proj/weight.npy"))
        self.ve_scale.data.copy_(npy(root / "ve_shared/scale.npy").reshape(()))
        for i in range(2):
            self.ve_layer_scales[i].data.copy_(
                npy(root / f"ve_layer_scales/{i}.npy").reshape(1)
            )
        sw = npy(root / "skip_weights.npy")
        if sw.ndim == 1:
            sw = sw.view(2, self.c)
        self.skip_weights.data.copy_(sw)
        self.qo_bank.data.copy_(npy(root / "qo_bank.npy"))
        self.kv_bank.data.copy_(npy(root / "kv_bank.npy"))
        self.mlp_up.data.copy_(npy(root / "mlp_up_bank.npy"))
        self.mlp_down.data.copy_(npy(root / "mlp_down_bank.npy"))
        for i, b in enumerate(self.blocks):
            base = root / f"blocks/{i}"
            b.attn.q_gain.data.copy_(npy(base / "attn/q_gain.npy"))
            b.attn.vr_lambda.data.copy_(npy(base / "attn/vr_lambda.npy"))
            b.attn_scale.data.copy_(npy(base / "attn_scale.npy"))
            b.mlp_scale.data.copy_(npy(base / "mlp_scale.npy"))
            b.resid_mix.data.copy_(npy(base / "resid_mix.npy"))

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        b, t = toks.shape
        c, L = self.c, self.L
        bg = bigram_hash(toks, self.bgv)
        x = self.tok_emb(toks)
        x = x + self.bigram_scale * self.bigram_proj(self.bigram_emb(bg))
        x = F.rms_norm(x, (c,))
        g = torch.sigmoid(self.smear_gate)[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        x = (1 - g) * x + g * x_prev
        x0 = x
        v0 = None
        skips = []
        for i, block in enumerate(self.blocks):
            if i >= L // 2:
                si = i - L // 2
                enc = L // 2 - 1 - si
                x = x + self.skip_weights[si][None, None, :] * skips[enc]
            ve = None
            if i in self.ve_layers:
                vi = self.ve_layers.index(i)
                ve = (
                    self.ve_proj(self.ve_emb(toks))
                    * self.ve_scale
                    * self.ve_layer_scales[vi]
                )
            q_w = self.qo_bank[i]
            out_w = self.qo_bank[L + i]
            k_w = self.kv_bank[i]
            v_w = self.kv_bank[L + i]
            up_w = self.mlp_up[i]
            down_w = self.mlp_down[i]
            x, raw_v = block(
                x, x0, q_w, k_w, v_w, out_w, up_w, down_w, v0=v0, ve=ve
            )
            if i == 0:
                v0 = raw_v
            if i < L // 2:
                skips.append(x)
        x = F.rms_norm(x, (c,))
        logits = F.linear(x, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--start-step", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=Path("out/bisect_mps"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() or args.device == "cpu" else "cpu")
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable; falling back to CPU")
        device = torch.device("cpu")

    model = SotaToy().to(device)
    model.load_npy_tree(args.weights)
    print(f"loaded {args.weights} → {device}")

    # Param groups matching metal/python
    matrix, embed, scalar = [], [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ("qo_bank", "kv_bank", "mlp_up", "mlp_down")):
            matrix.append(p)
        elif any(k in n for k in ("tok_emb", "bigram_emb", "ve_emb")):
            embed.append(p)
        else:
            scalar.append(p)

    # Use AdamW for all on MPS (Muon optional); still probes weight poisoning.
    opt = torch.optim.AdamW(
        [
            {"params": embed, "lr": 0.035},
            {"params": scalar, "lr": 0.025},
            {"params": matrix, "lr": 0.025},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.04,
    )

    shards = sorted(args.data_dir.glob("*train*.bin"))
    assert shards, f"no train shards in {args.data_dir}"
    tokens = load_shard(shards[0])
    print(f"shard {shards[0].name}: {len(tokens)} tokens")

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.jsonl"
    cursor = 0
    B, T = args.batch, args.seq_len
    need = B * T + 1

    with metrics_path.open("w") as mf:
        for i in range(args.iters):
            step = args.start_step + i
            if cursor + need > len(tokens):
                cursor = 0
            chunk = tokens[cursor : cursor + need]
            cursor += B * T
            x = torch.tensor(chunk[:-1].reshape(B, T), dtype=torch.long, device=device)
            y = torch.tensor(chunk[1:].reshape(B, T), dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3).item()
            opt.step()
            dt = (time.perf_counter() - t0) * 1e3

            # Scalar L2 telemetry
            resid = [b.resid_mix.detach().float().norm().item() for b in model.blocks]
            vr = [
                (b.attn.vr_lambda.detach().float().norm().item() if hasattr(b.attn, "vr_lambda") else 0.0)
                for b in model.blocks
            ]
            row = {
                "step": step,
                "loss": float(loss.item()),
                "grad_norm_global": float(gnorm),
                "clip_factor": float(0.3 / max(gnorm, 1e-6)) if gnorm > 0.3 else 1.0,
                "step_ms": dt,
                "divergence": {"resid_mix": resid, "vr_lambda": vr},
            }
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            if i % 10 == 0 or i + 1 == args.iters:
                print(
                    f"step {step:5d} | loss {row['loss']:.4f} | gnorm {gnorm:.4f} "
                    f"| resid_mix {[round(v, 2) for v in resid]} | {dt:.0f} ms"
                )
                if not math.isfinite(gnorm) or gnorm > 50:
                    print("BISECT VERDICT: EXPLODES (weights poisoned before dump)")
                    return
        print("BISECT VERDICT: STABLE through continue (metal post-dump dynamics suspect)")


if __name__ == "__main__":
    main()
