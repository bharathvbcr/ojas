#!/usr/bin/env python3
"""Minimal Apple-layout stateful decode export probe for arch_02.

Uses 3D KV caches [1, T_max, Hkv*D] per layer (named k_cache_L{i} / v_cache_L{i})
matching coremltools' toy pattern more closely than 5D stacked caches.

Torch math parity remains in arch02_kv.Arch02KV / decode_kv_reference.py.
This module only exists to land an .mlpackage when 5D StateType convert fails.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from export_coreml import SOTA, apply_partial_rope, rms_norm


class Arch02KVFlatExport(nn.Module):
    """q_len=1 decode with per-layer 3D caches for Core ML StateType."""

    def __init__(self, cfg: dict = SOTA, seq_len: int = 32):
        super().__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        C = cfg["model_dim"]
        V = cfg["vocab_size"]
        L = cfg["num_layers"]
        H = cfg["num_heads"]
        Hkv = cfg["num_kv_heads"]
        D = cfg["head_dim"]
        kv = Hkv * D
        mlp = cfg["mlp_dim"]
        rd = cfg["rope_dims"]

        self.tok_emb = nn.Embedding(V, C)
        self.bigram_emb = nn.Embedding(cfg["bigram_vocab"], cfg["bigram_dim"])
        self.bigram_proj = nn.Linear(cfg["bigram_dim"], C, bias=False)
        self.bigram_scale = nn.Parameter(torch.tensor(0.05))
        self.smear_gate = nn.Parameter(torch.zeros(C))
        self.ve_emb = nn.Embedding(V, cfg["ve_dim"])
        self.ve_proj = nn.Linear(cfg["ve_dim"], kv, bias=False)
        self.ve_scale = nn.Parameter(torch.tensor(0.1))
        self.ve_layer_scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in cfg["ve_layers"]]
        )
        self.skip_weights = nn.Parameter(torch.ones(L // 2, C))
        self.qo_bank = nn.Parameter(torch.empty(2 * L, C, C))
        self.kv_bank = nn.Parameter(torch.empty(2 * L, kv, C))
        self.mlp_up = nn.Parameter(torch.empty(L, mlp, C))
        self.mlp_down = nn.Parameter(torch.empty(L, C, mlp))
        self.q_gain = nn.ParameterList([nn.Parameter(torch.ones(H)) for _ in range(L)])
        self.vr_lambda = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.5, 0.5])) for _ in range(L)]
        )
        self.attn_scale = nn.ParameterList([nn.Parameter(torch.ones(C)) for _ in range(L)])
        self.mlp_scale = nn.ParameterList([nn.Parameter(torch.ones(C)) for _ in range(L)])
        self.resid_mix = nn.ParameterList(
            [nn.Parameter(torch.stack((torch.ones(C), torch.zeros(C)))) for _ in range(L)]
        )
        inv = 1.0 / (
            cfg["rope_base"] ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        self.register_buffer("rope_cos", freqs.cos()[None, :, None, :], persistent=True)
        self.register_buffer("rope_sin", freqs.sin()[None, :, None, :], persistent=True)

        # Apple-style 3D caches per layer: [1, T_max, Hkv*D]
        for i in range(L):
            self.register_buffer(f"k_cache_L{i}", torch.zeros(1, seq_len, kv), persistent=True)
            self.register_buffer(f"v_cache_L{i}", torch.zeros(1, seq_len, kv), persistent=True)
        self.register_buffer("v0_cache", torch.zeros(1, seq_len, kv), persistent=True)
        self.register_buffer("prev_stem", torch.zeros(1, C), persistent=True)

    def load_from_kv(self, kv_mod: nn.Module) -> None:
        missing, unexpected = self.load_state_dict(kv_mod.state_dict(), strict=False)
        # Ignore cache shape mismatches between 5D stacked and flat 3D.
        _ = missing, unexpected

    def forward(
        self, input_ids: torch.Tensor, bigram_ids: torch.Tensor, causal_mask: torch.Tensor
    ) -> torch.Tensor:
        c = self.cfg
        C = c["model_dim"]
        L = c["num_layers"]
        H, Hkv, D = c["num_heads"], c["num_kv_heads"], c["head_dim"]
        kv = Hkv * D
        rd = c["rope_dims"]
        group = H // Hkv
        eps = 1.1920929e-07
        softcap = float(c["logit_softcap"])
        n_enc = L // 2
        B = 1

        end_step = causal_mask.shape[-1]
        past = end_step - input_ids.shape[-1]
        ids = input_ids.long()
        bid = bigram_ids.long()

        x = self.tok_emb(ids)
        x = x + self.bigram_proj(self.bigram_emb(bid)) * self.bigram_scale
        x = rms_norm(x, eps)
        g = torch.sigmoid(self.smear_gate).view(1, 1, C)
        pre = x
        x = (1.0 - g) * x + g * self.prev_stem.view(1, 1, C)
        self.prev_stem.copy_(pre[0])
        x0 = x

        skips: list[torch.Tensor] = []
        for layer in range(L):
            if layer >= n_enc:
                skip = skips.pop()
                x = x + self.skip_weights[layer - n_enc].view(1, 1, C) * skip

            mix = self.resid_mix[layer]
            x_in = mix[0].view(1, 1, C) * x + mix[1].view(1, 1, C) * x0
            ln = 1.0 / math.sqrt(layer + 1) if c["ln_scale"] else 1.0
            attn_in = rms_norm(x_in, eps) * ln

            q = F.linear(attn_in, self.qo_bank[layer]).view(B, 1, H, D)
            k = F.linear(attn_in, self.kv_bank[layer]).view(B, 1, Hkv, D)
            v = F.linear(attn_in, self.kv_bank[L + layer])
            if layer in c["ve_layers"]:
                ve_idx = c["ve_layers"].index(layer)
                h = self.ve_emb(ids)
                v = v + self.ve_proj(h) * self.ve_scale * self.ve_layer_scales[ve_idx]
            v = v.view(B, 1, Hkv, D)
            raw_v = v

            if layer == 0:
                self.v0_cache[:, past:end_step, :] = raw_v.reshape(B, 1, kv)
                v_sdpa = raw_v
            else:
                lam = self.vr_lambda[layer]
                v0_slice = self.v0_cache[:, past:end_step, :].view(B, 1, Hkv, D)
                v_sdpa = lam[0] * v0_slice + lam[1] * v

            cos = self.rope_cos[:, past:end_step, :, :]
            sin = self.rope_sin[:, past:end_step, :, :]
            q = rms_norm(q)
            k = rms_norm(k)
            q = apply_partial_rope(q, cos, sin, rd)
            k = apply_partial_rope(k, cos, sin, rd)
            q = q * self.q_gain[layer].view(1, 1, H, 1)

            k_buf = getattr(self, f"k_cache_L{layer}")
            v_buf = getattr(self, f"v_cache_L{layer}")
            k_buf[:, past:end_step, :] = k.reshape(B, 1, kv)
            v_buf[:, past:end_step, :] = v_sdpa.reshape(B, 1, kv)

            k_all = k_buf[:, :end_step, :].view(B, end_step, Hkv, D).transpose(1, 2)
            v_all = v_buf[:, :end_step, :].view(B, end_step, Hkv, D).transpose(1, 2)
            q_t = q.transpose(1, 2)
            if group > 1:
                k_t = (
                    k_all.unsqueeze(2)
                    .expand(B, Hkv, group, end_step, D)
                    .reshape(B, H, end_step, D)
                )
                v_t = (
                    v_all.unsqueeze(2)
                    .expand(B, Hkv, group, end_step, D)
                    .reshape(B, H, end_step, D)
                )
            else:
                k_t, v_t = k_all, v_all

            y = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=causal_mask.unsqueeze(1)
            )
            y = y.transpose(1, 2)
            if layer >= L - c["xsa_last_n"]:
                y_g = y.reshape(B, 1, Hkv, group, D)
                v_norm = raw_v.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).rsqrt()
                vn = (raw_v * v_norm).unsqueeze(-2)
                proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
                y = (y_g - proj).reshape(B, 1, H, D)

            attn_out = F.linear(y.reshape(B, 1, C), self.qo_bank[L + layer])
            x_mid = x_in + self.attn_scale[layer].view(1, 1, C) * attn_out
            mlp_in = rms_norm(x_mid, eps) * ln
            h = F.leaky_relu(F.linear(mlp_in, self.mlp_up[layer]), negative_slope=0.5)
            mlp_out = F.linear(h * h, self.mlp_down[layer])
            x = x_mid + self.mlp_scale[layer].view(1, 1, C) * mlp_out
            if layer < n_enc:
                skips.append(x)

        x = rms_norm(x, eps)
        logits = F.linear(x, self.tok_emb.weight)
        return softcap * torch.tanh(logits / softcap)


def try_export_flat_mlpackage(weights: Path, out_dir: Path, seq_len: int) -> Path | None:
    import coremltools as ct
    import shutil

    from arch02_kv import Arch02KV

    kv = Arch02KV(seq_len=seq_len)
    kv.load_npy_tree(weights)
    flat = Arch02KVFlatExport(seq_len=seq_len)
    # Copy overlapping parameter tensors.
    src = {k: v for k, v in kv.state_dict().items() if not k.startswith(("k_cache", "v_cache", "v0_cache", "prev_stem"))}
    flat.load_state_dict(src, strict=False)
    flat.eval()

    ids = torch.zeros(1, 1, dtype=torch.int32)
    bg = torch.zeros(1, 1, dtype=torch.int32)
    mask = torch.zeros(1, 1, 1, dtype=torch.float32)
    traced = torch.jit.trace(flat, (ids, bg, mask))

    L = SOTA["num_layers"]
    kv_dim = SOTA["num_kv_heads"] * SOTA["head_dim"]
    states = []
    for i in range(L):
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(1, seq_len, kv_dim), dtype=np.float16),
                name=f"k_cache_L{i}",
            )
        )
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(shape=(1, seq_len, kv_dim), dtype=np.float16),
                name=f"v_cache_L{i}",
            )
        )
    states.append(
        ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, seq_len, kv_dim), dtype=np.float16),
            name="v0_cache",
        )
    )
    states.append(
        ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, SOTA["model_dim"]), dtype=np.float16),
            name="prev_stem",
        )
    )

    path = out_dir / "arch02_sota_decode_step_fp16.mlpackage"
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="bigram_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=(1, 1, 1), dtype=np.float16),
            ],
            outputs=[ct.TensorType(name="logits")],
            states=states,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        if path.exists():
            shutil.rmtree(path)
        mlmodel.save(str(path))
        return path
    except Exception as e:
        print(f"flat Core ML convert failed: {e}")
        return None
