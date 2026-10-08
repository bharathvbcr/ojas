#!/usr/bin/env python3
"""Torch incremental KV decode for arch_02 (stateful Core ML precursor).

Caches post-mix V, layer-0 raw_v (v0), and smear prev_stem so token-by-token
decode matches full-seq Arch02Infer prefill. No architecture redesign —
correct state is enough.

Used by:
  - scripts/decode_kv_reference.py (parity gate)
  - scripts/export_coreml.py --stateful-kv (MLState dual packages)
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from export_coreml import SOTA, apply_partial_rope, rms_norm


class Arch02KV(nn.Module):
    """Prefill + decode_step with explicit KV / v0 / smear state.

    State buffers (B=1, fixed T_max):
      k_cache[L,1,Hkv,T_max,D]
      v_cache[L,1,Hkv,T_max,D]   # post-VE, post-v0-mix (what SDPA reads)
      v0_cache[1,Hkv,T_max,D]    # layer-0 raw_v
      prev_stem[1,C]             # pre-smear stem of previous token
    """

    def __init__(self, cfg: dict = SOTA, seq_len: int = 64):
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
            [
                nn.Parameter(torch.stack((torch.ones(C), torch.zeros(C))))
                for _ in range(L)
            ]
        )

        half = rd // 2
        inv = 1.0 / (
            cfg["rope_base"] ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        self.register_buffer("rope_cos", freqs.cos()[None, :, None, :], persistent=True)
        self.register_buffer("rope_sin", freqs.sin()[None, :, None, :], persistent=True)

        # Stateful caches (Core ML StateType names must match).
        self.register_buffer(
            "k_cache", torch.zeros(L, 1, Hkv, seq_len, D), persistent=True
        )
        self.register_buffer(
            "v_cache", torch.zeros(L, 1, Hkv, seq_len, D), persistent=True
        )
        self.register_buffer(
            "v0_cache", torch.zeros(1, Hkv, seq_len, D), persistent=True
        )
        self.register_buffer("prev_stem", torch.zeros(1, C), persistent=True)
        self._cache_len = 0

    def load_npy_tree(self, root: Path) -> None:
        from export_coreml import Arch02Infer

        # Reuse Arch02Infer loader into a twin, then copy weights.
        twin = Arch02Infer(cfg=self.cfg, seq_len=self.seq_len)
        twin.load_npy_tree(root)
        missing, unexpected = self.load_state_dict(twin.state_dict(), strict=False)
        # Buffers (caches / rope) may differ; rope should match via same seq_len.
        bad = [k for k in missing if not k.startswith(("k_cache", "v_cache", "v0_cache", "prev_stem"))]
        if bad or unexpected:
            raise RuntimeError(f"weight load incomplete: missing={bad} unexpected={unexpected}")

    def reset_state(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.v0_cache.zero_()
        self.prev_stem.zero_()
        self._cache_len = 0

    def _ve(self, ids: torch.Tensor, ve_idx: int) -> torch.Tensor:
        h = self.ve_emb(ids)
        h = self.ve_proj(h) * self.ve_scale * self.ve_layer_scales[ve_idx]
        return h

    def _stem_token(
        self, ids: torch.Tensor, bid: torch.Tensor, use_prev: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed + bigram + RMS + smear for tokens [B,Tq]. Returns (x, x0) [B,Tq,C]."""
        c = self.cfg
        C = c["model_dim"]
        eps = 1.1920929e-07
        x = self.tok_emb(ids)
        bh = self.bigram_emb(bid)
        x = x + self.bigram_proj(bh) * self.bigram_scale
        x = rms_norm(x, eps)
        g = torch.sigmoid(self.smear_gate).view(1, 1, C)
        if use_prev:
            # Decode: first of the chunk mixes with prev_stem; later positions
            # mix with previous in-chunk stem (pre-smear).
            B, Tq, _ = x.shape
            prevs = []
            cur_prev = self.prev_stem.view(1, 1, C).expand(B, 1, C)
            for i in range(Tq):
                xi = x[:, i : i + 1, :]
                xi_s = (1.0 - g) * xi + g * cur_prev
                prevs.append(xi_s)
                cur_prev = xi  # next smear looks at pre-smear of current
            x_s = torch.cat(prevs, dim=1)
            # Update prev_stem to last pre-smear token in chunk.
            self.prev_stem.copy_(x[:, -1, :])
            return x_s, x_s
        # Prefill-style smear inside the chunk (no external prev).
        B, T, _ = x.shape
        x_prev = torch.cat(
            [torch.zeros(B, 1, C, dtype=x.dtype, device=x.device), x[:, : T - 1]], dim=1
        )
        x_s = (1.0 - g) * x + g * x_prev
        if T > 0:
            self.prev_stem.copy_(x[:, -1, :])
        return x_s, x_s

    def _attn_write(
        self,
        x: torch.Tensor,
        layer: int,
        ids: torch.Tensor,
        t0: int,
        t1: int,
    ) -> torch.Tensor:
        """Attend query window [t0,t1) writing K/V into cache; return attn out [B,Tq,C]."""
        c = self.cfg
        B = 1
        Tq = t1 - t0
        C = c["model_dim"]
        H, Hkv, D = c["num_heads"], c["num_kv_heads"], c["head_dim"]
        L = c["num_layers"]
        rd = c["rope_dims"]
        group = H // Hkv

        q = F.linear(x, self.qo_bank[layer]).view(B, Tq, H, D)
        k = F.linear(x, self.kv_bank[layer]).view(B, Tq, Hkv, D)
        v = F.linear(x, self.kv_bank[L + layer])

        if layer in c["ve_layers"]:
            ve_idx = c["ve_layers"].index(layer)
            v = v + self._ve(ids, ve_idx)
        v = v.view(B, Tq, Hkv, D)
        raw_v = v

        if layer == 0:
            self.v0_cache[:, :, t0:t1, :] = raw_v.transpose(1, 2)
            v_sdpa = raw_v
        else:
            lam = self.vr_lambda[layer]
            v0_slice = self.v0_cache[:, :, t0:t1, :].transpose(1, 2)  # [B,Tq,Hkv,D]
            v_sdpa = lam[0] * v0_slice + lam[1] * v

        # RoPE at absolute positions.
        cos = self.rope_cos[:, t0:t1, :, :]
        sin = self.rope_sin[:, t0:t1, :, :]
        q = rms_norm(q)
        k = rms_norm(k)
        q = apply_partial_rope(q, cos, sin, rd)
        k = apply_partial_rope(k, cos, sin, rd)
        q = q * self.q_gain[layer].view(1, 1, H, 1)

        # Write caches (K + post-mix V).
        self.k_cache[layer, :, :, t0:t1, :] = k.transpose(1, 2)
        self.v_cache[layer, :, :, t0:t1, :] = v_sdpa.transpose(1, 2)

        end = t1
        k_all = self.k_cache[layer, :, :, :end, :]  # [1,Hkv,end,D]
        v_all = self.v_cache[layer, :, :, :end, :]

        q_t = q.transpose(1, 2)
        if group > 1:
            k_t = (
                k_all.unsqueeze(2)
                .expand(B, Hkv, group, end, D)
                .reshape(B, H, end, D)
            )
            v_t = (
                v_all.unsqueeze(2)
                .expand(B, Hkv, group, end, D)
                .reshape(B, H, end, D)
            )
        else:
            k_t, v_t = k_all, v_all

        # Causal: queries at absolute positions t0..t1-1 against keys 0..end-1.
        # Build additive mask [B,1,Tq,end]: -inf where key_pos > query_abs_pos.
        q_pos = torch.arange(t0, t1, device=x.device).view(Tq, 1)
        k_pos = torch.arange(end, device=x.device).view(1, end)
        causal = k_pos > q_pos
        attn_mask = torch.zeros(B, 1, Tq, end, dtype=x.dtype, device=x.device)
        attn_mask = attn_mask.masked_fill(causal.view(1, 1, Tq, end), float("-inf"))

        y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
        y = y.transpose(1, 2)

        if layer >= L - c["xsa_last_n"]:
            y_g = y.reshape(B, Tq, Hkv, group, D)
            v_norm = raw_v.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).rsqrt()
            vn = (raw_v * v_norm).unsqueeze(-2)
            proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
            y = (y_g - proj).reshape(B, Tq, H, D)

        y = y.reshape(B, Tq, C)
        return F.linear(y, self.qo_bank[L + layer])

    def _run_window(
        self, ids: torch.Tensor, bid: torch.Tensor, t0: int, use_prev_stem: bool
    ) -> torch.Tensor:
        """Run layers for tokens at absolute [t0, t0+Tq); return logits [B,Tq,V]."""
        c = self.cfg
        Tq = ids.shape[1]
        t1 = t0 + Tq
        if t1 > self.seq_len:
            raise ValueError(f"window [{t0},{t1}) exceeds T_max={self.seq_len}")
        C = c["model_dim"]
        L = c["num_layers"]
        eps = 1.1920929e-07
        softcap = float(c["logit_softcap"])
        n_enc = L // 2

        x, x0 = self._stem_token(ids.long(), bid.long(), use_prev=use_prev_stem)
        skips: list[torch.Tensor] = []

        for layer in range(L):
            if layer >= n_enc:
                skip = skips.pop()
                x = x + self.skip_weights[layer - n_enc].view(1, 1, C) * skip

            mix = self.resid_mix[layer]
            x_in = mix[0].view(1, 1, C) * x + mix[1].view(1, 1, C) * x0
            ln = 1.0 / math.sqrt(layer + 1) if c["ln_scale"] else 1.0
            attn_in = rms_norm(x_in, eps) * ln
            attn_out = self._attn_write(attn_in, layer, ids.long(), t0, t1)
            x_mid = x_in + self.attn_scale[layer].view(1, 1, C) * attn_out
            mlp_in = rms_norm(x_mid, eps) * ln
            h = F.leaky_relu(F.linear(mlp_in, self.mlp_up[layer]), negative_slope=0.5)
            mlp_out = F.linear(h * h, self.mlp_down[layer])
            x = x_mid + self.mlp_scale[layer].view(1, 1, C) * mlp_out
            if layer < n_enc:
                skips.append(x)

        x = rms_norm(x, eps)
        logits = F.linear(x, self.tok_emb.weight)
        self._cache_len = t1
        return softcap * torch.tanh(logits / softcap)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, bigram_ids: torch.Tensor) -> torch.Tensor:
        """Fill caches from empty; return full logits [1,T,V]."""
        self.reset_state()
        T = input_ids.shape[1]
        return self._run_window(input_ids, bigram_ids, t0=0, use_prev_stem=False)

    @torch.no_grad()
    def decode_step(
        self, input_ids: torch.Tensor, bigram_ids: torch.Tensor, t: int | None = None
    ) -> torch.Tensor:
        """One-token decode at absolute position t (default: cache_len)."""
        if input_ids.shape[1] != 1:
            raise ValueError("decode_step expects q_len=1")
        if t is None:
            t = self._cache_len
        use_prev = t > 0
        if t == 0:
            self.reset_state()
        return self._run_window(input_ids, bigram_ids, t0=t, use_prev_stem=use_prev)

    def forward(
        self,
        input_ids: torch.Tensor,
        bigram_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Core ML decode graph: q_len=1, shape-derived cache slices (no Python ints).

        `causal_mask` is `[1, 1, end_step]`; `end_step = past_kv_len + 1`.
        In-place buffer updates match Apple's stateful KV toy pattern so
        coremltools can lower slices. Torch parity still uses decode_step().
        """
        return self._decode_shaped(input_ids, bigram_ids, causal_mask)

    def _decode_shaped(
        self,
        input_ids: torch.Tensor,
        bigram_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """q_len=1 decode using only tensor.shape for cache indexing."""
        c = self.cfg
        C = c["model_dim"]
        L = c["num_layers"]
        H, Hkv, D = c["num_heads"], c["num_kv_heads"], c["head_dim"]
        rd = c["rope_dims"]
        group = H // Hkv
        eps = 1.1920929e-07
        softcap = float(c["logit_softcap"])
        n_enc = L // 2
        B = 1

        end_step = causal_mask.shape[-1]
        past = end_step - input_ids.shape[-1]  # q_len

        ids = input_ids.long()
        bid = bigram_ids.long()

        # Stem + smear with prev_stem (host zeros state on first step).
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
                v = v + self._ve(ids, ve_idx)
            v = v.view(B, 1, Hkv, D)
            raw_v = v

            if layer == 0:
                self.v0_cache[:, :, past:end_step, :] = raw_v.transpose(1, 2)
                v_sdpa = raw_v
            else:
                lam = self.vr_lambda[layer]
                v0_slice = self.v0_cache[:, :, past:end_step, :].transpose(1, 2)
                v_sdpa = lam[0] * v0_slice + lam[1] * v

            cos = self.rope_cos[:, past:end_step, :, :]
            sin = self.rope_sin[:, past:end_step, :, :]
            q = rms_norm(q)
            k = rms_norm(k)
            q = apply_partial_rope(q, cos, sin, rd)
            k = apply_partial_rope(k, cos, sin, rd)
            q = q * self.q_gain[layer].view(1, 1, H, 1)

            self.k_cache[layer, :, :, past:end_step, :] = k.transpose(1, 2)
            self.v_cache[layer, :, :, past:end_step, :] = v_sdpa.transpose(1, 2)

            k_all = self.k_cache[layer, :, :, :end_step, :]
            v_all = self.v_cache[layer, :, :, :end_step, :]
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

            # causal_mask: [1,1,end] → [1,1,1,end] for SDPA
            attn_mask = causal_mask.unsqueeze(1)
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
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


def copy_weights_from_infer(dst: Arch02KV, src: nn.Module) -> None:
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    bad = [
        k
        for k in missing
        if not k.startswith(("k_cache", "v_cache", "v0_cache", "prev_stem"))
    ]
    if bad:
        raise RuntimeError(f"copy_weights missing: {bad}; unexpected={unexpected}")
