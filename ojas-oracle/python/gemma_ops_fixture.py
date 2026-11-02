"""Golden fixture generator for Gemma Embedding ops (Milestone 4, Feature 26).

    /Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/gemma_ops_fixture.py \
        --out ojas-oracle/fixtures/gemma_ops/golden.txt

Generates reference golden text fixture containing:
1. `gelu_tanh` forward & backward
2. `seq_mean` masked sequence mean pooling
3. `l2_normalize` with Matryoshka dimension truncation
4. `rms_qkv_norm` with learned Q/K weights and weightless V
5. `swa` non-causal symmetric sliding window attention (|i - j| <= W)

Format `ojas-gemma-ops-golden-v1`: a line-oriented text file.
A case starts with `case <op> <name> <param...>`
followed by `t <name> <dtype> <dims> <hex...>`
and ends with `end`.

Reference oracle only (house policy): nothing here ships.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch import Tensor

FORMAT = "ojas-gemma-ops-golden-v1"
KAPPA = math.sqrt(2.0 / math.pi)
CUBIC = 0.044715


def hexbits_f32(t: Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1).to(torch.float32)
    return " ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in flat.view(torch.int32))


def hexbits_f64(t: Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1).to(torch.float64)
    return " ".join(f"{int(v) & 0xFFFFFFFFFFFFFFFF:016x}" for v in flat.view(torch.int64))


def hexbits_u32(t: Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1).to(torch.int64)
    return " ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in flat)


def tline(name: str, t: Tensor) -> str:
    if t.dtype in (torch.int32, torch.int64, torch.uint8):
        dtype = "u32"
        bits = hexbits_u32(t)
    elif t.dtype == torch.float64:
        dtype = "f64"
        bits = hexbits_f64(t)
    else:
        dtype = "f32"
        bits = hexbits_f32(t)
    dims = ",".join(str(d) for d in t.shape) or "-"
    return f"t {name} {dtype} {dims} {bits}"


def eps_bits(eps: float | None) -> str:
    if eps is None:
        return "-"
    return f"{int(torch.tensor(eps, dtype=torch.float32).view(torch.int32)) & 0xFFFFFFFF:08x}"


# ---- 1. GELU Tanh ----

def gelu_tanh_fwd_f64(x: Tensor) -> Tensor:
    u = 2.0 * KAPPA * (x + CUBIC * (x ** 3))
    s = torch.sigmoid(u)
    return x * s


def gelu_tanh_bwd_f64(x: Tensor, gy: Tensor) -> Tensor:
    u = 2.0 * KAPPA * (x + CUBIC * (x ** 3))
    s = torch.sigmoid(u)
    du = 2.0 * KAPPA * (1.0 + 3.0 * CUBIC * (x ** 2))
    ds = s * (1.0 - s) * du
    return gy * (s + x * ds)


def gen_gelu_cases(gen: torch.Generator) -> list[str]:
    lines = []
    cases = [
        ("standard", torch.randn(4, 16, generator=gen, dtype=torch.float32)),
        ("tails", torch.tensor([
            [-15.0, -12.0, -8.0, -5.0, -3.0, -2.0, -1.0, -0.5],
            [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0],
        ], dtype=torch.float32)),
        ("near_zero", torch.tensor([
            [-0.1, -1e-2, -1e-4, -1e-7, 0.0, 1e-7, 1e-4, 1e-2],
            [-0.05, -0.01, -0.001, -0.0, 0.0, 0.001, 0.01, 0.05],
        ], dtype=torch.float32)),
        ("batch_matrix", torch.randn(2, 3, 16, generator=gen, dtype=torch.float32) * 2.0),
    ]

    for name, x in cases:
        gy = torch.randn(x.shape, generator=gen, dtype=torch.float32)
        x_64 = x.to(torch.float64)
        gy_64 = gy.to(torch.float64)

        y_64 = gelu_tanh_fwd_f64(x_64)
        dx_64 = gelu_tanh_bwd_f64(x_64, gy_64)

        y = y_64.to(torch.float32)
        dx = dx_64.to(torch.float32)

        lines.append(f"case gelu_tanh {name} -")
        lines.append(tline("x", x))
        lines.append(tline("gy", gy))
        lines.append(tline("y", y))
        lines.append(tline("dx", dx))
        lines.append(tline("y_64", y_64))
        lines.append(tline("dx_64", dx_64))
        lines.append("end")

    return lines


# ---- 2. Sequence Mean Pooling ----

def seq_mean_f64(x: Tensor, key_lens: Tensor) -> Tensor:
    b, t, d = x.shape
    out = torch.zeros(b, d, dtype=torch.float64)
    for i in range(b):
        length = int(key_lens[i].item())
        valid_len = min(length, t)
        if valid_len > 0:
            active = x[i, :valid_len, :].to(torch.float64)
            # In f64, exact mean over active tokens
            out[i] = active.sum(dim=0) / float(length)
        else:
            out[i] = 0.0
    return out


def gen_seq_mean_cases(gen: torch.Generator) -> list[str]:
    lines = []
    cases = [
        ("ragged", 3, 8, 16, torch.tensor([5, 8, 3], dtype=torch.int32), 1.0),
        ("full", 2, 10, 32, torch.tensor([10, 10], dtype=torch.int32), 1.0),
        ("zero_len", 2, 4, 8, torch.tensor([0, 3], dtype=torch.int32), 1.0),
    ]

    for name, b, t, d, key_lens, scale in cases:
        x = torch.randn(b, t, d, generator=gen, dtype=torch.float32) * scale
        x_64 = x.to(torch.float64)
        y_64 = seq_mean_f64(x_64, key_lens)
        y = y_64.to(torch.float32)

        lines.append(f"case seq_mean {name} -")
        lines.append(tline("x", x))
        lines.append(tline("key_lens", key_lens))
        lines.append(tline("y", y))
        lines.append(tline("y_64", y_64))
        lines.append("end")

    # Catastrophic cancellation case: [1e7, 1.0, -1e7] vs [1.0, 1.0, 1.0]
    x_prec = torch.tensor([
        [[1e7, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [-1e7, 2.0, 3.0, 4.0]],
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
    ], dtype=torch.float32)
    key_lens_prec = torch.tensor([3, 3], dtype=torch.int32)
    y_64_prec = seq_mean_f64(x_prec.to(torch.float64), key_lens_prec)
    y_prec = y_64_prec.to(torch.float32)

    lines.append("case seq_mean cancellation -")
    lines.append(tline("x", x_prec))
    lines.append(tline("key_lens", key_lens_prec))
    lines.append(tline("y", y_prec))
    lines.append(tline("y_64", y_64_prec))
    lines.append("end")

    # A length past the end of the sequence: row 0 asks for 6 of 4 tokens, so
    # the divisor is the token count actually summed, min(6, 4) = 4, and the
    # mean of a row of ones is 1.0. Row 1 keeps junk past its length to pin
    # that padding is read by neither the divisor nor the sum. Written out
    # rather than drawn, so the cases above keep their draws from `gen`.
    x_over = torch.zeros(2, 4, 4, dtype=torch.float32)
    x_over[0] = 1.0
    x_over[1, :2] = 2.0
    x_over[1, 2:] = 7.0
    key_lens_over = torch.tensor([6, 2], dtype=torch.int32)
    y_64_over = seq_mean_f64(x_over.to(torch.float64), key_lens_over)
    y_over = y_64_over.to(torch.float32)

    lines.append("case seq_mean over_len -")
    lines.append(tline("x", x_over))
    lines.append(tline("key_lens", key_lens_over))
    lines.append(tline("y", y_over))
    lines.append(tline("y_64", y_64_over))
    lines.append("end")

    return lines


# ---- 3. Matryoshka L2 Normalization ----

def l2_norm_f64(x: Tensor, eps: float) -> Tensor:
    norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    div = torch.clamp(norm, min=eps)
    return x / div


def gen_l2_norm_cases(gen: torch.Generator) -> list[str]:
    lines = []
    eps = 1e-12
    cases = [
        ("embed_768", 4, 768),
        ("matryoshka_512", 4, 512),
        ("matryoshka_256", 4, 256),
        ("matryoshka_128", 4, 128),
        ("small_dim", 2, 8),
    ]

    for name, n, d in cases:
        x = torch.randn(n, d, generator=gen, dtype=torch.float32)
        x_64 = x.to(torch.float64)
        y_64 = l2_norm_f64(x_64, eps)
        y = y_64.to(torch.float32)

        lines.append(f"case l2_normalize {name} {eps_bits(eps)}")
        lines.append(tline("x", x))
        lines.append(tline("y", y))
        lines.append(tline("y_64", y_64))
        lines.append("end")

    # Zero row case
    x_zero = torch.zeros(2, 16, dtype=torch.float32)
    x_zero[1] = torch.randn(16, generator=gen, dtype=torch.float32)
    y_64_zero = l2_norm_f64(x_zero.to(torch.float64), eps)
    y_zero = y_64_zero.to(torch.float32)

    lines.append(f"case l2_normalize zero_row {eps_bits(eps)}")
    lines.append(tline("x", x_zero))
    lines.append(tline("y", y_zero))
    lines.append(tline("y_64", y_64_zero))
    lines.append("end")

    return lines


# ---- 4. Per-Head V RMSNorm ----

def rms_qkv_norm_f64(q: Tensor, k: Tensor, v: Tensor, w_q: Tensor, w_k: Tensor, eps: float):
    # q: [B, H, T, D], k: [B, H_kv, T, D], v: [B, H_kv, T, D]
    # w_q: [D], w_k: [D]
    q_rms = torch.rsqrt(torch.mean(q ** 2, dim=-1, keepdim=True) + eps)
    q_norm = q * q_rms * w_q

    k_rms = torch.rsqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + eps)
    k_norm = k * k_rms * w_k

    # v is normalized weightlessly (w_v = 1.0)
    v_rms = torch.rsqrt(torch.mean(v ** 2, dim=-1, keepdim=True) + eps)
    v_norm = v * v_rms

    return q_norm, k_norm, v_norm


def gen_rms_qkv_cases(gen: torch.Generator) -> list[str]:
    lines = []
    cases = [
        ("mha", 2, 4, 4, 8, 16, 1e-6),
        ("gqa", 2, 4, 1, 12, 16, 1e-6),
        ("vit_eps", 1, 2, 2, 4, 32, 1e-5),
    ]

    for name, b, h, h_kv, t, d, eps in cases:
        q = torch.randn(b, h, t, d, generator=gen, dtype=torch.float32)
        k = torch.randn(b, h_kv, t, d, generator=gen, dtype=torch.float32)
        v = torch.randn(b, h_kv, t, d, generator=gen, dtype=torch.float32)
        w_q = torch.randn(d, generator=gen, dtype=torch.float32) * 0.1 + 1.0
        w_k = torch.randn(d, generator=gen, dtype=torch.float32) * 0.1 + 1.0

        q_64, k_64, v_64 = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
        w_q_64, w_k_64 = w_q.to(torch.float64), w_k.to(torch.float64)

        q_n_64, k_n_64, v_n_64 = rms_qkv_norm_f64(q_64, k_64, v_64, w_q_64, w_k_64, eps)
        q_n = q_n_64.to(torch.float32)
        k_n = k_n_64.to(torch.float32)
        v_n = v_n_64.to(torch.float32)

        lines.append(f"case rms_qkv_norm {name} {eps_bits(eps)}")
        lines.append(tline("q", q))
        lines.append(tline("k", k))
        lines.append(tline("v", v))
        lines.append(tline("w_q", w_q))
        lines.append(tline("w_k", w_k))
        lines.append(tline("q_norm", q_n))
        lines.append(tline("k_norm", k_n))
        lines.append(tline("v_norm", v_n))
        lines.append(tline("q_norm_64", q_n_64))
        lines.append(tline("k_norm_64", k_n_64))
        lines.append(tline("v_norm_64", v_n_64))
        lines.append("end")

    return lines


# ---- 5. Symmetric Non-Causal Sliding Window Attention ----

def swa_f64(q: Tensor, k: Tensor, v: Tensor, key_lens: Tensor, radius: int, scale: float) -> Tensor:
    b, h, t_q, d = q.shape
    _, h_kv, t_k, _ = k.shape
    gqa_ratio = h // h_kv

    out = torch.zeros(b, h, t_q, d, dtype=torch.float64)

    for bi in range(b):
        length = int(key_lens[bi].item())
        for hi in range(h):
            h_kvi = hi // gqa_ratio
            for i in range(t_q):
                if i >= length:
                    # Query row is padding, zero output
                    continue

                # Find valid key window: |i - j| <= radius and j < min(length, t_k)
                j_min = max(0, i - radius)
                j_max = min(length, t_k, i + radius + 1)

                if j_min >= j_max:
                    continue

                scores = []
                for j in range(j_min, j_max):
                    dot = torch.dot(q[bi, hi, i, :], k[bi, h_kvi, j, :])
                    scores.append(dot * scale)

                scores_t = torch.tensor(scores, dtype=torch.float64)
                probs = torch.softmax(scores_t, dim=0)

                acc = torch.zeros(d, dtype=torch.float64)
                for idx, j in enumerate(range(j_min, j_max)):
                    acc += probs[idx] * v[bi, h_kvi, j, :]

                out[bi, hi, i, :] = acc

    return out


def gen_swa_cases(gen: torch.Generator) -> list[str]:
    lines = []
    cases = [
        ("basic", 2, 2, 2, 8, 16, 2, 1.0, torch.tensor([5, 8], dtype=torch.int32)),
        ("gqa", 2, 4, 2, 6, 16, 3, 1.0, torch.tensor([4, 6], dtype=torch.int32)),
        ("wide_window", 1, 2, 2, 10, 16, 16, 1.0, torch.tensor([10], dtype=torch.int32)),
        ("w1_nearest", 1, 1, 1, 6, 8, 1, 1.0, torch.tensor([6], dtype=torch.int32)),
    ]

    for name, b, h, h_kv, t, d, radius, scale, key_lens in cases:
        q = torch.randn(b, h, t, d, generator=gen, dtype=torch.float32)
        k = torch.randn(b, h_kv, t, d, generator=gen, dtype=torch.float32)
        v = torch.randn(b, h_kv, t, d, generator=gen, dtype=torch.float32)

        q_64 = q.to(torch.float64)
        k_64 = k.to(torch.float64)
        v_64 = v.to(torch.float64)

        y_64 = swa_f64(q_64, k_64, v_64, key_lens, radius, scale)
        y = y_64.to(torch.float32)

        lines.append(f"case swa {name} {radius} {eps_bits(scale)}")
        lines.append(tline("q", q))
        lines.append(tline("k", k))
        lines.append(tline("v", v))
        lines.append(tline("key_lens", key_lens))
        lines.append(tline("y", y))
        lines.append(tline("y_64", y_64))
        lines.append("end")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    # Deterministic seed
    gen = torch.Generator().manual_seed(20261101)

    lines = [
        FORMAT,
        f"torch {torch.__version__.split('+')[0]}",
    ]

    lines.extend(gen_gelu_cases(gen))
    lines.extend(gen_seq_mean_cases(gen))
    lines.extend(gen_l2_norm_cases(gen))
    lines.extend(gen_rms_qkv_cases(gen))
    lines.extend(gen_swa_cases(gen))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
