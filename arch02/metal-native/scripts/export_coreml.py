#!/usr/bin/env python3
"""Phase 5: arch_02 sota-toy → Core ML .mlpackage (inference only).

Loads Python-layout .npy weights (golden/weights_init or train EMA dump),
builds an ANE-oriented single-forward graph, exports via coremltools, optional
int8/int6 palettization, and benches on-device latency.

`--stateful-kv` exports experimental MLState decode packages after
`decode_reference_ok` + `decode_kv_reference_ok` (torch KV parity). Prefill-only
remains the default ANE latency path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sota-toy config (matches metal-native ModelConfig::sota_toy)
# ---------------------------------------------------------------------------

SOTA = dict(
    vocab_size=1024,
    num_layers=4,
    model_dim=128,
    num_heads=4,
    num_kv_heads=2,
    head_dim=32,
    mlp_dim=384,
    bigram_vocab=512,
    bigram_dim=48,
    ve_dim=24,
    rope_dims=8,
    rope_base=10000.0,
    logit_softcap=30.0,
    ve_layers=(2, 3),
    xsa_last_n=2,
    ln_scale=True,
)


def _load_npy(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path).astype(np.float32))


def rms_norm(x: torch.Tensor, eps: float = torch.finfo(torch.float32).eps) -> torch.Tensor:
    # Explicit (ANE-friendly) RMSNorm — avoid F.rms_norm for broader Core ML coverage.
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dims: int
) -> torch.Tensor:
    # x: [B,T,H,D], cos/sin: [1,T,1,half]
    x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
    half = rope_dims // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((x_rope, x_pass), dim=-1)


def bigram_hash_np(tokens: np.ndarray, bigram_vocab: int) -> np.ndarray:
    """Host-side bigram hash (matches train). Kept off the Core ML graph for ANE."""
    t = tokens.astype(np.int32)
    mod = bigram_vocab - 1
    out = np.empty_like(t)
    out[..., 0] = mod
    out[..., 1:] = np.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
    return out.astype(np.int32)


def bigram_hash_torch(tokens: torch.Tensor, bigram_vocab: int) -> torch.Tensor:
    t = tokens.to(torch.int64)
    mod = bigram_vocab - 1
    out = torch.empty_like(t)
    out[..., 0] = mod
    out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
    return out


class Arch02Infer(nn.Module):
    """Single-forward inference graph for Core ML export (fixed B=1, T).

    Inputs: input_ids [1,T] int32, bigram_ids [1,T] int32 (host-computed hash).
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
        # Banks stored Python-style [out, in] for F.linear
        self.qo_bank = nn.Parameter(torch.empty(2 * L, C, C))
        self.kv_bank = nn.Parameter(torch.empty(2 * L, kv, C))
        self.mlp_up = nn.Parameter(torch.empty(L, mlp, C))
        self.mlp_down = nn.Parameter(torch.empty(L, C, mlp))

        self.q_gain = nn.ParameterList(
            [nn.Parameter(torch.ones(H)) for _ in range(L)]
        )
        self.vr_lambda = nn.ParameterList(
            [nn.Parameter(torch.tensor([0.5, 0.5])) for _ in range(L)]
        )
        self.attn_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(C)) for _ in range(L)]
        )
        self.mlp_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(C)) for _ in range(L)]
        )
        self.resid_mix = nn.ParameterList(
            [nn.Parameter(torch.stack((torch.ones(C), torch.zeros(C)))) for _ in range(L)]
        )

        half = rd // 2
        inv = 1.0 / (
            cfg["rope_base"] ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd)
        )
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        self.register_buffer("rope_cos", freqs.cos()[None, :, None, :], persistent=True)
        self.register_buffer("rope_sin", freqs.sin()[None, :, None, :], persistent=True)

    def load_npy_tree(self, root: Path) -> None:
        c = self.cfg
        L = c["num_layers"]
        self.tok_emb.weight.data.copy_(_load_npy(root / "tok_emb/weight.npy"))
        self.bigram_emb.weight.data.copy_(_load_npy(root / "bigram/embed/weight.npy"))
        self.bigram_proj.weight.data.copy_(_load_npy(root / "bigram/proj/weight.npy"))
        self.bigram_scale.data.copy_(_load_npy(root / "bigram/scale.npy").reshape(()))
        self.smear_gate.data.copy_(_load_npy(root / "smear/gate.npy"))

        self.ve_emb.weight.data.copy_(_load_npy(root / "ve_shared/embed/weight.npy"))
        self.ve_proj.weight.data.copy_(_load_npy(root / "ve_shared/proj/weight.npy"))
        self.ve_scale.data.copy_(_load_npy(root / "ve_shared/scale.npy").reshape(()))
        for i in range(len(c["ve_layers"])):
            self.ve_layer_scales[i].data.copy_(
                _load_npy(root / f"ve_layer_scales/{i}.npy").reshape(1)
            )

        sw = _load_npy(root / "skip_weights.npy")
        if sw.ndim == 1:
            sw = sw.view(L // 2, c["model_dim"])
        self.skip_weights.data.copy_(sw)

        self.qo_bank.data.copy_(_load_npy(root / "qo_bank.npy"))
        self.kv_bank.data.copy_(_load_npy(root / "kv_bank.npy"))
        self.mlp_up.data.copy_(_load_npy(root / "mlp_up_bank.npy"))
        self.mlp_down.data.copy_(_load_npy(root / "mlp_down_bank.npy"))

        for i in range(L):
            base = root / f"blocks/{i}"
            self.q_gain[i].data.copy_(_load_npy(base / "attn/q_gain.npy"))
            self.vr_lambda[i].data.copy_(_load_npy(base / "attn/vr_lambda.npy"))
            self.attn_scale[i].data.copy_(_load_npy(base / "attn_scale.npy"))
            self.mlp_scale[i].data.copy_(_load_npy(base / "mlp_scale.npy"))
            self.resid_mix[i].data.copy_(_load_npy(base / "resid_mix.npy"))

    def _ve(self, ids: torch.Tensor, ve_idx: int) -> torch.Tensor:
        h = self.ve_emb(ids)
        h = self.ve_proj(h) * self.ve_scale * self.ve_layer_scales[ve_idx]
        return h  # [B,T,kv]

    def _attn(
        self,
        x: torch.Tensor,
        layer: int,
        ids: torch.Tensor,
        v0: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Fixed export shapes — never unpack .shape (avoids aten::Int Core ML bugs).
        c = self.cfg
        B = 1
        T = self.seq_len
        C = c["model_dim"]
        H, Hkv, D = c["num_heads"], c["num_kv_heads"], c["head_dim"]
        L = c["num_layers"]
        rd = c["rope_dims"]
        group = H // Hkv

        q = F.linear(x, self.qo_bank[layer]).view(B, T, H, D)
        k = F.linear(x, self.kv_bank[layer]).view(B, T, Hkv, D)
        v = F.linear(x, self.kv_bank[L + layer])  # [B,T,kv]

        if layer in c["ve_layers"]:
            ve_idx = c["ve_layers"].index(layer)
            v = v + self._ve(ids, ve_idx)
        v = v.view(B, T, Hkv, D)
        raw_v = v

        if v0 is not None:
            lam = self.vr_lambda[layer]
            v = lam[0] * v0 + lam[1] * v

        q = rms_norm(q)
        k = rms_norm(k)
        q = apply_partial_rope(q, self.rope_cos, self.rope_sin, rd)
        k = apply_partial_rope(k, self.rope_cos, self.rope_sin, rd)
        q = q * self.q_gain[layer].view(1, 1, H, 1)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        if group > 1:
            k_t = k_t.unsqueeze(2).expand(B, Hkv, group, T, D).reshape(B, H, T, D)
            v_t = v_t.unsqueeze(2).expand(B, Hkv, group, T, D).reshape(B, H, T, D)

        y = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, is_causal=True
        )
        y = y.transpose(1, 2)  # [B,T,H,D]

        if layer >= L - c["xsa_last_n"]:
            y_g = y.reshape(B, T, Hkv, group, D)
            v_norm = raw_v.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).rsqrt()
            vn = (raw_v * v_norm).unsqueeze(-2)
            proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
            y = (y_g - proj).reshape(B, T, H, D)

        y = y.reshape(B, T, C)
        out = F.linear(y, self.qo_bank[L + layer])
        ret_v0 = raw_v if layer == 0 else None
        return out, ret_v0

    def forward(self, input_ids: torch.Tensor, bigram_ids: torch.Tensor) -> torch.Tensor:
        """input_ids/bigram_ids: int32 [1, T] → logits float32 [1, T, V] (post softcap)."""
        c = self.cfg
        # Embedding accepts int32 in Core ML; keep Long only for torch Embedding.
        ids = input_ids.long()
        bid = bigram_ids.long()
        B = 1
        T = self.seq_len
        C = c["model_dim"]
        L = c["num_layers"]
        eps = 1.1920929e-07
        softcap = float(c["logit_softcap"])

        x = self.tok_emb(ids)
        bh = self.bigram_emb(bid)
        x = x + self.bigram_proj(bh) * self.bigram_scale
        x = rms_norm(x, eps)
        g = torch.sigmoid(self.smear_gate).view(1, 1, C)
        x_prev = torch.cat(
            [torch.zeros(B, 1, C, dtype=x.dtype, device=x.device), x[:, : T - 1]], dim=1
        )
        x = (1.0 - g) * x + g * x_prev
        x0 = x

        n_enc = L // 2
        skips: list[torch.Tensor] = []
        v0: torch.Tensor | None = None

        for layer in range(L):
            if layer >= n_enc:
                skip = skips.pop()
                x = x + self.skip_weights[layer - n_enc].view(1, 1, C) * skip

            mix = self.resid_mix[layer]
            x_in = mix[0].view(1, 1, C) * x + mix[1].view(1, 1, C) * x0
            ln = 1.0 / math.sqrt(layer + 1) if c["ln_scale"] else 1.0
            attn_in = rms_norm(x_in, eps) * ln
            attn_out, raw = self._attn(attn_in, layer, ids, v0)
            if layer == 0:
                v0 = raw
            x_mid = x_in + self.attn_scale[layer].view(1, 1, C) * attn_out
            mlp_in = rms_norm(x_mid, eps) * ln
            h = F.leaky_relu(F.linear(mlp_in, self.mlp_up[layer]), negative_slope=0.5)
            mlp_out = F.linear(h * h, self.mlp_down[layer])
            x = x_mid + self.mlp_scale[layer].view(1, 1, C) * mlp_out
            if layer < n_enc:
                skips.append(x)

        x = rms_norm(x, eps)
        logits = F.linear(x, self.tok_emb.weight)  # tied
        return softcap * torch.tanh(logits / softcap)


# ---------------------------------------------------------------------------
# Export / quant / bench
# ---------------------------------------------------------------------------


def export_stateful_kv_packages(
    weights: Path,
    out_dir: Path,
    seq_len: int,
) -> dict:
    """Export decode-step artifacts (torchscript always; mlpackage best-effort).

    Prefill package remains the non-stateful Arch02Infer baseline. Decode uses
    Arch02KV buffers as Core ML states when convert succeeds. Host can warm
    state via Arch02KV.prefill or T× decode_step.
    """
    import coremltools as ct

    from arch02_kv import Arch02KV

    out_dir.mkdir(parents=True, exist_ok=True)
    kv = Arch02KV(seq_len=seq_len)
    kv.load_npy_tree(weights)
    kv.eval()
    kv.reset_state()

    # Always ship TorchScript + state schema (torch KV parity is the correctness gate).
    ts_path = out_dir / "arch02_sota_decode_step.pt"
    example_ids = torch.zeros(1, 1, dtype=torch.int32)
    example_bg = torch.zeros(1, 1, dtype=torch.int32)
    example_mask = torch.zeros(1, 1, 1, dtype=torch.float32)
    traced = torch.jit.trace(kv, (example_ids, example_bg, example_mask))
    traced.save(str(ts_path))
    print(f"wrote {ts_path}")

    # Best-effort Core ML: fixed shapes (end_step=1) avoid RangeDim slice bugs;
    # multi-step decode stays on torch / host-warmed state for now.
    path = out_dir / "arch02_sota_decode_step_fp16.mlpackage"
    convert_err = None
    ml_path = None
    try:
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(kv.k_cache.shape), dtype=np.float16),
                name="k_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(kv.v_cache.shape), dtype=np.float16),
                name="v_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(kv.v0_cache.shape), dtype=np.float16),
                name="v0_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=tuple(kv.prev_stem.shape), dtype=np.float16),
                name="prev_stem",
            ),
        ]
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
        ml_path = path
        print(f"wrote {path}")
    except Exception as e:
        convert_err = str(e)
        print(f"Core ML stateful convert failed (torch package still valid): {e}")
        # Fallback: Apple-style 3D per-layer caches.
        try:
            from arch02_kv_flat_export import try_export_flat_mlpackage

            flat = try_export_flat_mlpackage(weights, out_dir, seq_len)
            if flat is not None:
                ml_path = flat
                convert_err = f"5D failed ({e}); flat 3D ok"
                print(f"wrote flat {flat}")
        except Exception as e2:
            convert_err = f"{e}; flat fallback: {e2}"
            print(f"flat fallback also failed: {e2}")

    meta = {
        "kv_cache": True,
        "torchscript_package": str(ts_path.name),
        "decode_package": str(ml_path.name) if ml_path else None,
        "convert_error": convert_err,
        "state_tensors": {
            "k_cache": list(kv.k_cache.shape),
            "v_cache": list(kv.v_cache.shape),
            "v0_cache": list(kv.v0_cache.shape),
            "prev_stem": list(kv.prev_stem.shape),
        },
        "warmup": "host: Arch02KV.prefill then copy buffers, or T× decode_step",
        "torch_parity": "scripts/decode_kv_reference.py",
        "status": "torch_ok_mlpackage_"
        + ("ok" if ml_path else "pending"),
    }
    (out_dir / "stateful_kv_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_mlpackage(
    model: Arch02Infer,
    out_dir: Path,
    seq_len: int,
    compute_units: str = "ALL",
) -> Path:
    import coremltools as ct

    model.eval()
    example_ids = torch.zeros(1, seq_len, dtype=torch.int32)
    example_bg = torch.zeros(1, seq_len, dtype=torch.int32)
    traced = torch.jit.trace(model, (example_ids, example_bg))

    cu = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }[compute_units]

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, seq_len), dtype=np.int32),
            ct.TensorType(name="bigram_ids", shape=(1, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=cu,
    )
    path = out_dir / "arch02_sota_fp16.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    mlmodel.save(str(path))
    return path


def palettize(mlpackage: Path, out_dir: Path, nbits: int) -> Path | None:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    mlmodel = ct.models.MLModel(str(mlpackage))
    cfg = OptimizationConfig(
        global_config=OpPalettizerConfig(mode="kmeans", nbits=nbits),
    )
    try:
        q = palettize_weights(mlmodel, cfg)
    except Exception as e:
        print(f"palettize int{nbits} failed: {e}")
        return None
    path = out_dir / f"arch02_sota_int{nbits}_palettized.mlpackage"
    if path.exists():
        shutil.rmtree(path)
    q.save(str(path))
    return path


def _predict_feed(seq_len: int) -> dict:
    ids = np.zeros((1, seq_len), dtype=np.int32)
    return {
        "input_ids": ids,
        "bigram_ids": bigram_hash_np(ids, SOTA["bigram_vocab"]),
    }


def bench_mlpackage(path: Path, seq_len: int, repeats: int = 50, warmup: int = 10) -> dict:
    import coremltools as ct

    results = {}
    x = _predict_feed(seq_len)
    for label, cu in [
        ("ALL", ct.ComputeUnit.ALL),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
    ]:
        try:
            m = ct.models.MLModel(str(path), compute_units=cu)
            for _ in range(warmup):
                m.predict(x)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                m.predict(x)
                times.append((time.perf_counter() - t0) * 1000.0)
            ms = float(np.median(times))
            results[label] = {
                "ms_forward_median": ms,
                "tok_per_s": (seq_len / (ms / 1000.0)) if ms > 0 else 0.0,
                "ok": True,
            }
        except Exception as e:
            results[label] = {"ok": False, "error": str(e)}
    return results


def inspect_compute_plan(path: Path) -> dict:
    """Best-effort ANE/GPU/CPU op placement via compute plan API."""
    try:
        import coremltools as ct

        m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        plan = m.get_compute_plan() if hasattr(m, "get_compute_plan") else None
        if plan is None:
            return {"note": "compute plan API unavailable in this coremltools build"}
        # Summarize device preferences if structure present
        summary = {"raw_type": type(plan).__name__}
        return summary
    except Exception as e:
        return {"note": f"compute plan unavailable: {e}"}


def write_readme(
    out_dir: Path,
    weight_source: str,
    seq_len: int,
    packages: dict,
    bench: dict,
    bpb_meta: dict,
) -> None:
    lines = [
        "# arch_02 Core ML export (Phase 5)",
        "",
        "Inference / deployment only — Core ML does **not** train this architecture",
        "(Muon, custom Metal bwd, on-device EMA).",
        "",
        "## Training BPB (not at parity)",
        "",
        f"- metal-native final EMA sliding BPB: **{bpb_meta.get('final_ema_sliding_bpb', 'n/a')}**",
        f"- Best live sliding BPB (step ~1999): **{bpb_meta.get('best_live_bpb', 2.2587)}**",
        f"- 3070 Ti CUDA reference: **{bpb_meta.get('reference_3070ti', 1.9944)}**",
        "- Late-run grad explosion caused the final EMA to diverge; export uses the",
        "  best *available* weight dump (see provenance below), not a CUDA-parity EMA.",
        "",
        "## Provenance",
        "",
        f"- Weight source: `{weight_source}`",
        f"- Export seq_len (fixed): **{seq_len}** (B=1)",
        "- Graph: **single-forward prefill** (causal SDPA). Stateful KV-cache deferred —",
        "  VE reinjection, smear, U-net skips, and layer-0 value residual couple the",
        "  full sequence; a correct KV decode needs a dedicated redesign.",
        "",
        "## Packages",
        "",
    ]
    for k, v in packages.items():
        lines.append(f"- `{k}`: `{v}`" if v else f"- `{k}`: *(failed)*")
    lines += [
        "",
        "## Load / run (Python)",
        "",
        "```bash",
        "source .venv/bin/activate  # Python 3.12 + coremltools",
        "python - <<'PY'",
        "import coremltools as ct, numpy as np",
        "from scripts.export_coreml import bigram_hash_np, SOTA",
        f"m = ct.models.MLModel('{packages.get('fp16', 'arch02_sota_fp16.mlpackage')}',",
        "                       compute_units=ct.ComputeUnit.ALL)",
        f"ids = np.zeros((1, {seq_len}), np.int32)",
        "bg = bigram_hash_np(ids, SOTA['bigram_vocab'])",
        "out = m.predict({'input_ids': ids, 'bigram_ids': bg})",
        "print(out['logits'].shape)  # (1, T, 1024)",
        "PY",
        "```",
        "",
        "Bigram XOR hash runs on the **host** (not in the mlpackage) so the graph stays",
        "ANE-friendly.",
        "",
        "## Quantization",
        "",
        "- int8 and int6 **weight palettization** (k-means LUT) via",
        "  `coremltools.optimize.coreml.palettize_weights` when supported.",
        "- Activations remain float; this is not activation quantization / QAT.",
        "",
        "## ANE / compute units",
        "",
        "- Preferred ops: Embedding, Linear/matmul, SDPA (causal), RMSNorm (explicit),",
        "  leaky-ReLU + square MLP, tanh softcap.",
        "- Likely **GPU/CPU fallback**: GQA expand/reshape chains, XSA normalize,",
        "  some reduce paths — inspect with Instruments / compute plan.",
        "- Bigram hash is host-side (not a Core ML op).",
        "- Bench below exercises ALL / CPU_AND_NE / CPU_AND_GPU / CPU_ONLY.",
        "",
        "## Latency (this machine)",
        "",
        "```json",
        json.dumps(bench, indent=2),
        "```",
        "",
        "## Known limits",
        "",
        "1. Training BPB not at CUDA parity (see above).",
        "2. No stateful KV-cache package in this export.",
        "3. Fixed shape `(1, T)`; re-export to change T.",
        "4. burn-port untouched; this path is metal-native → npy → Core ML only.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "golden" / "weights_init",
        help="Python-layout npy tree (EMA dump or golden/weights_init)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "coreml_export",
    )
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--bench-repeats", type=int, default=40)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument(
        "--bpb-metrics",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "out"
        / "sota_seed1337"
        / "metrics.jsonl",
    )
    ap.add_argument(
        "--stateful-kv",
        action="store_true",
        help="Export MLState decode package (requires decode_reference_ok + "
        "decode_kv_reference_ok; see STATEFUL_KV_CORE_AI.md)",
    )
    args = ap.parse_args()

    if args.stateful_kv:
        marker = args.out / "decode_reference_ok"
        kv_marker = args.out / "decode_kv_reference_ok"
        if not marker.exists():
            raise SystemExit(
                "--stateful-kv requires a passing decode reference first:\n"
                "  python scripts/decode_reference.py --weights ...\n"
                f"(expected marker {marker}; see out/coreml_export/STATEFUL_KV_CORE_AI.md)"
            )
        if not kv_marker.exists():
            raise SystemExit(
                "--stateful-kv requires torch KV parity first:\n"
                "  python scripts/decode_kv_reference.py --weights ...\n"
                f"(expected marker {kv_marker})"
            )
        # Prefer workspace-local caches (sandbox / CI friendly).
        cache = args.out / ".ct_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache)
        os.environ["PYTHONUSERBASE"] = str(cache)
        print("exporting stateful KV decode package…")
        meta = export_stateful_kv_packages(args.weights, args.out, args.seq_len)
        print(json.dumps(meta, indent=2))
        if not meta.get("torchscript_package"):
            raise SystemExit("stateful-kv export failed to write torchscript package")
        return

    # Prefer workspace-local caches (sandbox / CI friendly).
    cache = args.out / ".ct_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["PYTHONUSERBASE"] = str(cache)

    args.out.mkdir(parents=True, exist_ok=True)

    bpb_meta = {
        "final_ema_sliding_bpb": 2.7951,
        "best_live_bpb": 2.2587,
        "reference_3070ti": 1.9944,
    }
    if args.bpb_metrics.exists():
        for line in args.bpb_metrics.read_text().splitlines():
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "final_ema_sliding_bpb" in j:
                bpb_meta["final_ema_sliding_bpb"] = j["final_ema_sliding_bpb"]
                bpb_meta["reference_3070ti"] = j.get("reference_3070ti", 1.9944)

    print(f"loading weights from {args.weights}")
    model = Arch02Infer(seq_len=args.seq_len)
    model.load_npy_tree(args.weights)
    model.eval()

    # Quick torch sanity
    with torch.no_grad():
        ids = torch.randint(0, SOTA["vocab_size"], (1, args.seq_len), dtype=torch.int32)
        bg = bigram_hash_torch(ids, SOTA["bigram_vocab"]).to(torch.int32)
        logits = model(ids, bg)
        assert logits.shape == (1, args.seq_len, SOTA["vocab_size"]), logits.shape
        print(f"torch forward ok: {tuple(logits.shape)} finite={torch.isfinite(logits).all().item()}")

    print("converting → mlprogram…")
    fp16 = export_mlpackage(model, args.out, args.seq_len)
    print(f"wrote {fp16}")

    packages = {"fp16": str(fp16.name)}
    for nbits in (8, 6):
        print(f"palettizing int{nbits}…")
        p = palettize(fp16, args.out, nbits)
        packages[f"int{nbits}"] = str(p.name) if p else None
        if p:
            print(f"wrote {p}")

    bench = {}
    if not args.skip_bench:
        print("benchmarking…")
        # Prefer int8 if present else fp16
        target = args.out / (packages["int8"] or packages["fp16"])
        if not target.exists():
            target = fp16
        bench = {
            "package": str(target.name),
            "seq_len": args.seq_len,
            "compute_units": bench_mlpackage(target, args.seq_len, repeats=args.bench_repeats),
            "compute_plan": inspect_compute_plan(target),
        }
        (args.out / "bench.json").write_text(json.dumps(bench, indent=2))
        print(json.dumps(bench, indent=2))

    # Copy weight provenance note
    (args.out / "weight_source.txt").write_text(
        f"{args.weights.resolve()}\n"
        "Note: sota_seed1337 did not persist EMA buffers; default export uses\n"
        "golden/weights_init unless --weights points at an ema_weights dump.\n"
    )

    write_readme(
        args.out,
        weight_source=str(args.weights.resolve()),
        seq_len=args.seq_len,
        packages=packages,
        bench=bench,
        bpb_meta=bpb_meta,
    )
    (args.out / "export_meta.json").write_text(
        json.dumps(
            {
                "packages": packages,
                "seq_len": args.seq_len,
                "weight_source": str(args.weights.resolve()),
                "bpb": bpb_meta,
                "kv_cache": False,
                "reason_no_kv": "default path is prefill-only; use --stateful-kv",
            },
            indent=2,
        )
    )
    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
