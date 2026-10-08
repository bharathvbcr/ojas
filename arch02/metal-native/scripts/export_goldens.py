#!/usr/bin/env python3
"""Export f32 golden tensors for arch_02 metal-native parity gates.

Sota shapes (from burn-port ModelConfig::sota_toy / run_toy_3070ti.py):
  L=4 C=128 H=4 Hkv=2 hd=32 rope=8/32 mlp=384 V=1024 B=16 T=256
  VALUE_RESIDUAL=1 GATED_ATTENTION=0, seed=1337, pure f32 (no bf16 autocast).

Usage (from repo root or any cwd):
  python arch02/metal-native/scripts/export_goldens.py
  python .../export_goldens.py --out /path/to/golden

Writes under metal-native/golden/:
  fwd/*.npy, grads/*.npy, optim_step3/*.npy, weights_init/*.npy, manifest.json, README.md
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
METAL_NATIVE_DIR = SCRIPT_DIR.parent
ARCH_DIR = METAL_NATIVE_DIR.parent
NATIVE_TRAIN_PATH = ARCH_DIR / "train_gpt_sprint_native.py"
DEFAULT_OUT = METAL_NATIVE_DIR / "golden"

SEED = 1337
BATCH = 16
SEQ_LEN = 256
NUM_OPTIM_STEPS = 3

# Sota env (must be set before importing Hyperparameters / constructing GPT).
SOTA_ENV: dict[str, str] = {
    "SEED": str(SEED),
    "NUM_LAYERS": "4",
    "MODEL_DIM": "128",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "3",
    "VOCAB_SIZE": "1024",
    "TRAIN_SEQ_LEN": str(SEQ_LEN),
    "BIGRAM_VOCAB_SIZE": "512",
    "BIGRAM_DIM": "48",
    "XSA_LAST_N": "2",
    "XSA_MODE": "paper",
    "XSA_VALUE_SOURCE": "mixed",
    "ROPE_DIMS": "8",
    "LN_SCALE": "1",
    "VE_ENABLED": "1",
    "VE_DIM": "24",
    "VE_LAYERS": "2,3",
    "VALUE_RESIDUAL": "1",
    "GATED_ATTENTION": "0",
    "TIE_EMBEDDINGS": "1",
    "LOGIT_SOFTCAP": "30.0",
    "QK_GAIN_INIT": "1.5",
    "TIED_EMBED_INIT_STD": "0.005",
    "MATRIX_LR": "0.025",
    "SCALAR_LR": "0.025",
    "TIED_EMBED_LR": "0.035",
    "MUON_MOMENTUM": "0.95",
    "MUON_MOMENTUM_WARMUP_START": "0.92",
    "MUON_MOMENTUM_WARMUP_STEPS": "1500",
    "MUON_BACKEND_STEPS": "5",
    "MUON_WD": "0.04",
    "ADAM_WD": "0.04",
    "BETA1": "0.9",
    "BETA2": "0.95",
    "ADAM_EPS": "1e-8",
    "GRAD_CLIP_NORM": "0.3",
    "WARMDOWN_ITERS": "0",
    "WARMUP_STEPS": "0",
    "MTP_NUM_HEADS": "0",
    "DTG_ENABLED": "0",
    "QAT_ENABLED": "0",
    "ARCH_VARIANT": "baseline",
    "SPRINT_FLASH_BACKEND": "sdpa_fallback",
}


def _apply_sota_env() -> None:
    for k, v in SOTA_ENV.items():
        os.environ[k] = v


def _install_flash_attn_fallback() -> str:
    """Install SDPA-backed flash_attn_interface before native import."""
    try:
        import flash_attn_interface  # noqa: F401

        os.environ["SPRINT_FLASH_BACKEND"] = "flash_attn_interface"
        return "flash_attn_interface"
    except ImportError:
        pass

    fallback = types.ModuleType("flash_attn_interface")

    def flash_attn_func(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        **_: Any,
    ) -> Tensor:
        del dropout_p
        qh = q.permute(0, 2, 1, 3).contiguous()
        kh = k.permute(0, 2, 1, 3).contiguous()
        vh = v.permute(0, 2, 1, 3).contiguous()
        if qh.size(1) != kh.size(1):
            if qh.size(1) % kh.size(1) != 0:
                raise ValueError(
                    f"GQA fallback expected Q heads divisible by KV heads, "
                    f"got {qh.size(1)} and {kh.size(1)}"
                )
            repeat = qh.size(1) // kh.size(1)
            kh = kh.repeat_interleave(repeat, dim=1)
            vh = vh.repeat_interleave(repeat, dim=1)
        # Force math SDPA for deterministic CPU/MPS goldens.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                out = F.scaled_dot_product_attention(
                    qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=softmax_scale
                )
        except Exception:
            out = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=softmax_scale
            )
        return out.permute(0, 2, 1, 3).contiguous()

    fallback.flash_attn_func = flash_attn_func
    sys.modules["flash_attn_interface"] = fallback
    os.environ["SPRINT_FLASH_BACKEND"] = "sdpa_fallback"
    return "sdpa_fallback"


def _load_native() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("arch02_sprint_native", NATIVE_TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {NATIVE_TRAIN_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer deterministic math where available.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _zeropower_f32(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz5 in f32 (reference uses bf16; goldens stay f32)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


def _patch_muon_f32(native: types.ModuleType) -> None:
    """Keep Muon momentum + NS5 in f32 (reference uses bf16 for both)."""
    native.zeropower_via_newtonschulz5 = _zeropower_f32

    @torch.no_grad()
    def step_f32(self, closure=None):  # type: ignore[no-untyped-def]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            for m in self._bank_meta:
                p = m["p"]
                if p.grad is None:
                    continue
                # f32 deviation: reference does p.grad.bfloat16()
                g = p.grad.float()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = native.zeropower_via_newtonschulz5(update, steps=backend_steps)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(update.to(dtype=p.dtype), alpha=-lr * m["scale"])
        return loss

    native.Muon.step = step_f32  # type: ignore[method-assign]


def _make_batches(device: torch.device, vocab: int, n: int) -> list[tuple[Tensor, Tensor]]:
    """Deterministic synthetic token batches (no FineWeb dependency)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    batches: list[tuple[Tensor, Tensor]] = []
    for i in range(n):
        # Distinct streams per batch index while remaining seed-fixed.
        g.manual_seed(SEED + 10_000 * (i + 1))
        tokens = torch.randint(0, vocab, (BATCH, SEQ_LEN + 1), generator=g, dtype=torch.int64)
        x = tokens[:, :-1].contiguous().to(device)
        y = tokens[:, 1:].contiguous().to(device)
        batches.append((x, y))
    return batches


def save_tensor(out_dir: Path, rel: str, t: Tensor | float | int, manifest: dict[str, Any], *, group: str, how: str) -> None:
    path = out_dir / rel
    if not path.suffix:
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(t, (float, int)):
        # Scalars stored as float32 for a uniform parity dtype (except int counters).
        if isinstance(t, float):
            arr = np.asarray(t, dtype=np.float32)
        else:
            arr = np.asarray(t, dtype=np.int64)
        shape = list(arr.shape)
        dtype = str(arr.dtype)
    else:
        arr = t.detach().float().cpu().contiguous().numpy()
        shape = list(arr.shape)
        dtype = "float32"
    np.save(path, arr)
    key = rel[:-4] if rel.endswith(".npy") else rel
    key = key.replace("\\", "/")
    file_rel = (key + ".npy") if not rel.endswith(".npy") else rel.replace("\\", "/")
    digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    manifest["tensors"][key] = {
        "file": file_rel,
        "shape": shape,
        "dtype": dtype,
        "group": group,
        "how": how,
        "sha256_16": digest,
    }


@torch.no_grad()
def capture_forward(model: nn.Module, input_ids: Tensor, target_ids: Tensor) -> dict[str, Tensor | float]:
    """Instrumented baseline backbone + head, mirroring GPT._forward_baseline_backbone."""
    outs: dict[str, Tensor | float] = {}
    x = model.tok_emb(input_ids)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = model.smear(x)
    outs["fwd/stem_after_smear"] = x.detach().clone()
    x0 = x
    v0 = None
    skips: list[Tensor] = []
    ve_cache: dict[str, Tensor] = {}

    for i in range(model.num_encoder_layers):
        attn_out, mlp_out, x, raw_v = _block_with_parts(model, i, x, x0, input_ids, ve_cache, v0)
        outs[f"fwd/layer{i}_attn_out"] = attn_out.detach().clone()
        outs[f"fwd/layer{i}_mlp_out"] = mlp_out.detach().clone()
        outs[f"fwd/layer{i}_x"] = x.detach().clone()
        if v0 is None and raw_v is not None:
            v0 = raw_v
            outs["fwd/v0"] = v0.detach().clone()
        skips.append(x)

    for i in range(model.num_decoder_layers):
        bi = model.num_encoder_layers + i
        if skips:
            skip = skips.pop()
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
            outs[f"fwd/layer{bi}_after_skip"] = x.detach().clone()
        attn_out, mlp_out, x, _ = _block_with_parts(model, bi, x, x0, input_ids, ve_cache, v0)
        outs[f"fwd/layer{bi}_attn_out"] = attn_out.detach().clone()
        outs[f"fwd/layer{bi}_mlp_out"] = mlp_out.detach().clone()
        outs[f"fwd/layer{bi}_x"] = x.detach().clone()

    x = model.final_norm(x)
    outs["fwd/final_norm"] = x.detach().clone()
    x_flat = x.reshape(-1, x.size(-1))
    logits_pre = model._project_logits(x_flat)
    logits_post = model.logit_softcap * torch.tanh(logits_pre / model.logit_softcap)
    outs["fwd/logits_pre_softcap"] = logits_pre.detach().clone().reshape(input_ids.shape[0], input_ids.shape[1], -1)
    outs["fwd/logits_post_softcap"] = logits_post.detach().clone().reshape(input_ids.shape[0], input_ids.shape[1], -1)
    targets = target_ids.reshape(-1)
    loss = F.cross_entropy(logits_post.float(), targets, reduction="mean")
    outs["fwd/loss"] = float(loss.item())
    return outs


def _block_with_parts(
    model: nn.Module,
    layer_idx: int,
    x: Tensor,
    x0: Tensor,
    input_ids: Tensor,
    ve_cache: dict[str, Tensor],
    v0: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Return (attn_out, mlp_out, x_out, raw_v) for one block."""
    n = model.num_layers
    block = model.blocks[layer_idx]
    ve = model._get_ve(layer_idx, input_ids, ve_cache)
    mix = block.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out, raw_v = block.attn(
        block.attn_norm(x_in) * block.ln_scale_factor,
        model.qo_bank[layer_idx],
        model.kv_bank[layer_idx],
        model.kv_bank[n + layer_idx],
        model.qo_bank[n + layer_idx],
        v_embed=ve,
        v0=v0,
    )
    x_mid = x_in + block.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
    mlp_out = block.mlp(
        block.mlp_norm(x_mid) * block.ln_scale_factor,
        model.mlp_up_bank[layer_idx],
        model.mlp_down_bank[layer_idx],
    )
    x_out = x_mid + block.mlp_scale.to(dtype=x_mid.dtype)[None, None, :] * mlp_out
    return attn_out, mlp_out, x_out, raw_v


def _build_optimizers(native: types.ModuleType, model: nn.Module, args: Any) -> tuple[Any, Any, Any, list]:
    matrix_params = [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]
    block_named_params = list(model.blocks.named_parameters())
    patterns = native.CONTROL_TENSOR_NAME_PATTERNS
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in patterns)
    ]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)
    scalar_params.append(model.smear.gate)
    if model.bigram is not None:
        scalar_params.append(model.bigram.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if model.bigram is not None:
        tok_params.append({"params": [model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if model.bigram.proj is not None:
            scalar_params.append(model.bigram.proj.weight)
    if model.ve_shared is not None:
        tok_params.append({"params": [model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if model.ve_shared.proj is not None:
            scalar_params.append(model.ve_shared.proj.weight)
        scalar_params.append(model.ve_shared.scale)
        for s in model.ve_layer_scales:
            scalar_params.append(s)

    # fused=True is CUDA-only; goldens often run on CPU/MPS.
    fused = bool(torch.cuda.is_available())
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=fused,
    )
    optimizer_muon = native.Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=fused,
    )
    return optimizer_tok, optimizer_muon, optimizer_scalar, scalar_params


def _muon_momentum(args: Any, step: int) -> float:
    if args.muon_momentum_warmup_steps <= 0:
        return float(args.muon_momentum)
    frac = min(step / args.muon_momentum_warmup_steps, 1.0)
    return (1.0 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum


def _export_adam_state(opt: torch.optim.Optimizer, name_map: dict[int, str], out_dir: Path, manifest: dict, prefix: str) -> None:
    for p, state in opt.state.items():
        if not state:
            continue
        pname = name_map.get(id(p), f"param_{id(p)}")
        safe = pname.replace(".", "/")
        if "exp_avg" in state:
            save_tensor(
                out_dir,
                f"{prefix}/{safe}_exp_avg.npy",
                state["exp_avg"],
                manifest,
                group="optim_step3",
                how=f"AdamW exp_avg (1st moment) after {NUM_OPTIM_STEPS} steps; param={pname}",
            )
        if "exp_avg_sq" in state:
            save_tensor(
                out_dir,
                f"{prefix}/{safe}_exp_avg_sq.npy",
                state["exp_avg_sq"],
                manifest,
                group="optim_step3",
                how=f"AdamW exp_avg_sq (2nd moment) after {NUM_OPTIM_STEPS} steps; param={pname}",
            )
        if "step" in state:
            step_t = state["step"]
            step_val = float(step_t.item()) if torch.is_tensor(step_t) else float(step_t)
            save_tensor(
                out_dir,
                f"{prefix}/{safe}_step.npy",
                step_val,
                manifest,
                group="optim_step3",
                how=f"AdamW step counter after {NUM_OPTIM_STEPS} steps; param={pname}",
            )


def _param_name_map(model: nn.Module) -> dict[int, str]:
    return {id(p): n for n, p in model.named_parameters()}


def export(out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("fwd", "grads", "optim_step3", "weights_init", "inputs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    _apply_sota_env()
    flash_backend = _install_flash_attn_fallback()
    native = _load_native()
    _patch_muon_f32(native)

    device = _pick_device()
    _seed_everything(SEED)

    # Rebuild Hyperparameters from env (class body already ran at import — re-read attrs).
    # Hyperparameters fields are set at class definition time from os.environ, which we
    # set before import, so args should already match SOTA_ENV.
    args = native.Hyperparameters()
    # Force critical fields in case import order raced.
    args.seed = SEED
    args.num_layers = 4
    args.model_dim = 128
    args.num_heads = 4
    args.num_kv_heads = 2
    args.mlp_mult = 3.0
    args.vocab_size = 1024
    args.train_seq_len = SEQ_LEN
    args.bigram_vocab_size = 512
    args.bigram_dim = 48
    args.xsa_last_n = 2
    args.rope_dims = 8
    args.ln_scale = True
    args.ve_enabled = True
    args.ve_dim = 24
    args.ve_layers = "2,3"
    args.value_residual = True
    args.gated_attention = False
    args.tie_embeddings = True

    model = native.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=int(args.mlp_mult) if float(args.mlp_mult).is_integer() else args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        xsa_mode=args.xsa_mode,
        xsa_value_source=args.xsa_value_source,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=False,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=False,
        value_residual=True,
        arch_variant="baseline",
    ).to(device).float()
    native.restore_low_dim_params_to_fp32(model)
    model.train()

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "flash_backend": flash_backend,
        "dtype_policy": "f32_throughout",
        "reference": str(NATIVE_TRAIN_PATH),
        "config": {
            "preset": "sota",
            "num_layers": 4,
            "model_dim": 128,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 32,
            "rope_dims": 8,
            "mlp_dim": 384,
            "vocab_size": 1024,
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "value_residual": True,
            "gated_attention": False,
            "bigram_vocab_size": 512,
            "bigram_dim": 48,
            "ve_dim": 24,
            "ve_layers": [2, 3],
            "xsa_last_n": 2,
            "ln_scale": True,
            "logit_softcap": 30.0,
        },
        "weight_layout": {
            "python_torch": "[out, in] — nn.Linear / F.linear convention; banks shaped "
            "(bank, out_features, in_features). qo_bank[i]: Q [C,C]; qo_bank[L+i]: Out [C,C]; "
            "kv_bank[i]: K [kv_dim,C]; kv_bank[L+i]: V [kv_dim,C]; "
            "mlp_up [mlp,C]; mlp_down [C,mlp].",
            "burn_metal": "[in, out] — x.matmul(w). Rust parity harness must transpose "
            "2D slices (swap last two dims of each bank matrix) when loading these goldens.",
            "bank_shapes_python": {
                "qo_bank": [8, 128, 128],
                "kv_bank": [8, 64, 128],
                "mlp_up_bank": [4, 384, 128],
                "mlp_down_bank": [4, 128, 384],
            },
            "bank_shapes_burn_equivalent": {
                "qo": [8, 128, 128],
                "kv": [8, 128, 64],
                "up": [4, 128, 384],
                "down": [4, 384, 128],
            },
            "muon_scale": "scale = sqrt(max(1, out/in)) = sqrt(max(1, shape[-2]/shape[-1])) on Python layout",
        },
        "deviations_from_reference": [
            "No bf16 autocast; model + activations + grads kept in float32.",
            "Muon Newton-Schulz5 and momentum buffers run in float32 (reference uses bfloat16 for both).",
            "Muon.step patched to single-process f32 path (no reduce-scatter / all-gather).",
            "Attention via SDPA MATH backend when flash_attn_interface is unavailable (SPRINT_FLASH_BACKEND=sdpa_fallback).",
            "Synthetic input_ids/targets from torch.randint(seed=1337+10000*batch_idx); no FineWeb shards.",
            "AdamW fused=False on non-CUDA devices.",
            "EMA / SWA not exported (optimizer-state gate only).",
        ],
        "protocol": [
            "1. Init GPT (sota) at seed 1337, f32.",
            "2. Save initial weights.",
            "3. Batch0: capture module forward outputs (no grad).",
            "4. Batch0: forward+backward, clip_grad_norm_(0.3), save per-param grads (pre-optim).",
            "5. Run 3 full optimizer steps (tok AdamW → scalar AdamW → Muon) on batches 0,1,2.",
            "6. Save Muon momentum buffers + AdamW moments after step 3.",
        ],
        "tensors": {},
    }

    # --- initial weights ---
    for name, p in model.named_parameters():
        save_tensor(
            out_dir,
            f"weights_init/{name.replace('.', '/')}.npy",
            p,
            manifest,
            group="weights_init",
            how=f"Initial parameter after GPT._init_weights; torch layout; name={name}",
        )

    batches = _make_batches(device, args.vocab_size, NUM_OPTIM_STEPS)
    for bi, (xb, yb) in enumerate(batches):
        for label, tensor in ((f"input_ids_batch{bi}", xb), (f"target_ids_batch{bi}", yb)):
            path = out_dir / "inputs" / f"{label}.npy"
            arr = tensor.detach().cpu().numpy().astype(np.int64)
            np.save(path, arr)
            manifest["tensors"][f"inputs/{label}"] = {
                "file": f"inputs/{label}.npy",
                "shape": list(tensor.shape),
                "dtype": "int64",
                "group": "inputs",
                "how": f"Synthetic {label} (int64); generator seed SEED+10000*(batch_idx+1)",
                "sha256_16": hashlib.sha256(arr.tobytes()).hexdigest()[:16],
            }
    x0, y0 = batches[0]

    # --- forward capture ---
    model.eval()
    with torch.no_grad():
        fwd = capture_forward(model, x0, y0)
    model.train()
    for key, val in fwd.items():
        # key already like fwd/stem_after_smear
        save_tensor(out_dir, f"{key}.npy", val, manifest, group="fwd", how=f"Forward capture: {key}")

    # Verify loss matches training forward
    loss_check = model(x0, y0)
    save_tensor(
        out_dir,
        "fwd/loss_train_forward.npy",
        float(loss_check.detach().item()),
        manifest,
        group="fwd",
        how="GPT.forward CE loss on batch0 (should match fwd/loss)",
    )

    optimizer_tok, optimizer_muon, optimizer_scalar, _scalar_params = _build_optimizers(native, model, args)
    name_map = _param_name_map(model)

    # --- grads after 1 training step (backward+clip, before optim) ---
    for opt in (optimizer_tok, optimizer_muon, optimizer_scalar):
        opt.zero_grad(set_to_none=True)
    loss = model(x0, y0)
    loss.backward()
    if args.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        save_tensor(
            out_dir,
            f"grads/{name.replace('.', '/')}.npy",
            p.grad,
            manifest,
            group="grads",
            how=f"Gradient after 1 fwd+bwd on batch0, post clip_grad_norm_({args.grad_clip_norm}); name={name}",
        )
    save_tensor(
        out_dir,
        "grads/loss_step1.npy",
        float(loss.detach().item()),
        manifest,
        group="grads",
        how="Training loss for the grad-producing forward (batch0)",
    )

    # Complete step 1 optim, then steps 2..NUM_OPTIM_STEPS
    for step in range(NUM_OPTIM_STEPS):
        if step > 0:
            for opt in (optimizer_tok, optimizer_muon, optimizer_scalar):
                opt.zero_grad(set_to_none=True)
            xb, yb = batches[step]
            loss = model(xb, yb)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        mom = _muon_momentum(args, step)
        for group in optimizer_muon.param_groups:
            group["momentum"] = mom
            group["lr"] = group["base_lr"]  # warmdown=0 → lr_mul=1
        for opt in (optimizer_tok, optimizer_scalar):
            for group in opt.param_groups:
                group["lr"] = group["base_lr"]

        optimizer_muon.launch_reduce_scatters()
        optimizer_tok.step()
        optimizer_scalar.step()
        optimizer_muon.step()

    # --- optimizer states after 3 steps ---
    # Muon momentum buffers
    bank_param_names = {
        id(model.qo_bank): "qo_bank",
        id(model.kv_bank): "kv_bank",
        id(model.mlp_up_bank): "mlp_up_bank",
        id(model.mlp_down_bank): "mlp_down_bank",
    }
    for p, state in optimizer_muon.state.items():
        bname = bank_param_names.get(id(p), f"bank_{id(p)}")
        if "momentum_buffer" in state:
            save_tensor(
                out_dir,
                f"optim_step3/muon/{bname}_momentum_buffer.npy",
                state["momentum_buffer"].float(),
                manifest,
                group="optim_step3",
                how=(
                    f"Muon momentum_buffer after {NUM_OPTIM_STEPS} steps (Nesterov path, f32). "
                    f"Bank={bname}."
                ),
            )
    # Also dump bank meta scales if built
    if getattr(optimizer_muon, "_built", False):
        for m in optimizer_muon._bank_meta:
            bname = bank_param_names.get(id(m["p"]), "unknown")
            save_tensor(
                out_dir,
                f"optim_step3/muon/{bname}_scale.npy",
                float(m["scale"]),
                manifest,
                group="optim_step3",
                how=f"Muon LR scale = sqrt(max(1, out/in)) for bank {bname}",
            )

    _export_adam_state(optimizer_tok, name_map, out_dir, manifest, "optim_step3/adamw_embed")
    _export_adam_state(optimizer_scalar, name_map, out_dir, manifest, "optim_step3/adamw_scalar")

    # Post-step params (helpful for step-level parity later)
    for name, p in model.named_parameters():
        save_tensor(
            out_dir,
            f"optim_step3/params/{name.replace('.', '/')}.npy",
            p,
            manifest,
            group="optim_step3",
            how=f"Parameter values after {NUM_OPTIM_STEPS} optimizer steps; name={name}",
        )

    # Write manifest + README
    manifest_path = out_dir / "manifest.json"
    # Deduplicate any accidental duplicate keys from earlier helper — already unique by path key.
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    readme = f"""# arch_02 metal-native golden tensors

Generated by `metal-native/scripts/export_goldens.py`.

## Config

Sota toy: L=4 C=128 H=4 Hkv=2 hd=32 rope=8/32 mlp=384 V=1024 B={BATCH} T={SEQ_LEN}, seed={SEED}, **f32**.

## Weight layout

| Framework | Linear weight layout | Matmul |
|-----------|---------------------|--------|
| **Python / these `.npy` files** | `[out, in]` | `F.linear(x, W)` ≡ `x @ W.T` |
| **Burn / metal-native** | `[in, out]` | `x.matmul(W)` |

When loading bank matrices into Rust, **transpose the last two dimensions** of each
2D slice (Python `kv_bank` is `[8, 64, 128]` → Burn `[8, 128, 64]`).

## Groups

- `inputs/` — synthetic batch 0 tokens
- `weights_init/` — parameters right after init
- `fwd/` — stem after smear, per-layer attn/mlp outs, logits pre/post softcap, loss
- `grads/` — per-param grads after 1 fwd+bwd + grad clip (before optim)
- `optim_step3/` — Muon momentum buffers, AdamW moments, params after 3 steps

## Regenerate

```bash
python arch02/metal-native/scripts/export_goldens.py
```

## Deviations

See `manifest.json` → `deviations_from_reference`. Notably: f32 NS5 (not bf16),
SDPA math fallback if FlashAttention-3 is missing, synthetic tokens.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    n_tensors = len(manifest["tensors"])
    print(f"Wrote {n_tensors} tensors → {out_dir}")
    print(f"device={device} flash={flash_backend} loss_fwd={fwd['fwd/loss']:.6f}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output golden directory")
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
