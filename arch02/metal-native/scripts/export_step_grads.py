#!/usr/bin/env python3
"""Export post-clip grads before each of 3 optim steps (for optim-only parity).

Writes to /tmp/metal_native_grads_steps/step{k}/*.npy (does NOT modify golden/).
Used by `optim_step3_parity_vs_goldens` so param gates isolate on-device optim
from Phase-2 AdamW amplification of sub-1e-9 grad noise on near-zero elements.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/tmp/metal_native_grads_steps")


def main() -> None:
    spec = importlib.util.spec_from_file_location("eg", ROOT / "scripts/export_goldens.py")
    eg = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(eg)

    eg._apply_sota_env()
    eg._install_flash_attn_fallback()
    native = eg._load_native()
    eg._patch_muon_f32(native)
    device = eg._pick_device()
    eg._seed_everything(eg.SEED)
    args = native.Hyperparameters()
    args.seed = eg.SEED
    args.num_layers = 4
    args.model_dim = 128
    args.num_heads = 4
    args.num_kv_heads = 2
    args.mlp_mult = 3.0
    args.vocab_size = 1024
    args.train_seq_len = eg.SEQ_LEN
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
        vocab_size=1024,
        num_layers=4,
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=3,
        tie_embeddings=True,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=512,
        bigram_dim=48,
        xsa_last_n=2,
        xsa_mode=args.xsa_mode,
        xsa_value_source=args.xsa_value_source,
        rope_dims=8,
        ln_scale=True,
        dtg=False,
        ve_enabled=True,
        ve_dim=24,
        ve_layers="2,3",
        gated_attention=False,
        value_residual=True,
        arch_variant="baseline",
    ).to(device).float()
    native.restore_low_dim_params_to_fp32(model)
    model.train()
    eg._seed_everything(eg.SEED)
    batches = eg._make_batches(device, 1024, 3)
    opt_tok, opt_muon, opt_scalar, _ = eg._build_optimizers(native, model, args)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for step in range(3):
        for opt in (opt_tok, opt_muon, opt_scalar):
            opt.zero_grad(set_to_none=True)
        xb, yb = batches[step]
        loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        sub = OUT / f"step{step}"
        sub.mkdir()
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            rel = sub / f"{name.replace('.', '/')}.npy"
            rel.parent.mkdir(parents=True, exist_ok=True)
            np.save(rel, p.grad.detach().float().cpu().contiguous().numpy())
        print(f"step{step} loss={float(loss.detach()):.4f} -> {sub}")
        mom = eg._muon_momentum(args, step)
        for group in opt_muon.param_groups:
            group["momentum"] = mom
            group["lr"] = group["base_lr"]
        for opt in (opt_tok, opt_scalar):
            for group in opt.param_groups:
                group["lr"] = group["base_lr"]
        opt_muon.launch_reduce_scatters()
        opt_tok.step()
        opt_scalar.step()
        opt_muon.step()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
