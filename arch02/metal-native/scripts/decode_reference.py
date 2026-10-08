#!/usr/bin/env python3
"""Decode math reference for arch_02 stateful KV (Core ML / Core AI).

Gate: causal one-token decode (prefix prefill) must match full-seq prefill
logits at every position t. This is the correctness contract before
`--stateful-kv` export can emit MLState K/V packages.

Usage:
  python scripts/decode_reference.py [--weights PATH] [--seq-len 64] [--tol 2e-4]

Writes `out/coreml_export/decode_reference_ok` on success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_coreml import Arch02Infer, SOTA, bigram_hash_torch  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "golden" / "weights_init",
    )
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--tol", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # Prefill model at full seq; decode checks use matching seq_len truncations.
    model = Arch02Infer(seq_len=args.seq_len)
    model.load_npy_tree(args.weights)
    model.eval()

    ids = torch.randint(0, SOTA["vocab_size"], (1, args.seq_len), dtype=torch.int32)
    bg = bigram_hash_torch(ids, SOTA["bigram_vocab"]).to(torch.int32)
    pref = model(ids, bg)

    max_err = 0.0
    for t in range(args.seq_len):
        # Causal decode reference = prefill on prefix 0..=t, take last position.
        # Stateful KV must reproduce this; VE / smear / v0 / skips are inside.
        mt = Arch02Infer(seq_len=t + 1)
        mt.load_npy_tree(args.weights)
        mt.eval()
        dec = mt(ids[:, : t + 1], bg[:, : t + 1])[:, -1, :]
        err = (dec - pref[:, t, :]).abs().max().item()
        max_err = max(max_err, err)
        if err > args.tol:
            print(f"FAIL t={t} max_abs={err:.6e} tol={args.tol}")
            return 1

    print(f"PASS decode_reference seq_len={args.seq_len} max_abs={max_err:.6e}")
    marker = (
        Path(__file__).resolve().parents[1]
        / "out"
        / "coreml_export"
        / "decode_reference_ok"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"max_abs={max_err}\nseq_len={args.seq_len}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
