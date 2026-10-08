#!/usr/bin/env python3
"""KV incremental decode parity vs Arch02Infer prefill.

Extends decode_reference.py: after causal prefix contract, prove Arch02KV
prefill + token-by-token decode_step match full-seq logits.

Usage:
  python scripts/decode_kv_reference.py [--weights PATH] [--seq-len 32] [--tol 2e-4]

Writes `out/coreml_export/decode_kv_reference_ok` on success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch02_kv import Arch02KV  # noqa: E402
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
    infer = Arch02Infer(seq_len=args.seq_len)
    infer.load_npy_tree(args.weights)
    infer.eval()

    kv = Arch02KV(seq_len=args.seq_len)
    kv.load_npy_tree(args.weights)
    kv.eval()

    ids = torch.randint(0, SOTA["vocab_size"], (1, args.seq_len), dtype=torch.int32)
    bg = bigram_hash_torch(ids, SOTA["bigram_vocab"]).to(torch.int32)
    pref = infer(ids, bg)

    # 1) KV prefill vs Arch02Infer
    kv_pref = kv.prefill(ids, bg)
    err_pref = (kv_pref - pref).abs().max().item()
    if err_pref > args.tol:
        print(f"FAIL kv prefill vs infer max_abs={err_pref:.6e}")
        return 1
    print(f"PASS kv prefill vs infer max_abs={err_pref:.6e}")

    # 2) Token-by-token decode from empty vs prefill
    kv.reset_state()
    max_dec = 0.0
    for t in range(args.seq_len):
        step = kv.decode_step(ids[:, t : t + 1], bg[:, t : t + 1], t=t)
        err = (step[:, 0, :] - pref[:, t, :]).abs().max().item()
        max_dec = max(max_dec, err)
        if err > args.tol:
            print(f"FAIL decode_step t={t} max_abs={err:.6e}")
            return 1
    print(f"PASS decode_step vs prefill max_abs={max_dec:.6e}")

    # 3) Prefill half then decode rest
    mid = args.seq_len // 2
    kv.reset_state()
    _ = kv.prefill(ids[:, :mid], bg[:, :mid])
    max_tail = 0.0
    for t in range(mid, args.seq_len):
        step = kv.decode_step(ids[:, t : t + 1], bg[:, t : t + 1], t=t)
        err = (step[:, 0, :] - pref[:, t, :]).abs().max().item()
        max_tail = max(max_tail, err)
        if err > args.tol:
            print(f"FAIL prefill+decode t={t} max_abs={err:.6e}")
            return 1
    print(f"PASS prefill+decode tail max_abs={max_tail:.6e}")

    marker = (
        Path(__file__).resolve().parents[1]
        / "out"
        / "coreml_export"
        / "decode_kv_reference_ok"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"prefill={err_pref}\ndecode={max_dec}\ntail={max_tail}\nseq_len={args.seq_len}\n"
    )
    print(f"wrote {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
