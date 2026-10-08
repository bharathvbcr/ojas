#!/usr/bin/env python3
"""MLX DFlash golden token streams + decode tok/s for gemma-metal parity gates.

Writes `bench/results/dflash_parity_mlx_golden.json` with:
  - short-prompt DFlash + greedy (mlx-lm) token id streams
  - accept totals / mean accept length
  - decode tok/s (quiet, after warm)

Requires `~/.venvs/dflash32` (mlx 0.32 + dflash). Exact verify ⇒ DFlash tokens
must equal greedy on the *same* MLX target; cross-stack vs gemma-metal may diverge.

Usage:
    ~/.venvs/dflash32/bin/python bench/dflash_parity_golden.py
    ~/.venvs/dflash32/bin/python bench/dflash_parity_golden.py --max-tokens 32
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate as mlx_greedy_stream
from mlx_lm.sample_utils import make_sampler
from dflash.model_mlx import load_draft, stream_generate as dflash_stream

TARGET = "mlx-community/gemma-4-31b-it-4bit"
DRAFT = "z-lab/gemma-4-31B-it-DFlash"
BLOCK = 5
PROMPTS = [
    {"id": "say_hi", "user": "Say hi"},
    {"id": "two_plus_two", "user": "What is 2+2? Answer with one digit."},
]


def chat_prompt(tok, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True,
        tokenize=False,
    )


def run_dflash(model, draft, tok, prompt: str, block: int, max_tokens: int):
    texts, token_ids, accepts = [], [], []
    last = None
    for r in dflash_stream(
        model, draft, tok, prompt, block_size=block, max_tokens=max_tokens, temperature=0.0
    ):
        texts.append(r.text)
        token_ids.extend(list(r.tokens))
        accepts.append(int(r.accepted))
        last = r
    return {
        "text": "".join(texts),
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
        "accepts": accepts,
        "mean_accept": (sum(accepts) / len(accepts)) if accepts else 0.0,
        "decode_tok_s": float(last.generation_tps) if last else 0.0,
        "peak_memory_gb": float(last.peak_memory) if last else None,
    }


def run_greedy(model, tok, prompt: str, max_tokens: int):
    texts, token_ids = [], []
    last = None
    sampler = make_sampler(temp=0.0)
    for r in mlx_greedy_stream(
        model, tok, prompt, max_tokens=max_tokens, sampler=sampler
    ):
        texts.append(r.text)
        token_ids.append(int(r.token))
        last = r
    tps = float(getattr(last, "generation_tps", 0.0) or 0.0) if last else 0.0
    return {
        "text": "".join(texts),
        "token_ids": token_ids,
        "n_tokens": len(token_ids),
        "decode_tok_s": tps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--skip-greedy", action="store_true")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dflash_parity_mlx_golden.json"

    t0 = time.perf_counter()
    model, tok = mlx_load(TARGET)
    draft = load_draft(DRAFT)
    nn.quantize(
        draft,
        group_size=64,
        bits=4,
        class_predicate=lambda p, m: isinstance(m, nn.Linear)
        and m.weight.shape[-1] % 64 == 0,
    )
    draft.bind(model)
    print(f"# loaded in {time.perf_counter()-t0:.1f}s (block={args.block})", flush=True)

    # Build prompts once.
    built = []
    for p in PROMPTS:
        prompt = chat_prompt(tok, p["user"])
        prompt_ids = tok.encode(
            prompt,
            add_special_tokens=tok.bos_token is None
            or not prompt.startswith(tok.bos_token or ""),
        )
        built.append({**p, "prompt": prompt, "prompt_ids": prompt_ids})

    # Greedy first (before DFlash patches target for capture).
    greeds = {}
    if not args.skip_greedy:
        warm_g = chat_prompt(tok, "hi")
        _ = run_greedy(model, tok, warm_g, 4)
        mx.clear_cache()
        for p in built:
            print(f"# greedy prompt={p['id']} max_tokens={args.max_tokens}", flush=True)
            greeds[p["id"]] = run_greedy(model, tok, p["prompt"], args.max_tokens)
            print(
                f"  greedy={greeds[p['id']]['decode_tok_s']:.1f} tok/s "
                f"n={greeds[p['id']]['n_tokens']}",
                flush=True,
            )
            mx.clear_cache()

    # Quiet DFlash warm + runs.
    warm = chat_prompt(tok, "hi")
    _ = run_dflash(model, draft, tok, warm, args.block, 8)
    mx.clear_cache()

    runs = []
    for p in built:
        print(f"# dflash prompt={p['id']} max_tokens={args.max_tokens}", flush=True)
        dflash = run_dflash(model, draft, tok, p["prompt"], args.block, args.max_tokens)
        greedy = greeds.get(p["id"])
        match = None
        if greedy is not None:
            n = min(len(dflash["token_ids"]), len(greedy["token_ids"]))
            match = dflash["token_ids"][:n] == greedy["token_ids"][:n]
            first_mismatch = None
            if not match:
                for i in range(n):
                    if dflash["token_ids"][i] != greedy["token_ids"][i]:
                        first_mismatch = i
                        break
            print(
                f"  dflash={dflash['decode_tok_s']:.1f} tok/s mean_accept={dflash['mean_accept']:.2f} "
                f"greedy={greedy['decode_tok_s']:.1f} match_prefix={match}"
                + (f" mismatch@{first_mismatch}" if first_mismatch is not None else ""),
                flush=True,
            )
        else:
            print(
                f"  dflash={dflash['decode_tok_s']:.1f} tok/s mean_accept={dflash['mean_accept']:.2f}",
                flush=True,
            )
        runs.append(
            {
                "id": p["id"],
                "user": p["user"],
                "prompt_n_tokens": len(p["prompt_ids"]),
                "dflash": dflash,
                "greedy": greedy,
                "dflash_matches_greedy_prefix": match,
            }
        )
        mx.clear_cache()

    body = {
        "runtime": "mlx+dflash",
        "mlx": "0.32.0",
        "target": TARGET,
        "draft": DRAFT,
        "block_size": args.block,
        "max_tokens": args.max_tokens,
        "host": "Apple M5 Pro",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": (
            "DFlash exact verify ⇒ should match greedy on same MLX target. "
            "gemma-metal cross-stack stream may diverge (different Q4/kernels)."
        ),
        "runs": runs,
    }
    out_path.write_text(json.dumps(body, indent=2) + "\n")
    print(f"# wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
