#!/usr/bin/env python3
"""Fastest measured local Gemma-4-31B inference on this M5 Pro.

MLX 0.32 + DFlash block speculative decoding + 4-bit draft.
Measured on M5 Pro (2026-07-13, interleaved + drift-checked):
    plain mlx-lm 31B decode      ~12.7 tok/s
    + DFlash spec-decode (bs=8)  ~17.6         (block verify amortizes overhead)
    + 4-bit draft                ~18.6  (+6%, zero quality impact - verify is exact)
    + mlx 0.32.0 (M5 NAX GEMM)   ~27.8  (1.49x: M=8 verify matmuls 1.5-2x faster)
Total: ~2.2x over plain; clears the >=15 gate and the >=25 "+MTP" stretch gate.
IMPORTANT: mlx must be 0.32+ — dflash's own [mlx] extra pins 0.31.2, install base
dflash then mlx separately (see below).

Spec-decode is EXACT: the target verifies every token, so output is identical to
greedy 31B regardless of the draft. Draft quant/config only affects speed.

Setup (one time — venv already exists at ~/.venvs/dflash32):
    python3 -m venv ~/.venvs/dflash32
    ~/.venvs/dflash32/bin/pip install mlx==0.32.0 mlx-lm==0.31.3
    ~/.venvs/dflash32/bin/pip install "dflash @ git+https://github.com/z-lab/dflash"
    hf download mlx-community/gemma-4-31b-it-4bit
    hf download z-lab/gemma-4-31B-it-DFlash

Run:
    ~/.venvs/dflash32/bin/python dflash_fast_31b.py "Your prompt here"
"""
import sys, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from dflash.model_mlx import load_draft, stream_generate

TARGET = "mlx-community/gemma-4-31b-it-4bit"
DRAFT = "z-lab/gemma-4-31B-it-DFlash"
BLOCK_SIZE = 5      # tuned for M5 Pro + mlx 0.32: fine-sweep {3..8,12..32} → 5 wins
                    # (37.2 vs 27.8 @ 8 vs 17.5 @ 12; NAX small-M verify favors short blocks)
QUANT_DRAFT = True  # 4-bit draft: +6% decode, 2.2 GB less, zero quality impact


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain speculative decoding in 3 sentences."
    t0 = time.perf_counter()
    model, tok = load(TARGET)
    draft = load_draft(DRAFT)
    if QUANT_DRAFT:
        nn.quantize(draft, group_size=64, bits=4,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
    draft.bind(model)
    print(f"# loaded in {time.perf_counter()-t0:.1f}s "
          f"(draft {'4-bit' if QUANT_DRAFT else 'bf16'}, block_size={BLOCK_SIZE})\n", flush=True)

    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    last = None
    for r in stream_generate(model, draft, tok, text, block_size=BLOCK_SIZE, max_tokens=512, temperature=0.0):
        print(r.text, end="", flush=True)
        last = r
    if last:
        print(f"\n\n# {last.generation_tps:.1f} tok/s decode · {last.generation_tokens} tokens · "
              f"peak {last.peak_memory:.1f} GB")


if __name__ == "__main__":
    main()
