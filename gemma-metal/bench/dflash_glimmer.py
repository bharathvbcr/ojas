"""DFlash block speculative decoding for Muse Glimmer 30B on MLX.

Mirrors the protocol in `dflash.model_mlx.stream_generate` (anchor + mask block, exact
argmax verify, longest-accepted-prefix) but drives mlx-vlm's Muse Glimmer target, whose
call surface differs from mlx-lm's: logits arrive as `LanguageModelOutput.logits` and the
decoder layers live under `model.language_model.model.layers`.

Exact verify means greedy DFlash output must be *token-identical* to plain greedy decode.
`--check` asserts that rather than trusting it.

    ~/.venvs/glimmer/bin/python dflash_glimmer.py --check
    ~/.venvs/glimmer/bin/python dflash_glimmer.py --block-size 5 --max-tokens 512
"""

from __future__ import annotations

import argparse
import time
from typing import Any, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache
from mlx_vlm import load

import muse_glimmer_dflash as mgd

TARGET = "mlx-community/Muse-Glimmer-30B-4bit"
DRAFT = mgd.DRAFT_REPO


class _LayerHook:
    """Records a decoder layer's output so the drafter can condition on it."""

    def __init__(self, layer, idx, storage):
        self._layer, self._idx, self._storage = layer, idx, storage

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        self._storage[self._idx] = out[0] if isinstance(out, tuple) else out
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _patch_target(model, layer_ids):
    text_model = model.language_model.model
    if hasattr(text_model, "_hidden_states"):
        return text_model
    text_model._hidden_states = [None] * len(layer_ids)
    for slot, lid in enumerate(layer_ids):
        text_model.layers[lid] = _LayerHook(
            text_model.layers[lid], slot, text_model._hidden_states
        )
    return text_model


def _forward(model, text_model, ids, cache):
    """Target forward returning (last-position-aligned logits, concatenated hidden states)."""
    logits = model.language_model(ids, cache=cache).logits
    hidden = mx.concatenate(text_model._hidden_states, axis=-1)
    return logits, hidden


def greedy_generate(model, tokenizer, prompt_ids, max_tokens):
    cache = model.language_model.make_cache()
    logits = model.language_model(prompt_ids, cache=cache).logits
    token = mx.argmax(logits[:, -1, :], axis=-1).item()
    out = [token]
    tic = time.perf_counter()
    for _ in range(max_tokens - 1):
        if token in tokenizer.eos_token_ids:
            break
        logits = model.language_model(mx.array([[token]]), cache=cache).logits
        token = mx.argmax(logits[:, -1, :], axis=-1).item()
        out.append(token)
    mx.eval(mx.array(out))
    return out, len(out) / (time.perf_counter() - tic)


def dflash_generate(model, draft, tokenizer, prompt_ids, max_tokens, block_size):
    text_model = _patch_target(model, draft.config.target_layer_ids)
    draft.bind(model)
    mask_id = draft.config.mask_token_id

    target_cache = model.language_model.make_cache()
    draft_cache = draft.make_cache()
    if not can_trim_prompt_cache(target_cache):
        raise RuntimeError("target cache does not support trimming; rollback impossible")

    logits, hidden = _forward(model, text_model, prompt_ids, target_cache)
    token = mx.argmax(logits[:, -1, :], axis=-1).item()
    out = [token]
    prompt_size = int(prompt_ids.shape[1])
    n = 1
    blocks = 0
    accepted_total = 0

    tic = time.perf_counter()
    while n < max_tokens:
        bs = min(block_size, max_tokens - n + 1)
        if bs <= 1:
            break

        block = mx.array([[out[-1]] + [mask_id] * (bs - 1)])
        draft_logits = draft(block, hidden, draft_cache, logits_start=1)
        overshoot = draft_cache[0].offset - (prompt_size + n - 1)
        if overshoot > 0:
            trim_prompt_cache(draft_cache, overshoot)
        draft_tokens = mx.argmax(draft_logits, axis=-1)
        mx.async_eval(draft_tokens)

        verify_input = mx.concatenate([mx.array([[out[-1]]]), draft_tokens], axis=1)
        logits, hidden = _forward(model, text_model, verify_input, target_cache)
        target_tokens = mx.argmax(logits, axis=-1)
        mx.async_eval(target_tokens, hidden)

        d_list = draft_tokens[0].tolist()
        t_list = target_tokens[0].tolist()
        accepted = next(
            (i for i in range(len(d_list)) if d_list[i] != t_list[i]), len(d_list)
        )
        new_tokens = (d_list[:accepted] + [t_list[accepted]])[: max_tokens - n]

        blocks += 1
        accepted_total += accepted
        out.extend(new_tokens)
        n += len(new_tokens)

        eos_idx = next(
            (i for i, t in enumerate(new_tokens) if t in tokenizer.eos_token_ids), None
        )
        if eos_idx is not None:
            del out[len(out) - len(new_tokens) + eos_idx + 1 :]
            break

        trim = bs - accepted - 1
        if trim > 0:
            trim_prompt_cache(target_cache, trim)
        hidden = hidden[:, : accepted + 1, :]

    elapsed = time.perf_counter() - tic
    mean_accept = (accepted_total / blocks + 1) if blocks else 0.0
    return out, len(out) / elapsed, mean_accept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=None, help="default: drafter config")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt", default="Write a Python function that merges two sorted lists.")
    ap.add_argument("--check", action="store_true", help="assert DFlash == greedy token-for-token")
    args = ap.parse_args()

    model, processor = load(TARGET)
    tokenizer = processor.tokenizer
    draft = mgd.load_draft(DRAFT, target_config={"num_hidden_layers": len(model.language_model.layers)})
    block_size = args.block_size or draft.config.block_size

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = mx.array([tokenizer.encode(text)])
    print(f"target={TARGET}\ndraft={DRAFT}\nblock_size={block_size} prompt_tokens={prompt_ids.shape[1]}\n")

    spec_tokens, spec_tps, accept = dflash_generate(
        model, draft, tokenizer, prompt_ids, args.max_tokens, block_size
    )
    print(f"DFlash  {spec_tps:6.1f} tok/s   mean_accept={accept:.2f}   tokens={len(spec_tokens)}")

    base_tokens, base_tps = greedy_generate(model, tokenizer, prompt_ids, args.max_tokens)
    print(f"greedy  {base_tps:6.1f} tok/s                      tokens={len(base_tokens)}")
    print(f"speedup {spec_tps / base_tps:6.2f}x   peak={mx.get_peak_memory() / 1e9:.1f} GB")

    if args.check:
        limit = min(len(spec_tokens), len(base_tokens))
        diff = next(
            (i for i in range(limit) if spec_tokens[i] != base_tokens[i]), None
        )
        if diff is None and len(spec_tokens) == len(base_tokens):
            print(f"\nEXACTNESS: PASS — {limit} tokens identical to greedy")
        else:
            print(f"\nEXACTNESS: FAIL — first divergence at index {diff}")
            print("  greedy:", tokenizer.decode(base_tokens[:limit])[:300])
            print("  dflash:", tokenizer.decode(spec_tokens[:limit])[:300])
            raise SystemExit(1)

    print("\n--- sample ---")
    print(tokenizer.decode(spec_tokens)[:600])


if __name__ == "__main__":
    main()
