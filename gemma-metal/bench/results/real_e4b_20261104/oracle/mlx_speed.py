"""Throwaway baseline: mlx_lm greedy decode tok/s on the real E4B 4-bit checkpoint.
Prompt = the same 6 ids; N new tokens; R repeats in one process after one warm-up.
Reports per-run decode tok/s (generation phase only, from mlx_lm's own response stats).

usage: python -I mlx_speed.py MODEL_DIR N R
"""
import sys

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

model_dir, n, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model, tok = load(model_dir)
tok.eos_token_ids = []  # time a fixed N tokens: never stop early
prompt = [2, 818, 5279, 529, 7001, 563]
sampler = make_sampler(temp=0.0)


def run():
    last = None
    for r in stream_generate(model, tok, prompt, max_tokens=n, sampler=sampler):
        last = r
    return last


run()  # warm-up
for i in range(reps):
    r = run()
    print(f"run {i}: gen {r.generation_tokens} tok at {r.generation_tps:.2f} tok/s; prompt {r.prompt_tps:.1f} tok/s; peak {r.peak_memory:.2f} GB")
