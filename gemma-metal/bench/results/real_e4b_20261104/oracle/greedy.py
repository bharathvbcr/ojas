"""Throwaway oracle: mlx_lm greedy decode of the MLX 4-bit Gemma 4 E4B checkpoint on fixed ids.

usage: python -I greedy.py MODEL_DIR N id1 id2 ...
Prints the new ids, their text, and the top-5 logits of the first step.
"""
import sys

import mlx.core as mx
from mlx_lm import load

model_dir, n = sys.argv[1], int(sys.argv[2])
prompt = [int(x) for x in sys.argv[3:]]
model, tok = load(model_dir)
print("model class", type(model).__module__, type(model).__name__)
ids = list(prompt)
cache = None
from mlx_lm.models.cache import make_prompt_cache

cache = make_prompt_cache(model)
x = mx.array([ids])
new = []
for step in range(n):
    logits = model(x, cache=cache)[0, -1].astype(mx.float32)
    if step == 0:
        top = mx.argsort(-logits)[:5].tolist()
        print("step0 top5", [(t, round(logits[t].item(), 4)) for t in top])
        print("step0 logits min/max", logits.min().item(), logits.max().item())
    t = int(mx.argmax(logits).item())
    new.append(t)
    x = mx.array([[t]])
print("new ids", new)
print("new text", repr(tok.decode(new)))
