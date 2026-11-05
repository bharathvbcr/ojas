"""Throwaway oracle: every Gemma 4 decoder layer's output (post layer_scalar) at every prompt
position, from mlx_lm on the MLX 4-bit checkpoint. Same JSON shape as gemma-metal's
`diag_tok --dump-hidden`: {"prompt", "argmax", "hidden": [pos][layer][H]}.

usage: python -I dump_hidden.py MODEL_DIR OUT.json id1 id2 ...
"""
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import gemma4_text

model_dir, out = sys.argv[1], sys.argv[2]
prompt = [int(x) for x in sys.argv[3:]]
model, _ = load(model_dir)

captured = []
orig = gemma4_text.DecoderLayer.__call__


def wrapped(self, *a, **kw):
    r = orig(self, *a, **kw)
    captured.append(r[0][0].astype(mx.float32))  # [L, H]
    return r


gemma4_text.DecoderLayer.__call__ = wrapped
logits = model(mx.array([prompt]))[0, -1].astype(mx.float32)
mx.eval(logits, *captured)
n_layers = len(captured)
hidden = [[captured[l][p].tolist() for l in range(n_layers)] for p in range(len(prompt))]
json.dump({"prompt": prompt, "argmax": int(mx.argmax(logits).item()), "hidden": hidden}, open(out, "w"))
print(f"dumped {len(prompt)} positions x {n_layers} layers to {out}; argmax {int(mx.argmax(logits).item())}")
