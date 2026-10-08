#!/usr/bin/env python3
"""Ground-truth DFlash intermediates for the native accept-bug localization.
Fixed input; single target forward + one draft block. Dumps checksums + samples of
target_hidden (the 6 capture layers), h_ctx, and the draft's proposed block tokens.
Native `src/dflash.rs` should reproduce these (see docs/dflash_draft_contract.md ★1)."""
import json
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load as mlx_load
from dflash.model_mlx import load_draft, _patch_model
import dflash.model_mlx as dfm

# --- Monkeypatch DFlashAttention to capture per-layer post-RoPE q/ctx_keys +
#     the RoPE offsets it uses (localizes contract ★2/★3 when native diffs). ---
_ATTN_TRACE = []
_orig_attn = dfm.DFlashAttention.__call__
def _traced_attn(self, x, x_ctx, rope, cache):
    B, Lq, _ = x.shape
    S = x_ctx.shape[1]
    q_off = cache.offset + S
    ctx_off = cache.offset
    out = _orig_attn(self, x, x_ctx, rope, cache)
    _ATTN_TRACE.append({"S": int(S), "Lq": int(Lq), "q_rope_offset": int(q_off),
                        "ctx_rope_offset": int(ctx_off), "is_sliding": bool(self.is_sliding),
                        "sliding_window": (int(self.sliding_window) if self.sliding_window else None)})
    return out
dfm.DFlashAttention.__call__ = _traced_attn

TARGET="mlx-community/gemma-4-31b-it-4bit"; DRAFT="z-lab/gemma-4-31B-it-DFlash"
INPUT=[2,105,4368,1246]   # matches native bench prompt
BS=5

def cksum(a):
    a=a.astype(mx.float32).flatten()
    return {"sum":round(float(a.sum()),4),"absmean":round(float(mx.abs(a).mean()),6),
            "first8":[round(float(x),5) for x in a[:8].tolist()],"shape":list(a.shape)}

model, tok = mlx_load(TARGET)
draft = load_draft(DRAFT); draft.bind(model)
_patch_model(model, draft.config.target_layer_ids)   # installs _LayerHook capture

x = mx.array([INPUT])
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)
logits = model(x, cache)
mx.eval(logits)
hid = model._hidden_states                 # list of 6 tensors [1,T,H] (layer OUTPUTS)
target_hidden = mx.concatenate(hid, axis=-1)   # [1,T,6H]

# h_ctx = hidden_norm(fc(concat))  -- the conditioning the draft consumes
h_ctx = draft.hidden_norm(draft.fc(target_hidden))
mx.eval(h_ctx)

# draft proposes block-1 tokens from [last_input_tok, mask*(BS-1)] conditioned on h_ctx
mask_id = int(draft.config.mask_token_id)
block = mx.array([[INPUT[-1]] + [mask_id]*(BS-1)])
draft_cache = make_prompt_cache(draft)
dl = draft(block, target_hidden, draft_cache, logits_start=1)
proposed = mx.argmax(dl, axis=-1)[0].tolist()
mx.eval(dl)

out={
 "input_ids":INPUT, "target_layer_ids":list(draft.config.target_layer_ids),
 "block_size":BS, "mask_token_id":mask_id,
 "embed_scale":round(float(draft.embed_scale),5),
 "target_hidden_per_layer":[cksum(hid[i][:, -1, :]) for i in range(len(hid))],  # last-pos, each capture layer
 "h_ctx_lastrow":cksum(h_ctx[:, -1, :]),
 "draft_logits_lastpos":cksum(dl[:, -1, :]),
 "proposed_block_tokens":proposed,
 "target_next_argmax":int(mx.argmax(logits[:, -1, :]).item()),
 "fc_out_lastrow":cksum(draft.fc(target_hidden)[:, -1, :]),      # pre-hidden_norm (isolate fc vs norm)
 "draft_attn_offsets_per_layer":_ATTN_TRACE,                     # ★2/★3: per-layer q/ctx RoPE offsets + sliding
 "note":"Native src/dflash.rs must reproduce: target_hidden (layer OUTPUTS at target_layer_ids), fc_out, h_ctx, per-layer RoPE offsets (q=cache.offset+S, ctx=cache.offset), and proposed_block_tokens. First divergence = bug layer. See docs/dflash_draft_contract.md.",
}
print(json.dumps(out, indent=2))
open("golden_intermediates_31b.json","w").write(json.dumps(out, indent=2))
