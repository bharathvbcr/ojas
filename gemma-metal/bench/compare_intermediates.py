#!/usr/bin/env python3
"""Localize the native DFlash accept≈0 bug: diff MLX golden vs native intermediates,
report the FIRST divergent tensor (that's the bug's layer). GPU-free.

  python compare_intermediates.py golden_intermediates_31b.json native_intermediates_31b.json

MLX golden is produced by bench/golden_intermediates.py. The native side must emit a JSON
with the SAME keys/shapes (see SCHEMA below) from src/dflash.rs on input [2,105,4368,1246]:
  - target_hidden_per_layer[6]: {sum, absmean, first8, shape}  (post-layer_scalar residual, last pos)
  - fc_out_lastrow, h_ctx_lastrow, draft_logits_lastpos: same cksum dict
  - draft_attn_offsets_per_layer[5]: {S, Lq, q_rope_offset, ctx_rope_offset, is_sliding, sliding_window}
  - proposed_block_tokens: [int]  (bs-1)
  - target_next_argmax: int

Comparison order follows docs/dflash_draft_contract.md so the first mismatch names the suspect:
  target_hidden (capture) -> fc_out (fc) -> h_ctx (hidden_norm) -> attn offsets (★2/★3)
  -> draft_logits -> proposed tokens.
"""
import json, sys, math

CKSUM_ORDER = [  # (label, accessor, contract-ref)
    ("target_hidden_per_layer", "★1 capture tap / feed"),
    ("fc_out_lastrow", "fc projection"),
    ("h_ctx_lastrow", "hidden_norm"),
    ("draft_logits_lastpos", "draft forward output"),
]

def cks_close(a, b, rtol=2e-2, atol=1e-2):
    """Compare two cksum dicts. Uses absmean + sum + first8 with tolerance."""
    for k in ("absmean", "sum"):
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            return False, f"missing {k}"
        if abs(va - vb) > atol + rtol * max(abs(va), abs(vb)):
            return False, f"{k}: golden {va} vs native {vb}"
    if a.get("shape") != b.get("shape"):
        return False, f"shape {a.get('shape')} vs {b.get('shape')}"
    fa, fb = a.get("first8", []), b.get("first8", [])
    for i, (x, y) in enumerate(zip(fa, fb)):
        if abs(x - y) > atol + rtol * max(abs(x), abs(y)):
            return False, f"first8[{i}]: {x} vs {y}"
    return True, "ok"

def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    g = json.load(open(sys.argv[1])); n = json.load(open(sys.argv[2]))
    print("=== DFlash intermediate diff (golden vs native) ===")
    first_bad = None

    # 1) per-layer captured hidden
    gh, nh = g["target_hidden_per_layer"], n.get("target_hidden_per_layer", [])
    for i in range(len(gh)):
        ok, why = (cks_close(gh[i], nh[i]) if i < len(nh) else (False, "native missing layer"))
        lid = g["target_layer_ids"][i]
        print(f"  target_hidden[layer {lid}]: {'OK' if ok else 'DIVERGE — ' + why}")
        if not ok and first_bad is None:
            first_bad = (f"target_hidden[layer {lid}]", "★1 capture tap/feed", why)

    # 2) scalar cksum fields in contract order
    for key, ref in CKSUM_ORDER[1:]:
        if key in g and key in n:
            ok, why = cks_close(g[key], n[key])
            print(f"  {key}: {'OK' if ok else 'DIVERGE — ' + why}  ({ref})")
            if not ok and first_bad is None:
                first_bad = (key, ref, why)
        else:
            print(f"  {key}: native MISSING (emit it)")

    # 3) RoPE offsets per draft layer (★2/★3)
    go, no = g.get("draft_attn_offsets_per_layer", []), n.get("draft_attn_offsets_per_layer", [])
    for i in range(len(go)):
        gi = go[i]; ni = no[i] if i < len(no) else {}
        keys = ("q_rope_offset", "ctx_rope_offset", "S")
        bad = [k for k in keys if gi.get(k) != ni.get(k)]
        tag = "OK" if not bad else "DIVERGE — " + ", ".join(f"{k}: {gi.get(k)} vs {ni.get(k)}" for k in bad)
        print(f"  draft L{i} offsets (sliding={gi.get('is_sliding')}): {tag}")
        if bad and first_bad is None:
            first_bad = (f"draft L{i} RoPE offset", "★2/★3 offset", tag)

    # 4) proposed tokens
    gp, np_ = g.get("proposed_block_tokens"), n.get("proposed_block_tokens")
    print(f"  proposed_block_tokens: golden {gp} vs native {np_} — {'MATCH' if gp == np_ else 'DIFFER'}")

    print("\n=== VERDICT ===")
    if first_bad:
        print(f"FIRST DIVERGENCE: {first_bad[0]}  ->  suspect: {first_bad[1]}")
        print(f"  detail: {first_bad[2]}")
        print("  Fix this layer; everything downstream diverges from it.")
    elif gp != np_:
        print("Intermediates all match but proposed tokens differ -> sampling/argmax or logits-softcap in draft head.")
    else:
        print("Full match — draft is correct; if accept still ≈0 the bug is in the VERIFY path "
              "(target argmax offset / seed), not the draft. See contract closing note.")

if __name__ == "__main__":
    main()
