"""Oracle for gemma-metal's assistant drafter (`MtpSession::assistant_step`).

Reruns every step dumped by `tests/real_e4b_mtp_oracle.rs` through transformers'
`Gemma4AssistantForCausalLM` in float32, with the same inputs (target embedding,
target hidden, position, target shared sliding / global K/V), and compares the
drafted token and the projected hidden state.

    python -I mtp_assistant_oracle.py ASSISTANT_DIR DUMP_DIR
"""

import json
import sys

import numpy as np
import torch
from transformers import Gemma4AssistantForCausalLM


def load(dump, case, name, shape=None):
    a = np.fromfile(f"{dump}/c{case}_{name}.f32", dtype="<f4")
    return torch.from_numpy(a if shape is None else a.reshape(shape))


def main():
    asst_dir, dump = sys.argv[1], sys.argv[2]
    model = Gemma4AssistantForCausalLM.from_pretrained(
        asst_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    meta = json.load(open(f"{dump}/meta.json"))
    worst_rel = 0.0
    token_ok = 0
    for row in meta:
        c, kc = row["case"], row["kv_case"]
        st, sh, sd = row["sliding"]
        gt, gh, gd = row["global"]
        # Rust stores [T, heads, dim]; HF wants [batch, heads, T, dim].
        kv = {
            "sliding_attention": (
                load(dump, kc, "sk", (st, sh, sd)).permute(1, 0, 2)[None],
                load(dump, kc, "sv", (st, sh, sd)).permute(1, 0, 2)[None],
            ),
            "full_attention": (
                load(dump, kc, "gk", (gt, gh, gd)).permute(1, 0, 2)[None],
                load(dump, kc, "gv", (gt, gh, gd)).permute(1, 0, 2)[None],
            ),
        }
        embed = load(dump, c, "embed")
        hidden = load(dump, c, "hidden")
        inputs = torch.cat([embed, hidden])[None, None]
        with torch.no_grad():
            out = model(
                inputs_embeds=inputs,
                position_ids=torch.tensor([[row["pos"]]]),
                shared_kv_states=kv,
                attention_mask=None,
            )
        logits = out.logits[0, -1]
        tok = int(logits.argmax())
        top = torch.topk(logits, 3)
        proj = out.last_hidden_state[0, -1]
        rust = load(dump, c, "projected")
        rel = float((proj - rust).norm() / proj.norm())
        worst_rel = max(worst_rel, rel)
        token_ok += tok == row["drafted"]
        print(
            f"case {c} step {row['step']} pos {row['pos']} input {row['input_token']}: "
            f"hf {tok} rust {row['drafted']} {'OK' if tok == row['drafted'] else 'DIFF'} "
            f"top3 {top.indices.tolist()} margin {float(top.values[0] - top.values[1]):.4f} "
            f"projected rel L2 {rel:.2e}"
        )
    print(f"tokens {token_ok}/{len(meta)} match, worst projected rel L2 {worst_rel:.2e}")


if __name__ == "__main__":
    main()
