"""Torch reference for ojas's Qwen3.5 tape fixture (ojas-model/tests/fixtures/qwen35_tiny).

    /opt/homebrew/opt/python@3.14/bin/python3.14 ojas-oracle/python/qwen35_wide_fixture.py

A random `Qwen3_5ForCausalLM` of the 2B's shape family, widened past what
tessl's single-sequence fixture covers so the Rust tape is checked against
transformers' autograd on every axis a fine-tune uses:

- batch 2 (two sequences through one tape);
- two GDN heads (key = value heads = 2) of 128, so the head axis of the
  delta rule, the conv channels and the fused projections are not trivial;
- vocab 8320, past the default fused-CE column tile of 8192, so the tied
  head's second (partial) tile and the gradient rows it writes are checked;
- `-100` labels: row 0 masks its first 20 positions (a prompt), row 1 masks
  its last 15 (right padding). The loss is transformers' own
  `cross_entropy(ignore_index=-100)`: the mean over the unmasked positions.

Row 1's padding is not attention-masked: the tower is causal, so a position
never sees a later token and the padding only changes which targets count.

Writes config.json, model.safetensors (bf16), and reference.safetensors:
`ids` and `labels` (int64 [B, T], labels as transformers takes them, shifted
inside the loss), `loss` (f32, the f64 loss rounded) and every parameter's
f32 gradient under its Hugging Face name. Requires torch 2.13 and
transformers 5.15 (`modeling_qwen3_5`); every random draw is seeded, and the
parameters are rounded to bf16 so the checkpoint the Rust side loads and the
float32 model torch differentiates hold the same values.
"""
import json
import os

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "ojas-model", "tests", "fixtures", "qwen35_tiny")
BATCH, SEQ, VOCAB, GDN_HEADS = 2, 70, 8320, 2
SEED = 20261012


def config():
    return mq.Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=GDN_HEADS,
        linear_num_value_heads=GDN_HEADS,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=VOCAB,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )


def reinit(model, gen):
    """Every parameter at a scale where it matters, then rounded to bf16."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.25)
            elif name.endswith("conv1d.weight"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("A_log"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.5 - 1.0)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=gen))
            elif p.dim() == 1:
                p.copy_(0.3 * torch.randn(p.shape, generator=gen))  # zero-centred norms, (1 + w)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) / p.shape[1] ** 0.5)
            p.copy_(p.to(torch.bfloat16).float())


def main():
    os.makedirs(OUT, exist_ok=True)
    gen = torch.Generator().manual_seed(SEED)
    cfg = config()
    torch.manual_seed(0)
    model = mq.Qwen3_5ForCausalLM(cfg).float()
    reinit(model, gen)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=gen)
    # Targets in the second column tile (ids 8192..VOCAB) in both rows.
    for b in range(BATCH):
        pos = torch.randperm(SEQ - 1, generator=gen)[:6] + 1
        ids[b, pos] = torch.randint(8192, VOCAB, (6,), generator=gen)
    labels = ids.clone()
    labels[0, :20] = -100
    labels[1, -15:] = -100
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, labels=labels)
    out.loss.backward()
    loss = out.loss.item()
    counted = int((labels[:, 1:] != -100).sum())
    print(f"loss {loss:.6f} over {counted} of {BATCH * (SEQ - 1)} positions")
    state = {n: p.detach().to(torch.bfloat16).contiguous() for n, p in model.named_parameters()}
    save_file(state, os.path.join(OUT, "model.safetensors"))
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(json.loads(cfg.to_json_string()), f, indent=1, sort_keys=True)
    ref = {
        "ids": ids.to(torch.int64).contiguous(),
        "labels": labels.to(torch.int64).contiguous(),
        "loss": torch.tensor([loss], dtype=torch.float32),
    }
    for n, p in model.named_parameters():
        ref[n] = p.grad.detach().float().contiguous()
    save_file(ref, os.path.join(OUT, "reference.safetensors"))
    print(f"{len(ref) - 3} gradients -> {OUT}")


if __name__ == "__main__":
    main()
