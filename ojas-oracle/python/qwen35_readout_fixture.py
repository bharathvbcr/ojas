"""Torch reference for ojas's Qwen3.5 readout seam (ojas-model/tests/fixtures/qwen35_tiny_readout).

    /opt/homebrew/opt/python@3.14/bin/python3.14 ojas-oracle/python/qwen35_readout_fixture.py

The base model is the widened tiny Qwen3.5 (`qwen35_tiny`, written by
`qwen35_wide_fixture.py`), loaded from its bf16 checkpoint into float32 by the
LoRA fixture's `build_model` (imported, not copied), so ojas and torch hold
the same base values. Two groups of references, every one on CPU, fp32, one
thread, deterministic algorithms:

- `gather.*`  `index_select` of rows of a random `[B, T, D]` at one position
              per example, and the gradient it scatters for a random upstream
              `[B, D]`: what `gather_rows` forward and backward must equal.
- `row.*`     the answer head Lappi trains host-side, per example run alone at
              its own length (no padding at all, the definition a padded batch
              must reproduce): the final-norm hidden state `h` at the answer
              position, then `W17 = embed[answer_ids] + D17` (`D17` non-zero),
              the 17-way logits `h @ W17^T`, their mean cross-entropy over the
              batch, and the gradients of that loss with respect to `D17` and
              to `h` (`dh`, the rows of dH at the answer positions). The tied
              embedding is held fixed, as it is frozen on ojas's tape.
              `row.full_loss` is the full-vocabulary cross-entropy at the same
              positions (`h @ embed^T`, the 17 answer ids' tokens as targets),
              a report-only value with no gradient.

`ids` is `[B, T]` right-padded with `PAD`; `lengths` the true lengths. The
padded columns hold arbitrary tokens: the tower is causal, so they cannot
reach a row's own positions, and the references do not see them at all.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
import qwen35_lora_fixture as lora  # noqa: E402  (build_model: the bf16 checkpoint as f32)

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "..", "ojas-model", "tests", "fixtures", "qwen35_tiny_readout")
SEED = 20261016
VOCAB, HIDDEN = 8320, 64
LENGTHS = [24, 17, 9, 1]  # the last row is one token long, answered at position 0
POSITIONS = [23, 10, 4, 0]  # a row's last position, two interior ones, and 0
SEQ = max(LENGTHS)
TARGETS = [3, 16, 0, 9]
N_ANSWERS = 17
D17_SCALE = 0.05


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.makedirs(OUT, exist_ok=True)
    gen = torch.Generator().manual_seed(SEED)
    model = lora.build_model()
    for p in model.parameters():
        p.requires_grad_(False)
    embed = model.model.embed_tokens.weight.detach()
    batch = len(LENGTHS)

    ids = torch.randint(0, VOCAB, (batch, SEQ), generator=gen)
    for b, n in enumerate(LENGTHS):
        ids[b, n:] = torch.randint(0, VOCAB, (SEQ - n,), generator=gen)  # padding, never read
    answer_ids = torch.randperm(VOCAB, generator=gen)[:N_ANSWERS]
    d17 = D17_SCALE * torch.randn(N_ANSWERS, HIDDEN, generator=gen)
    out = {
        "ids": ids.to(torch.int64).contiguous(),
        "lengths": torch.tensor(LENGTHS, dtype=torch.int64),
        "positions": torch.tensor(POSITIONS, dtype=torch.int64),
        "targets": torch.tensor(TARGETS, dtype=torch.int64),
        "answer_ids": answer_ids.to(torch.int64).contiguous(),
        "d17": d17.contiguous(),
    }

    # row.*: each example alone, at its own length.
    rows, hiddens = [], []
    for b, n in enumerate(LENGTHS):
        h = model.model(input_ids=ids[b : b + 1, :n]).last_hidden_state[0]  # [n, D], final-normed
        hiddens.append(h)
        rows.append(h[POSITIONS[b]].detach().clone())
    h_rows = torch.stack(rows).requires_grad_(True)
    d17_leaf = d17.clone().requires_grad_(True)
    w17 = embed[answer_ids] + d17_leaf
    logits = h_rows @ w17.T
    targets = torch.tensor(TARGETS)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    out["row.h"] = h_rows.detach().contiguous()
    out["row.loss"] = torch.tensor([loss.item()], dtype=torch.float32)
    out["row.loss_each"] = F.cross_entropy(logits.detach(), targets, reduction="none").contiguous()
    out["row.d17_grad"] = d17_leaf.grad.contiguous()
    out["row.dh"] = h_rows.grad.contiguous()
    full = h_rows.detach() @ embed.T
    out["row.full_loss"] = torch.tensor(
        [F.cross_entropy(full, answer_ids[targets]).item()], dtype=torch.float32
    )
    # Zero D17: the head is the frozen embedding rows alone.
    logits0 = h_rows.detach() @ embed[answer_ids].T
    out["row.loss_d0"] = torch.tensor([F.cross_entropy(logits0, targets).item()], dtype=torch.float32)

    # gather.*: index_select of rows of [B, T, D] and its scattered gradient.
    gh = torch.randn(batch, SEQ, HIDDEN, generator=gen).requires_grad_(True)
    flat = torch.tensor([b * SEQ + p for b, p in enumerate(POSITIONS)])
    up = torch.randn(batch, HIDDEN, generator=gen)
    picked = gh.reshape(-1, HIDDEN).index_select(0, flat)
    (picked * up).sum().backward()
    out["gather.h"] = gh.detach().contiguous()
    out["gather.up"] = up.contiguous()
    out["gather.out"] = picked.detach().contiguous()
    out["gather.dh"] = gh.grad.contiguous()

    save_file(out, os.path.join(OUT, "readout.safetensors"))
    print(f"row loss {loss.item():.7f} (D17 = 0: {out['row.loss_d0'].item():.7f}), "
          f"full-vocab {out['row.full_loss'].item():.7f}, {len(out)} tensors -> {OUT}")


if __name__ == "__main__":
    main()
