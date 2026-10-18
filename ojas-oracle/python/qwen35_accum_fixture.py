"""Torch reference for ojas's multi-row gradient accumulation (card ft-b5e64557, criteria 15/16):
ojas-model/tests/fixtures/qwen35_tiny_accum.

    /opt/homebrew/opt/python@3.14/bin/python3.14 -P ojas-oracle/python/qwen35_accum_fixture.py

Lappi's loop runs one tape per row and sums the adapter gradients of the rows
of a step; the CE is weighted `scale = 1/N_total_rows` (a sum, not a mean), and
a head or D17 on the host hands the tower a fixed upstream gradient `dh` at
some hidden positions. This fixture is that step in torch, on the tiny Qwen3.5
(`qwen35_tiny`) with the LoRA fixture's targets, rank 4 and alpha 8, A and B
drawn exactly as `qwen35_lora_fixture.py` draws them (the same seeds, so they
equal that fixture's `init.*` and `bnz.*`):

- three rows of widths 5, 9 and 13 tokens, cut from the fixture's ids (flattened)
  at offsets 0, 3 and 7;
- per row, CE targets at `positions` (random vocabulary ids in `targets`) and
  `dh` at `hidden_at` (hidden positions, drawn independently of the CE ones);
- the backward objective of a row = `scale * sum_CE(logits[positions], targets)
  + sum(h[hidden_at] * dh)`, `h` the final-norm hidden state and
  `logits = lm_head(h)` (tied head); `loss.<i>` is the UNSCALED `sum_CE`, the
  value `TapeStepProvider::accumulate` returns;
- `backward()` once per row, gradients left to sum in `.grad` across rows.

fp32, CPU, one thread, deterministic algorithms. Written next to this
into `--out` as `accum.safetensors` and `manifest.json`, in the layout
`ojas-model/tests/qwen35_step_provider.rs` reads
(`three_row_accumulate_matches_the_torch_fixture`):

- `ids.<i>`, `positions.<i>`, `targets.<i>`, `hidden_at.<i>`: I64 (ids `[T_i]`,
  the others `[n_i]` / `[m_i]`); `dh.<i>`: F32 `[m_i, hidden]`;
- `scale`: F32 `[1]`, `1/rows`; `loss.<i>`: F32 `[1]`, row i's `sum_CE`;
- `grad.<hf module>.lora_A.weight` `[r, in]` and `grad.<hf module>.lora_B.weight`
  `[out, r]`: dA and dB summed over the rows. The torch `lora_B` is the one
  fused Hugging Face `[out, r]` weight (`in_proj_qkv` as `[q|k|v]`, `q_proj` with
  query and gate rows interleaved per head), which is the order
  `LoraTarget::fuse_b` (ojas-model/src/qwen35/params.rs) rebuilds from the
  model's parts, so no regrouping is needed;
- the adapters are NOT written: the test reads `init.*` and `bnz.*` from
  `qwen35_tiny_lora`, which this script reproduces from the same seeds;
- the manifest holds the torch version, the arguments, the widths and the
  sha256 of `accum.safetensors`.

A position outside its row is refused before any model is built.
"""
import argparse
import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "..", "..", "ojas-model", "tests", "fixtures", "qwen35_tiny")
OUT = os.path.join(HERE, "..", "..", "ojas-model", "tests", "fixtures", "qwen35_tiny_accum")
RANK, ALPHA = 4, 8
SEED = 20261013  # the LoRA fixture's: A from SEED, B from SEED + 1
DATA_SEED = SEED + 7
WIDTHS = (5, 9, 13)
OFFSETS = (0, 3, 7)
CE_PER_ROW = 3
DH_PER_ROW = 2
DH_STD = 0.1

TARGET_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


class LoRALinear(nn.Module):
    """lora.py's LoRALinear, as qwen35_lora_fixture.py has it."""

    def __init__(self, base, *, r, alpha):
        super().__init__()
        self.base = base
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        device = base.weight.device
        self.lora_A = nn.utils.skip_init(nn.Linear, base.in_features, r, bias=False, device=device, dtype=torch.float32)
        self.lora_B = nn.utils.skip_init(nn.Linear, r, base.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        result = self.base(x)
        update = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scaling
        return result + update.to(result.dtype)


def is_target(name):
    return any(name == s or name.endswith("." + s) for s in TARGET_SUFFIXES)


def attach_lora(model, *, seed, r, alpha):
    targets = [(n, m) for n, m in model.named_modules() if is_target(n)]
    assert targets and all(isinstance(m, nn.Linear) for _, m in targets)
    for p in model.parameters():
        p.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = []
    for name, module in targets:
        wrapped = LoRALinear(module, r=r, alpha=alpha)
        a = torch.empty(r, module.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        with torch.no_grad():
            wrapped.lora_A.weight.copy_(a)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, wrapped)
        names.append(name)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.weight.requires_grad_(True)
            m.lora_B.weight.requires_grad_(True)
    return names


def adapters(model):
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            yield f"{name}.lora_A.weight", m.lora_A.weight
            yield f"{name}.lora_B.weight", m.lora_B.weight


def build_model(base):
    cfg = mq.Qwen3_5TextConfig.from_dict(json.load(open(os.path.join(base, "config.json"))))
    cfg._attn_implementation = "eager"
    model = mq.Qwen3_5ForCausalLM(cfg).float()
    state = load_file(os.path.join(base, "model.safetensors"))
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    assert not unexpected and missing == ["lm_head.weight"], (missing, unexpected)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return model


def check_positions(label, positions, width):
    """Refuse a position outside `[0, width)` or a repeated one."""
    for p in positions:
        if not 0 <= p < width:
            raise SystemExit(f"{label}: position {p} is outside the row of width {width}")
    if len(set(positions)) != len(positions):
        raise SystemExit(f"{label}: repeated position in {positions}")


def draw_rows(flat_ids, vocab, hidden):
    gen = torch.Generator().manual_seed(DATA_SEED)
    rows = []
    for width, offset in zip(WIDTHS, OFFSETS):
        if offset + width > flat_ids.numel():
            raise SystemExit(f"row of width {width} at offset {offset} exceeds the {flat_ids.numel()} fixture ids")
        ce_pos = sorted(torch.randperm(width, generator=gen)[:CE_PER_ROW].tolist())
        dh_pos = sorted(torch.randperm(width, generator=gen)[:DH_PER_ROW].tolist())
        check_positions("ce_pos", ce_pos, width)
        check_positions("dh_pos", dh_pos, width)
        rows.append({
            "ids": flat_ids[offset:offset + width].clone().long(),
            "positions": torch.tensor(ce_pos, dtype=torch.int64),
            "targets": torch.randint(0, vocab, (CE_PER_ROW,), generator=gen, dtype=torch.int64),
            "hidden_at": torch.tensor(dh_pos, dtype=torch.int64),
            "dh": (DH_STD * torch.randn(DH_PER_ROW, hidden, generator=gen)).contiguous(),
        })
    return rows


def row_loss(model, row, scale):
    h = model.model(input_ids=row["ids"].unsqueeze(0)).last_hidden_state[0]
    logits = model.lm_head(h)
    ce = F.cross_entropy(logits[row["positions"]], row["targets"], reduction="sum")
    return scale * ce + (h[row["hidden_at"]] * row["dh"]).sum(), ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE, help="base checkpoint dir (default: qwen35_tiny)")
    ap.add_argument("--out", default=OUT, help="output dir")
    ap.add_argument("--alpha", type=int, default=ALPHA, help="LoRA alpha at rank 4")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    cfg = json.load(open(os.path.join(args.base, "config.json")))
    text = cfg.get("text_config", cfg)
    vocab, hidden = text["vocab_size"], text["hidden_size"]
    ids = load_file(os.path.join(args.base, "reference.safetensors"))["ids"]
    rows = draw_rows(ids.flatten(), vocab, hidden)
    scale = 1.0 / len(rows)

    model = build_model(args.base)
    names = attach_lora(model, seed=SEED, r=RANK, alpha=args.alpha)
    out = {}
    gen = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for n, p in adapters(model):
            if n.endswith("lora_B.weight"):
                p.copy_(0.05 * torch.randn(p.shape, generator=gen))

    model.zero_grad(set_to_none=True)
    losses = []
    for i, row in enumerate(rows):
        objective, ce = row_loss(model, row, scale)
        objective.backward()
        losses.append(ce.item())
        for k, v in row.items():
            out[f"{k}.{i}"] = v
        out[f"loss.{i}"] = torch.tensor([losses[-1]], dtype=torch.float32)
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
    for n, p in adapters(model):
        assert p.grad is not None, n
        out[f"grad.{n}"] = p.grad.detach().clone().contiguous()
    out["scale"] = torch.tensor([scale], dtype=torch.float32)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "accum.safetensors")
    save_file(out, path, metadata={"torch": torch.__version__, "rank": str(RANK), "alpha": str(args.alpha)})
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "torch": torch.__version__,
        "args": {"base": os.path.basename(os.path.normpath(args.base)), "alpha": args.alpha, "rank": RANK},
        "seed": SEED, "data_seed": DATA_SEED, "widths": list(WIDTHS), "offsets": list(OFFSETS),
        "ce_per_row": CE_PER_ROW, "dh_per_row": DH_PER_ROW, "dh_std": DH_STD, "scale": scale,
        "targets": names, "losses": losses, "accum.safetensors.sha256": digest,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"losses {[f'{x:.6f}' for x in losses]}, {len(out)} tensors, sha256 {digest[:12]} -> {args.out}")


if __name__ == "__main__":
    main()
