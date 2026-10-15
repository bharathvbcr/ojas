"""Torch reference for ojas's LoRA-on-the-Tape tests (ojas-model/tests/fixtures/qwen35_tiny_lora).

    /opt/homebrew/opt/python@3.14/bin/python3.14 ojas-oracle/python/qwen35_lora_fixture.py
    /opt/homebrew/opt/python@3.14/bin/python3.14 ojas-oracle/python/qwen35_lora_fixture.py \
        --alpha 4 --out ojas-model/tests/fixtures/qwen35_tiny_lora_s1

The oracle is the plain-torch LoRA arm of Lappi's `python/qd_train/lora.py`
(Fable D7), not peft and not mlx-lm. `LoRALinear` and `attach_lora` below
are that file's, reduced to what the fixture needs and otherwise verbatim:
`y = base(x) + (alpha/r) lora_B(lora_A(x))`, `A` `kaiming_uniform_(a=sqrt(5))`
from one seeded CPU generator visiting the targets in `named_modules()`
order, `B` zero, every base parameter frozen, adapters f32, targets matched
on the dotted module-name tail (attention q/k/v/o, GDN in_proj_qkv / in_proj_z
/ out_proj, MLP gate/up/down). It has no dropout path.

The base model is the widened tiny Qwen3.5 (`qwen35_tiny`, written by
`qwen35_wide_fixture.py`), loaded from its bf16 checkpoint into float32, so
ojas and torch hold the same base values. Rank 4 and alpha 8 (scaling 2, so
a dropped scaling cannot pass); `--alpha 4` writes the scaling-1 twin
(`qwen35_tiny_lora_s1`), the pre-registered run's alpha == r, where ojas
skips the scale node. The seed is shared, so both fixtures hold the same A,
B and data and differ only by the scale. Four cases, every one on CPU, fp32, one
thread, deterministic algorithms:

- `c.*`   B = 0 (the init): the loss equals the base model's.
- `a.*`   B random (`bnz.*`): the LoRA loss and every adapter gradient.
          Base parameters have no gradient (asserted `grad is None`).
- `b.*`   a 20-step AdamW trajectory from the init (A random, B = 0) on the
          same batch every step: the loss before each update, and the final
          adapters. AdamW lr 5e-3, betas (0.9, 0.999), eps 1e-8, weight decay
          0.01 on every adapter. `b.final64.*` is the same trajectory in
          float64 (same fp32-drawn A, same data): the exact answer the fp32
          arms are both approximating. Adam's step is scale-free, so
          a gradient error of 1e-6 of a tensor's peak becomes a parameter
          error of several 1e-5 after 20 steps; `b.final` vs `b.final64`
          is torch's own noise floor on this trajectory.
- `lin.*` one LoRA linear alone, x/W/A/B/gy given: y, merged W, and
          dX, dA, dB of `sum(y * gy)`.

Adapter tensors are named `<hf module>.lora_A.weight` / `.lora_B.weight`.
"""
import argparse
import json
import math
import os

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

HERE = os.path.dirname(__file__)
BASE = os.path.join(HERE, "..", "..", "ojas-model", "tests", "fixtures", "qwen35_tiny")
OUT = os.path.join(HERE, "..", "..", "ojas-model", "tests", "fixtures", "qwen35_tiny_lora")
RANK, ALPHA = 4, 8
SEED = 20261013
STEPS = 20
LR, WD = 5e-3, 0.01

TARGET_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


class LoRALinear(nn.Module):
    """lora.py's LoRALinear."""

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
    """lora.py's attach_lora; returns the wrapped module names."""
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


def build_model():
    cfg = mq.Qwen3_5TextConfig.from_dict(json.load(open(os.path.join(BASE, "config.json"))))
    cfg._attn_implementation = "eager"
    model = mq.Qwen3_5ForCausalLM(cfg).float()
    state = load_file(os.path.join(BASE, "model.safetensors"))
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    assert not unexpected and missing == ["lm_head.weight"], (missing, unexpected)
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    return model


def batch():
    ref = load_file(os.path.join(BASE, "reference.safetensors"))
    return ref["ids"], ref["labels"], ref["loss"].item()


def loss_of(model, ids, labels):
    return model(input_ids=ids, labels=labels).loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=int, default=ALPHA,
                    help="LoRA alpha at rank 4 (default 8: scaling 2; 4 is scaling 1, the run's)")
    ap.add_argument("--out", default=OUT, help="where to write lora.safetensors")
    args = ap.parse_args()
    alpha, out_dir = args.alpha, args.out
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.makedirs(out_dir, exist_ok=True)
    ids, labels, base_loss = batch()
    out = {}

    # c: B = 0, the init.
    model = build_model()
    base = loss_of(model, ids, labels).item()
    assert abs(base - base_loss) <= 1e-6 * abs(base_loss), (base, base_loss)
    names = attach_lora(model, seed=SEED, r=RANK, alpha=alpha)
    init = {n: p.detach().clone() for n, p in adapters(model)}
    c_loss = loss_of(model, ids, labels).item()
    out["c.loss"] = torch.tensor([c_loss], dtype=torch.float32)
    out["c.base_loss"] = torch.tensor([base], dtype=torch.float32)
    for n, v in init.items():
        if n.endswith("lora_A.weight"):
            out[f"init.{n}"] = v.contiguous()
    print(f"base loss {base:.7f}, B=0 loss {c_loss:.7f}, {len(names)} targets")

    # a: B random.
    gen = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for n, p in adapters(model):
            if n.endswith("lora_B.weight"):
                p.copy_(0.05 * torch.randn(p.shape, generator=gen))
    for n, p in adapters(model):
        if n.endswith("lora_B.weight"):
            out[f"bnz.{n}"] = p.detach().clone().contiguous()
    model.zero_grad(set_to_none=True)
    a_loss = loss_of(model, ids, labels)
    a_loss.backward()
    out["a.loss"] = torch.tensor([a_loss.item()], dtype=torch.float32)
    for n, p in adapters(model):
        assert p.grad is not None
        out[f"a.grad.{n}"] = p.grad.detach().clone().contiguous()
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
    print(f"a: loss {a_loss.item():.7f}, {len(frozen)} frozen parameters without a gradient")

    # b: 20 AdamW steps from the init (A random, B = 0).
    model = build_model()
    attach_lora(model, seed=SEED, r=RANK, alpha=alpha)
    params = [p for _, p in adapters(model)]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
    losses = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = loss_of(model, ids, labels)
        loss.backward()
        losses.append(loss.item())
        opt.step()
    out["b.losses"] = torch.tensor(losses, dtype=torch.float32)
    for n, p in adapters(model):
        out[f"b.final.{n}"] = p.detach().clone().contiguous()
    print("b: losses", [f"{x:.5f}" for x in losses[:3]], "...", f"{losses[-1]:.5f}")

    # b64: the same trajectory in float64.
    m64 = build_model().double()
    attach_lora(m64, seed=SEED, r=RANK, alpha=alpha)
    with torch.no_grad():
        for n, p in adapters(m64):
            if n.endswith("lora_A.weight"):
                p.copy_(init[n].double())
    for m in m64.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.double()
            m.lora_B.double()
    opt64 = torch.optim.AdamW([p for _, p in adapters(m64)], lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
    for _ in range(STEPS):
        opt64.zero_grad(set_to_none=True)
        loss_of(m64, ids, labels).backward()
        opt64.step()
    for n, p in adapters(m64):
        out[f"b.final64.{n}"] = p.detach().float().contiguous()  # fp64 result rounded once to f32

    # lin: one LoRA linear alone, at the fixture's scaling.
    g = torch.Generator().manual_seed(SEED + 2)
    rows, inn, outf = 12, 40, 24
    w = torch.randn(outf, inn, generator=g) / inn ** 0.5
    base_lin = nn.Linear(inn, outf, bias=False)
    with torch.no_grad():
        base_lin.weight.copy_(w)
    lin = LoRALinear(base_lin, r=RANK, alpha=alpha)
    base_lin.weight.requires_grad_(False)
    a = torch.randn(RANK, inn, generator=g) / inn ** 0.5
    b = 0.3 * torch.randn(outf, RANK, generator=g)
    with torch.no_grad():
        lin.lora_A.weight.copy_(a)
        lin.lora_B.weight.copy_(b)
    x = torch.randn(rows, inn, generator=g).requires_grad_(True)
    gy = torch.randn(rows, outf, generator=g)
    y = lin(x)
    (y * gy).sum().backward()
    assert base_lin.weight.grad is None
    out.update({
        "lin.x": x.detach().contiguous(), "lin.w": w.contiguous(), "lin.a": a.contiguous(),
        "lin.b": b.contiguous(), "lin.gy": gy.contiguous(), "lin.y": y.detach().contiguous(),
        "lin.dx": x.grad.contiguous(), "lin.da": lin.lora_A.weight.grad.contiguous(),
        "lin.db": lin.lora_B.weight.grad.contiguous(),
        "lin.merged": (w + (alpha / RANK) * (b @ a)).contiguous(),
    })

    meta = {"torch": torch.__version__, "rank": str(RANK), "alpha": str(alpha), "seed": str(SEED),
            "steps": str(STEPS), "lr": str(LR), "weight_decay": str(WD), "targets": json.dumps(names)}
    save_file(out, os.path.join(out_dir, "lora.safetensors"), metadata=meta)
    print(f"{len(out)} tensors -> {out_dir}")


if __name__ == "__main__":
    main()
