"""Goldens for the attention variants the tiny MHA fixtures do not cover.

    python3.14 ojas-oracle/python/variant_fixture.py --dir ojas-oracle/fixtures/variants

Writes one file per variant, each a complete forward and backward of the
tiny GPT (2 layers, d=64, 4 heads x 16, SwiGLU 192, V=256, T=32, B=2) under
nanolab's own model, fp32, CPU, one thread, deterministic algorithms:

- `gqa.safetensors`: grouped-query attention, `n_kv_head = 2` for 4 query
  heads. nanolab widens K and V with `repeat_interleave(rep, dim=1)`
  (mixers.py:283-285), so query head `h` reads KV head `h // rep`.
- `swa.safetensors`: sliding-window attention, window 8 on every layer, no
  sinks (`mixer="swa"`, `swa_window=8`, `swa_sinks=0`; mixers.py:98-151). A
  query at `t` sees keys `t - 8 < j <= t`, which is ojas's
  `Backend::causal_sdpa_forward(.., window = Some(8))`. MHA, so a window bug
  is not confused with a GQA one.

Each file holds, as F32 (ids as I64): `param/<name>` the parameters (nanolab
`state_dict` names, the tied head not stored), `grad/<name>` d(loss)/dθ with
loss seed 1, `x`, `y`, and `logits`. `__metadata__` has `ojas.spec`
(`ojas-spec-v1`, or `ojas-spec-v2` with `window`) and `ojas.oracle`
(provenance, `kind = "variant"`, the loss and its f32 bits).

Why the parameters are not nanolab's init. nanolab zero-initialises
`o_proj` and `ffn.down` (`zero_init_proj`), so at the init every attention
and FFN gradient upstream of them is exactly zero and a gradient golden
would check none of the attention path. The parameters here are the nanolab
init (`zero_init_proj=False`, so `o_proj` and `ffn.down` are N(0, 0.02) and
every gradient flows) with each tensor that nanolab initialises to a
constant moved off it by a fixed seeded draw: norm weights and the
QK-norm weights 1 + 0.1 N(0,1), the gate bias, the gate weight and
`vr_lambda` 0.1 N(0,1) (gate and `vr_lambda` start at exactly 0, which would
leave the gate at 0.5 and the value blend at 0.5 for every head and layer).

Reference oracle only (house policy): nothing here ships. Two runs write the
same bytes (the generator refuses to write otherwise).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import golden  # noqa: E402

from nanolab.config import Config  # noqa: E402

GENERATOR = "ojas-oracle/python/variant_fixture.py"
SEED = 1337
PERTURB_SEED = 0x0DA5_0002
BATCH = 2
WINDOW = 8
# Offsets into `golden.synthetic_tokens()` of the two windows of T+1 ids.
STARTS = (0, 1001)

VARIANTS = {
    "gqa": dict(n_kv_head=2, mixer="attention"),
    "swa": dict(n_kv_head=4, mixer="swa", swa_window=WINDOW, swa_sinks=0),
}


def variant_config(**overrides) -> Config:
    """The tiny model in fp32, eager, CPU, with `zero_init_proj` off and
    `overrides`. Every architecture flag but the two a variant changes is
    checked against what ojas implements (`common.REQUIRED_FLAGS`)."""
    base = dict(common.TINY)
    base.update(seed=SEED, dtype="fp32", compile=False, device="cpu",
                zero_init_proj=False)
    base.update(overrides)
    cfg = Config(**base)
    free = {"mixer", "zero_init_proj"}
    for key, want in common.REQUIRED_FLAGS.items():
        if key not in free and getattr(cfg, key) != want:
            raise SystemExit(f"config {key}={getattr(cfg, key)!r}: ojas implements only {want!r}")
    if cfg.mixer not in ("attention", "swa"):
        raise SystemExit(f"mixer {cfg.mixer!r} is not a variant of attention")
    return cfg


def perturbed_state(cfg: Config) -> dict[str, torch.Tensor]:
    """nanolab's init at `cfg.seed` with every constant-initialised tensor
    moved by a fixed seeded draw (see the module comment)."""
    model = common.build_seeded_model(cfg)
    state = common.export_state(model)
    gen = torch.Generator().manual_seed(PERTURB_SEED)
    out = {}
    for name in sorted(state):
        t = state[name]
        if bool((t == 1).all()):
            t = 1.0 + 0.1 * torch.randn(t.shape, generator=gen, dtype=torch.float32)
        elif bool((t == 0).all()):
            t = 0.1 * torch.randn(t.shape, generator=gen, dtype=torch.float32)
        out[name] = t.contiguous()
    return out


def spec_of(cfg: Config) -> dict:
    spec = common.spec_from_config(cfg)
    if cfg.mixer == "swa":
        spec["format"] = "ojas-spec-v2"
        spec["window"] = cfg.swa_window
    return spec


def check_names(state: dict, spec: dict) -> None:
    want = [name for name, *_ in common.expected_params(spec)]
    if sorted(want) != sorted(state):
        raise SystemExit(f"§2 names disagree with nanolab: {sorted(set(want) ^ set(state))}")
    for name, shape, *_ in common.expected_params(spec):
        if list(state[name].shape) != shape:
            raise SystemExit(f"{name}: shape {list(state[name].shape)}, §2 says {shape}")


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    tokens = golden.synthetic_tokens().astype(np.int64)
    t = common.TINY["block_size"]
    rows = [tokens[s:s + t + 1] for s in STARTS]
    seq = torch.from_numpy(np.stack(rows))
    return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()


def forward_and_grads(cfg: Config, state: dict, x, y):
    model = common.build_seeded_model(cfg)
    common.load_state(model, state)
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss_g = model(x, y)
    loss_g.backward()
    grads, no_grad = {}, []
    for name, p in model.named_parameters():
        if name == "lm_head.weight":
            continue  # tied: its gradient is tok_emb's, already accumulated there
        if p.grad is None:
            no_grad.append(name)
        else:
            grads[name] = p.grad.detach().clone().contiguous()
    return logits.detach().contiguous(), float(loss.detach()), grads, no_grad


def build(name: str):
    cfg = variant_config(**VARIANTS[name])
    spec = spec_of(cfg)
    state = perturbed_state(cfg)
    check_names(state, spec)
    x, y = batch()
    logits, loss, grads, no_grad = forward_and_grads(cfg, state, x, y)
    for gname, g in grads.items():
        if not bool(g.any()):
            raise SystemExit(f"{name}: gradient of {gname} is exactly zero; the fixture "
                             "would check nothing through it")
    # The variant must matter: the same weights under plain MHA causal
    # attention give other logits (window), and a different KV-head order
    # gives other logits (GQA).
    ref_cfg = variant_config(n_kv_head=cfg.n_kv_head, mixer="attention")
    ref_logits, _, _, _ = forward_and_grads(ref_cfg, state, x, y)
    if cfg.mixer == "swa":
        gap = float((logits - ref_logits).abs().max())
        if gap < 1e-3:
            raise SystemExit(f"swa: window 8 changes logits by only {gap:e}")
    else:
        swapped = dict(state)
        for i in range(cfg.n_layer):
            for proj in ("k_proj", "v_proj"):
                key = f"blocks.{i}.mixer.{proj}.weight"
                w = state[key]
                half = w.shape[0] // 2
                swapped[key] = torch.cat([w[half:], w[:half]]).contiguous()
        sw_logits, _, _, _ = forward_and_grads(cfg, swapped, x, y)
        gap = float((logits - sw_logits).abs().max())
        if gap < 1e-3:
            raise SystemExit(f"gqa: swapping the KV heads changes logits by only {gap:e}")
    return cfg, spec, state, x, y, logits, loss, grads, no_grad, gap


def write(d: Path, name: str) -> None:
    cfg, spec, state, x, y, logits, loss, grads, no_grad, gap = build(name)
    tensors = {"x": x, "y": y, "logits": logits}
    tensors.update({f"param/{n}": t for n, t in state.items()})
    tensors.update({f"grad/{n}": g for n, g in grads.items()})
    prov = common.provenance(
        GENERATOR, cfg, kind="variant", variant=name, seed=SEED,
        perturb_seed=PERTURB_SEED, starts=list(STARTS),
        loss=common.finite_float(loss), loss_f32_bits=golden.f32_bits(loss),
        loss_seed=1.0, no_grad=no_grad, variant_logit_gap=gap)
    meta = {"ojas.spec": common.canonical_json(spec),
            "ojas.oracle": common.canonical_json(prov)}
    path = d / f"{name}.safetensors"
    common.write_safetensors(path, tensors, meta)
    print(f"{name}: loss {loss:.7f}, {len(grads)} gradients, "
          f"variant logit gap {gap:.3e}, {path.stat().st_size} bytes", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    common.setup_determinism()
    args.dir.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        write(args.dir, name)


if __name__ == "__main__":
    main()
