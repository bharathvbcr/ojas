"""One stock nanolab Muon step per case: the bf16 Newton-Schulz golden.

    python3.14 ojas-oracle/python/muon_step.py --out ojas-oracle/fixtures/tiny/muon_step_bf16.safetensors

Each case runs `nanolab.optim.Muon.step` once, unpatched, so its
Newton-Schulz iterate is `G.bfloat16()` (optim.py:46), on a parameter, a
gradient and a non-zero momentum buffer drawn from a seeded generator. The
file holds, per case `c<i>`, the inputs `p`, `g`, `m` and torch's results
`p_after`, `m_after`. Metadata lists each case's shape and hyperparameters.

The shapes cover the three orientations Muon distinguishes: square, tall
(rows > cols, transposed before the iteration and back after) and wide. The
last case turns off weight decay and Nesterov.

Reference oracle only (house policy): CPU, one thread, deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from nanolab.optim import Muon  # noqa: E402

GENERATOR = "ojas-oracle/python/muon_step.py"
SEED = 20261006

# (rows, cols, lr, momentum, weight_decay, nesterov)
CASES = [
    (64, 64, 0.025, 0.99, 0.1, True),
    (96, 32, 0.025, 0.99, 0.1, True),
    (32, 96, 0.025, 0.99, 0.1, True),
    (64, 64, 0.02, 0.95, 0.0, False),
]


def one_case(gen: torch.Generator, rows, cols, lr, mom, wd, nesterov):
    p = torch.randn(rows, cols, generator=gen) * 0.02
    g = torch.randn(rows, cols, generator=gen) * 1e-3
    m = torch.randn(rows, cols, generator=gen) * 1e-3
    param = torch.nn.Parameter(p.clone())
    param.grad = g.clone()
    opt = Muon([param], lr=lr, momentum=mom, nesterov=nesterov, ns_steps=5,
               weight_decay=wd)
    opt.state[param]["momentum_buffer"] = m.clone()
    calls = {"n": 0}
    stock = sys.modules["nanolab.optim"].zeropower_via_newtonschulz5

    def counted(*args, **kwargs):
        calls["n"] += 1
        return stock(*args, **kwargs)

    sys.modules["nanolab.optim"].zeropower_via_newtonschulz5 = counted
    try:
        opt.step()
    finally:
        sys.modules["nanolab.optim"].zeropower_via_newtonschulz5 = stock
    if calls["n"] != 1:
        raise SystemExit(f"expected one stock NS5 call, saw {calls['n']}")
    m_after = opt.state[param]["momentum_buffer"].detach().clone()
    return p, g, m, param.detach().clone(), m_after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    common.setup_determinism()
    gen = torch.Generator().manual_seed(SEED)
    tensors: dict[str, torch.Tensor] = {}
    cases = []
    for i, (rows, cols, lr, mom, wd, nesterov) in enumerate(CASES):
        p, g, m, p_after, m_after = one_case(gen, rows, cols, lr, mom, wd, nesterov)
        for name, t in [("p", p), ("g", g), ("m", m), ("p_after", p_after),
                        ("m_after", m_after)]:
            tensors[f"c{i}.{name}"] = t.contiguous()
        cases.append({"rows": rows, "cols": cols, "lr": lr, "momentum": mom,
                      "weight_decay": wd, "nesterov": nesterov})
    meta = common.provenance(GENERATOR, kind="muon_step", ns5="bf16", seed=SEED,
                             cases=cases)
    common.write_safetensors(args.out, tensors,
                             {"ojas.oracle": common.canonical_json(meta)})
    print(json.dumps({"out": str(args.out), "cases": len(cases)}))


main()
