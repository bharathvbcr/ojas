"""Export nanolab's GPT initialisation as `model.safetensors` for ojas.

    python3.14 ojas-oracle/python/export_init.py --tiny --out ojas-oracle/fixtures/tiny/init.safetensors
    python3.14 ojas-oracle/python/export_init.py --out ojas-oracle/generated/124m/model.safetensors

Builds `GPT(Config(...))` exactly as nanolab's trainer does (set_seed, then
build_model), on CPU in fp32, and writes its state_dict under the
framework-design.md §2 naming rules: `_orig_mod.` stripped, only `tok_emb`
stored (the tied `lm_head` is checked bit-equal first), the spec JSON in
`__metadata__["ojas.spec"]` and provenance in `__metadata__["ojas.oracle"]`.

Before writing, every row of the §2 init table is checked against the built
model (names, shapes, zero/one inits exactly, N(0, 0.02) statistically, and
the Muon/AdamW group against nanolab's own `_split_params`). Any mismatch
aborts; `--report` prints the measured statistics as JSON.

The default (124M) export is ~0.5 GB. It is a generated artifact: write it
under ojas-oracle/generated/ (gitignored), never into fixtures/.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

GENERATOR = "ojas-oracle/python/export_init.py"
INIT_STD = 0.02
SIGMAS = 6.0


def verify_init_table(model, spec: dict) -> list[dict]:
    """Check §2's table against the model. Returns the measured rows."""
    state = common.export_state(model)
    expected = common.expected_params(spec)
    names = [name for name, *_ in expected]
    if sorted(names) != sorted(state):
        missing = sorted(set(names) - set(state))
        extra = sorted(set(state) - set(names))
        raise SystemExit(f"§2 names disagree with nanolab: missing {missing}, extra {extra}")
    if model.lm_head.weight.data_ptr() != model.tok_emb.weight.data_ptr():
        raise SystemExit("lm_head is not tied to tok_emb")
    groups = common.param_groups(model)
    rows = []
    failures = []
    for name, shape, init, group in expected:
        t = state[name].double()
        n = t.numel()
        row = {
            "name": name, "shape": list(t.shape), "init": init, "group": group,
            "nanolab_group": groups[name],
            "mean": t.mean().item(), "std": t.std(unbiased=False).item() if n > 1 else 0.0,
            "min": t.min().item(), "max": t.max().item(),
            "zeros": int((t == 0).sum().item()), "numel": n,
        }
        ok = list(t.shape) == shape and groups[name] == group
        if init == "zeros":
            ok &= row["zeros"] == n
        elif init == "ones":
            ok &= bool((t == 1).all().item())
        else:
            # Sample mean and std of n draws from N(0, s): bounds at SIGMAS
            # standard errors (mean: s/sqrt(n); std: about s/sqrt(2n)).
            ok &= abs(row["mean"]) <= SIGMAS * INIT_STD / math.sqrt(n)
            # The std bound already separates N(0, 0.02) from a zeroed tensor.
            # Exact zeros are not refused: torch's CPU normal_ draws a few at
            # 124M scale (1 of 38.6M in tok_emb at seed 1337).
            ok &= abs(row["std"] / INIT_STD - 1.0) <= SIGMAS / math.sqrt(2 * n)
        row["ok"] = bool(ok)
        rows.append(row)
        if not ok:
            failures.append(name)
    if failures:
        raise SystemExit(f"§2 init table disagrees with nanolab for {failures}")
    return rows


def export(tiny: bool, seed: int, out: Path) -> list[dict]:
    common.setup_determinism()
    cfg = common.make_config(tiny, seed)
    model = common.build_seeded_model(cfg)
    spec = common.spec_from_config(cfg)
    rows = verify_init_table(model, spec)
    state = common.export_state(model)
    meta = {
        "ojas.spec": common.canonical_json(spec),
        "ojas.oracle": common.canonical_json(common.provenance(
            GENERATOR, cfg, kind="init", seed=seed,
            config_name="tiny" if tiny else "nanolab-124m")),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    common.write_safetensors(out, state, meta)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true",
                    help="2 layers, d=64, 4x16 heads, hidden 192, V=256, T=32")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", action="store_true",
                    help="print the measured init-table rows as JSON")
    args = ap.parse_args()
    out = args.out.resolve()
    if not args.tiny and common.FIXTURE_DIR in out.parents:
        raise SystemExit("the 124M export is a generated artifact; write it outside fixtures/")
    rows = export(args.tiny, args.seed, out)
    if args.report:
        print(common.canonical_json(rows))
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(rows)} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
