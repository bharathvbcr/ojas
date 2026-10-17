"""Golden for ojas-infer's host sampler (`sample_token`), computed with torch.

    python3.14 ojas-oracle/python/sampler_golden.py --out ojas-oracle/fixtures/sampler/golden.txt

torch's RNG is not ojas's, so the comparison does not go through a seed.
For each case the reference takes fixed logits and settings and computes,
with torch ops in float64:

1. the candidates: finite logits ranked by (logit descending, index
   ascending) with `torch.sort(stable=True)`, cut to `top_k`;
2. the temperature softmax of the candidates (`softmax(z / t)`);
3. the top-p prefix: the shortest prefix whose cumulative mass reaches
   `top_p`, renormalised;
4. for a list of fixed uniforms `u`, the inverse-CDF choice
   `searchsorted(cdf, u, right=True)`, i.e. the first candidate whose
   renormalised cumulative probability exceeds `u`.

The Rust test (`ojas-infer/tests/sampler_golden.rs`) drives the sampler with
those same uniforms (by placing `SplitMix64` at the state whose next
`next_f64` is `u`) and requires the same ids, and recovers the sampler's own
cumulative probabilities by bisecting its choice function over `u`, which it
compares with `cdf` to a stated tolerance.

`temperature` and `top_p` are f32 in the sampler, so the reference uses the
f64 value of the f32 (`np.float32(x)`), not the decimal literal. Every case
is checked for margin: no uniform within 1e-7 of a cdf step and no
cumulative mass within 1e-6 of `top_p`, so a boundary that f32 rounding in
the sampler's exponential could move is never what a case decides on.

Greedy (`temperature == 0`) is `torch.argmax` where the row has no tie and
the stable sort's first element otherwise (lowest index of the largest
value, `+0.0` and `-0.0` equal), which is `rank_logits`'s rule.

Format `ojas-sampler-golden-v1`: a line-oriented text file (no JSON
dependency in ojas-infer). Each case is `case <name>` then one `key v v ...`
line each for `logits` (f32 bit patterns, hex), `temperature` and `top_p`
(f32 bits, hex, or `none`), `top_k` (integer or `none`), `rank_ids` (the
kept candidates, rank order), `cdf` (f64, the renormalised cumulative
probability of each kept candidate), `uniforms` (f64) and `choices` (ids),
then `end`.

Reference oracle only (house policy): nothing here ships.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FORMAT = "ojas-sampler-golden-v1"
NEG_INF = float("-inf")
UNIFORM_MARGIN = 1e-7
TOP_P_MARGIN = 1e-6
MASK64 = (1 << 64) - 1


def splitmix_uniforms(seed: int, n: int) -> list[float]:
    """`n` uniforms in [0, 1) from SplitMix64's top 53 bits (the sampler's
    own generator, used here only to spread the test points)."""
    state, out = seed & MASK64, []
    for _ in range(n):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        z ^= z >> 31
        out.append((z >> 11) / float(1 << 53))
    return out


def f32_bits(x: float) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def rank(logits: torch.Tensor) -> torch.Tensor:
    """Indices of the finite logits by (value desc, index asc)."""
    finite = torch.isfinite(logits)
    if bool((logits == float("inf")).any()) or bool(torch.isnan(logits).any()):
        raise SystemExit("case logits must not hold NaN or +inf")
    masked = torch.where(finite, logits, torch.full_like(logits, NEG_INF))
    order = torch.sort(masked, descending=True, stable=True).indices
    return order[: int(finite.sum())]


def reference(logits: torch.Tensor, temperature: float, top_k, top_p, uniforms):
    """`(rank_ids, cdf, choices)` of one case."""
    order = rank(logits)
    if temperature == 0.0:
        lead = int(order[0])
        if bool(torch.isfinite(logits).all()) and int((logits == logits.max()).sum()) == 1:
            if int(torch.argmax(logits)) != lead:
                raise SystemExit("torch.argmax disagrees with the stable sort")
        return [lead], [1.0], [lead] * len(uniforms)
    cand = order if top_k is None else order[:top_k]
    t = float(np.float32(temperature))
    probs = torch.softmax(logits[cand].double() / t, dim=0)
    keep = len(probs)
    if top_p is not None:
        tp = float(np.float32(top_p))
        cum = torch.cumsum(probs, dim=0)
        for c in cum.tolist():
            if abs(c - tp) < TOP_P_MARGIN:
                raise SystemExit(f"cumulative mass {c} is within {TOP_P_MARGIN} of top_p {tp}")
        reached = (cum >= tp).nonzero()
        keep = int(reached[0]) + 1 if len(reached) else len(probs)
    kept = probs[:keep]
    cdf = torch.cumsum(kept / kept.sum(), dim=0)
    steps = cdf.tolist()
    choices = []
    for u in uniforms:
        for s in steps[:-1]:  # the last step is 1.0: every u < 1 is below it
            if abs(u - s) < UNIFORM_MARGIN:
                raise SystemExit(f"uniform {u} is within {UNIFORM_MARGIN} of cdf step {s}")
        i = int(torch.searchsorted(cdf, torch.tensor(u, dtype=torch.float64), right=True))
        choices.append(int(cand[min(i, keep - 1)]))
    return [int(i) for i in cand[:keep]], steps, choices


def normal_row(seed: int, n: int, scale: float) -> list[float]:
    rs = np.random.RandomState(seed)  # legacy generator: stable across numpy versions
    return [float(np.float32(v)) for v in rs.normal(0.0, scale, n)]


def cases():
    us = splitmix_uniforms(0x5EED_0001, 24) + [0.0, 1.0 - 2.0 ** -53]
    ties = [3.0, 2.0, 2.0, 2.0, 0.5, 2.0]
    spread = normal_row(11, 16, 2.0)
    return [
        # name, logits, temperature, top_k, top_p
        ("greedy_plain", normal_row(1, 12, 2.0), 0.0, None, None),
        ("greedy_tied_max", [1.0, 4.0, 0.5, 4.0, 4.0, -2.0], 0.0, None, None),
        ("greedy_signed_zero", [-1.0, -0.0, 0.0, -2.0], 0.0, None, None),
        ("greedy_masked_leader", [NEG_INF, 0.25, NEG_INF, 0.75, 0.5], 0.0, None, None),
        ("temperature_only", spread, 0.7, None, None),
        ("hot_temperature", spread, 1.9, None, None),
        ("top_k_5", normal_row(12, 20, 1.5), 1.3, 5, None),
        ("top_k_1", normal_row(13, 10, 1.5), 0.9, 1, None),
        ("top_k_cuts_a_tie", ties, 1.0, 2, None),
        ("top_p_0_9", normal_row(14, 24, 1.2), 0.8, None, 0.9),
        ("top_p_small", normal_row(15, 24, 1.0), 1.0, None, 0.4),
        ("top_k_8_top_p_0_8", normal_row(16, 32, 1.1), 1.0, 8, 0.8),
        ("masked_row", [NEG_INF, 1.5, 0.5, NEG_INF, 1.0, -0.5, NEG_INF], 0.6, None, 0.95),
        ("vocab_256", normal_row(17, 256, 1.0), 0.9, 40, 0.95),
        ("top_k_above_vocab", normal_row(18, 6, 1.0), 1.1, 50, None),
    ], us


def render(name, logits, temperature, top_k, top_p, uniforms) -> list[str]:
    t = torch.tensor(logits, dtype=torch.float32)
    ids, cdf, choices = reference(t, temperature, top_k, top_p, uniforms)
    hexf = lambda xs: " ".join(f"{f32_bits(x):08x}" for x in xs)
    return [
        f"case {name}",
        f"logits {hexf(logits)}",
        f"temperature {f32_bits(temperature):08x}",
        f"top_k {'none' if top_k is None else top_k}",
        f"top_p {'none' if top_p is None else format(f32_bits(top_p), '08x')}",
        "rank_ids " + " ".join(map(str, ids)),
        "cdf " + " ".join(repr(c) for c in cdf),
        "uniforms " + " ".join(repr(u) for u in uniforms),
        "choices " + " ".join(map(str, choices)),
        "end",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    common.setup_determinism()
    rows, uniforms = cases()
    lines = [FORMAT, f"torch {torch.__version__}", f"uniforms_per_case {len(uniforms)}"]
    for name, logits, temperature, top_k, top_p in rows:
        lines += render(name, logits, temperature, top_k, top_p, uniforms)
    for line in lines:
        if "nan" in line.lower().split():
            raise SystemExit("NaN in the golden")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(args.out)
    spread_choices = {n: len(set(map(int, next(l for l in render(n, lg, t, k, p, uniforms)
                                                if l.startswith("choices")).split()[1:])))
                      for n, lg, t, k, p in rows}
    print(f"{len(rows)} cases, {len(uniforms)} uniforms each; distinct choices per case: "
          f"{spread_choices}", file=sys.stderr)


if __name__ == "__main__":
    main()
