"""Golden for `max_pool2d_forward` / `max_pool2d_backward`, computed with torch.

    /Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/max_pool_fixture.py \
        --out ojas-oracle/fixtures/max_pool/golden.txt

Each case runs `F.max_pool2d(x, kernel, stride)` (no padding, no dilation,
floor mode, values only) forward and `y.backward(gy)` in float32 on the CPU.
Max pooling does no arithmetic on the values it passes, and the backward only
adds the `gy` of the windows an input wins, so the expected tensors are exact
and ojas is held to them BIT FOR BIT (`ojas-metal/tests/max_pool.rs` for the
Metal backend and the CPU backend).

The cases include ties, since the tie rule is the part that can differ:
torch credits the FIRST maximum of a window in (kh, kw) order. Every case is
also checked here against a pure-python reference of that rule, so a torch
that changed it would stop the fixture being written rather than change the
golden. A plane of equal values, values drawn from three integers, and a mix
of +0.0 and -0.0 (equal under `>`, distinct bit patterns) are the tie cases.

Format `ojas-maxpool-golden-v1`: a line-oriented text file. A case is
`case <name> <n> <c> <h> <w> <kh> <kw> <sh> <sw>`, then one `t <name> f32
<dims, comma separated> <hex...>` line per tensor (f32 bit patterns of 8 hex
digits, row-major: `x`, `gy`, the forward output `y` and the input gradient
`dx`), then `end`.

Reference oracle only (house policy): nothing here ships.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

FORMAT = "ojas-maxpool-golden-v1"


def hexbits(t: Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1)
    assert flat.dtype == torch.float32
    return " ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in flat.view(torch.int32))


def tline(name: str, t: Tensor) -> str:
    dims = ",".join(str(d) for d in t.shape)
    return f"t {name} f32 {dims} {hexbits(t)}"


def reference(x: Tensor, gy: Tensor, k: tuple[int, int], s: tuple[int, int]):
    """First-maximum max pool and its gather-form gradient, in plain python."""
    n, c, h, w = x.shape
    oh, ow = (h - k[0]) // s[0] + 1, (w - k[1]) // s[1] + 1
    y = torch.zeros(n, c, oh, ow, dtype=torch.float32)
    dx = torch.zeros_like(x)
    for b in range(n):
        for ch in range(c):
            for oy in range(oh):
                for ox in range(ow):
                    best = None
                    for i in range(k[0]):
                        for j in range(k[1]):
                            at = (oy * s[0] + i, ox * s[1] + j)
                            if best is None or x[b, ch, at[0], at[1]] > x[b, ch, best[0], best[1]]:
                                best = at
                    y[b, ch, oy, ox] = x[b, ch, best[0], best[1]]
                    dx[b, ch, best[0], best[1]] += gy[b, ch, oy, ox]
    return y, dx


def make_cases(gen: torch.Generator):
    def randn(*shape):
        return torch.randn(shape, generator=gen, dtype=torch.float32)

    def ints(*shape):
        return torch.randint(0, 3, shape, generator=gen).to(torch.float32)

    zeros = torch.randint(0, 2, (1, 2, 6, 6), generator=gen)
    signed_zeros = torch.where(zeros == 0, torch.tensor(0.0), torch.tensor(-0.0))
    yield "neck_like_2x2_s2", randn(1, 3, 8, 8), (2, 2), (2, 2)
    yield "floor_remainder_odd", randn(2, 2, 7, 9), (2, 2), (2, 2)
    yield "overlap_3x3_s1", randn(1, 2, 6, 7), (3, 3), (1, 1)
    yield "overlap_3x3_s2", randn(1, 2, 9, 9), (3, 3), (2, 2)
    yield "rect_2x3_s1x2", randn(1, 2, 5, 8), (2, 3), (1, 2)
    yield "stride_gt_kernel_2_s3", randn(1, 1, 7, 7), (2, 2), (3, 3)
    yield "window_is_image", randn(2, 1, 4, 5), (4, 5), (1, 1)
    yield "ties_constant_plane", torch.ones(1, 1, 4, 4), (2, 2), (2, 2)
    yield "ties_three_integers", ints(1, 3, 6, 6), (2, 2), (2, 2)
    yield "ties_overlap_integers", ints(1, 2, 7, 7), (3, 3), (1, 1)
    yield "ties_signed_zeros", signed_zeros, (2, 2), (2, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(20261031)
    lines = [FORMAT, f"torch {torch.__version__.split('+')[0]}"]
    for name, x, k, s in make_cases(gen):
        x = x.clone().requires_grad_(True)
        y = F.max_pool2d(x, kernel_size=k, stride=s)
        gy = torch.randn(y.shape, generator=gen, dtype=torch.float32)
        y.backward(gy)
        want_y, want_dx = reference(x.detach(), gy, k, s)
        assert torch.equal(y.detach().view(torch.int32), want_y.view(torch.int32)), name
        assert torch.equal(x.grad.view(torch.int32), want_dx.view(torch.int32)), name
        n, c, h, w = x.shape
        lines.append(f"case {name} {n} {c} {h} {w} {k[0]} {k[1]} {s[0]} {s[1]}")
        lines.append(tline("x", x.detach()))
        lines.append(tline("gy", gy))
        lines.append(tline("y", y.detach()))
        lines.append(tline("dx", x.grad))
        lines.append("end")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
