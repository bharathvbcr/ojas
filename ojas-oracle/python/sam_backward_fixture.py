"""Golden for the SAM backward ops (GroupNorm and bilinear resampling), computed with torch autograd.

    /Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/sam_backward_fixture.py \
        --out ojas-oracle/fixtures/sam_bwd/golden.txt

Each case runs one op forward and backward in torch on the CPU, once in
float32 (what torch computes) and once in float64 from the same float32
inputs (the value torch's float32 is an approximation of). The Rust side
(`ojas-cpu/tests/sam_backward_golden.rs`) compares Ojas's gradients with both:
the float32 pair records the error against torch, the float64 pair shows
whether a gap is torch's rounding or Ojas's.

The ops are torch's own:

- `gn`: `F.group_norm(x, groups, weight, bias, eps)`, the functional form of
  `nn.GroupNorm(groups, channels, eps)`.
- `bil`: `F.interpolate(x, size=(out_h, out_w), mode="bilinear",
  align_corners=...)`.

Every case also checks, here, that torch's float64 gradients equal an
independent closed form to 1e-9, so a fixture that disagrees with the
formulas is never written: GroupNorm's from the normalized values
(`dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))` per group),
bilinear's from the resampling matrix, which is the op applied to the unit
images (the op is linear, so its adjoint is that matrix's transpose).

Format `ojas-sam-bwd-golden-v1`: a line-oriented text file. A case is
`case <op> <name> <eps bits, or -> <param>` where `param` is the group count
for `gn` and 0 or 1 (`align_corners`) for `bil`, then one
`t <name> <f32|f64> <dims, comma separated, or -> <hex...>` line per tensor
(f32 bit patterns of 8 hex digits, f64 of 16, row-major), then `end`. The
tensors are the inputs (`x`, `w`, `b`, `gy`), the forward output `y`, and the
gradients `dx`, `dw`, `db`, each with a `_64` copy from the float64 run. A
`bil` case's output size is the trailing two dimensions of `gy`.

Reference oracle only (house policy): nothing here ships.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

FORMAT = "ojas-sam-bwd-golden-v1"


def hexbits(t: Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1)
    if flat.dtype == torch.float32:
        return " ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in flat.view(torch.int32))
    assert flat.dtype == torch.float64
    return " ".join(f"{int(v) & 0xFFFFFFFFFFFFFFFF:016x}" for v in flat.view(torch.int64))


def eps_bits(eps: float | None) -> str:
    if eps is None:
        return "-"
    return f"{int(torch.tensor(eps, dtype=torch.float32).view(torch.int32)) & 0xFFFFFFFF:08x}"


def tline(name: str, t: Tensor) -> str:
    dtype = "f32" if t.dtype == torch.float32 else "f64"
    dims = ",".join(str(d) for d in t.shape) or "-"
    return f"t {name} {dtype} {dims} {hexbits(t)}"


def randn(gen: torch.Generator, *shape: int, scale: float = 1.0, shift: float = 0.0) -> Tensor:
    return torch.randn(shape, generator=gen, dtype=torch.float32) * scale + shift


def run(fn, inputs: dict[str, Tensor], gy: Tensor):
    """`fn(**inputs)` forward in float32 and float64 and its backward with `gy`."""
    out = {}
    for tag, dtype in (("", torch.float32), ("_64", torch.float64)):
        leaves = {k: v.detach().to(dtype).requires_grad_(True) for k, v in inputs.items()}
        y = fn(**leaves)
        y.backward(gy.to(dtype))
        out["y" + tag] = y.detach()
        for k, leaf in leaves.items():
            out["d" + k + tag] = leaf.grad.detach()
    return out


def emit(op: str, name: str, eps: float | None, param: int, ins: dict[str, Tensor], gy: Tensor,
         res: dict[str, Tensor]):
    lines = [f"case {op} {name} {eps_bits(eps)} {param}"]
    for k, v in ins.items():
        lines.append(tline(k, v))
    lines.append(tline("gy", gy))
    for k, v in res.items():
        lines.append(tline(k, v))
    lines.append("end")
    return lines


def close(got: Tensor, want: Tensor, what: str):
    assert torch.allclose(got, want, rtol=1e-9, atol=1e-12), f"{what}: torch float64 disagrees with the closed form"


# ---- closed forms in float64, to check torch against ----


def gn_closed(x, w, groups, eps, gy):
    n, c = x.shape[0], x.shape[1]
    rest = x.shape[2:]
    g = x.reshape(n, groups, -1)
    mean = g.mean(-1, keepdim=True)
    var = ((g - mean) ** 2).mean(-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    xhat = (g - mean) * rstd
    xhat_c = xhat.reshape(n, c, *rest)
    dxhat = (gy * w.reshape(1, c, *([1] * len(rest)))).reshape(n, groups, -1)
    dx = rstd * (dxhat - dxhat.mean(-1, keepdim=True) - xhat * (dxhat * xhat).mean(-1, keepdim=True))
    axes = [0] + list(range(2, x.dim()))
    return dx.reshape(x.shape), (gy * xhat_c).sum(axes), gy.sum(axes)


def bilinear_matrix(h: int, w: int, oh: int, ow: int, align: bool) -> Tensor:
    """`[h*w, oh*ow]`: row i is the resampling of the i-th unit image."""
    basis = torch.eye(h * w, dtype=torch.float64).reshape(h * w, 1, h, w)
    out = F.interpolate(basis, size=(oh, ow), mode="bilinear", align_corners=align)
    return out.reshape(h * w, oh * ow)


def gn_case(name, shape, groups, eps, gen, shift=0.0, scale=1.0):
    c = shape[1]
    x = randn(gen, *shape, scale=scale, shift=shift)
    w = randn(gen, c, scale=0.5, shift=1.0)
    b = randn(gen, c, scale=0.5)
    gy = randn(gen, *shape)
    res = run(lambda x, w, b: F.group_norm(x, groups, w, b, eps), {"x": x, "w": w, "b": b}, gy)
    dx, dw, db = gn_closed(x.double(), w.double(), groups, eps, gy.double())
    close(res["dx_64"], dx, name + " dx")
    close(res["dw_64"], dw, name + " dw")
    close(res["db_64"], db, name + " db")
    return emit("gn", name, eps, groups, {"x": x, "w": w, "b": b}, gy, res)


def bil_case(name, shape, out_size, align, gen):
    n, c, h, w = shape
    oh, ow = out_size
    x = randn(gen, *shape)
    gy = randn(gen, n, c, oh, ow)
    res = run(
        lambda x: F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=align),
        {"x": x},
        gy,
    )
    m = bilinear_matrix(h, w, oh, ow, align)
    want = (gy.double().reshape(n, c, oh * ow) @ m.T).reshape(n, c, h, w)
    close(res["dx_64"], want, name + " dx")
    return emit("bil", name, None, int(align), {"x": x}, gy, res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(11)
    lines = [FORMAT, f"torch {torch.__version__}"]

    # GroupNorm: one group over everything, two groups, one channel a group
    # (SAM's 8 groups at 8 channels), two channels a group at 8 groups, a
    # smaller eps, one position a channel, a mean far from zero, a batch of 1.
    lines += gn_case("gn_g1", (2, 8, 3, 5), 1, 1e-5, gen)
    lines += gn_case("gn_g2", (2, 8, 3, 5), 2, 1e-5, gen)
    lines += gn_case("gn_g8", (2, 8, 3, 5), 8, 1e-5, gen)
    lines += gn_case("gn_c16_g8", (2, 16, 3, 5), 8, 1e-5, gen)
    lines += gn_case("gn_eps_1e6", (2, 8, 3, 5), 4, 1e-6, gen)
    lines += gn_case("gn_spatial_one", (3, 8, 1, 1), 2, 1e-5, gen)
    lines += gn_case("gn_mean_offset", (2, 8, 3, 5), 2, 1e-5, gen, shift=100.0, scale=0.5)
    lines += gn_case("gn_batch_one", (1, 12, 4, 4), 3, 1e-5, gen)

    # Bilinear, both align_corners: up and down on a rectangle, a 1 x N edge
    # up and down, the identity, and an exact 2x reduction.
    for align in (False, True):
        tag = "align" if align else "half"
        lines += bil_case(f"bil_up_{tag}", (1, 2, 5, 7), (9, 13), align, gen)
        lines += bil_case(f"bil_down_{tag}", (1, 2, 5, 7), (3, 4), align, gen)
        lines += bil_case(f"bil_1xn_up_{tag}", (1, 1, 1, 4), (1, 9), align, gen)
        lines += bil_case(f"bil_1xn_down_{tag}", (1, 2, 1, 9), (1, 4), align, gen)
        lines += bil_case(f"bil_identity_{tag}", (1, 1, 3, 3), (3, 3), align, gen)
        lines += bil_case(f"bil_halve_{tag}", (2, 1, 4, 4), (2, 2), align, gen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    cases = sum(1 for line in lines if line.startswith("case "))
    print(f"wrote {args.out} ({cases} cases, torch {torch.__version__})")


if __name__ == "__main__":
    main()
