"""Golden for the encoder backward ops, computed with torch autograd.

    /Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/encoder_backward_fixture.py \
        --out ojas-oracle/fixtures/encoder_bwd/golden.txt

Each case runs one op forward and backward in torch on the CPU, once in
float32 (what torch computes) and once in float64 from the same float32
inputs (the value torch's float32 is an approximation of). The Rust side
(`ojas-cpu/tests/encoder_backward.rs`) compares Ojas's gradients with both:
the float32 pair records the error against torch, the float64 pair shows
whether a gap is torch's rounding or Ojas's.

The ops are torch's own:

- `ln`: `F.layer_norm(x, (dim,), weight, bias, eps)`.
- `cln`: SAM's `LayerNorm2d`: the channel axis moved last, `F.layer_norm`, and
  moved back.
- `gelu`: `F.gelu(x)` (the erf form, `approximate="none"`).
- `lin`: `F.linear(x, weight, bias)`.
- `badd`, `bmul`: `a + b` and `a * b` with torch's broadcasting; autograd sums
  each gradient back to its operand's shape.
- `softmax`: `F.softmax(x, dim=-1)`.

Every case also checks, here, that torch's float64 gradients equal the closed
forms below to 1e-9, so a fixture that disagrees with the formulas is never
written.

Format `ojas-encoder-bwd-golden-v1`: a line-oriented text file. A case is
`case <op> <name> <eps bits>` (`-` when the op has no eps), then one `t <name>
<f32|f64> <dims, comma separated, or -> <hex...>` line per tensor (f32 bit
patterns of 8 hex digits, f64 of 16, row-major), then `end`. The tensors are
the inputs (`x`, `w`, `b`, `a`, `gy`), the forward output `y`, and the
gradients `dx`, `dw`, `db`, `da` with a `_64` suffix for the float64 run.

Reference oracle only (house policy): nothing here ships.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

FORMAT = "ojas-encoder-bwd-golden-v1"


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


# ---- the ops under test, as torch defines them ----


def op_ln(eps: float):
    return lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b, eps)


def op_cln(eps: float):
    def f(x, w, b):
        moved = x.movedim(1, -1)
        y = F.layer_norm(moved, (moved.shape[-1],), w, b, eps)
        return y.movedim(-1, 1)

    return f


# ---- closed forms in float64, to check torch against ----


def ln_closed(x, w, b, eps, gy):
    dim = x.shape[-1]
    mean = x.mean(-1, keepdim=True)
    var = ((x - mean) ** 2).mean(-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    xhat = (x - mean) * rstd
    dxhat = gy * w
    dx = rstd * (dxhat - dxhat.mean(-1, keepdim=True) - xhat * (dxhat * xhat).mean(-1, keepdim=True))
    lead = tuple(range(x.dim() - 1))
    return dx, (gy * xhat).sum(lead), gy.sum(lead)


def gelu_closed(x, gy):
    cdf = 0.5 * (1.0 + torch.erf(x / 2**0.5))
    pdf = torch.exp(-0.5 * x * x) / (2 * torch.pi) ** 0.5
    return gy * (cdf + x * pdf)


def softmax_closed(x, gy):
    y = torch.softmax(x, -1)
    return y * (gy - (gy * y).sum(-1, keepdim=True))


def unbroadcast(g: Tensor, shape) -> Tensor:
    return g.sum_to_size(shape) if len(shape) else g.sum()


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


def emit(op: str, name: str, eps: float | None, ins: dict[str, Tensor], gy: Tensor, res: dict[str, Tensor]):
    lines = [f"case {op} {name} {eps_bits(eps)}"]
    for k, v in ins.items():
        lines.append(tline(k, v))
    lines.append(tline("gy", gy))
    for k, v in res.items():
        lines.append(tline(k, v))
    lines.append("end")
    return lines


def close(got: Tensor, want: Tensor, what: str):
    assert torch.allclose(got, want, rtol=1e-9, atol=1e-12), f"{what}: torch float64 disagrees with the closed form"


def norm_case(op, name, shape, eps, gen, affine_scale=1.0, shift=0.0, scale=1.0):
    dim = shape[-1] if op == "ln" else shape[1]
    x = randn(gen, *shape, scale=scale, shift=shift)
    w = randn(gen, dim, scale=affine_scale, shift=1.0)
    b = randn(gen, dim, scale=0.5)
    gy = randn(gen, *shape)
    fn = op_ln(eps) if op == "ln" else op_cln(eps)
    res = run(lambda x, w, b: fn(x, w, b), {"x": x, "w": w, "b": b}, gy)
    # Closed form check, float64.
    x64, w64, b64, gy64 = (t.double() for t in (x, w, b, gy))
    if op == "ln":
        dx, dw, db = ln_closed(x64, w64, b64, eps, gy64)
    else:
        mx, mg = x64.movedim(1, -1), gy64.movedim(1, -1)
        dx, dw, db = ln_closed(mx, w64, b64, eps, mg)
        dx = dx.movedim(-1, 1)
    close(res["dx_64"], dx, name + " dx")
    close(res["dw_64"], dw, name + " dw")
    close(res["db_64"], db, name + " db")
    return emit(op, name, eps, {"x": x, "w": w, "b": b}, gy, res)


def gelu_case(name, x: Tensor, gen):
    gy = randn(gen, *x.shape)
    res = run(lambda x: F.gelu(x), {"x": x}, gy)
    close(res["dx_64"], gelu_closed(x.double(), gy.double()), name + " dx")
    return emit("gelu", name, None, {"x": x}, gy, res)


def lin_case(name, x_shape, out, gen):
    inn = x_shape[-1]
    x = randn(gen, *x_shape)
    w = randn(gen, out, inn, scale=0.5)
    b = randn(gen, out)
    gy = randn(gen, *x_shape[:-1], out)
    res = run(lambda x, w, b: F.linear(x, w, b), {"x": x, "w": w, "b": b}, gy)
    x64, w64, gy64 = x.double(), w.double(), gy.double()
    flat_x, flat_g = x64.reshape(-1, inn), gy64.reshape(-1, out)
    close(res["dx_64"], (flat_g @ w64).reshape(x_shape), name + " dx")
    close(res["dw_64"], flat_g.T @ flat_x, name + " dw")
    close(res["db_64"], flat_g.sum(0), name + " db")
    return emit("lin", name, None, {"x": x, "w": w, "b": b}, gy, res)


def bin_case(op, name, a_shape, b_shape, gen):
    a = randn(gen, *a_shape)
    b = randn(gen, *b_shape)
    out_shape = torch.broadcast_shapes(a_shape, b_shape)
    gy = randn(gen, *out_shape)
    fn = (lambda a, b: a + b) if op == "badd" else (lambda a, b: a * b)
    res = run(fn, {"a": a, "b": b}, gy)
    a64, b64, gy64 = a.double(), b.double(), gy.double()
    if op == "badd":
        want_a, want_b = unbroadcast(gy64, a_shape), unbroadcast(gy64, b_shape)
    else:
        want_a, want_b = unbroadcast(gy64 * b64, a_shape), unbroadcast(gy64 * a64, b_shape)
    close(res["da_64"], want_a, name + " da")
    close(res["db_64"], want_b, name + " db")
    return emit(op, name, None, {"a": a, "b": b}, gy, res)


def softmax_case(name, x: Tensor, gen):
    gy = randn(gen, *x.shape)
    res = run(lambda x: F.softmax(x, dim=-1), {"x": x}, gy)
    close(res["dx_64"], softmax_closed(x.double(), gy.double()), name + " dx")
    return emit("softmax", name, None, {"x": x}, gy, res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(7)
    lines = [FORMAT, f"torch {torch.__version__}"]

    # LayerNorm: ViT/DETR eps, CLIP eps, a row mean far from zero, eps 0 on
    # rows that have variance, a dimension of 1, and a leading-axes case.
    lines += norm_case("ln", "ln_basic", (5, 16), 1e-5, gen)
    lines += norm_case("ln", "ln_vit_eps", (2, 3, 24), 1e-6, gen)
    lines += norm_case("ln", "ln_mean_offset", (4, 32), 1e-5, gen, shift=1000.0, scale=0.5)
    lines += norm_case("ln", "ln_eps_zero", (3, 12), 0.0, gen)
    lines += norm_case("ln", "ln_dim_one", (6, 1), 1e-5, gen)
    lines += norm_case("ln", "ln_tiny_scale", (4, 20), 1e-6, gen, scale=1e-3)
    lines += norm_case("cln", "cln_nchw", (2, 6, 3, 4), 1e-6, gen)
    lines += norm_case("cln", "cln_nc", (3, 5), 1e-6, gen)
    lines += norm_case("cln", "cln_one_position", (1, 8, 1, 1), 1e-5, gen)
    lines += norm_case("cln", "cln_nlc", (2, 4, 9), 1e-5, gen, shift=-50.0)

    # GELU: a spread, the saturation edges (|x| near 6 * sqrt(2) = 8.5, where
    # erf saturates) and far tails.
    xs = torch.cat(
        [
            randn(gen, 64, scale=3.0),
            torch.tensor(
                [0.0, -0.0, 1e-8, -1e-8, 0.5, -0.5, 1.0, -1.0, 4.9, -4.9, 8.4, -8.4, 8.6, -8.6, 10.0, -10.0,
                 14.0, -14.0, 14.5, -14.5, 15.0, -15.0, 30.0, -30.0, 1e30, -1e30],
                dtype=torch.float32,
            ),
        ]
    )
    lines += gelu_case("gelu_spread", xs.reshape(10, 9), gen)

    # Linear with bias.
    lines += lin_case("lin_2d", (4, 7), 5, gen)
    lines += lin_case("lin_3d", (2, 3, 7), 4, gen)
    lines += lin_case("lin_one_row", (1, 9), 6, gen)

    # Broadcast add and multiply: a bias row, size-1 axes on both sides, a
    # scalar, a column, equal shapes.
    for op in ("badd", "bmul"):
        lines += bin_case(op, f"{op}_row", (3, 8), (8,), gen)
        lines += bin_case(op, f"{op}_both_stretch", (2, 1, 4), (1, 3, 1), gen)
        lines += bin_case(op, f"{op}_scalar", (), (5,), gen)
        lines += bin_case(op, f"{op}_column", (1,), (4, 1), gen)
        lines += bin_case(op, f"{op}_equal", (3, 4), (3, 4), gen)
        lines += bin_case(op, f"{op}_rank3_lead", (6, 2, 3), (2, 1), gen)

    # Softmax: ordinary rows, a peaked row, a dimension of 1, a wide row.
    lines += softmax_case("softmax_basic", randn(gen, 4, 10, scale=2.0), gen)
    lines += softmax_case("softmax_rank3", randn(gen, 2, 3, 7), gen)
    peaked = randn(gen, 3, 8)
    peaked[0, 0] = 100.0
    peaked[1, 3] = -100.0
    peaked[2] = 30.0
    lines += softmax_case("softmax_peaked", peaked, gen)
    lines += softmax_case("softmax_dim_one", randn(gen, 5, 1), gen)
    lines += softmax_case("softmax_wide", randn(gen, 2, 513, scale=3.0), gen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    cases = sum(1 for line in lines if line.startswith("case "))
    print(f"wrote {args.out} ({cases} cases)")


if __name__ == "__main__":
    main()
