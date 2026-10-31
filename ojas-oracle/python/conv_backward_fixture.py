"""Golden for the convolution backward ops, computed with torch autograd.

    /Users/bharath/.venvs/torch/bin/python -I ojas-oracle/python/conv_backward_fixture.py \
        --out ojas-oracle/fixtures/conv_bwd/golden.txt

One case per SAM configuration: the patch embedding (kernel 14, stride 14),
1 x 1, 3 x 3 with padding 1, the strided 3 x 3 (stride 2, padding 1), the
depthwise 7 x 7 (padding 3), and the transposed 2 x 2 stride 2, each with and
without a bias. Each case runs `F.conv2d` / `F.conv_transpose2d` forward and
backward in torch on the CPU, once in float32 (what torch computes) and once
in float64 from the same float32 inputs (the value torch's float32 is an
approximation of). The Rust side (`ojas-metal/tests/conv_backward.rs`)
compares the gradients of the CPU and Metal backends with both.

Every case also runs `torch.autograd.gradcheck` on the float64 function (a
central difference against the analytic backward), so a fixture whose
gradients torch itself cannot reproduce numerically is never written.

Format `ojas-conv-bwd-golden-v1`: a line-oriented text file, the layout of
`ojas-encoder-bwd-golden-v1`. A case is `case <op> <name> <key=value ...>`
(`stride=h,w padding=h,w groups=n bias=0|1`), then one `t <name>
<f32|f64> <dims, comma separated> <hex...>` line per tensor (f32 bit patterns
of 8 hex digits, f64 of 16, row-major), then `end`. The tensors are the
inputs `x`, `w`, `b` (with a bias), `gy`, the forward output `y`, and the
gradients `dx`, `dw`, `db`, each also with a `_64` suffix for the float64 run.

Reference oracle only (house policy): nothing here ships.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

FORMAT = "ojas-conv-bwd-golden-v1"


def hexbits(t: Tensor) -> str:
    flat = t.detach().contiguous().reshape(-1)
    if flat.dtype == torch.float32:
        return " ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in flat.view(torch.int32))
    assert flat.dtype == torch.float64
    return " ".join(f"{int(v) & 0xFFFFFFFFFFFFFFFF:016x}" for v in flat.view(torch.int64))


def tline(name: str, t: Tensor) -> str:
    dtype = "f32" if t.dtype == torch.float32 else "f64"
    dims = ",".join(str(d) for d in t.shape)
    return f"t {name} {dtype} {dims} {hexbits(t)}"


def randn(gen: torch.Generator, *shape: int, scale: float = 1.0) -> Tensor:
    return torch.randn(shape, generator=gen, dtype=torch.float32) * scale


def pair(v: tuple[int, int]) -> str:
    return f"{v[0]},{v[1]}"


# name, op, input [n, c, h, w], weight shape, stride, padding, groups
CASES = [
    ("patch_embed_k14_s14", "conv2d", (1, 2, 28, 28), (3, 2, 14, 14), (14, 14), (0, 0), 1),
    ("pointwise_1x1", "conv2d", (2, 6, 5, 7), (4, 6, 1, 1), (1, 1), (0, 0), 1),
    ("conv3x3_p1", "conv2d", (2, 5, 6, 7), (4, 5, 3, 3), (1, 1), (1, 1), 1),
    ("conv3x3_s2_p1", "conv2d", (2, 4, 9, 8), (6, 4, 3, 3), (2, 2), (1, 1), 1),
    ("depthwise_7x7_p3", "conv2d", (2, 5, 10, 9), (5, 1, 7, 7), (1, 1), (3, 3), 5),
    ("convt_2x2_s2", "conv_transpose2d", (2, 5, 4, 3), (5, 3, 2, 2), (2, 2), (0, 0), 1),
]


def forward(op, stride, padding, groups):
    if op == "conv2d":
        return lambda x, w, b: F.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)
    return lambda x, w, b: F.conv_transpose2d(x, w, b, stride=stride)


def run(fn, inputs: dict[str, Tensor], gy: Tensor):
    """`fn` forward in float32 and float64 and its backward with `gy`."""
    out = {}
    for tag, dtype in (("", torch.float32), ("_64", torch.float64)):
        leaves = {k: v.detach().to(dtype).requires_grad_(True) for k, v in inputs.items()}
        y = fn(leaves["x"], leaves["w"], leaves.get("b"))
        y.backward(gy.to(dtype))
        out["y" + tag] = y.detach()
        for k, leaf in leaves.items():
            out["d" + k + tag] = leaf.grad.detach()
    return out


def gradchecked(fn, inputs: dict[str, Tensor], gy: Tensor) -> None:
    """Central-difference check of the float64 backward on the loss `sum(gy * y)`."""
    keys = list(inputs)
    leaves = [inputs[k].detach().to(torch.float64).requires_grad_(True) for k in keys]
    gy64 = gy.to(torch.float64)

    def loss(*args):
        named = dict(zip(keys, args))
        return (fn(named["x"], named["w"], named.get("b")) * gy64).sum()

    assert torch.autograd.gradcheck(loss, tuple(leaves), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gen = torch.Generator().manual_seed(20261031)
    lines = [FORMAT, f"torch {torch.__version__}"]
    for name, op, xs, ws, stride, padding, groups in CASES:
        out_ch = ws[0] if op == "conv2d" else ws[1]
        for with_bias in (False, True):
            x = randn(gen, *xs)
            w = randn(gen, *ws, scale=0.2)
            ins = {"x": x, "w": w}
            if with_bias:
                ins["b"] = randn(gen, out_ch)
            fn = forward(op, stride, padding, groups)
            y_shape = fn(x, w, ins.get("b")).shape
            gy = randn(gen, *y_shape)
            gradchecked(fn, ins, gy)
            res = run(fn, ins, gy)
            case = f"{name}_bias" if with_bias else name
            lines.append(
                f"case {op} {case} stride={pair(stride)} padding={pair(padding)} "
                f"groups={groups} bias={int(with_bias)}"
            )
            for k, v in ins.items():
                lines.append(tline(k, v))
            lines.append(tline("gy", gy))
            for k, v in res.items():
                lines.append(tline(k, v))
            lines.append("end")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(CASES) * 2} cases")


if __name__ == "__main__":
    main()
