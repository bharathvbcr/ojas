"""Golden for the padded `ojas_core::window_partition` / `window_merge`, computed with torch.

    python3.14 ojas-oracle/python/window_pad_fixture.py --out ojas-oracle/fixtures/window_pad/golden.txt

`window_partition` and `window_unpartition` below are SAM 3's own functions
(`sam3/model/vitdet.py`, the ViT trunk's windowed attention), copied without
change in their arithmetic: `F.pad(x, (0, 0, 0, pad_w, 0, pad_h))`, then
`view(B, Hp // w, w, Wp // w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, w, w, C)`,
and the inverse with the `[:, :H, :W, :]` crop. SAM's real shapes divide
(72 / 24), so the pad branch is exercised only here, on small shapes where
the height, the width, both, or neither does not divide the window, a window
is larger than the image, and an extent is 1.

The input is seeded normal noise, and the fixture records torch's windows and
torch's merge of them (which equals the input; the Rust side compares both).
Every case also checks, here, that the padded cells are +0.0 and that the
merge inverts the partition, so a fixture that disagrees with itself is never
written. Nothing in the comparison involves arithmetic, so the Rust side holds
the values to bit equality.

Format `ojas-window-pad-golden-v1`: a line-oriented text file. Each case is
`case <name> <batch> <height> <width> <channels> <window>`, then `x`,
`windows` and `merged`, each a list of f32 bit patterns in hex (row-major),
then `end`.

Reference oracle only (house policy): nothing here ships.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

FORMAT = "ojas-window-pad-golden-v1"

# (name, batch, height, width, channels, window)
CASES = [
    ("both_padded", 2, 5, 7, 3, 4),
    ("both_padded_square", 1, 7, 7, 2, 3),
    ("height_padded", 2, 5, 8, 3, 4),
    ("width_padded", 1, 8, 6, 2, 4),
    ("window_over_height", 1, 3, 9, 2, 5),
    ("window_over_width", 1, 9, 3, 2, 5),
    ("window_over_both", 2, 3, 2, 2, 5),
    ("height_one", 1, 1, 9, 2, 4),
    ("width_one", 2, 6, 1, 2, 3),
    ("one_by_one", 1, 1, 1, 1, 3),
    ("divides", 2, 4, 6, 3, 2),
    ("window_one", 1, 5, 5, 1, 1),
    ("window_is_image", 1, 4, 4, 1, 4),
]


def window_partition(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    """SAM 3 `sam3/model/vitdet.py::window_partition`."""
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> Tensor:
    """SAM 3 `sam3/model/vitdet.py::window_unpartition`."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def hex_bits(t: Tensor) -> str:
    flat = t.contiguous().reshape(-1).to(torch.float32).view(torch.int32)
    return " ".join(f"{int(v) & 0xFFFFFFFF:08x}" for v in flat)


def case_lines(index: int, name: str, b: int, h: int, w: int, c: int, win: int) -> list[str]:
    gen = torch.Generator().manual_seed(1000 + index)
    x = torch.randn(b, h, w, c, generator=gen, dtype=torch.float32)
    windows, pad_hw = window_partition(x, win)
    merged = window_unpartition(windows, win, pad_hw, (h, w))

    hp, wp = pad_hw
    assert windows.shape == (b * (hp // win) * (wp // win), win, win, c), name
    assert torch.equal(merged, x), f"{name}: merge does not invert the partition"
    # Every padded cell is +0.0: the windows hold exactly the input's values
    # plus zeros, and the zeros are positive.
    zeros = int((windows.reshape(-1).view(torch.int32) == 0).sum())
    assert zeros == windows.numel() - x.numel(), f"{name}: padding is not +0.0"
    return [
        f"case {name} {b} {h} {w} {c} {win}",
        f"x {hex_bits(x)}",
        f"windows {hex_bits(windows)}",
        f"merged {hex_bits(merged)}",
        "end",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lines = [FORMAT, f"torch {torch.__version__}"]
    for i, case in enumerate(CASES):
        lines += case_lines(i, *case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
