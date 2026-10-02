"""Replay ojas BatchSampler window starts as torch (x, y) batches.

The starts come from `ojas-oracle/examples/dump_batch_starts.rs` (format
`ojas-batch-starts-v1`, see that example's doc comment). This module turns them
into the tensors nanolab's `Batcher.batch` would return for the same windows:
it reads the token bin the way `Batcher` does (a little-endian uint16 memmap,
`data[start:start + T + 1]`, int64, `x = seq[:, :-1]`, `y = seq[:, 1:]`;
nanolab/data.py:356-442), so only the choice of windows differs from nanolab.

    python3.14 ojas-oracle/python/batches.py verify --starts S.json --bin tokens.bin

`verify` checks the bin against the dump's length and FNV-1a 64 and compares
every replayed window, byte for byte, with the `rows` that ojas-data's
`TokenBin` reader produced in the dump. It exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

FORMAT = "ojas-batch-starts-v1"
_INT_FIELDS = ("bin_header_bytes", "bin_tokens", "seed", "seq_len", "batch",
               "accum", "steps", "windows_per_epoch")


def fnv1a64(path: Path) -> int:
    h = 0xCBF29CE484222325
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            for byte in chunk:
                h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def load_starts(path: Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT:
        raise SystemExit(f"{path}: format is not {FORMAT}")
    for key in _INT_FIELDS:
        if not isinstance(doc.get(key), int) or doc[key] < 0:
            raise SystemExit(f"{path}: {key} is not a non-negative integer")
    n = doc["steps"] * doc["accum"] * doc["batch"]
    if len(doc["starts"]) != n:
        raise SystemExit(f"{path}: {len(doc['starts'])} starts, expected {n}")
    if "rows" in doc and len(doc["rows"]) != n * (doc["seq_len"] + 1):
        raise SystemExit(f"{path}: rows length disagrees with starts")
    return doc


class StartReplay:
    """Batches at the dumped starts, from the bin the dump was made from."""

    def __init__(self, starts: dict, bin_path: Path):
        self.doc = starts
        self.t = starts["seq_len"]
        self.b = starts["batch"]
        self.k = starts["accum"]
        self.steps = starts["steps"]
        header = starts["bin_header_bytes"]
        data = np.memmap(bin_path, dtype="<u2", mode="r", offset=header)
        if len(data) != starts["bin_tokens"]:
            raise SystemExit(f"{bin_path}: {len(data)} tokens, the dump says "
                             f"{starts['bin_tokens']}")
        digest = f"{fnv1a64(bin_path):016x}"
        if digest != starts["bin_fnv1a64"]:
            raise SystemExit(f"{bin_path}: FNV-1a {digest} != dump {starts['bin_fnv1a64']}")
        self.data = data
        self.pos = 0  # next micro-batch, for the nanolab Batcher contract

    def micro(self, step: int, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= step < self.steps and 0 <= micro < self.k):
            raise IndexError(f"step {step} micro {micro} outside the dump")
        first = (step * self.k + micro) * self.b
        width = self.t + 1
        buf = np.empty((self.b, width), dtype=np.uint16)
        for row, start in enumerate(self.doc["starts"][first:first + self.b]):
            window = self.data[start:start + width]
            if len(window) != width:
                raise SystemExit(f"start {start} runs past the bin")
            buf[row] = window
        seq = torch.from_numpy(np.asarray(buf, dtype=np.int64))
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    # nanolab Batcher contract (data.py:412), so `train(cfg, batchers=...)`
    # consumes the dump in order: one call per micro-batch.
    def batch(self, block_size=None, frontier=1.0):
        if (block_size or self.t) != self.t or frontier != 1.0:
            raise SystemExit("replay supports neither curriculum nor frontier")
        if self.pos >= self.steps * self.k:
            raise SystemExit("the dump has no more micro-batches")
        step, micro = divmod(self.pos, self.k)
        self.pos += 1
        return self.micro(step, micro)

    def state_dict(self) -> dict:
        return {"pos": self.pos}

    def load_state_dict(self, state: dict) -> None:
        self.pos = int(state["pos"])


def verify(starts_path: Path, bin_path: Path) -> int:
    doc = load_starts(starts_path)
    if "rows" not in doc:
        raise SystemExit("the dump has no rows; regenerate it with --rows")
    replay = StartReplay(doc, bin_path)
    width = replay.t + 1
    rows = np.asarray(doc["rows"], dtype=np.int64).reshape(-1, replay.b, width)
    checked = 0
    for step in range(replay.steps):
        for micro in range(replay.k):
            x, y = replay.micro(step, micro)
            want = torch.from_numpy(rows[step * replay.k + micro])
            if not (torch.equal(x, want[:, :-1]) and torch.equal(y, want[:, 1:])):
                raise SystemExit(f"step {step} micro {micro}: replay differs from ojas TokenBin")
            checked += 1
    return checked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--starts", type=Path, required=True)
    v.add_argument("--bin", type=Path, required=True)
    args = ap.parse_args()
    n = verify(args.starts, args.bin)
    print(f"{n} micro-batches identical to ojas TokenBin rows", file=sys.stderr)


if __name__ == "__main__":
    main()
