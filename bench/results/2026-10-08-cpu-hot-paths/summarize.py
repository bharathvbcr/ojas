"""Summarize run.sh output: per row, the minimum over rounds of each lane's
per-round minimum, the after/base ratio by that minimum and by the median of
the per-round ratios, and after/torch where torch has the row.

Analysis only (house policy: Python never ships).

    python3 summarize.py OUT_DIR > summary.md
"""

import glob
import os
import re
import statistics
import sys


def rows(path, prefix):
    out = {}
    for line in open(path):
        if not line.startswith(prefix):
            continue
        f = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
        if prefix == "OJAS_TAPE":
            key = ("tape", f["row"])
        else:
            key = (f["op"], f["dir"] + ("/" + f["variant"] if "variant" in f else ""))
        out[key] = float(f["min_ms"])
    return out


def main(out):
    lanes = {}
    for path in glob.glob(os.path.join(out, "*-*-*.txt")):
        m = re.match(r"(ops|tape)-(\d+)-(\w+)\.txt", os.path.basename(path))
        if not m:
            continue
        kind, rnd, lane = m.group(1), int(m.group(2)), m.group(3)
        prefix = {"ops": "TORCH_OP" if lane == "torch" else "OJAS_OP", "tape": "OJAS_TAPE"}[kind]
        lanes.setdefault(lane, {}).setdefault(rnd, {}).update(rows(path, prefix))
    rounds = sorted(lanes.get("after", {}))
    keys = sorted({k for r in lanes.get("after", {}).values() for k in r})
    print(f"rounds={len(rounds)}; cells are ms, min over rounds of each round's min-of-N\n")
    print("| row | dir | base | after | after/base (min) | after/base (median of rounds) | torch | after/torch |")
    print("| :-- | :-- | --: | --: | --: | --: | --: | --: |")
    for key in keys:
        b = [lanes["base"][r].get(key) for r in rounds if r in lanes.get("base", {})]
        a = [lanes["after"][r].get(key) for r in rounds]
        per = [x / y for x, y in zip(a, b) if x is not None and y is not None]
        bmin = min([v for v in b if v is not None], default=None)
        amin = min([v for v in a if v is not None], default=None)
        tkey = key
        if key[0] == "adamw_50304x768":
            tkey = (key[0], key[1] + "/fused")
        t = [lanes["torch"][r].get(tkey) for r in rounds if r in lanes.get("torch", {})]
        if all(v is None for v in t):
            t = [lanes["torch"][r].get((key[0], key[1] + "/default")) for r in rounds if r in lanes.get("torch", {})]
        tmin = min([v for v in t if v is not None], default=None)
        fmt = lambda v: "" if v is None else f"{v:.4f}"
        ratio = lambda x, y: "" if x is None or y is None else f"{x / y:.2f}"
        med = f"{statistics.median(per):.2f}" if per else ""
        print(f"| {key[0]} | {key[1]} | {fmt(bmin)} | {fmt(amin)} | {ratio(amin, bmin)} | {med} | {fmt(tmin)} | {ratio(amin, tmin)} |")


if __name__ == "__main__":
    main(sys.argv[1])
