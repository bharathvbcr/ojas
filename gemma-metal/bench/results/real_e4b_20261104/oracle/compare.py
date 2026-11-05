"""Throwaway: per layer and position, relative L2 of native vs oracle hidden; prints the table
and the first (pos, layer) over the threshold.

usage: python -I compare.py ORACLE.json NATIVE.json [threshold]
"""
import json
import math
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
assert a["prompt"] == b["prompt"], (a["prompt"], b["prompt"])
print("argmax oracle", a["argmax"], "native", b["argmax"])
first = None
for p in range(len(a["hidden"])):
    row = []
    for l in range(len(a["hidden"][p])):
        x, y = a["hidden"][p][l], b["hidden"][p][l]
        num = math.sqrt(sum((u - v) ** 2 for u, v in zip(x, y)))
        den = math.sqrt(sum(u * u for u in x)) or 1.0
        r = num / den
        row.append(r)
        if first is None and r > thr:
            first = (p, l, r, x[:6], y[:6])
    print(f"pos {p}: " + " ".join(f"{r:.3f}" for r in row))
print("first over", thr, ":", first)
