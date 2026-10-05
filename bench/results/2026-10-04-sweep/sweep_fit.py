"""Read a run_paired.sh output directory of the sweep_ rows and print markdown.

Throwaway analysis, never shipped. Usage: sweep_fit.py OUT_DIR

Per runtime and row: m = median over rounds of the per-round minimum, d =
median over rounds of the per-round median (the conventions of
docs/bench-gpu-vs-torch.md). Spread = max / min of the per-round medians.

silu: floor a = t(1); slope b from the two largest N; model crossover
N* = a / b (fixed part = per-value part). The direct crossover is where the
measured t(N) first reaches 2 * t(1), interpolated linearly between sweep
points; it needs no model.

decode: B requests batched (b) vs B batch-1 calls before one sync (x).
"""

import glob
import json
import os
import statistics
import sys

RUNTIMES = ("ojas-metal", "ojas-wgpu", "torch-mps")


def load(out):
    per = {}
    for path in sorted(glob.glob(os.path.join(out, "round*", "*.jsonl"))):
        r = int(os.path.basename(os.path.dirname(path))[5:])
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["row"].startswith("_"):
                continue
            per.setdefault((d["runtime"], d["row"]), {})[r] = d
    return per


def stats(per, rt, row):
    rounds = per.get((rt, row), {})
    bad = sorted({d["status"] for d in rounds.values() if d["status"] != "ok"})
    ok = [d for d in rounds.values() if d["status"] == "ok"]
    if bad or not ok:
        return {"status": ",".join(bad) or "missing", "n": len(ok)}
    mins = [d["min_ms"] for d in ok]
    meds = [d["median_ms"] for d in ok]
    return {"status": "ok", "n": len(ok), "m": statistics.median(mins),
            "d": statistics.median(meds), "spread": max(meds) / min(meds) - 1}


def f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def direct_crossover(pts, a):
    """First N where t(N) >= 2a, linear between sweep points."""
    for (n0, t0), (n1, t1) in zip(pts, pts[1:]):
        if t1 >= 2 * a > t0:
            return n0 + (2 * a - t0) * (n1 - n0) / (t1 - t0)
    return None


def main():
    out = sys.argv[1]
    per = load(out)
    rows = sorted({row for (_, row) in per})
    silu = sorted((int(r.split("_n")[1]), r) for r in rows if r.startswith("sweep_silu_n"))
    bs = sorted({int(r.split("_b")[1]) for r in rows if r.startswith("sweep_decode_b")})

    print("## silu_forward over N values\n")
    print("Cells: min-of-round-minimums median (m) / median-of-medians (d) ms, spread of per-round medians.\n")
    print("| N | " + " | ".join(RUNTIMES) + " | metal/torch d ratio |")
    print("| --: |" + " :-- |" * len(RUNTIMES) + " --: |")
    for n, row in silu:
        cells, s = [], {}
        for rt in RUNTIMES:
            s[rt] = stats(per, rt, row)
            st = s[rt]
            cells.append(f"{f(st['m'])} / {f(st['d'])} ({st['spread']:.0%})" if st["status"] == "ok" else st["status"])
        ratio = (s["torch-mps"]["d"] / s["ojas-metal"]["d"]) if s["torch-mps"]["status"] == s["ojas-metal"]["status"] == "ok" else None
        print(f"| {n} | " + " | ".join(cells) + f" | {f(ratio, 2)}x |")

    print("\n## Crossover where the per-value work equals the fixed cost\n")
    print("| runtime | stat | floor a = t(1) ms | slope b ns/value | model N* = a/b | direct N (t = 2a) |")
    print("| :-- | :-- | --: | --: | --: | --: |")
    for rt in RUNTIMES:
        for key in ("m", "d"):
            pts = []
            for n, row in silu:
                st = stats(per, rt, row)
                if st["status"] == "ok":
                    pts.append((n, st[key]))
            if len(pts) < 3 or pts[0][0] != 1:
                print(f"| {rt} | {key} | incomplete ({len(pts)} sizes) | | | |")
                continue
            a = pts[0][1]
            (n0, t0), (n1, t1) = pts[-2], pts[-1]
            b = (t1 - t0) / (n1 - n0)
            nstar = a / b if b > 0 else None
            dx = direct_crossover(pts, a)
            print(f"| {rt} | {key} | {f(a)} | {f(b * 1e6, 4)} | "
                  f"{'-' if nstar is None else f'{nstar / 1e6:.2f}M'} | "
                  f"{'-' if dx is None else f'{dx / 1e6:.2f}M'} |")

    print("\n## Decode: B requests batched (b) vs B batch-1 calls, one sync (x)\n")
    print("Cells: median-of-medians ms (spread). Per-request = cell / B. Extra per call = (x - b) / (B - 1).\n")
    print("| B | runtime | batched b | per request | split x | per request | extra per call |")
    print("| --: | :-- | --: | --: | --: | --: | --: |")
    for b in bs:
        for rt in RUNTIMES:
            sb = stats(per, rt, f"sweep_decode_b{b}")
            sx = stats(per, rt, f"sweep_decode_x{b}") if b > 1 else None
            cb = f"{f(sb['d'])} ({sb['spread']:.0%})" if sb["status"] == "ok" else sb["status"]
            pb = f(sb["d"] / b) if sb["status"] == "ok" else "-"
            if sx is None:
                cx, px, ex = "-", "-", "-"
            elif sx["status"] != "ok":
                cx, px, ex = sx["status"], "-", "-"
            else:
                cx, px = f"{f(sx['d'])} ({sx['spread']:.0%})", f(sx["d"] / b)
                ex = f((sx["d"] - sb["d"]) / (b - 1)) if sb["status"] == "ok" else "-"
            print(f"| {b} | {rt} | {cb} | {pb} | {cx} | {px} | {ex} |")


if __name__ == "__main__":
    main()
