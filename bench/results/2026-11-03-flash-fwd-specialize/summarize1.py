"""Job 1's verdicts by rules.md: per-round ratios against A, noise N from A2/A, control bar.

Usage: python3 -I summarize1.py <time.tsv>
"""

import statistics
import sys

rows = {}
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("\t")
    r, label, arm = f[0], f[2], f[3]
    stat = dict(x.split(" ", 1) for x in f[4:])
    rows[(r, label, arm)] = stat
rounds = sorted({k[0] for k in rows}, key=lambda s: int(s[1:]))


def med(r, label, arm):
    return float(rows[(r, label, arm)]["median"])


def mn(r, label, arm):
    return float(rows[(r, label, arm)]["min"])


bad = [k for k, s in rows.items() if k[2] != "ctl" and s["flash_d32"] != "[18, 0]"]
bad += [k for k, s in rows.items() if k[2] == "ctl" and s["flash_d32"] != "[0, 0]"]
print(f"kernel identity (gate 2): {'OK' if not bad else 'BROKEN ' + str(bad)}")

print("\ncontrol (gate 3): ctl median ratio over A, per round; median over rounds")
repeat = False
for v in ("B", "C", "A2"):
    rs = [med(r, v, "ctl") / med(r, "A", "ctl") for r in rounds]
    m = statistics.median(rs)
    ok = 0.95 <= m <= 1.05
    repeat |= not ok
    print(f"  {v}: {' '.join(f'{x:.3f}' for x in rs)}  median {m:.3f}  {'in bar' if ok else 'OUT OF BAR'}")
print(f"  window: {'repeat (CONTENDED)' if repeat else 'clean'}")

for arm in ("x36400", "x15600"):
    n = max(abs(med(r, "A2", arm) / med(r, "A", arm) - 1) for r in rounds)
    print(f"\n{arm}: A medians {' '.join(f'{med(r, 'A', arm):.2f}' for r in rounds)}")
    print(f"  noise N (max |A2/A - 1|) = {n:.3f}")
    for v in ("B", "C", "A2"):
        rs = [med(r, v, arm) / med(r, "A", arm) for r in rounds]
        if all(x < 1 - n for x in rs):
            verdict = "faster"
        elif all(x > 1 + n for x in rs):
            verdict = "slower"
        else:
            verdict = "not measured (inside the noise)"
        mins = min(mn(r, v, arm) for r in rounds) / min(mn(r, "A", arm) for r in rounds)
        print(f"  {v}/A: {' '.join(f'{x:.3f}' for x in rs)}  median {statistics.median(rs):.3f} "
              f"(range {min(rs):.3f}-{max(rs):.3f}); min-of-mins ratio {mins:.3f}; {verdict}")
