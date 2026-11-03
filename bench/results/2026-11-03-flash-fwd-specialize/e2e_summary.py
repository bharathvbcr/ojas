"""Summary of job 2 (rules.md, "End to end"): per round, the steady-frame median of SpikeTrack's
memory-attention stage and of the tracker step, A against the winner.

Usage: python3 -I e2e_summary.py <e2e dir> <winner label>. Steady = frame >= 8, as the CoreML A/B.
"""

import statistics
import sys
from pathlib import Path

STEADY_FROM = 8
MATTN = "tracker propagate, state 0: memory attention (device)"


def stage_rows(md):
    rows, inside = [], False
    for line in md.read_text().splitlines():
        if line == "```tsv":
            inside = True
            continue
        if inside and line == "```":
            break
        if inside and line != "frame\tstage\tms":
            f, stage, ms = line.split("\t")
            rows.append((int(f), stage, float(ms)))
    if not rows:
        raise SystemExit(f"{md}: no per-frame tsv block")
    return rows


def steady(d):
    rows = stage_rows(d / "82.profile-stages.md")
    if any("state 1" in s for _, s, _ in rows):
        raise SystemExit(f"{d}: a second tracker state ran")
    frames = sorted({f for f, _, _ in rows if f >= STEADY_FROM})
    mattn = {f: 0.0 for f in frames}
    step = {f: 0.0 for f in frames}
    for f, s, ms in rows:
        if f not in mattn:
            continue
        if s == MATTN:
            mattn[f] += ms
        if s.startswith("tracker propagate") or s.startswith("memory encode"):
            step[f] += ms
    missing = [f for f in frames if mattn[f] == 0.0]
    if missing:
        raise SystemExit(f"{d}: frames {missing} have no '{MATTN}' row")
    return statistics.median(mattn.values()), statistics.median(step.values()), len(frames)


def main():
    e, win = Path(sys.argv[1]), sys.argv[2]
    out = {}
    for r in (1, 2, 3):
        for arm in ("A", win):
            out[(r, arm)] = steady(e / f"r{r}-{arm}")
    print(f"# e2e: SpikeTrack 38e0a35, ojas A (286a444 + probe) vs {win}; steady frames >= {STEADY_FROM}")
    for i, name in ((0, "memory attention (device)"), (1, "tracker step")):
        a = [out[(r, "A")][i] for r in (1, 2, 3)]
        w = [out[(r, win)][i] for r in (1, 2, 3)]
        noise = max(max(a) / min(a) - 1, max(w) / min(w) - 1)
        ratios = [wi / ai for wi, ai in zip(w, a)]
        if all(x < 1 - noise for x in ratios):
            verdict = "faster"
        elif all(x > 1 + noise for x in ratios):
            verdict = "slower"
        else:
            verdict = "not measured (inside the noise)"
        print(f"{name}: A {' / '.join(f'{x:.1f}' for x in a)} ms; {win} {' / '.join(f'{x:.1f}' for x in w)} ms; "
              f"ratio {' / '.join(f'{x:.3f}' for x in ratios)}; noise {noise:.3f}; {verdict}")
    print("frames per run: " + ", ".join(f"r{r}-{arm} {out[(r, arm)][2]}" for r, arm in sorted(out)))


if __name__ == "__main__":
    main()
