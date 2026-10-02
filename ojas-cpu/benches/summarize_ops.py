"""Summarize interleaved bench_ops.rs / torch_ops.py rounds into one table.

Each round log holds the OJAS_OP lines of one bench_ops run or the TORCH_OP
lines of one torch_ops.py run. The parity log holds PARITY and PARITY_F64
lines from `torch_ops.py --parity`. Per row the statistic is the minimum over
rounds of each round's min-of-N; spread is (max round min - min round min) /
min round min, flagged above 15%. Ratio is torch / ojas, so below 1 means ojas
is slower. A row with no parity line prints MISSING.

House policy: Python here is analysis only; it ships nothing.

    python3 ojas-cpu/benches/summarize_ops.py --parity parity.txt round*.txt
"""

import argparse
from collections import defaultdict

PRIMARY = {"sdpa": "math", "block": "math"}
ALT = {"sdpa": "default", "block": "default", "adamw_768x768": "fused",
       "adamw_3072x768": "fused", "adamw_50304x768": "fused"}
SPREAD_FLAG = 0.15

# How many times each op row runs in one block step as bench_ops.rs composes
# it (forward plus the hand-written backward). Residual-gradient sums in the
# backward use add.fwd.
BLOCK_FWD = {("rms_norm", "fwd"): 2, ("linear_qkvo", "fwd"): 4, ("linear_up", "fwd"): 2,
             ("linear_down", "fwd"): 1, ("qk_norm", "fwd"): 1, ("rope", "fwd"): 2,
             ("vres", "fwd"): 1, ("permute", "fwd"): 4, ("sdpa", "fwd"): 1, ("gate", "fwd"): 1,
             ("add", "fwd"): 2, ("silu", "fwd"): 1, ("mul", "fwd"): 1}
BLOCK_BWD = {("rms_norm", "bwd"): 2, ("linear_qkvo", "bwd"): 4, ("linear_up", "bwd"): 2,
             ("linear_down", "bwd"): 1, ("qk_norm", "bwd"): 1, ("rope", "bwd"): 2,
             ("vres", "bwd"): 1, ("permute", "fwd"): 4, ("sdpa", "bwd"): 1, ("gate", "bwd"): 1,
             ("add", "fwd"): 6, ("silu", "bwd"): 1, ("mul", "bwd"): 1}


def kv(line):
    out = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse_rounds(paths):
    ojas = defaultdict(list)    # (op, dir, threads) -> [round mins]
    torch = defaultdict(list)   # (op, dir, variant) -> [round mins]
    shapes, order = {}, []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.startswith("OJAS_OP "):
                    d = kv(line)
                    key = (d["op"], d["dir"])
                    if key not in order:
                        order.append(key)
                    shapes[key] = d["shape"]
                    ojas[(d["op"], d["dir"], int(d["threads"]))].append(float(d["min_ms"]))
                elif line.startswith("TORCH_OP "):
                    d = kv(line)
                    torch[(d["op"], d["dir"], d["variant"])].append(float(d["min_ms"]))
    return ojas, torch, shapes, order


def parse_parity(path):
    rows = defaultdict(list)    # (op, dir, variant) -> [(out, ok, err, tol)]
    f64 = {}                    # (op, dir, out) -> dict
    if not path:
        return rows, f64
    with open(path) as f:
        for line in f:
            if line.startswith("PARITY "):
                d = kv(line)
                err = float(d.get("norm_err", "nan"))
                rows[(d["op"], d["dir"], d["variant"])].append(
                    (d["out"], d["ok"] == "True", err, d["tol"]))
            elif line.startswith("PARITY_F64 "):
                d = kv(line)
                f64[(d["op"], d["dir"], d["out"])] = d
    return rows, f64


def parity_cell(op, d, variant, rows, f64):
    lines = rows.get((op, d, variant))
    if not lines:
        return "MISSING"
    worst = max(err for _, _, err, _ in lines)
    tol = lines[0][3]
    bad = [(o, err) for o, ok, err, _ in lines if not ok]
    if not bad:
        return f"pass {worst:.1e} (tol {tol})"
    notes = []
    # Torch errors an f64 arbiter already pinned on torch: an output scaled by
    # that torch value (clip's gradients) inherits the same relative error.
    torch_off = [float(a["torch_err"]) for (o2, d2, _), a in f64.items()
                 if o2 == op and d2 == d and a["ojas_ok"] == "True"]
    for o, err in bad:
        arb = f64.get((op, d, o))
        if not arb and any(abs(err - te) <= 0.01 * te for te in torch_off):
            notes.append(f"{o}: {err:.1e}, inherits torch's f64 miss above")
        elif arb and arb["ojas_ok"] == "True":
            notes.append(f"{o}: torch off f64 by {float(arb['torch_err']):.1e}, ojas {float(arb['ojas_err']):.1e}")
        elif arb:
            notes.append(f"FAIL {o}: ojas off f64 by {float(arb['ojas_err']):.1e} (torch {float(arb['torch_err']):.1e})")
        else:
            notes.append(f"FAIL {o} {err:.1e}")
    status = "FAIL" if any(n.startswith("FAIL") for n in notes) else "pass vs f64"
    return f"{status} (tol {tol}); " + "; ".join(notes)


def best(series):
    if not series:
        return None, None
    lo, hi = min(series), max(series)
    return lo, (hi - lo) / lo if lo > 0 else 0.0


def fmt(ms):
    if ms is None:
        return "—"
    return f"{ms:.3f}" if ms < 10 else f"{ms:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", default="")
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()
    ojas, torch, shapes, order = parse_rounds(args.logs)
    prow, f64 = parse_parity(args.parity)
    threads = sorted({t for (_, _, t) in ojas})
    print(f"rounds: ojas {max((len(v) for v in ojas.values()), default=0)}, "
          f"torch {max((len(v) for v in torch.values()), default=0)}; times in ms, min over rounds")
    head = ["op", "dir", "shape"] + [f"ojas {t}t" for t in threads] + ["torch"]
    head += [f"ratio {t}t" for t in threads] + ["torch alt", "spread>15%", "parity"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    mins = {}
    for op, d in order:
        variant = PRIMARY.get(op, "default")
        flags = []
        cells = []
        for t in threads:
            m, s = best(ojas.get((op, d, t), []))
            mins[(op, d, t)] = m
            cells.append(fmt(m))
            if s is not None and s > SPREAD_FLAG:
                flags.append(f"ojas{t}t {s:.0%}")
        tm, ts = best(torch.get((op, d, variant), []))
        mins[(op, d, "torch")] = tm
        if ts is not None and ts > SPREAD_FLAG:
            flags.append(f"torch {ts:.0%}")
        ratios = []
        for t in threads:
            om = mins[(op, d, t)]
            ratios.append(f"{tm / om:.2f}" if tm and om else "—")
        alt = ALT.get(op)
        am, _ = best(torch.get((op, d, alt), [])) if alt else (None, None)
        alt_cell = f"{fmt(am)} ({alt})" if am is not None else "—"
        tcell = fmt(tm) + (f" ({variant})" if variant != "default" else "")
        row = [op, d, shapes[(op, d)], *cells, tcell, *ratios, alt_cell,
               ", ".join(flags) or "no", parity_cell(op, d, variant, prow, f64)]
        print("| " + " | ".join(row) + " |")

    for t in threads + ["torch"]:
        f = sum(n * (mins.get((o, d, t)) or 0) for (o, d), n in BLOCK_FWD.items())
        b = sum(n * (mins.get((o, d, t)) or 0) for (o, d), n in BLOCK_BWD.items())
        fwd = mins.get(("block", "fwd", t))
        step = mins.get(("block", "step", t))
        label = f"ojas {t}t" if t != "torch" else "torch"
        print(f"block, {label}: sum of op rows fwd {f:.2f} + bwd {b:.2f} = {f + b:.2f} ms; "
              f"measured fwd {fmt(fwd)}, step {fmt(step)}, step - fwd {fmt(step - fwd) if fwd and step else '—'}")


if __name__ == "__main__":
    main()
