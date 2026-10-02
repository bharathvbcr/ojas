"""Torch CPU oracle and timer for ojas-cpu/tests/bench_ops.rs.

Reads the inputs and the ojas outputs that `OJAS_BENCH_MODE=dump` wrote under
--dir, checks torch against them (--parity), and times the same ops with the
same statistic as the Rust side: two warmup calls, then min-of-N (--time).
Forward rows run under no_grad. Backward rows run the forward untimed each
call, then time `torch.autograd.grad`, so nothing accumulates into `.grad`.

House policy: Python here is only the torch oracle; it ships nothing.

    python3 ojas-cpu/benches/torch_ops.py --threads 6 --parity
    python3 ojas-cpu/benches/torch_ops.py --threads 6 --time
"""

import argparse
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

T, D, NH, HD, FF, V = 1024, 768, 12, 64, 2048, 50304
EPS = 1e-6

# Normalized tolerance: max|torch - ojas| / max|torch| must not exceed this.
# Chosen before the first comparison, by the arithmetic each op does.
TOL = {
    "linear_qkvo": 1e-5, "linear_up": 1e-5, "linear_down": 1e-5, "linear_lmhead": 1e-5,
    "linear_dec_qkvo": 1e-5, "linear_dec_up": 1e-5, "linear_dec_down": 1e-5,
    "embedding": 0.0, "permute": 0.0,
    "rms_norm": 1e-5, "qk_norm": 1e-5, "rope": 1e-6, "sdpa": 1e-5, "gate": 1e-5,
    "vres": 1e-6, "silu": 1e-6, "mul": 1e-6, "add": 0.0, "ce": 1e-5,
    "adamw_768x768": 1e-5, "adamw_3072x768": 1e-5, "adamw_50304x768": 1e-5,
    "muon_768x768": 1e-3, "muon_2048x768": 1e-3, "muon_3072x768": 1e-3,
    "clip": 1e-5, "block": 1e-4,
}

# Calls per measurement, the same as bench_ops.rs.
N = {
    "linear_lmhead": 5, "linear_dec_qkvo": 200, "linear_dec_up": 200, "linear_dec_down": 200,
    "embedding.bwd": 5, "ce": 5, "sdpa": 10, "block": 10,
    "adamw_768x768": 10, "adamw_3072x768": 10, "adamw_50304x768": 5,
    "muon_768x768": 10, "muon_2048x768": 5, "muon_3072x768": 5, "clip": 5,
}


def calls(op, d):
    return N.get(f"{op}.{d}", N.get(op, 20))


def load(case_dir):
    inputs, outs = {}, {}
    with open(os.path.join(case_dir, "manifest.txt")) as f:
        for line in f:
            parts = line.split()
            if parts[0] == "case":
                continue
            kind, name, dtype, shape = parts
            shape = () if shape == "_" else tuple(int(s) for s in shape.split(","))
            raw = np.fromfile(os.path.join(case_dir, f"{kind}.{name}.bin"),
                              dtype="<f4" if dtype == "f32" else "<u4").reshape(shape)
            if dtype == "u32":
                t = torch.from_numpy(raw.astype(np.int64))
            else:
                t = torch.from_numpy(raw.copy())
            (inputs if kind == "in" else outs)[name] = t
    return inputs, outs


def nograd(f):
    def timed(_state):
        with torch.no_grad():
            return f()
    return None, timed


def backward(f, diff, grads, names):
    """Untimed forward `f(*leaves)` with grad, then timed autograd.grad."""
    def prep():
        leaves = [t.detach().requires_grad_(True) for t in diff]
        ys = f(*leaves)
        ys = list(ys) if isinstance(ys, tuple) else [ys]
        return leaves, ys

    def timed(state):
        leaves, ys = state
        gs = torch.autograd.grad(ys, leaves, grads)
        return dict(zip(names, gs))
    return prep, timed


def rms_d(x, w):
    return F.rms_norm(x, (D,), w, eps=EPS)


def qk_norm(q, k, wq, wk):
    return F.rms_norm(q, (HD,), wq, eps=EPS), F.rms_norm(k, (HD,), wk, eps=EPS)


def rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos[None, :, None, :] + rot * sin[None, :, None, :]


def gate(x, w, b, attn):
    return attn * torch.sigmoid(F.linear(x, w, b)).unsqueeze(-1)


def vres(v, v0, lam):
    s = torch.sigmoid(lam)
    return (1 - s) * v + s * v0


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def ns5(update):
    a, b, c = 3.4445, -4.7750, 2.0315
    rows, cols = update.shape
    x = update.T if rows > cols else update
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        am = x @ x.T
        bm = b * am + c * (am @ am)
        x = a * x + bm @ x
    return x.T if rows > cols else x


MUON_LR, MUON_MOM, MUON_WD = 0.025, 0.99, 0.1


def muon_step(p, g, buf):
    """nanolab Muon.step in f32, the ojas CPU contract (optim.rs)."""
    buf.mul_(MUON_MOM).add_(g)
    upd = g.add(buf, alpha=MUON_MOM)
    o = ns5(upd)
    rows, cols = p.shape
    p.mul_(1 - MUON_LR * MUON_WD)
    p.add_(o, alpha=-MUON_LR * max(1.0, rows / cols) ** 0.5)


def cases(op, inp):
    """{(dir, variant): (prep, timed)} for one case."""
    if op.startswith("linear_"):
        x, w, gy = inp["x"], inp["w"], inp["gy"]
        return {
            ("fwd", "default"): nograd(lambda: {"y": F.linear(x, w)}),
            ("bwd", "default"): backward(F.linear, [x, w], [gy], ["gx", "gw"]),
        }
    if op == "embedding":
        table, ids, gy = inp["table"], inp["ids"], inp["gy"]
        return {
            ("fwd", "default"): nograd(lambda: {"y": F.embedding(ids, table)}),
            ("bwd", "default"): backward(lambda t: F.embedding(ids, t), [table], [gy], ["gtable"]),
        }
    if op == "rms_norm":
        x, w, gy = inp["x"], inp["w"], inp["gy"]
        return {
            ("fwd", "default"): nograd(lambda: {"y": rms_d(x, w)}),
            ("bwd", "default"): backward(rms_d, [x, w], [gy], ["gx", "gw"]),
        }
    if op == "qk_norm":
        q, k, wq, wk = inp["q"], inp["k"], inp["wq"], inp["wk"]
        return {
            ("fwd", "default"): nograd(lambda: dict(zip(["qn", "kn"], qk_norm(q, k, wq, wk)))),
            ("bwd", "default"): backward(qk_norm, [q, k, wq, wk], [inp["gq"], inp["gk"]],
                                         ["gq", "gk", "gwq", "gwk"]),
        }
    if op == "rope":
        x, gy, cos, sin = inp["x"], inp["gy"], inp["cos"], inp["sin"]
        return {
            ("fwd", "default"): nograd(lambda: {"y": rope(x, cos, sin)}),
            ("bwd", "default"): backward(lambda x: rope(x, cos, sin), [x], [gy], ["gx"]),
        }
    if op == "sdpa":
        q, k, v, gy = inp["q"], inp["k"], inp["v"], inp["gy"]
        out = {}
        for variant in ("math", "default"):
            def f(q, k, v, variant=variant):
                if variant == "math":
                    with sdpa_kernel(SDPBackend.MATH):
                        return sdpa(q, k, v)
                return sdpa(q, k, v)
            fwd = nograd(lambda f=f: {"y": f(q, k, v)})
            prep, timed = backward(f, [q, k, v], [gy], ["gq", "gk", "gv"])
            if variant == "math":
                # The math backend's backward runs where its graph was recorded;
                # pin it for the backward call too.
                def timed_math(state, timed=timed):
                    with sdpa_kernel(SDPBackend.MATH):
                        return timed(state)
                timed = timed_math
            out[("fwd", variant)] = fwd
            out[("bwd", variant)] = (prep, timed)
        return out
    if op == "gate":
        args = [inp["x"], inp["w"], inp["b"], inp["attn"]]
        return {
            ("fwd", "default"): nograd(lambda: {"y": gate(*args)}),
            ("bwd", "default"): backward(gate, args, [inp["gy"]], ["gx", "gw", "gb", "gattn"]),
        }
    if op == "vres":
        args = [inp["v"], inp["v0"], inp["lam"]]
        return {
            ("fwd", "default"): nograd(lambda: {"y": vres(*args)}),
            ("bwd", "default"): backward(vres, args, [inp["gy"]], ["gv", "gv0", "glam"]),
        }
    if op == "silu":
        x = inp["x"]
        return {
            ("fwd", "default"): nograd(lambda: {"y": F.silu(x)}),
            ("bwd", "default"): backward(F.silu, [x], [inp["gy"]], ["gx"]),
        }
    if op == "mul":
        a, b = inp["a"], inp["b"]
        return {
            ("fwd", "default"): nograd(lambda: {"y": a * b}),
            ("bwd", "default"): backward(lambda a, b: a * b, [a, b], [inp["gy"]], ["ga", "gb"]),
        }
    if op == "add":
        x, y = inp["x"], inp["y"]
        return {
            ("fwd", "default"): nograd(lambda: {"z": x + y}),
            ("bwd", "default"): backward(lambda x, y: x + y, [x, y], [inp["gy"]], ["gx", "gy"]),
        }
    if op == "permute":
        x = inp["x"]
        return {("fwd", "default"): nograd(lambda: {"y": x.permute(0, 2, 1, 3).contiguous()})}
    if op == "ce":
        logits, targets = inp["logits"], inp["targets"]
        return {
            ("fwd", "default"): nograd(lambda: {"loss": F.cross_entropy(logits, targets)}),
            ("bwd", "default"): backward(lambda lg: F.cross_entropy(lg, targets), [logits], [None],
                                         ["glogits"]),
        }
    if op.startswith("adamw_"):
        out = {}
        for variant, fused in (("default", None), ("fused", True)):
            def prep(fused=fused):
                p = inp["p"].clone()
                opt = torch.optim.AdamW([p], lr=6e-4, betas=(0.9, 0.95), eps=1e-8,
                                        weight_decay=0.0, fused=fused)
                p.grad = inp["g"].clone()
                opt.state[p] = {"step": torch.tensor(0.0), "exp_avg": inp["m1"].clone(),
                                "exp_avg_sq": inp["m2"].clone()}
                return p, opt

            def timed(state):
                p, opt = state
                opt.step()
                st = opt.state[p]
                return {"p": p.detach(), "m1": st["exp_avg"], "m2": st["exp_avg_sq"]}
            out[("step", variant)] = (prep, timed)
        return out
    if op.startswith("muon_"):
        def prep():
            return inp["p"].clone(), inp["g"], inp["mom"].clone()

        def timed(state):
            p, g, buf = state
            muon_step(p, g, buf)
            return {"p": p, "mom": buf}
        return {("step", "default"): (prep, timed)}
    if op == "clip":
        names = [n for n in inp]

        def prep():
            # Uncommitted pages: the parameter values are never read.
            ps = []
            for n in names:
                p = torch.empty(inp[n].shape)
                p.grad = inp[n].clone()
                ps.append(p)
            return ps

        def timed(ps):
            norm = torch.nn.utils.clip_grad_norm_(ps, 1.0)
            return {"norm": norm, "b0_q": ps[names.index("b0.q")].grad,
                    "norm_f": ps[names.index("norm_f")].grad}
        return {("step", "default"): (prep, timed)}
    if op == "block":
        return block_cases(inp)
    raise KeyError(op)


BLOCK_PARAMS = ["x", "wq", "wk", "wv", "wo", "wd", "wfg", "wfu", "wg", "bg", "wqn", "wkn",
                "wn1", "wn2", "lam", "v0"]
BLOCK_GRADS = ["gx", "gwq", "gwk", "gwv", "gwo", "gwd", "gwfg", "gwfu", "gwg", "gbg", "gwqn",
               "gwkn", "gwn1", "gwn2", "glam", "gv0"]


def block_forward(P, cos, sin):
    x = P["x"]
    h = F.rms_norm(x, (D,), P["wn1"], eps=EPS)
    q = F.linear(h, P["wq"]).view(1, T, NH, HD)
    k = F.linear(h, P["wk"]).view(1, T, NH, HD)
    v = F.linear(h, P["wv"]).view(1, T, NH, HD)
    qn = F.rms_norm(q, (HD,), P["wqn"], eps=EPS)
    kn = F.rms_norm(k, (HD,), P["wkn"], eps=EPS)
    qr, kr = rope(qn, cos, sin), rope(kn, cos, sin)
    vr = vres(v, P["v0"], P["lam"])
    y = sdpa(qr.transpose(1, 2), kr.transpose(1, 2), vr.transpose(1, 2)).transpose(1, 2)
    g = gate(h.view(1, T, D), P["wg"], P["bg"], y).reshape(T, D)
    x1 = x + F.linear(g, P["wo"])
    h2 = F.rms_norm(x1, (D,), P["wn2"], eps=EPS)
    m = F.silu(F.linear(h2, P["wfg"])) * F.linear(h2, P["wfu"])
    return x1 + F.linear(m, P["wd"])


def block_cases(inp):
    cos, sin, gy = inp["cos"], inp["sin"], inp["gy"]
    out = {}
    for variant in ("math", "default"):
        def run(fn, variant=variant):
            if variant == "math":
                with sdpa_kernel(SDPBackend.MATH):
                    return fn()
            return fn()

        def fwd(_state, run=run):
            with torch.no_grad():
                return run(lambda: {"out": block_forward(inp, cos, sin)})

        def step(_state, run=run):
            def go():
                P = {n: (t.detach().requires_grad_(True) if n in BLOCK_PARAMS else t)
                     for n, t in inp.items()}
                o = block_forward(P, cos, sin)
                gs = torch.autograd.grad(o, [P[n] for n in BLOCK_PARAMS], gy)
                r = {"out": o.detach()}
                r.update(dict(zip(BLOCK_GRADS, gs)))
                return r
            return run(go)
        out[("fwd", variant)] = (None, fwd)
        out[("step", variant)] = (None, step)
    return out


def parity_metric(op, d, outs, got):
    """One PARITY line per dumped ojas output of this direction."""
    lines = []
    tol = TOL[op]
    for full, want in outs.items():
        dd, name = full.split(".", 1)
        if dd != d:
            continue
        if name not in got:
            lines.append(f"PARITY op={op} dir={d} out={name} status=MISSING_TORCH tol={tol:g} ok=False")
            continue
        ref = got[name].detach().to(torch.float32).reshape(want.shape)
        ojas = want
        if op.startswith("muon_") and name == "p":
            # Compare the orthogonalized update, not the parameter it is added to.
            rows, cols = ref.shape
            alpha = -MUON_LR * max(1.0, rows / cols) ** 0.5
            p0 = I_CACHE["p"] * (1 - MUON_LR * MUON_WD)
            ref, ojas = (ref - p0) / alpha, (ojas - p0) / alpha
            name = "update"
        elif op.startswith("adamw_") and name == "p":
            ref, ojas = ref - I_CACHE["p"], ojas - I_CACHE["p"]
            name = "delta"
        diff = (ref.double() - ojas.double()).abs().max().item()
        scale = ref.double().abs().max().item()
        err = diff / scale if scale > 0 else diff
        ok = err <= tol
        lines.append(f"PARITY op={op} dir={d} out={name} max_abs={diff:.3e} scale={scale:.3e} "
                     f"norm_err={err:.3e} tol={tol:g} ok={ok}")
    return lines


def f64_arbiter(op, inp, outs):
    """Reductions where torch f32 and ojas disagree past tolerance: compare
    both to an f64 evaluation of the same inputs. Lines are PARITY_F64."""
    tol = TOL[op]
    lines = []
    if op == "clip":
        sq = sum(float((g.double() ** 2).sum()) for g in inp.values())
        exact = sq ** 0.5
        ps = []
        for g in inp.values():
            p = torch.empty(g.shape)
            p.grad = g.clone()
            ps.append(p)
        torch_norm = float(torch.nn.utils.clip_grad_norm_(ps, 1.0))
        pairs = [("norm", exact, float(outs["step.norm"]), torch_norm)]
    elif op == "vres":
        s = torch.sigmoid(inp["lam"].double())[0].item()
        exact = float((inp["gy"].double() * (inp["v0"].double() - inp["v"].double())).sum()) * s * (1 - s)
        leaves = [x.detach().requires_grad_(True) for x in (inp["v"], inp["v0"], inp["lam"])]
        torch_glam = float(torch.autograd.grad(vres(*leaves), leaves, inp["gy"])[2][0])
        pairs = [("glam", exact, float(outs["bwd.glam"][0]), torch_glam)]
    else:
        return lines
    d = "step" if op == "clip" else "bwd"
    for name, exact, ojas, ref in pairs:
        oe, te = abs(ojas - exact) / abs(exact), abs(ref - exact) / abs(exact)
        lines.append(f"PARITY_F64 op={op} dir={d} out={name} exact={exact:.9e} ojas={ojas:.9e} "
                     f"torch={ref:.9e} ojas_err={oe:.3e} torch_err={te:.3e} tol={tol:g} ojas_ok={oe <= tol}")
    return lines


I_CACHE = {}


def bench(op, prep, timed, n):
    for _ in range(2):
        timed(prep() if prep else None)
    ns = []
    for _ in range(n):
        state = prep() if prep else None
        t0 = time.perf_counter_ns()
        r = timed(state)
        ns.append(time.perf_counter_ns() - t0)
        del r, state
    ns.sort()
    return ns[0] / 1e6, ns[len(ns) // 2] / 1e6


ORDER = ["linear_qkvo", "linear_up", "linear_down", "linear_lmhead", "linear_dec_qkvo",
         "linear_dec_up", "linear_dec_down", "embedding", "rms_norm",
         "qk_norm", "rope", "sdpa", "gate", "vres", "silu", "mul", "add", "permute", "ce",
         "adamw_768x768", "adamw_3072x768", "adamw_50304x768", "muon_768x768", "muon_2048x768",
         "muon_3072x768", "clip", "block"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "../../target-lane-measure/bench_ops"))
    ap.add_argument("--ops", default="")
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = torch.__config__.show().replace(",", " ").split()
    blas = [tok for tok in cfg if tok.startswith(("BLAS_INFO=", "LAPACK_INFO=", "USE_OPENMP="))]
    probe = torch.zeros(1, NH, T, HD)
    try:
        choice = SDPBackend(torch._fused_sdp_choice(probe, probe, probe, is_causal=True)).name
    except (AttributeError, RuntimeError, ValueError) as e:
        choice = f"unavailable({type(e).__name__})"
    print(f"TORCH_META torch={torch.__version__} threads={torch.get_num_threads()} "
          f"interop={torch.get_num_interop_threads()} sdpa_default={choice} pid={os.getpid()} "
          f"blas={' | '.join(blas).replace(' ', '_')}")
    ops = [o for o in ORDER if not args.ops or o in args.ops.split(",")]
    for op in ops:
        inp, outs = load(os.path.join(args.dir, op))
        I_CACHE.clear()
        I_CACHE.update(inp)
        table = cases(op, inp)
        if args.parity:
            for (d, variant), (prep, timed) in table.items():
                got = timed(prep() if prep else None)
                for line in parity_metric(op, d, outs, got):
                    print(line.replace(f"dir={d}", f"dir={d} variant={variant}"))
                # Mutating ops (adamw, muon, clip) work on copies made in prep,
                # so every variant starts from the dumped state.
            for line in f64_arbiter(op, inp, outs):
                print(line)
        if args.time:
            for (d, variant), (prep, timed) in table.items():
                n = calls(op, d)
                mn, md = bench(op, prep, timed, n)
                print(f"TORCH_OP op={op} dir={d} variant={variant} threads={torch.get_num_threads()} "
                      f"n={n} min_ms={mn:.4f} med_ms={md:.4f}", flush=True)
        del inp, outs, table
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"TORCH_RSS ops={','.join(ops)} maxrss_bytes={rss}")


if __name__ == "__main__":
    main()
