"""torch MPS reference for examples/metal_bench.rs: same ops, shapes and step.

/opt/homebrew/opt/python@3.14/bin/python3.14 ojas-metal/benches/torch_mps.py [iters]

Timing is wall time with torch.mps.synchronize() before the clock stops.
"resident" keeps inputs on the device; "with transfer" copies host tensors up,
runs the op and copies every output back (.cpu()). Values match the Rust
bench's: f32, causal attention via scaled_dot_product_attention, the same
reshape-not-permute head split for the full step, AdamW (foreach) and
clip_grad_norm_.
"""

import statistics
import sys
import time

import torch
import torch.nn.functional as F

DEV = torch.device("mps")
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def sync():
    torch.mps.synchronize()


def timed(fn, iters):
    for _ in range(2):
        fn()
        sync()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        ms.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ms)


def row(name, res, tr, flops):
    print(f"| {name} | {res:.3f} | {tr:.3f} | {flops / (res * 1e-3) / 1e12:.2f} |")


def rand(*shape):
    return torch.rand(*shape) * 2 - 1


def linear(rows, kin, nout):
    xh, wh, gh = rand(rows, kin), rand(nout, kin), rand(rows, nout)
    x, w, g = xh.to(DEV), wh.to(DEV), gh.to(DEV)
    flops = 2.0 * rows * kin * nout
    res = timed(lambda: F.linear(x, w), ITERS)
    tr = timed(lambda: F.linear(xh.to(DEV), wh.to(DEV)).cpu(), ITERS)
    row(f"linear fwd {rows}x{kin}x{nout}", res, tr, flops)

    def bwd(x, w, g):
        return g @ w, g.t() @ x

    res = timed(lambda: bwd(x, w, g), ITERS)
    tr = timed(lambda: [t.cpu() for t in bwd(xh.to(DEV), wh.to(DEV), gh.to(DEV))], ITERS)
    row(f"linear bwd {rows}x{kin}x{nout}", res, tr, 2 * flops)


def attention(b, h, t, d):
    hs = [rand(b, h, t, d) for _ in range(4)]

    def fwd(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def bwd(q, k, v, g):
        q, k, v = (z.detach().requires_grad_() for z in (q, k, v))
        o = fwd(q, k, v)
        return torch.autograd.grad(o, (q, k, v), g)

    dv = [z.to(DEV) for z in hs]
    flops = 2.0 * 2.0 * b * h * d * t * t / 2.0
    res = timed(lambda: fwd(*dv[:3]), ITERS)
    tr = timed(lambda: fwd(*(z.to(DEV) for z in hs[:3])).cpu(), ITERS)
    row(f"attn fwd B{b} H{h} T{t} D{d}", res, tr, flops)
    # torch's backward reruns the forward under autograd; that is its cost.
    res = timed(lambda: bwd(*dv), ITERS)
    tr = timed(lambda: [z.cpu() for z in bwd(*(z.to(DEV) for z in hs))], ITERS)
    row(f"attn bwd B{b} H{h} T{t} D{d}", res, tr, 2.5 * flops)


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def full_step(b, t, d, v, iters):
    h = d // 64
    g = torch.Generator().manual_seed(0)

    def p(*s):
        return ((torch.rand(*s, generator=g) * 2 - 1) * 0.05).to(DEV).requires_grad_()

    params = [p(v, d), torch.ones(d, device=DEV, requires_grad=True), p(d, d), p(d, d), p(d, d),
              p(h, d), torch.zeros(h, device=DEV, requires_grad=True), p(d, d), p(v, d)]
    emb, rw, wq, wk, wv, gw, gb, wo, head = params
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    tok_h = torch.tensor([(i * 2654435761) % v for i in range(b * t)]).view(b, t)
    tgt_h = ((tok_h + 1) % v).view(-1)

    def step(tok, tgt):
        x = F.embedding(tok, emb)
        hn = rms(x, rw)
        q = F.linear(hn, wq).reshape(b, h, t, 64)
        k = F.linear(hn, wk).reshape(b, h, t, 64)
        vv = F.linear(hn, wv).reshape(b, h, t, 64)
        a = F.scaled_dot_product_attention(q, k, vv, is_causal=True).reshape(b, t, h, 64)
        gate = torch.sigmoid(F.linear(hn, gw, gb)).unsqueeze(-1)
        o = F.linear((a * gate).reshape(b, t, d), wo)
        s = F.silu(x + o)
        loss = F.cross_entropy(F.linear(s, head).view(b * t, v), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        return loss

    tok, tgt = tok_h.to(DEV), tgt_h.to(DEV)
    res = timed(lambda: step(tok, tgt), iters)
    tr = timed(lambda: step(tok_h.to(DEV), tgt_h.to(DEV)).item(), iters)
    mm = 4 * d * d + v * d
    flops = 6.0 * b * t * mm + 2.0 * 2.0 * b * d * t * t / 2.0 * 3.5
    row(f"full step B{b} T{t} d{d} V{v}", res, tr, flops)


def main():
    torch.manual_seed(0)
    print(f"torch {torch.__version__} mps, median of {ITERS}")
    print("| op | resident ms | with transfer ms | TFLOP/s (resident) |")
    print("|---|---:|---:|---:|")
    linear(512, 768, 768)
    linear(2048, 2048, 2048)
    for t in (128, 512, 2048):
        attention(4, 8, t, 64)
    for d in (512, 768):
        full_step(4, 128, d, 50304, min(ITERS, 10))


if __name__ == "__main__":
    main()
