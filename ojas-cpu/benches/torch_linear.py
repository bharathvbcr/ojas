"""Time torch CPU linear and causal attention forward and backward.

Shapes match ojas-cpu tests/bench_cpu.rs (linear_wall_time_matrix and
attention_wall_time_matrix) and tests/linear_kernel.rs::linear_scale_wall_time.

    python3 ojas-cpu/benches/torch_linear.py --threads 6
    python3 ojas-cpu/benches/torch_linear.py --threads 18 --attention
"""

import argparse
import time

import torch
import torch.nn.functional as F


def median(samples):
    samples = sorted(samples)
    return samples[len(samples) // 2]


def bench_linear():
    shapes = [(64, 64, 128), (256, 256, 256), (512, 768, 768), (2048, 2048, 2048)]
    for rows, kin, nout in shapes:
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        x = torch.randn(rows, kin, generator=g).requires_grad_(True)
        w = torch.randn(nout, kin, generator=g).requires_grad_(True)
        gy = torch.randn(rows, nout, generator=g)
        calls = 4 if rows >= 2048 else 8 if rows >= 512 else 30
        fwd, bwd = [], []
        for i in range(calls + 1):
            x.grad = None
            w.grad = None
            t0 = time.perf_counter()
            y = F.linear(x, w)
            fwd_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            y.backward(gy)
            bwd_s = time.perf_counter() - t1
            assert torch.isfinite(y).all()
            if i > 0:
                fwd.append(fwd_s)
                bwd.append(bwd_s)
        print(
            f"TORCH_LINEAR threads={torch.get_num_threads()} rows={rows} kin={kin} nout={nout} "
            f"fwd_ms={median(fwd) * 1e3:.4f} bwd_ms={median(bwd) * 1e3:.4f}"
        )


def bench_attention():
    b, h, d = 4, 8, 64
    for t in (128, 512, 2048):
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        q = torch.randn(b, h, t, d, generator=g).requires_grad_(True)
        k = torch.randn(b, h, t, d, generator=g).requires_grad_(True)
        v = torch.randn(b, h, t, d, generator=g).requires_grad_(True)
        gy = torch.randn(b, h, t, d, generator=g)
        calls = 2 if t >= 2048 else 6
        fwd, bwd = [], []
        for i in range(calls + 1):
            q.grad = k.grad = v.grad = None
            t0 = time.perf_counter()
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            fwd_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            y.backward(gy)
            bwd_s = time.perf_counter() - t1
            if i > 0:
                fwd.append(fwd_s)
                bwd.append(bwd_s)
        print(
            f"TORCH_ATTN threads={torch.get_num_threads()} B={b} H={h} T={t} D={d} "
            f"fwd_ms={median(fwd) * 1e3:.3f} bwd_ms={median(bwd) * 1e3:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print("threads", torch.get_num_threads(), "torch", torch.__version__)
    bench_linear()
    if args.attention:
        bench_attention()


if __name__ == "__main__":
    main()
