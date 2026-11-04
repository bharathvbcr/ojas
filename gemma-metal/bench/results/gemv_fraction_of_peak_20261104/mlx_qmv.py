"""Throwaway probe for ojas ft-5b2a72ee criterion 3: MLX's quantized matmul at the same E4B shapes as
gemv.swift (affine Q4, group 64, bf16 x of one row), each timed batch walking distinct weight copies
totalling >= 256 MB, min/median per call over rounds. Nothing here ships.

    python3 -I mlx_qmv.py <rounds>
"""

import sys
import time

import mlx.core as mx

SHAPES = [
    ("mlp gate/up", 10240, 2560),
    ("mlp down", 2560, 10240),
    ("attn q (sliding)", 2048, 2560),
    ("attn q (global)", 4096, 2560),
    ("attn k or v (sliding)", 512, 2560),
    ("attn o (sliding)", 2560, 2048),
    ("lm_head (tied)", 262144, 2560),
]
G = 64


def bank(rows, cols):
    w_bytes = rows * cols // 2
    sb_bytes = rows * (cols // G) * 4  # bf16 scale + bf16 bias, as tessl's bank
    copies = max(1, (256 << 20) // (w_bytes + sb_bytes) + 1)
    mats = []
    for c in range(copies):
        w = mx.random.normal((rows, cols), key=mx.random.key(rows * 7 + cols + c)).astype(mx.bfloat16) * 0.02
        wq, s, b = mx.quantize(w, group_size=G, bits=4)
        mats.append((wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)))
        mx.eval(mats[-1])
        del w
    x = (mx.random.normal((1, cols), key=mx.random.key(1)) ).astype(mx.bfloat16)
    mx.eval(x)
    return copies, w_bytes + sb_bytes, mats, x


def run(copies, mats, x):
    mx.synchronize()
    t0 = time.perf_counter()
    ys = [mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=G, bits=4) for wq, s, b in mats]
    mx.eval(ys)
    mx.synchronize()
    return (time.perf_counter() - t0) / copies


def main():
    rounds = int(sys.argv[1])
    print(f"mlx {mx.__version__}, device {mx.default_device()}")
    banks = [(name, rows, cols, *bank(rows, cols)) for name, rows, cols in SHAPES]
    for b in banks:
        run(b[3], b[5], b[6])
    times = {b[0]: [] for b in banks}
    for _ in range(rounds):
        for name, rows, cols, copies, nbytes, mats, x in banks:
            times[name].append(run(copies, mats, x))
    print("shape | rows x cols | bytes (w + sb) | copies/batch | min us | median us | max us | GB/s at min (wall, includes encode)")
    for name, rows, cols, copies, nbytes, mats, x in banks:
        t = sorted(times[name])
        print(f"{name} | {rows} x {cols} | {nbytes / 1e6:.2f} MB | {copies} | {t[0] * 1e6:.1f} | {t[len(t) // 2] * 1e6:.1f} | "
              f"{t[-1] * 1e6:.1f} | {nbytes / t[0] / 1e9:.1f}")


if __name__ == "__main__":
    main()
