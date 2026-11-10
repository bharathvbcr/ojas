"""Golden arrays for ojas's MLX-exact affine quantiser (ojas-core/src/quant.rs).

    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/mlx_affine_goldens.py
    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/mlx_affine_goldens.py --out DIR

Every case is a bf16 matrix quantised and dequantised by mlx on the GPU stream
(`stream=mx.gpu` on every call: mx.cpu disagrees with mx.gpu on ~0.2-0.4% of codes, and the
GPU is mlx's default and what `qwen35_fakequant_base.py` produced), once for 8 bits and once
for 4 bits, group size 64:

    wq, scales, biases = mx.quantize(w, group_size=64, bits=B, stream=mx.gpu)
    deq = mx.dequantize(wq, scales, biases, group_size=64, bits=B, stream=mx.gpu)

One safetensors file per case holds `w` (BF16 [rows, cols]) and, for q in {q8, q4},
`q.wq` (U32 [rows, cols*bits/32]), `q.scales` and `q.biases` (BF16 [rows, cols/64]) and
`q.deq` (BF16 [rows, cols]). The per-group arithmetic depends only on the 64 values of the
group, so the cases use a few rows of each DISTINCT in-feature width of Qwen3.5-2B-Base's wrapped
linears (derived below from the local config and the safetensors header), and a width-64 case
holds one named group kind per row so a failing row names what broke.

`manifest.json` records the mlx version, stream, seed, calls, the 2B widths with their source,
the cases (rows, group kinds, how many .5 ties the data actually lands on) and the sha256 of
every file. A group whose range overflows f32 (+bf16 max with -bf16 max) has no finite scale:
the probe's mlx outputs go in the manifest, not in a golden, because ojas refuses it.

The bf16 inputs are built from integer bit patterns (no float cast that the GPU could flush),
so subnormal bf16 values reach mlx intact.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import struct
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qwen35_fakequant_base import write_safetensors  # noqa: E402  (the one safetensors writer)

OUT = os.path.join(HERE, "..", "..", "ojas-core", "tests", "fixtures", "mlx_affine")
SEED = 20261110
GROUP_SIZE = 64
BITS = (8, 4)
SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/*/")
WRAPPED = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
CALLS = {
    "quantize": "mx.quantize(w, group_size=64, bits=B, stream=mx.gpu)",
    "dequantize": "mx.dequantize(wq, scales, biases, group_size=64, bits=B, stream=mx.gpu)",
}
f32 = np.float32
BF16_MAX = float(np.array([0x7F7F0000], dtype=np.uint32).view(f32)[0])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bf16_bits(x):
    """f32 -> bf16 bits by integer round-to-nearest-even (no float op that could flush)."""
    u = np.ascontiguousarray(x, dtype=f32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF).astype(np.uint16)


def widen(bits):
    return (bits.astype(np.uint32) << 16).view(f32)


def to_mx(bits):
    return mx.array(np.ascontiguousarray(bits, dtype=np.uint16)).view(mx.bfloat16)


def from_mx(a):
    assert a.dtype == mx.bfloat16, a.dtype
    return np.array(a.view(mx.uint16)).astype("<u2")


def u32_of(a):
    assert a.dtype == mx.uint32, a.dtype
    return np.array(a).astype("<u4")


def wide_widths():
    """The distinct in-feature widths of the 2B wrapped linears, with their source."""
    snaps = sorted(glob.glob(SNAPSHOT_GLOB))
    if not snaps:
        raise SystemExit(f"no Qwen3.5-2B-Base snapshot at {SNAPSHOT_GLOB}")
    snap = snaps[-1]
    with open(os.path.join(snap, "config.json")) as f:
        text = json.load(f)["text_config"]
    shard = sorted(glob.glob(os.path.join(snap, "*.safetensors")))[0]
    with open(shard, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    shapes = {}
    for name, e in header.items():
        m = re.fullmatch(r"model\.language_model\.layers\.\d+\.(.+)\.weight", name)
        if m and m.group(1) in WRAPPED:
            assert e["dtype"] == "BF16", name
            shapes.setdefault(m.group(1), set()).add(tuple(e["shape"]))
    assert set(shapes) == set(WRAPPED), set(WRAPPED) - set(shapes)
    assert all(len(v) == 1 for v in shapes.values()), shapes
    per_kind = {k: list(next(iter(v))) for k, v in sorted(shapes.items())}
    distinct_shapes = sorted({tuple(v) for v in per_kind.values()})
    widths = sorted({v[1] for v in per_kind.values()})
    assert widths == sorted({text["hidden_size"], text["intermediate_size"]}), (widths, text)
    return {
        "source": "config.json text_config.hidden_size / intermediate_size, confirmed against "
                  "the safetensors header shapes of the wrapped linears",
        "snapshot": os.path.basename(snap.rstrip("/")),
        "wrapped_linear_shapes_out_in": per_kind,
        "distinct_shapes_out_in": [list(s) for s in distinct_shapes],
        "distinct_in_features": widths,
    }


# --- group kinds: each returns 64 f32 values (rounded to bf16 afterwards) ---------------------

def tiny(sign, k):
    return sign * f32(2.0) ** f32(-133 + k)  # bf16 subnormals: 2^-133 is the smallest


def group_kinds(rng):
    g = GROUP_SIZE
    randn = lambda s: rng.standard_normal(g).astype(f32) * f32(s)  # noqa: E731
    kinds = {}

    def add(name, values):
        v = np.asarray(values, dtype=f32)
        assert v.shape == (g,), (name, v.shape)
        kinds[name] = v

    add("zero", np.zeros(g))
    add("const_pos", np.full(g, 0.5))
    add("const_neg", np.full(g, -0.5))
    add("const_1e-30", np.full(g, 1e-30))
    add("const_neg_zero", np.full(g, -0.0))
    add("all_negative", -np.abs(randn(0.02)) - f32(1e-4))
    add("all_positive", np.abs(randn(0.02)) + f32(1e-4))
    add("randn_0.02", randn(0.02))
    add("randn_1e-5", randn(1e-5))
    add("randn_1e-3", randn(1e-3))
    add("randn_100", randn(100.0))
    add("wide_dynamic", randn(1.0) * (10.0 ** rng.uniform(-3, 3, g)).astype(f32))
    add("linspace_pm1", np.linspace(-1, 1, g))
    add("integer_grid", np.round(np.linspace(-8, 8, g)))
    add("linspace_0_1", np.linspace(0, 1, g))
    # Values on .5 code boundaries. q8: range 127.5 -> scale 0.5, w = 0.25 + 0.5k lands on x.5.
    tie = np.zeros(g)
    tie[1] = 127.5
    tie[2:] = 0.25 + 0.5 * rng.integers(0, 128, g - 2)
    add("ties_q8_scale_half", tie)
    add("ties_q8_scale_half_neg", -tie)
    # q8 range 255 -> scale 1, w = k + 0.5.
    tie = np.zeros(g)
    tie[1] = 255.0
    tie[2:] = rng.integers(0, 127, g - 2) + 0.5
    add("ties_q8_scale_one", tie)
    add("ties_q8_scale_one_neg", -tie)
    # q4 range 15 -> scale 1, w = k + 0.5.
    tie = np.zeros(g)
    tie[1] = 15.0
    tie[2:] = rng.integers(0, 15, g - 2) + 0.5
    add("ties_q4_scale_one", tie)
    add("ties_q4_scale_one_neg", -tie)
    # Straddling zero so the edge is not 0: min -127.5, max 127.5, w on x.25/x.75.
    tie = 0.25 + 0.5 * rng.integers(-128, 127, g)
    tie[0], tie[1] = -127.5, 127.5
    add("ties_straddle", tie)
    for name, v in (("outlier_pos", 1.0), ("outlier_neg", -1.0), ("outlier_big", 1e4)):
        x = randn(0.01)
        x[rng.integers(0, g)] = v
        add(name, x)
    x = randn(0.02)
    x[5] = BF16_MAX
    add("max_finite_pos", x)
    x = randn(0.02)
    x[5] = -BF16_MAX
    add("max_finite_neg", x)
    x = np.full(g, BF16_MAX)
    add("max_finite_const", x)
    add("max_finite_const_neg", -x)
    # Smaller than the 1e-7 scale floor but normal: the at_zero branch with signed -0.0 codes.
    add("below_floor", randn(1e-8))
    add("below_floor_pos_edge", np.abs(randn(1e-8)))
    x = randn(1e-8)
    x[0] = 4.9e-8
    add("near_half_floor", x)
    # bf16 subnormals.
    add("subnormal_const_pos", np.full(g, 1e-39))
    add("subnormal_const_neg", np.full(g, -3e-40))
    add("subnormal_only_mixed_sign", tiny(1, 0) * rng.integers(-60, 60, g).astype(f32))
    x = randn(0.02)
    x[[3, 17, 40]] = [1e-39, -3e-40, tiny(1, 3)]
    add("subnormal_mixed_with_normals", x)
    x = np.zeros(g)
    x[7] = -3e-40
    add("subnormal_neg_with_zeros", x)
    x = np.zeros(g)
    x[7] = 3e-40
    add("subnormal_pos_with_zeros", x)
    x = np.zeros(g)
    x[7], x[8] = -3e-40, 1e-39
    add("subnormal_both_with_zeros", x)
    x = np.full(g, 1e-39)
    x[0] = -1e-30
    add("subnormal_with_tiny_normal", x)
    return kinds


OVERFLOW_PROBE = np.zeros(GROUP_SIZE, dtype=f32)
OVERFLOW_PROBE[0], OVERFLOW_PROBE[1] = BF16_MAX, -BF16_MAX


def count_ties(wb, bits):
    """Elements whose affine code lands on x.5 (round-half-away is exercised), f32, no FTZ."""
    n_bins = f32((1 << bits) - 1)
    g = wb.reshape(-1, GROUP_SIZE)
    w_min = np.minimum(g.min(axis=1), f32(BF16_MAX))
    w_max = np.maximum(g.max(axis=1), f32(0.0))
    with np.errstate(all="ignore"):
        scale = np.maximum((w_max - w_min) / n_bins, f32(1e-7))
        side = np.abs(w_min) > np.abs(w_max)
        scale = np.where(side, scale, -scale).astype(f32)
        edge = np.where(side, w_min, w_max)
        q0 = np.where(edge / scale >= 0, np.floor(edge / scale + f32(0.5)), np.ceil(edge / scale - f32(0.5)))
        scale = np.where(q0 == 0, scale, edge / q0).astype(f32)
        bias = np.where(q0 == 0, f32(0), edge).astype(f32)
        x = ((g - bias[:, None]) / scale[:, None]).astype(f32)
    frac = np.abs(x) - np.floor(np.abs(x))
    return int(np.sum(np.isfinite(x) & (frac == f32(0.5))))


def quantise_case(bits_arr):
    w = to_mx(bits_arr)
    out = {"w": bits_arr.astype("<u2")}
    ties = {}
    for bits in BITS:
        wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=bits, stream=mx.gpu)
        deq = mx.dequantize(wq, scales, biases, group_size=GROUP_SIZE, bits=bits, stream=mx.gpu)
        mx.eval(wq, scales, biases, deq)
        rows, cols = bits_arr.shape
        assert wq.shape == (rows, cols * bits // 32), wq.shape
        assert scales.shape == biases.shape == (rows, cols // GROUP_SIZE)
        assert scales.dtype == biases.dtype == deq.dtype == mx.bfloat16
        out[f"q{bits}.wq"] = u32_of(wq)
        out[f"q{bits}.scales"] = from_mx(scales)
        out[f"q{bits}.biases"] = from_mx(biases)
        out[f"q{bits}.deq"] = from_mx(deq)
        ties[f"q{bits}"] = count_ties(widen(bits_arr), bits)
    return out, ties


def dequant_params_case(rng):
    """mx.dequantize on hand-made scales/biases, subnormal and signed-zero ones included.

    The quantiser never emits a subnormal scale or bias, but a loaded checkpoint could carry one;
    the GPU's treatment of it is pinned here, not guessed.
    """
    scale_pool = np.array([1e-7, 3e-40, 1e-39, -3e-40, -0.0, 0.0, 0.5, -0.25, 2.0 ** -126, -1e-7, 1e4], dtype=f32)
    bias_pool = np.array([0.0, 1e-39, -3e-40, -0.0, 0.25, -1e-3, 2.0 ** -126, -(2.0 ** -126), 1.5 * 2.0 ** -126,
                          -1.5 * 2.0 ** -126, 1e-7, -2.0, 100.0], dtype=f32)
    # Every (scale, bias) pair: s*q + b can land on a subnormal (s = 2^-126, q = 1, b = -1.5 * 2^-126).
    pairs = [(s, b) for s in scale_pool for b in bias_pool]
    rows, cols = len(pairs), GROUP_SIZE
    s_bits = bf16_bits(np.array([[s] for s, _ in pairs], dtype=f32))
    b_bits = bf16_bits(np.array([[b] for _, b in pairs], dtype=f32))
    out = {}
    for bits in BITS:
        wq = rng.integers(0, 2 ** 32, (rows, cols * bits // 32), dtype=np.uint64).astype("<u4")
        deq = mx.dequantize(mx.array(wq), to_mx(s_bits), to_mx(b_bits), group_size=GROUP_SIZE, bits=bits, stream=mx.gpu)
        mx.eval(deq)
        out[f"q{bits}.wq"], out[f"q{bits}.scales"], out[f"q{bits}.biases"] = wq, s_bits.astype("<u2"), b_bits.astype("<u2")
        out[f"q{bits}.deq"] = from_mx(deq)
    return out


def st_entry(name, arr):
    dtype = {"<u2": "BF16", "<u4": "U32"}[arr.dtype.str]
    if name.endswith((".wq",)):
        assert dtype == "U32"
    return dtype, list(arr.shape), np.ascontiguousarray(arr).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    wide = wide_widths()
    kinds = group_kinds(rng)
    names = list(kinds)

    cases = {}  # name -> (bf16 bits [rows, cols], row labels)
    # One kind per row at width 64: a failing row names its kind.
    rows = np.stack([kinds[n] for n in names])
    cases["kinds_w64"] = (bf16_bits(rows), names)
    # Every kind in turn across the groups of a row, at each 2B in-feature width.
    for w in wide["distinct_in_features"]:
        ng = w // GROUP_SIZE
        mixed = np.zeros((3, w), dtype=f32)
        labels = []
        for r in range(3):
            row = [names[(r * 7 + j) % len(names)] for j in range(ng)]
            labels.append(row)
            mixed[r] = np.concatenate([kinds[n] for n in row])
        cases[f"mixed_w{w}"] = (bf16_bits(mixed), labels)
        rn = (rng.standard_normal((8 if w <= 2048 else 4, w)) * 0.02).astype(f32)
        cases[f"randn_w{w}"] = (bf16_bits(rn), "randn*0.02, bf16-rounded")

    manifest = {
        "mlx_version": mx.__version__,
        "stream": "mx.gpu (passed explicitly on every mx.quantize / mx.dequantize call)",
        "seed": SEED,
        "rng": "numpy.random.default_rng(seed); kinds first, then the randn cases in width order",
        "group_size": GROUP_SIZE,
        "bits": list(BITS),
        "mode": "affine (mlx default)",
        "calls": CALLS,
        "input_construction": "f32 values rounded to bf16 by integer round-to-nearest-even on the "
                              "bit pattern; subnormal bf16 values reach mlx intact",
        "qwen35_2b_base": wide,
        "group_kinds": names,
        "cases": {},
    }
    for name, (bits_arr, labels) in cases.items():
        out, ties = quantise_case(bits_arr)
        path = os.path.join(args.out, f"{name}.safetensors")
        write_safetensors(path, {k: st_entry(k, v) for k, v in out.items()}, None)
        manifest["cases"][name] = {
            "file": f"{name}.safetensors",
            "rows": int(bits_arr.shape[0]),
            "cols": int(bits_arr.shape[1]),
            "row_labels": labels,
            "elements_landing_on_a_half_code": ties,
            "sha256": sha256(path),
        }

    params = dequant_params_case(rng)
    path = os.path.join(args.out, "deq_params_w64.safetensors")
    write_safetensors(path, {k: st_entry(k, v) for k, v in params.items()}, None)
    manifest["cases"]["deq_params_w64"] = {
        "file": "deq_params_w64.safetensors", "rows": len(params["q8.scales"]), "cols": GROUP_SIZE,
        "note": "dequantize only: random packed codes, one row per (scale, bias) pair of a pool that includes bf16 subnormals, +-0.0 and +-2^-126; no `w`",
        "sha256": sha256(path),
    }

    # The overflow probe: what mlx does when w_max - w_min overflows f32.
    probe = bf16_bits(OVERFLOW_PROBE.reshape(1, GROUP_SIZE))
    pw = to_mx(probe)
    manifest["overflow_probe"] = {"note": "ojas refuses this group (non-finite range); recorded for the record, not a golden",
                                  "group": "[+bf16 max, -bf16 max, 0 x62]"}
    for bits in BITS:
        wq, s, b = mx.quantize(pw, group_size=GROUP_SIZE, bits=bits, stream=mx.gpu)
        mx.eval(wq, s, b)
        manifest["overflow_probe"][f"q{bits}"] = {
            "scale_bf16_bits": hex(int(from_mx(s)[0, 0])),
            "bias_bf16_bits": hex(int(from_mx(b)[0, 0])),
            "wq_u32": [int(x) for x in u32_of(wq)[0][: 64 * bits // 32]],
        }

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    total = sum(os.path.getsize(os.path.join(args.out, p)) for p in os.listdir(args.out))
    print(f"mlx {mx.__version__}: {len(cases)} cases, {total} bytes -> {args.out}")


if __name__ == "__main__":
    main()
