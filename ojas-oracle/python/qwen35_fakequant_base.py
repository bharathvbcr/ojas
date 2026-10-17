"""Fake-quant q8 base for ojas's quantised-LoRA fixture (ojas-model/tests/fixtures/qwen35_tiny_fq8).

    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/qwen35_fakequant_base.py
    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/qwen35_fakequant_base.py \
        --src DIR --out DIR

The Lappi decision model trains a LoRA over a q8 base. This writes the tiny
Qwen3.5 base (`qwen35_tiny`, bf16) with exactly the ten LoRA-wrapped linear
kinds (`self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`,
`mlp.{gate,up,down}_proj`, 13 tensors in the tiny model) replaced by

    wq, scales, biases = mx.quantize(W_bf16, group_size=64, bits=8)   # affine
    W' = mx.dequantize(wq, scales, biases, group_size=64, bits=8)

`W_bf16` is the checkpoint's bf16 tensor loaded as a bf16 mlx array. mlx
returns `scales` and `biases` in the input's dtype, so quantising from the
bf16 array makes both bf16 (asserted below), and `W'` is bf16 as well: every
value is exactly representable in f32, so ojas and torch (which load bf16
into f32) hold the same base. Every other tensor, `config.json` and the
`ids`/`labels` of `reference.safetensors` are byte copies. The output
`reference.safetensors` carries only `ids` and `labels` (no `loss`, no
gradients: those belong to the unquantised base). `manifest.json` records
the mlx version, the call, the quantised tensor names with their in-features
and max |W' - W|, and the sha256 of every input and output file.

mlx's affine quantiser needs the last dimension to be a multiple of the group
size, so a wrapped linear whose in-features is not a multiple of 64 is
refused. The tiny base's wrapped in-features are 64 (qkv, z, q, k, v, gate,
up), 128 (down) and 256 / 512 (GDN out_proj, attention o_proj).

Then regenerate the torch references over it:

    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/qwen35_lora_fixture.py \
        --base ojas-model/tests/fixtures/qwen35_tiny_fq8 \
        --out ojas-model/tests/fixtures/qwen35_tiny_lora_fq8
"""
import argparse
import hashlib
import json
import os
import re
import struct

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "ojas-model", "tests", "fixtures")
SRC = os.path.join(FIXTURES, "qwen35_tiny")
OUT = os.path.join(FIXTURES, "qwen35_tiny_fq8")
GROUP_SIZE, BITS = 64, 8
CALL = "mx.dequantize(*mx.quantize(w, group_size=64, bits=8), group_size=64, bits=8)"

# ojas-model's LORA_KINDS suffixes (qwen35/params.rs), the wrapped linears.
TARGET_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors(path):
    """(ordered {name: (dtype, shape, raw bytes)}, metadata or None)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        data = f.read()
    meta = header.pop("__metadata__", None)
    tensors = {}
    for name, e in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        lo, hi = e["data_offsets"]
        tensors[name] = (e["dtype"], e["shape"], data[lo:hi])
    return tensors, meta


def write_safetensors(path, tensors, meta):
    header, offset = {}, 0
    if meta is not None:
        header["__metadata__"] = meta
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, raw in tensors.values():
            f.write(raw)


def is_target(name):
    m = re.fullmatch(r"model\.layers\.\d+\.(.+)\.weight", name)
    return m is not None and m.group(1) in TARGET_SUFFIXES


def to_mx_bf16(shape, raw):
    bits = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    return mx.array(bits).view(mx.bfloat16)


def from_mx_bf16(a):
    assert a.dtype == mx.bfloat16, a.dtype
    return np.array(a.view(mx.uint16)).astype("<u2").tobytes()


def fakequant(shape, raw, name="tensor"):
    """dequantize(quantize(W)) with bf16 scales and biases; returns (raw bf16, max abs error).

    Refuses (SystemExit, never a NaN in the manifest) a non-finite weight, a
    non-finite result, and any group whose result is further from W than the
    affine q8 step allows: that is how a group whose scale collapsed to zero
    (a constant group dequantised to 0 instead of to its value) shows up.
    """
    w = to_mx_bf16(shape, raw)
    w32 = np.array(w.astype(mx.float32))
    if not np.isfinite(w32).all():
        raise SystemExit(f"{name}: non-finite weight")
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    assert scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16, (scales.dtype, biases.dtype)
    deq = mx.dequantize(wq, scales, biases, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(deq)
    d32 = np.array(deq.astype(mx.float32))
    if not np.isfinite(d32).all():
        raise SystemExit(f"{name}: non-finite after quantisation")
    g, dg = w32.reshape(-1, GROUP_SIZE), d32.reshape(-1, GROUP_SIZE)
    step = (g.max(axis=1) - g.min(axis=1)) / (2**BITS - 1)
    allowed = 1.5 * step + np.abs(g).max(axis=1) * 2.0**-7  # one step plus bf16 scale/bias/result rounding
    worst = np.abs(dg - g).max(axis=1)
    if (worst > allowed).any():
        bad = int(np.argmax(worst - allowed))
        raise SystemExit(f"{name}: group {bad} is off by {worst[bad]:.3e}, more than the q8 step allows ({allowed[bad]:.3e})")
    return from_mx_bf16(deq), float(np.abs(d32 - w32).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="the bf16 tiny base dir")
    ap.add_argument("--out", default=OUT, help="the fake-quant base dir to write")
    args = ap.parse_args()
    if os.path.realpath(args.src) == os.path.realpath(args.out):
        raise SystemExit("--src and --out are the same directory: the source would be overwritten")
    for f in ("config.json", "model.safetensors", "reference.safetensors"):
        if not os.path.isfile(os.path.join(args.src, f)):
            raise SystemExit(f"{os.path.join(args.src, f)} is missing")
    os.makedirs(args.out, exist_ok=True)

    tensors, meta = read_safetensors(os.path.join(args.src, "model.safetensors"))
    out, quantised = {}, []
    for name, (dtype, shape, raw) in tensors.items():
        if not is_target(name):
            out[name] = (dtype, shape, raw)
            continue
        if dtype != "BF16" or len(shape) != 2:
            raise SystemExit(f"{name}: expected a 2-D BF16 weight, found {dtype} {shape}")
        if shape[1] % GROUP_SIZE:
            raise SystemExit(f"{name}: in-features {shape[1]} is not a multiple of {GROUP_SIZE}")
        new, err = fakequant(shape, raw, name)
        assert len(new) == len(raw)
        out[name] = (dtype, shape, new)
        quantised.append({"name": name, "shape": shape, "in_features": shape[1],
                          "max_abs_error": err, "changed": new != raw})
    if not quantised:
        raise SystemExit("no wrapped linear found")
    assert not any("in_proj_a" in q["name"] or "in_proj_b" in q["name"] for q in quantised)

    ref, _ = read_safetensors(os.path.join(args.src, "reference.safetensors"))
    missing = [k for k in ("ids", "labels") if k not in ref]
    if missing:
        raise SystemExit(f"reference.safetensors has no {missing}")
    ref_out = {k: ref[k] for k in ("ids", "labels")}

    model_out = os.path.join(args.out, "model.safetensors")
    write_safetensors(model_out, out, meta)
    write_safetensors(os.path.join(args.out, "reference.safetensors"), ref_out, None)
    with open(os.path.join(args.src, "config.json"), "rb") as f:
        config = f.read()
    with open(os.path.join(args.out, "config.json"), "wb") as f:
        f.write(config)

    # Verify the written model: quantised tensors are bf16 values of the call, the rest are byte copies.
    back, _ = read_safetensors(model_out)
    assert list(back) == list(tensors)
    names = {q["name"] for q in quantised}
    for name, (dtype, shape, raw) in back.items():
        assert dtype == tensors[name][0] and shape == tensors[name][1], name
        if name not in names:
            assert raw == tensors[name][2], f"{name} is not a byte copy"

    files = ("config.json", "model.safetensors", "reference.safetensors")
    manifest = {
        "mlx_version": mx.__version__,
        "call": CALL,
        "mode": "affine",
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "scale_bias_dtype": "bfloat16 (quantised from the bf16 array; asserted)",
        "source_dir": "qwen35_tiny",
        "quantised": quantised,
        "copied_byte_identical": sorted(set(tensors) - names),
        "input_sha256": {f: sha256(os.path.join(args.src, f)) for f in files},
        "output_sha256": {f: sha256(os.path.join(args.out, f)) for f in files},
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    worst = max(q["max_abs_error"] for q in quantised)
    print(f"mlx {mx.__version__}: {len(quantised)} tensors fake-quantised (worst |dW| {worst:.3e}) -> {args.out}")


if __name__ == "__main__":
    main()
