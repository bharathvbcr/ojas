"""Fake-quant q8 base for ojas's quantised-LoRA fixture (ojas-model/tests/fixtures/qwen35_tiny_fq8).

    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/qwen35_fakequant_base.py
    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/qwen35_fakequant_base.py \
        --src DIR --out DIR
    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/qwen35_fakequant_base.py \
        --packed-out [DIR]     # also write the packed q8 base (see below)

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

`--packed-out` also writes the same quantisation as a packed base
(`ojas-model/tests/fixtures/qwen35_tiny_q8`), from the SAME `mx.quantize` call that made the
fake-quant weight: every wrapped linear `<prefix>.weight` becomes the three tensors mlx_lm names a
quantised linear, `<prefix>.weight` (U32 `[out, in*8/32]`, the packed codes), `<prefix>.scales` and
`<prefix>.biases` (BF16 `[out, in/64]`), in that order at the position of the original; every other
tensor is a byte copy; `config.json` is the source's plus mlx_lm's `"quantization": {"group_size": 64,
"bits": 8}`; and `manifest.json` records the mlx version, `stream=mx.gpu`, the call and the sha256
of every input and output file. ojas-model's `qwen35_q8_host` test checks that ojas's host
quantiser reproduces these bytes from `qwen35_tiny` and its dequantiser reproduces the fq8 weights.
Every mx.quantize / mx.dequantize call passes `stream=mx.gpu` explicitly (mlx's default stream; the
fq8 bytes are the same with or without it).

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
import shutil
import struct

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "ojas-model", "tests", "fixtures")
SRC = os.path.join(FIXTURES, "qwen35_tiny")
OUT = os.path.join(FIXTURES, "qwen35_tiny_fq8")
PACKED_OUT = os.path.join(FIXTURES, "qwen35_tiny_q8")
GROUP_SIZE, BITS = 64, 8
CALL = "mx.dequantize(*mx.quantize(w, group_size=64, bits=8), group_size=64, bits=8)"
PACKED_CALL = "wq, scales, biases = mx.quantize(w, group_size=64, bits=8, stream=mx.gpu)"

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
    """dequantize(quantize(W)) with bf16 scales and biases.

    Returns (raw bf16, max abs error, (packed U32 bytes, scales bf16 bytes, biases bf16 bytes)).

    Refuses (SystemExit, never a NaN in the manifest) a non-finite weight, a
    non-finite result, and any group whose result is further from W than the
    affine q8 step allows: that is how a group whose scale collapsed to zero
    (a constant group dequantised to 0 instead of to its value) shows up.
    """
    w = to_mx_bf16(shape, raw)
    w32 = np.array(w.astype(mx.float32))
    if not np.isfinite(w32).all():
        raise SystemExit(f"{name}: non-finite weight")
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS, stream=mx.gpu)
    assert scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16, (scales.dtype, biases.dtype)
    deq = mx.dequantize(wq, scales, biases, group_size=GROUP_SIZE, bits=BITS, stream=mx.gpu)
    mx.eval(wq, scales, biases, deq)
    assert wq.dtype == mx.uint32 and wq.shape == (shape[0], shape[1] * BITS // 32), (wq.dtype, wq.shape)
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
    packed = (np.array(wq).astype("<u4").tobytes(), from_mx_bf16(scales), from_mx_bf16(biases))
    return from_mx_bf16(deq), float(np.abs(d32 - w32).max()), packed


def write_packed(args, tensors, packed_out, quantised_names, quantised, files):
    """The packed q8 base: mlx_lm's quantised-linear tensors from the same mx.quantize call."""
    out_dir = args.packed_out
    if os.path.realpath(out_dir) in (os.path.realpath(args.src), os.path.realpath(args.out)):
        raise SystemExit("--packed-out is the same directory as --src or --out")
    os.makedirs(out_dir, exist_ok=True)
    model_out = os.path.join(out_dir, "model.safetensors")
    _, meta = read_safetensors(os.path.join(args.src, "model.safetensors"))
    write_safetensors(model_out, packed_out, meta)
    shutil.copyfile(os.path.join(args.out, "reference.safetensors"), os.path.join(out_dir, "reference.safetensors"))
    with open(os.path.join(args.src, "config.json")) as f:
        config_text = f.read()
    config = json.loads(config_text)
    assert json.dumps(config, indent=1) == config_text, "config.json is not in the 1-space json.dumps layout"
    config["quantization"] = {"group_size": GROUP_SIZE, "bits": BITS}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(json.dumps(dict(sorted(config.items())), indent=1))
    back, _ = read_safetensors(model_out)
    for name, (dtype, shape, raw) in tensors.items():
        if name in quantised_names:
            prefix = name[: -len(".weight")]
            assert back[name][0] == "U32" and back[name][1] == [shape[0], shape[1] * BITS // 32], name
            for suffix in ("scales", "biases"):
                got = back[f"{prefix}.{suffix}"]
                assert got[0] == "BF16" and got[1] == [shape[0], shape[1] // GROUP_SIZE], (name, suffix)
        else:
            assert back[name] == tensors[name], f"{name} is not a byte copy"
    assert len(back) == len(tensors) + 2 * len(quantised_names)
    manifest = {
        "mlx_version": mx.__version__,
        "stream": "mx.gpu (passed explicitly)",
        "call": PACKED_CALL,
        "mode": "affine",
        "group_size": GROUP_SIZE,
        "bits": BITS,
        "layout": "mlx_lm quantised linear: <prefix>.weight U32 [out, in*bits/32], <prefix>.scales and "
                  "<prefix>.biases BF16 [out, in/group_size]; config.json carries mlx_lm's quantization block",
        "source_dir": "qwen35_tiny",
        "fakequant_dir": "qwen35_tiny_fq8 (mx.dequantize of the same call)",
        "quantised": [{"name": q["name"], "shape": q["shape"], "in_features": q["in_features"]} for q in quantised],
        "copied_byte_identical": sorted(set(tensors) - quantised_names),
        "input_sha256": {f: sha256(os.path.join(args.src, f)) for f in files},
        "output_sha256": {f: sha256(os.path.join(out_dir, f)) for f in files},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"packed q8 base -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="the bf16 tiny base dir")
    ap.add_argument("--out", default=OUT, help="the fake-quant base dir to write")
    ap.add_argument("--packed-out", nargs="?", const=PACKED_OUT, default=None,
                    help="also write the packed q8 base to DIR (default %(const)s)")
    args = ap.parse_args()
    if os.path.realpath(args.src) == os.path.realpath(args.out):
        raise SystemExit("--src and --out are the same directory: the source would be overwritten")
    for f in ("config.json", "model.safetensors", "reference.safetensors"):
        if not os.path.isfile(os.path.join(args.src, f)):
            raise SystemExit(f"{os.path.join(args.src, f)} is missing")
    os.makedirs(args.out, exist_ok=True)

    tensors, meta = read_safetensors(os.path.join(args.src, "model.safetensors"))
    out, quantised, packed_out = {}, [], {}
    for name, (dtype, shape, raw) in tensors.items():
        if not is_target(name):
            out[name] = (dtype, shape, raw)
            packed_out[name] = (dtype, shape, raw)
            continue
        if dtype != "BF16" or len(shape) != 2:
            raise SystemExit(f"{name}: expected a 2-D BF16 weight, found {dtype} {shape}")
        if shape[1] % GROUP_SIZE:
            raise SystemExit(f"{name}: in-features {shape[1]} is not a multiple of {GROUP_SIZE}")
        new, err, (wq_raw, scales_raw, biases_raw) = fakequant(shape, raw, name)
        assert len(new) == len(raw)
        out[name] = (dtype, shape, new)
        prefix = name[: -len(".weight")]
        groups = shape[1] // GROUP_SIZE
        packed_out[name] = ("U32", [shape[0], shape[1] * BITS // 32], wq_raw)
        packed_out[prefix + ".scales"] = ("BF16", [shape[0], groups], scales_raw)
        packed_out[prefix + ".biases"] = ("BF16", [shape[0], groups], biases_raw)
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
    if args.packed_out:
        write_packed(args, tensors, packed_out, names, quantised, files)
    worst = max(q["max_abs_error"] for q in quantised)
    print(f"mlx {mx.__version__}: {len(quantised)} tensors fake-quantised (worst |dW| {worst:.3e}) -> {args.out}")


if __name__ == "__main__":
    main()
