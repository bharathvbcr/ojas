"""Refusals and matching rules of qwen35_fakequant_base.py (needs mlx).

    /Users/bharath/.venvs/ml/bin/python3 -I ojas-oracle/python/test_qwen35_fakequant_base.py
"""
import importlib.util
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("fq", os.path.join(HERE, "qwen35_fakequant_base.py"))
fq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fq)


def bf16_raw(values):
    """bf16 bytes of float32 `values` (exact for values that are already bf16)."""
    f = np.ascontiguousarray(values, dtype=np.float32)
    return (f.view(np.uint32) >> 16).astype("<u2").tobytes()


class FakeQuant(unittest.TestCase):
    def test_ten_suffixes_match_exactly_and_only_under_model_layers(self):
        hits = [
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.layers.0.linear_attn.in_proj_z.weight",
            "model.layers.0.linear_attn.out_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
        misses = [
            "model.layers.0.linear_attn.in_proj_a.weight",
            "model.layers.0.linear_attn.in_proj_b.weight",
            "model.layers.0.self_attn.out_proj.weight",
            "model.layers.0.linear_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.bias",
            "model.layers.0.xmlp.gate_proj.weight",
            "model.visual.blocks.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "model.layers.0.self_attn.q_proj.weight.extra",
        ]
        for n in hits:
            self.assertTrue(fq.is_target(n), n)
        for n in misses:
            self.assertFalse(fq.is_target(n), n)

    def test_a_constant_group_survives_and_a_normal_group_is_close(self):
        rng = np.random.default_rng(0)
        w = rng.standard_normal((2, 128)).astype(np.float32)
        w[0, :64] = 0.5  # a constant group: scale would be 0
        w[1, :64] = 0.0  # an all-zero group
        raw = bf16_raw(w)
        new, err = fq.fakequant([2, 128], raw, "t")
        got = (np.frombuffer(new, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(2, 128)
        want = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(2, 128)
        self.assertTrue(np.isfinite(got).all())
        self.assertTrue(np.allclose(got[0, :64], want[0, :64], rtol=2**-6))
        self.assertTrue(np.all(got[1, :64] == 0.0))
        self.assertLess(err, 0.1)

    def test_non_finite_weights_are_refused(self):
        for bad in (np.nan, np.inf, -np.inf):
            w = np.ones((1, 64), dtype=np.float32)
            w[0, 5] = bad
            with self.assertRaises(SystemExit):
                fq.fakequant([1, 64], bf16_raw(w), "t")

    def test_src_equal_to_out_is_refused(self):
        argv = sys.argv
        try:
            sys.argv = ["x", "--src", HERE, "--out", HERE]
            with self.assertRaises(SystemExit):
                fq.main()
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
