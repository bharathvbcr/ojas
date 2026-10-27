//! The fixture machinery of `ojas_oracle::sam_fixtures`, run on small
//! synthetic fixtures whose "torch" outputs are computed here in f64, so the
//! glue (layouts, table splitting, head slicing, the gate itself) is checked
//! on every run. The real file is `tests/sam_fixtures_cpu.rs`.

mod support;

use ojas_core::{erf_f64, Budget, Numerics, OjasError};
use ojas_cpu::CpuBackend;
use ojas_oracle::sam_fixtures::{
    attention, axial_tables, compare, gelu, group_norm, group_norm_header,
    layer_norm_affine_recovered, rope_interleaved, skipped_ops, Fixture, Kind, Tolerance, GELU_TOL,
    ROPE_THETA,
};
use ojas_oracle::sam_fixtures::{
    conv2d, conv_transpose2d, layer_norm_rows, layer_norm_weighted, linear_bias, qkv_linear,
    weight_name, weighted_ops, ConvCase, VIT_LN_EPS, WEIGHTS_PREFIX,
};
use support::{build, f32_raw, Raw};

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 28)).with_numerics(Numerics::Exact)
}

/// Deterministic values in `[-amp, amp)`.
fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

const TIGHT: Tolerance = Tolerance {
    max_abs: 1e-6,
    max_rel: 1e-3,
    min_cosine: 1.0 - 1e-9,
};

#[test]
fn the_gate_passes_equal_tensors_and_fails_a_perturbed_one() {
    let want = values(64, 1, 3.0);
    let ok = compare("op", "d", Kind::Direct, &want, &want, TIGHT).unwrap();
    assert!(ok.pass);
    assert_eq!((ok.max_abs, ok.max_rel), (0.0, 0.0));
    assert!((ok.cosine - 1.0).abs() < 1e-15);

    let mut got = want.clone();
    got[7] += 1e-3;
    let bad = compare("op", "d", Kind::Direct, &got, &want, TIGHT).unwrap();
    assert!(!bad.pass);
    assert!((bad.max_abs - 1e-3).abs() < 1e-6, "{}", bad.max_abs);
    // Each bound alone can fail the gate.
    let loose = Tolerance {
        max_abs: 1.0,
        max_rel: 1e9,
        min_cosine: -1.0,
    };
    assert!(
        compare("op", "d", Kind::Direct, &got, &want, loose)
            .unwrap()
            .pass
    );
    for tighter in [
        Tolerance {
            max_abs: 1e-6,
            ..loose
        },
        Tolerance {
            max_rel: 1e-6,
            ..loose
        },
        Tolerance {
            min_cosine: 1.0,
            ..loose
        },
    ] {
        assert!(
            !compare("op", "d", Kind::Direct, &got, &want, tighter)
                .unwrap()
                .pass
        );
    }
}

#[test]
fn an_unmeasurable_result_is_an_error_not_a_number() {
    let want = [1.0f32, 2.0];
    assert!(compare("op", "d", Kind::Direct, &[1.0], &want, TIGHT).is_err());
    assert!(matches!(
        compare("op", "d", Kind::Direct, &[1.0, f32::NAN], &want, TIGHT),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(matches!(
        compare("op", "d", Kind::Direct, &[1.0, f32::INFINITY], &want, TIGHT),
        Err(OjasError::NonFinite { .. })
    ));
    assert!(compare("op", "d", Kind::Direct, &want, &[1.0, f32::NAN], TIGHT).is_err());
    // Two zero tensors agree; a zero against a non-zero does not.
    let z = compare("op", "d", Kind::Direct, &[0.0; 4], &[0.0; 4], TIGHT).unwrap();
    assert!(z.pass && z.cosine == 1.0);
    let nz = compare("op", "d", Kind::Direct, &[0.0; 2], &want, TIGHT).unwrap();
    assert!(!nz.pass && nz.cosine == 0.0);
}

#[test]
fn gelu_matches_an_f64_erf_and_a_wrong_output_fails() {
    let x = values(24, 2, 4.0);
    let want: Vec<f32> = x
        .iter()
        .map(|&v| {
            let v = f64::from(v);
            (0.5 * v * (1.0 + erf_f64(v / std::f64::consts::SQRT_2))) as f32
        })
        .collect();
    let wrong: Vec<f32> = x.iter().map(|&v| v.max(0.0)).collect();
    let bytes = build(
        &[
            f32_raw("a.in", &[2, 3, 4], &x),
            f32_raw("a.out", &[2, 3, 4], &want),
            f32_raw("b.out", &[2, 3, 4], &wrong),
        ],
        &[],
    );
    let fx = Fixture::parse(&bytes).unwrap();
    let ok = gelu(&cpu(), &fx, "a.in", "a.out").unwrap();
    assert!(ok.pass, "{ok}");
    assert_eq!(ok.tolerance, GELU_TOL);
    assert!(!gelu(&cpu(), &fx, "a.in", "b.out").unwrap().pass);
    // A shape disagreement is refused.
    let skew = build(
        &[
            f32_raw("a.in", &[2, 3, 4], &x),
            f32_raw("a.out", &[24], &want),
        ],
        &[],
    );
    assert!(gelu(&cpu(), &Fixture::parse(&skew).unwrap(), "a.in", "a.out").is_err());
}

/// `softmax(q k^T / sqrt(d)) v` per `[b, h]` plane, in f64.
fn attention_f64(q: &[f32], k: &[f32], v: &[f32], planes: usize, t: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; planes * t * d];
    for p in 0..planes {
        for i in 0..t {
            let s: Vec<f64> = (0..t)
                .map(|j| {
                    (0..d)
                        .map(|e| {
                            f64::from(q[(p * t + i) * d + e]) * f64::from(k[(p * t + j) * d + e])
                        })
                        .sum::<f64>()
                        / (d as f64).sqrt()
                })
                .collect();
            let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - max).exp()).sum();
            for e in 0..d {
                let acc: f64 = (0..t)
                    .map(|j| (s[j] - max).exp() / z * f64::from(v[(p * t + j) * d + e]))
                    .sum();
                out[(p * t + i) * d + e] = acc as f32;
            }
        }
    }
    out
}

#[test]
fn attention_matches_f64_checks_only_the_heads_asked_for_and_fails_a_wrong_head() {
    let (b, h, t, d) = (2usize, 3usize, 5usize, 4usize);
    let n = b * h * t * d;
    let (q, k, v) = (values(n, 3, 1.0), values(n, 4, 1.0), values(n, 5, 1.0));
    let out = attention_f64(&q, &k, &v, b * h, t, d);
    let shape = [b, h, t, d];
    let mut corrupted = out.clone();
    // Head 2 of batch row 0 (planes are [b][h]): outside a 1-head check,
    // inside the full one.
    corrupted[(2 * t) * d] += 0.5;
    let bytes = build(
        &[
            f32_raw("blk.attn.sdpa.q", &shape, &q),
            f32_raw("blk.attn.sdpa.k", &shape, &k),
            f32_raw("blk.attn.sdpa.v", &shape, &v),
            f32_raw("blk.attn.sdpa.out", &shape, &corrupted),
            f32_raw("good.attn.sdpa.q", &shape, &q),
            f32_raw("good.attn.sdpa.k", &shape, &k),
            f32_raw("good.attn.sdpa.v", &shape, &v),
            f32_raw("good.attn.sdpa.out", &shape, &out),
        ],
        &[],
    );
    let fx = Fixture::parse(&bytes).unwrap();
    let be = cpu();
    let all = attention(&be, &fx, "good", None).unwrap();
    assert!(all.pass, "{all}");
    assert_eq!(all.elements, n);
    assert!(all.detail.contains("heads 3 of 3"), "{}", all.detail);
    let full = attention(&be, &fx, "blk", None).unwrap();
    assert!(!full.pass, "a wrong head must fail the full check: {full}");
    let one = attention(&be, &fx, "blk", Some(1)).unwrap();
    assert!(one.pass, "head 0 is intact: {one}");
    assert_eq!(one.elements, b * t * d);
    assert!(one.detail.contains("heads 1 of 3"), "{}", one.detail);
    assert!(attention(&be, &fx, "good", Some(0)).is_err());
    // A head count above H is the whole tensor.
    assert_eq!(attention(&be, &fx, "good", Some(99)).unwrap().elements, n);
}

/// Torch's `freqs_cis` `[t, half, 2]` of a `[grid_h, grid_w]` row-major grid, in
/// f64 from the formula of `sam3.sam.rope.compute_axial_cis`.
fn freqs_cis_f64(head_dim: usize, grid: [usize; 2], scale: f64) -> Vec<f32> {
    let (half, quarter) = (head_dim / 2, head_dim / 4);
    let freqs: Vec<f64> = (0..quarter)
        .map(|j| 1.0 / ROPE_THETA.powf((4 * j) as f64 / head_dim as f64))
        .collect();
    let mut out = Vec::new();
    for tok in 0..grid[0] * grid[1] {
        let (x, y) = (
            (tok % grid[1]) as f64 * scale,
            (tok / grid[1]) as f64 * scale,
        );
        for pair in 0..half {
            let (pos, f) = if pair < quarter {
                (x, freqs[pair])
            } else {
                (y, freqs[pair - quarter])
            };
            out.push((pos * f).cos() as f32);
            out.push((pos * f).sin() as f32);
        }
    }
    out
}

#[test]
fn axial_tables_and_rope_match_a_formula_fixture() {
    let (head_dim, grid) = (8usize, [3usize, 2usize]);
    let (t, half) = (6usize, 4usize);
    let freqs = freqs_cis_f64(head_dim, grid, 1.0);
    // Heads 2, batch 2: x in torch's [B, H, T, D].
    let (b, h) = (2usize, 2usize);
    let x = values(b * h * t * head_dim, 6, 2.0);
    let mut want = vec![0.0f32; x.len()];
    for plane in 0..b * h {
        for tok in 0..t {
            for pair in 0..half {
                let base = (plane * t + tok) * head_dim + 2 * pair;
                let (c, s) = (
                    f64::from(freqs[(tok * half + pair) * 2]),
                    f64::from(freqs[(tok * half + pair) * 2 + 1]),
                );
                let (re, im) = (f64::from(x[base]), f64::from(x[base + 1]));
                want[base] = (re * c - im * s) as f32;
                want[base + 1] = (re * s + im * c) as f32;
            }
        }
    }
    let wrong_freqs: Vec<f32> = freqs.iter().map(|v| v * 0.5).collect();
    let bytes = build(
        &[
            f32_raw("blk.attn.rope.freqs_cis", &[t, half, 2], &freqs),
            f32_raw("blk.attn.rope.q_in", &[b, h, t, head_dim], &x),
            f32_raw("blk.attn.rope.q_out", &[b, h, t, head_dim], &want),
            f32_raw("blk.attn.rope.k_in", &[b, h, t, head_dim], &x),
            f32_raw("blk.attn.rope.k_out", &[b, h, t, head_dim], &x),
            f32_raw("bad.attn.rope.freqs_cis", &[t, half, 2], &wrong_freqs),
        ],
        &[],
    );
    let fx = Fixture::parse(&bytes).unwrap();
    let be = cpu();
    let tables = axial_tables(&be, &fx, "blk", grid, 1.0).unwrap();
    assert!(tables.pass, "{tables}");
    let q = rope_interleaved(&be, &fx, "blk", "q").unwrap();
    assert!(q.pass, "{q}");
    // k_out is recorded un-rotated, so the rotation must be seen to differ.
    let k = rope_interleaved(&be, &fx, "blk", "k").unwrap();
    assert!(!k.pass, "an unrotated k_out must fail: {k}");
    // The wrong scale and the wrong table both fail the table check.
    assert!(!axial_tables(&be, &fx, "blk", grid, 0.5).unwrap().pass);
    assert!(!axial_tables(&be, &fx, "bad", grid, 1.0).unwrap().pass);
    // A freqs_cis whose rows do not match the tensor is refused.
    let skew = build(
        &[
            f32_raw(
                "s.attn.rope.freqs_cis",
                &[t - 1, half, 2],
                &freqs[..(t - 1) * half * 2],
            ),
            f32_raw("s.attn.rope.q_in", &[b, h, t, head_dim], &x),
            f32_raw("s.attn.rope.q_out", &[b, h, t, head_dim], &want),
        ],
        &[],
    );
    assert!(rope_interleaved(&be, &Fixture::parse(&skew).unwrap(), "s", "q").is_err());
}

/// `(x - mean) / sqrt(var + eps) * w + b` per row, in f64.
fn layer_norm_f64(x: &[f32], dim: usize, eps: f64, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(dim) {
        let r: Vec<f64> = row.iter().map(|&v| f64::from(v)).collect();
        let mean = r.iter().sum::<f64>() / dim as f64;
        let var = r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dim as f64;
        let rstd = 1.0 / (var + eps).sqrt();
        for (c, &v) in r.iter().enumerate() {
            out.push(((v - mean) * rstd * f64::from(w[c]) + f64::from(b[c])) as f32);
        }
    }
    out
}

#[test]
fn layer_norm_with_a_recovered_affine_part_sees_the_row_statistics() {
    let (rows, dim) = (64usize, 16usize);
    // Rows of very different spread, so the epsilon matters per row.
    let mut x = values(rows * dim, 7, 1.0);
    for (r, row) in x.chunks_exact_mut(dim).enumerate() {
        let spread = 0.01 + r as f32 * 0.1;
        row.iter_mut()
            .for_each(|v| *v = *v * spread + r as f32 * 0.3);
    }
    let w = values(dim, 8, 2.0);
    let b = values(dim, 9, 1.0);
    let good = layer_norm_f64(&x, dim, 1e-5, &w, &b);
    // The same op made with a hundred times the epsilon: not a per-channel
    // error, so the fit cannot absorb it.
    let off = layer_norm_f64(&x, dim, 1e-3, &w, &b);
    let bytes = build(
        &[
            f32_raw("x", &[rows, dim], &x),
            f32_raw("good", &[rows, dim], &good),
            f32_raw("off", &[rows, dim], &off),
        ],
        &[],
    );
    let fx = Fixture::parse(&bytes).unwrap();
    let be = cpu();
    let ok = layer_norm_affine_recovered(&be, &fx, "x", "good", 1e-5).unwrap();
    assert!(ok.pass, "{ok}");
    assert_eq!(ok.kind, Kind::Derived);
    let bad = layer_norm_affine_recovered(&be, &fx, "x", "off", 1e-5).unwrap();
    assert!(!bad.pass, "a wrong epsilon must be seen: {bad}");
}

#[test]
fn the_skipped_ops_are_named_while_the_file_has_no_weights() {
    let bytes = build(&[f32_raw("act", &[1], &[1.0])], &[]);
    let fx = Fixture::parse(&bytes).unwrap();
    let skipped = skipped_ops(&fx);
    let ops: Vec<&str> = skipped.iter().map(|s| s.op).collect();
    assert_eq!(
        ops,
        [
            "layer_norm_forward (direct)",
            "linear_bias_forward",
            "conv2d_forward",
            "conv_transpose2d_forward"
        ]
    );
    assert!(skipped.iter().all(|s| s.reason.contains("1 tensors")));
    // Once the file holds a weight, nothing is skipped on its account.
    let bytes = build(
        &[
            f32_raw("act", &[1], &[1.0]),
            f32_raw("norm.weight", &[1], &[1.0]),
        ],
        &[],
    );
    assert!(skipped_ops(&Fixture::parse(&bytes).unwrap()).is_empty());
}

#[test]
fn a_header_claiming_too_many_elements_is_refused_before_it_is_read() {
    let huge = Raw {
        name: "t".into(),
        dtype: "F32",
        shape: vec![1 << 20, 1 << 20],
        bytes: Vec::new(),
    };
    // The file is far shorter than the shape claims, so the reader refuses
    // it first; either way no tensor of that size is produced.
    let bytes = build(&[huge], &[]);
    let parsed = Fixture::parse(&bytes);
    assert!(parsed.is_err() || parsed.unwrap().tensor("t").is_err());
}

/// `(x - mean) / sqrt(var + eps) * w[c] + b[c]` per `(image, group)`, in f64.
fn group_norm_f64(
    x: &[f32],
    [n, c, hw]: [usize; 3],
    groups: usize,
    eps: f64,
    w: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let per = c / groups;
    let len = per * hw;
    let mut out = vec![0.0f32; x.len()];
    for img in 0..n {
        for g in 0..groups {
            let start = (img * c + g * per) * hw;
            let vals: Vec<f64> = x[start..start + len]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let mean = vals.iter().sum::<f64>() / len as f64;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / len as f64;
            let rstd = 1.0 / (var + eps).sqrt();
            for (i, v) in vals.iter().enumerate() {
                let ch = g * per + i / hw;
                out[start + i] = ((v - mean) * rstd * f64::from(w[ch]) + f64::from(b[ch])) as f32;
            }
        }
    }
    out
}

#[test]
fn group_norm_matches_an_f64_reference_and_sees_a_wrong_grouping_or_eps() {
    let (n, c, h, w) = (2usize, 8usize, 5usize, 6usize);
    let x = values(n * c * h * w, 21, 3.0);
    let (wt, bs) = (values(c, 22, 2.0), values(c, 23, 1.0));
    let good = group_norm_f64(&x, [n, c, h * w], 4, 1e-5, &wt, &bs);
    let other_groups = group_norm_f64(&x, [n, c, h * w], 2, 1e-5, &wt, &bs);
    let other_eps = group_norm_f64(&x, [n, c, h * w], 4, 1.0, &wt, &bs);
    let shape = [n, c, h, w];
    let bytes = build(
        &[
            f32_raw("gn.input", &shape, &x),
            f32_raw("gn.weight", &[c], &wt),
            f32_raw("gn.bias", &[c], &bs),
            f32_raw("gn.output", &shape, &good),
            f32_raw("bg.input", &shape, &x),
            f32_raw("bg.weight", &[c], &wt),
            f32_raw("bg.bias", &[c], &bs),
            f32_raw("bg.output", &shape, &other_groups),
            f32_raw("be.input", &shape, &x),
            f32_raw("be.weight", &[c], &wt),
            f32_raw("be.bias", &[c], &bs),
            f32_raw("be.output", &shape, &other_eps),
        ],
        &[("groups", "4"), ("eps", "1e-5")],
    );
    let fx = Fixture::parse(&bytes).unwrap();
    assert_eq!(group_norm_header(&fx).unwrap(), (4, 1e-5));
    let be = cpu();
    let ok = group_norm(&be, &fx, "gn", 4, 1e-5).unwrap();
    assert!(ok.pass, "{ok}");
    assert_eq!(ok.kind, Kind::Direct);
    assert!(!group_norm(&be, &fx, "bg", 4, 1e-5).unwrap().pass);
    assert!(!group_norm(&be, &fx, "be", 4, 1e-5).unwrap().pass);
    // The call must agree with the header, and a missing tensor is an error.
    assert!(group_norm(&be, &fx, "gn", 2, 1e-5).is_err());
    assert!(group_norm(&be, &fx, "gn", 4, 1e-6).is_err());
    assert!(group_norm(&be, &fx, "absent", 4, 1e-5).is_err());
}

#[test]
fn a_group_norm_fixture_without_a_usable_header_is_refused() {
    let x = values(8, 31, 1.0);
    let tensors = || {
        [
            f32_raw("gn.input", &[1, 2, 2, 2], &x),
            f32_raw("gn.weight", &[2], &[1.0, 1.0]),
            f32_raw("gn.bias", &[2], &[0.0, 0.0]),
            f32_raw("gn.output", &[1, 2, 2, 2], &x),
        ]
    };
    let be = cpu();
    for metadata in [
        &[][..],
        &[("groups", "2")],
        &[("eps", "1e-5")],
        &[("groups", "two"), ("eps", "1e-5")],
        &[("groups", "2"), ("eps", "tiny")],
    ] {
        let bytes = build(&tensors(), metadata);
        let fx = Fixture::parse(&bytes).unwrap();
        assert!(group_norm_header(&fx).is_err(), "{metadata:?}");
        assert!(group_norm(&be, &fx, "gn", 2, 1e-5).is_err(), "{metadata:?}");
    }
}

fn wname(module: &str, part: &str) -> String {
    format!("{WEIGHTS_PREFIX}{module}.{part}")
}

#[test]
fn weight_names_map_the_fixture_modules_to_the_checkpoint() {
    let p = "detector.backbone.vision_backbone.";
    for (module, part, want) in [
        (
            "trunk.blocks.0.attn.qkv",
            "weight",
            "trunk.blocks.0.attn.qkv.weight",
        ),
        (
            "trunk.blocks.7.mlp.fc2",
            "bias",
            "trunk.blocks.7.mlp.fc2.bias",
        ),
        ("trunk.ln_pre", "weight", "trunk.ln_pre.weight"),
        (
            "trunk.patch_embed.proj",
            "weight",
            "trunk.patch_embed.proj.weight",
        ),
        // The neck is `convs.N` in the checkpoint, with no `neck.` segment.
        ("neck.convs.0.conv_3x3", "weight", "convs.0.conv_3x3.weight"),
        (
            "neck.convs.0.dconv_2x2_1",
            "bias",
            "convs.0.dconv_2x2_1.bias",
        ),
        ("neck.convs.2.conv_1x1", "weight", "convs.2.conv_1x1.weight"),
    ] {
        assert_eq!(weight_name(module, part).unwrap(), format!("{p}{want}"));
    }
    for (module, part) in [
        ("neck.pos_enc.0", "weight"),
        ("tracker.model.x", "weight"),
        ("convs.0.conv_3x3", "weight"),
        ("trunk.blocks.0.norm1", "scale"),
        ("", "bias"),
    ] {
        assert!(weight_name(module, part).is_err(), "{module} {part}");
    }
}

fn mat(x: &[f64], rows: usize, cols: usize, w: &[f32], out: usize, b: &[f32]) -> Vec<f32> {
    // y[r][o] = sum_c x[r][c] * w[o][c] + b[o], in f64.
    let mut y = Vec::with_capacity(rows * out);
    for r in 0..rows {
        for o in 0..out {
            let dot: f64 = (0..cols)
                .map(|c| x[r * cols + c] * f64::from(w[o * cols + c]))
                .sum();
            y.push((dot + f64::from(b[o])) as f32);
        }
    }
    y
}

#[test]
fn layer_norm_and_linear_use_the_real_weights_and_see_a_wrong_bias() {
    let be = cpu();
    // LayerNorm with a real affine.
    let (rows, dim) = (6usize, 8usize);
    let x = values(rows * dim, 40, 2.0);
    let (w, b) = (values(dim, 41, 1.5), values(dim, 42, 1.0));
    let want = layer_norm_f64(&x, dim, 1e-6, &w, &b);
    // Linear: x [2, 3, 4] -> [2, 3, 5].
    let lx = values(24, 43, 2.0);
    let (lw, lb) = (values(20, 44, 0.5), values(5, 45, 1.0));
    let lx64: Vec<f64> = lx.iter().map(|&v| f64::from(v)).collect();
    let good = mat(&lx64, 6, 4, &lw, 5, &lb);
    let lb_bad: Vec<f32> = lb.iter().map(|v| v + 1e-2).collect();
    let bad_out = mat(&lx64, 6, 4, &lw, 5, &lb_bad);
    let acts = build(
        &[
            f32_raw("n.in", &[rows, dim], &x),
            f32_raw("n.out", &[rows, dim], &want),
            f32_raw("l.in", &[2, 3, 4], &lx),
            f32_raw("l.out", &[2, 3, 5], &good),
            f32_raw("l.bad", &[2, 3, 5], &bad_out),
        ],
        &[],
    );
    let weights = build(
        &[
            f32_raw(&wname("trunk.ln_pre", "weight"), &[dim], &w),
            f32_raw(&wname("trunk.ln_pre", "bias"), &[dim], &b),
            f32_raw(&wname("trunk.blocks.0.mlp.fc1", "weight"), &[5, 4], &lw),
            f32_raw(&wname("trunk.blocks.0.mlp.fc1", "bias"), &[5], &lb),
        ],
        &[],
    );
    let (a, wt) = (
        Fixture::parse(&acts).unwrap(),
        Fixture::parse(&weights).unwrap(),
    );
    let ln = layer_norm_weighted(&be, &a, &wt, "trunk.ln_pre", "n.in", "n.out", 1e-6).unwrap();
    assert!(ln.pass && ln.kind == Kind::Direct, "{ln}");
    let m = "trunk.blocks.0.mlp.fc1";
    let ok = linear_bias(&be, &a, &wt, m, "l.in", "l.out").unwrap();
    assert!(ok.pass, "{ok}");
    // A bias that is off by 1e-2 is outside LINEAR_TOL's 1e-3.
    let bad = linear_bias(&be, &a, &wt, m, "l.in", "l.bad").unwrap();
    assert!(!bad.pass, "{bad}");
    // A missing weight is an error, not a pass.
    assert!(linear_bias(&be, &a, &wt, "trunk.blocks.0.mlp.fc2", "l.in", "l.out").is_err());
    // A weight of the wrong shape is refused.
    assert!(linear_bias(&be, &a, &wt, m, "n.in", "n.out").is_err());
}

#[test]
fn qkv_cuts_windows_in_row_major_grid_order_and_a_global_block_does_not() {
    let be = cpu();
    // norm1 [1, 4, 4, 4]; windows of 2: [4, 2, 2, 4].
    let n1 = values(64, 50, 2.0);
    let (w, b) = (values(24, 51, 0.5), values(6, 52, 1.0));
    let at = |r: usize, c: usize, ch: usize| f64::from(n1[(r * 4 + c) * 4 + ch]);
    let mut windowed = Vec::new();
    for win in 0..4 {
        let (wi, wj) = (win / 2, win % 2);
        for r in 0..2 {
            for c in 0..2 {
                windowed.extend((0..4).map(|ch| at(2 * wi + r, 2 * wj + c, ch)));
            }
        }
    }
    let global: Vec<f64> = n1.iter().map(|&v| f64::from(v)).collect();
    let acts = build(
        &[
            f32_raw("trunk.blocks.0.norm1", &[1, 4, 4, 4], &n1),
            f32_raw(
                "trunk.blocks.0.attn.qkv",
                &[4, 2, 2, 6],
                &mat(&windowed, 16, 4, &w, 6, &b),
            ),
            f32_raw("trunk.blocks.7.norm1", &[1, 4, 4, 4], &n1),
            f32_raw(
                "trunk.blocks.7.attn.qkv",
                &[1, 4, 4, 6],
                &mat(&global, 16, 4, &w, 6, &b),
            ),
        ],
        &[],
    );
    let mut tensors = Vec::new();
    for block in ["trunk.blocks.0", "trunk.blocks.7"] {
        tensors.push(f32_raw(
            &wname(&format!("{block}.attn.qkv"), "weight"),
            &[6, 4],
            &w,
        ));
        tensors.push(f32_raw(
            &wname(&format!("{block}.attn.qkv"), "bias"),
            &[6],
            &b,
        ));
    }
    let weights = build(&tensors, &[]);
    let (a, wt) = (
        Fixture::parse(&acts).unwrap(),
        Fixture::parse(&weights).unwrap(),
    );
    let win = qkv_linear(&be, &a, &wt, "trunk.blocks.0", Some(2)).unwrap();
    assert!(win.pass, "{win}");
    // The windowed row says its input is derived.
    assert!(
        win.detail.contains("derived via window_partition"),
        "{}",
        win.detail
    );
    let glob = qkv_linear(&be, &a, &wt, "trunk.blocks.7", None).unwrap();
    assert!(glob.pass && !glob.detail.contains("derived"), "{glob}");
    // The wrong order (no window against a windowed output) fails.
    assert!(qkv_linear(&be, &a, &wt, "trunk.blocks.0", None).is_err());
}

#[test]
fn conv_and_conv_transpose_use_the_real_weights_layouts_and_nhwc() {
    let be = cpu();
    // Patch embedding: [1, 2, 6, 6], weight [4, 2, 2, 2], stride 2, no bias,
    // NCHW [1, 4, 3, 3] recorded as NHWC [1, 3, 3, 4].
    let x = values(72, 60, 2.0);
    let w = values(32, 61, 0.5);
    let px = |c: usize, r: usize, q: usize| f64::from(x[(c * 6 + r) * 6 + q]);
    let mut nchw = vec![0.0f32; 4 * 9];
    for o in 0..4 {
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0f64;
                for c in 0..2 {
                    for a in 0..2 {
                        for b in 0..2 {
                            acc += px(c, 2 * i + a, 2 * j + b)
                                * f64::from(w[((o * 2 + c) * 2 + a) * 2 + b]);
                        }
                    }
                }
                nchw[(o * 3 + i) * 3 + j] = acc as f32;
            }
        }
    }
    let mut nhwc = vec![0.0f32; 36];
    for o in 0..4 {
        for i in 0..3 {
            for j in 0..3 {
                nhwc[(i * 3 + j) * 4 + o] = nchw[(o * 3 + i) * 3 + j];
            }
        }
    }
    // A 3x3, padding 1 convolution with a bias: [1, 2, 4, 4] -> [1, 3, 4, 4].
    let cx = values(32, 62, 2.0);
    let (cw, cb) = (values(54, 63, 0.5), values(3, 64, 1.0));
    let mut conv = vec![0.0f32; 3 * 16];
    for o in 0..3 {
        for i in 0..4i64 {
            for j in 0..4i64 {
                let mut acc = f64::from(cb[o]);
                for c in 0..2 {
                    for a in 0..3i64 {
                        for b in 0..3i64 {
                            let (r, q) = (i + a - 1, j + b - 1);
                            if (0..4).contains(&r) && (0..4).contains(&q) {
                                acc += f64::from(cx[(c * 4 + r as usize) * 4 + q as usize])
                                    * f64::from(
                                        cw[((o * 2 + c) * 3 + a as usize) * 3 + b as usize],
                                    );
                            }
                        }
                    }
                }
                conv[(o * 4 + i as usize) * 4 + j as usize] = acc as f32;
            }
        }
    }
    // A 2x2 stride-2 transposed convolution: [1, 2, 3, 3], weight [2, 3, 2, 2].
    let tx = values(18, 65, 2.0);
    let (tw, tb) = (values(24, 66, 0.5), values(3, 67, 1.0));
    let mut tconv = vec![0.0f32; 3 * 36];
    for o in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                for a in 0..2 {
                    for b in 0..2 {
                        let mut acc = f64::from(tb[o]);
                        for c in 0..2 {
                            acc += f64::from(tx[(c * 3 + i) * 3 + j])
                                * f64::from(tw[((c * 3 + o) * 2 + a) * 2 + b]);
                        }
                        tconv[(o * 6 + 2 * i + a) * 6 + 2 * j + b] = acc as f32;
                    }
                }
            }
        }
    }
    let acts = build(
        &[
            f32_raw("p.in", &[1, 2, 6, 6], &x),
            f32_raw("p.out", &[1, 3, 3, 4], &nhwc),
            f32_raw("p.nchw", &[1, 4, 3, 3], &nchw),
            f32_raw("neck.convs.0.conv_3x3.in", &[1, 2, 4, 4], &cx),
            f32_raw("neck.convs.0.conv_3x3", &[1, 3, 4, 4], &conv),
            f32_raw("neck.convs.1.dconv_2x2.in", &[1, 2, 3, 3], &tx),
            f32_raw("neck.convs.1.dconv_2x2", &[1, 3, 6, 6], &tconv),
        ],
        &[],
    );
    let weights = build(
        &[
            f32_raw(
                &wname("trunk.patch_embed.proj", "weight"),
                &[4, 2, 2, 2],
                &w,
            ),
            f32_raw(&wname("convs.0.conv_3x3", "weight"), &[3, 2, 3, 3], &cw),
            f32_raw(&wname("convs.0.conv_3x3", "bias"), &[3], &cb),
            f32_raw(&wname("convs.1.dconv_2x2", "weight"), &[2, 3, 2, 2], &tw),
            f32_raw(&wname("convs.1.dconv_2x2", "bias"), &[3], &tb),
        ],
        &[],
    );
    let (a, wt) = (
        Fixture::parse(&acts).unwrap(),
        Fixture::parse(&weights).unwrap(),
    );
    let patch = ConvCase {
        module: "trunk.patch_embed.proj",
        input: "p.in",
        output: "p.out",
        params: ojas_core::Conv2dParams {
            stride: [2, 2],
            ..ojas_core::Conv2dParams::new()
        },
        has_bias: false,
        output_nhwc: true,
    };
    let r = conv2d(&be, &a, &wt, &patch).unwrap();
    assert!(
        r.pass && r.detail.contains("NHWC") && r.detail.contains("no bias"),
        "{r}"
    );
    // Without the permute the NCHW record matches, and the NHWC one does not.
    let plain = ConvCase {
        output: "p.nchw",
        output_nhwc: false,
        ..patch
    };
    assert!(conv2d(&be, &a, &wt, &plain).unwrap().pass);
    assert!(conv2d(
        &be,
        &a,
        &wt,
        &ConvCase {
            output_nhwc: false,
            ..patch
        }
    )
    .is_err());
    let three = ConvCase {
        module: "neck.convs.0.conv_3x3",
        input: "neck.convs.0.conv_3x3.in",
        output: "neck.convs.0.conv_3x3",
        params: ojas_core::Conv2dParams {
            padding: [1, 1],
            ..ojas_core::Conv2dParams::new()
        },
        has_bias: true,
        output_nhwc: false,
    };
    assert!(conv2d(&be, &a, &wt, &three).unwrap().pass);
    // No padding changes the output shape: refused, not a pass.
    let unpadded = ConvCase {
        params: ojas_core::Conv2dParams::new(),
        ..three
    };
    assert!(conv2d(&be, &a, &wt, &unpadded).is_err());
    let t = conv_transpose2d(
        &be,
        &a,
        &wt,
        "neck.convs.1.dconv_2x2",
        "neck.convs.1.dconv_2x2.in",
        "neck.convs.1.dconv_2x2",
        2,
    )
    .unwrap();
    assert!(t.pass, "{t}");
    // Stride 1 would not give the recorded size.
    assert!(conv_transpose2d(
        &be,
        &a,
        &wt,
        "neck.convs.1.dconv_2x2",
        "neck.convs.1.dconv_2x2.in",
        "neck.convs.1.dconv_2x2",
        1,
    )
    .is_err());
}

#[test]
fn a_weights_file_missing_names_is_an_error_not_an_empty_list() {
    let weights = build(
        &[f32_raw(
            &wname("trunk.ln_pre", "weight"),
            &[1024],
            &[0.0f32; 1024],
        )],
        &[],
    );
    let w = Fixture::parse(&weights).unwrap();
    assert!(weighted_ops(&w).is_err());
}

/// Rows whose variance is about 3e-4 (a mean of 10, a spread of 0.03), where
/// eps 1e-5 against 1e-6 moves the output by about 1.5 per cent: the case the
/// real file only catches through one row's relative error.
fn low_variance_norm_fixture() -> (Vec<u8>, Vec<u8>) {
    let (rows, dim) = (6usize, 16usize);
    let mut acts = Vec::new();
    let mut weights = Vec::new();
    let mut seed = 70u64;
    let mut modules = vec![("trunk.ln_pre".to_string(), "trunk.ln_pre.out".to_string())];
    for block in [0, 7] {
        for norm in ["norm1", "norm2"] {
            let m = format!("trunk.blocks.{block}.{norm}");
            modules.push((m.clone(), m));
        }
    }
    for (module, out) in &modules {
        seed += 1;
        let x: Vec<f32> = values(rows * dim, seed, 0.03)
            .iter()
            .map(|v| v + 10.0)
            .collect();
        let w: Vec<f32> = values(dim, seed + 100, 0.5)
            .iter()
            .map(|v| v + 1.0)
            .collect();
        let b = values(dim, seed + 200, 1.0);
        // "torch": eps 1e-5, in f64.
        let want = layer_norm_f64(&x, dim, 1e-5, &w, &b);
        acts.push(f32_raw(&format!("{out}.in"), &[rows, dim], &x));
        acts.push(f32_raw(out, &[rows, dim], &want));
        weights.push(f32_raw(&wname(module, "weight"), &[dim], &w));
        weights.push(f32_raw(&wname(module, "bias"), &[dim], &b));
    }
    (build(&acts, &[]), build(&weights, &[]))
}

#[test]
fn a_reverted_vit_ln_eps_fails_without_the_real_files() {
    assert_eq!(VIT_LN_EPS, 1e-5);
    let be = cpu();
    let (acts, weights) = low_variance_norm_fixture();
    let (a, w) = (
        Fixture::parse(&acts).unwrap(),
        Fixture::parse(&weights).unwrap(),
    );
    // The helper, at the constant, matches torch's eps on every norm.
    let rows = layer_norm_rows(&be, &a, &w).unwrap();
    assert_eq!(rows.len(), 5);
    assert!(rows.iter().all(|r| r.pass), "{rows:#?}");
    // The same rows at the old eps 1e-6 fail LAYER_NORM_TOL, abs and rel.
    for (module, input, output) in [
        ("trunk.ln_pre", "trunk.ln_pre.out.in", "trunk.ln_pre.out"),
        (
            "trunk.blocks.0.norm1",
            "trunk.blocks.0.norm1.in",
            "trunk.blocks.0.norm1",
        ),
        (
            "trunk.blocks.7.norm2",
            "trunk.blocks.7.norm2.in",
            "trunk.blocks.7.norm2",
        ),
    ] {
        let old = layer_norm_weighted(&be, &a, &w, module, input, output, 1e-6).unwrap();
        assert!(!old.pass && old.max_abs > 1e-4, "{old}");
    }
}

#[test]
fn a_transposed_square_weight_is_seen_as_a_wrong_answer_not_a_shape_error() {
    let be = cpu();
    // proj is the one square weight in SAM's trunk: [4, 4] here.
    let x = values(24, 80, 2.0);
    let (w, b) = (values(16, 81, 0.5), values(4, 82, 1.0));
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let want = mat(&x64, 6, 4, &w, 4, &b);
    let mut wt = vec![0.0f32; 16];
    for o in 0..4 {
        for c in 0..4 {
            wt[c * 4 + o] = w[o * 4 + c];
        }
    }
    let acts = build(
        &[
            f32_raw("p.in", &[2, 3, 4], &x),
            f32_raw("p.out", &[2, 3, 4], &want),
        ],
        &[],
    );
    let mk = |weight: &[f32]| {
        build(
            &[
                f32_raw(
                    &wname("trunk.blocks.0.attn.proj", "weight"),
                    &[4, 4],
                    weight,
                ),
                f32_raw(&wname("trunk.blocks.0.attn.proj", "bias"), &[4], &b),
            ],
            &[],
        )
    };
    let (good, bad) = (mk(&w), mk(&wt));
    let a = Fixture::parse(&acts).unwrap();
    let m = "trunk.blocks.0.attn.proj";
    let ok = linear_bias(&be, &a, &Fixture::parse(&good).unwrap(), m, "p.in", "p.out").unwrap();
    assert!(ok.pass, "{ok}");
    // The shape is right, so only the comparison can refuse it.
    let transposed =
        linear_bias(&be, &a, &Fixture::parse(&bad).unwrap(), m, "p.in", "p.out").unwrap();
    assert!(!transposed.pass, "{transposed}");
}

#[test]
fn fixture_shape_reads_the_header_and_refuses_an_unknown_name() {
    let bytes = build(&[f32_raw("a", &[2, 3, 4], &values(24, 90, 1.0))], &[]);
    let fx = Fixture::parse(&bytes).unwrap();
    assert_eq!(fx.shape("a").unwrap(), vec![2, 3, 4]);
    assert!(fx.shape("missing").is_err());
}
