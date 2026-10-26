//! The fixture machinery of `ojas_oracle::sam_fixtures`, run on small
//! synthetic fixtures whose "torch" outputs are computed here in f64, so the
//! glue (layouts, table splitting, head slicing, the gate itself) is checked
//! on every run. The real file is `tests/sam_fixtures_cpu.rs`.

mod support;

use ojas_core::{erf_f64, Budget, Numerics, OjasError};
use ojas_cpu::CpuBackend;
use ojas_oracle::sam_fixtures::{
    attention, axial_tables, compare, gelu, layer_norm_affine_recovered, rope_interleaved,
    skipped_ops, Fixture, Kind, Tolerance, GELU_TOL, ROPE_THETA,
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
