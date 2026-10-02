//! Shared by `tests/device_gdn_published.rs` (the device) and
//! `tests/device_gdn_published_mirror.rs` (the host mirror): the published
//! corpus **embedded in the test binary** (the device binary runs on the box,
//! where this checkout's paths do not exist), the float64 reference driven on
//! a case, and the judges that apply `gdn_host::published_bounds`.
//!
//! The including test declares `mod reference;` before `mod
//! device_gdn_published_common;`.
#![allow(dead_code)]

use ojas_qwen35_cuda::gdn_host::{
    published_bounds, rel_of_max, GdnPublishedCase, GdnPublishedOutputs,
};
use ojas_qwen35_cuda::gdn_plan::GDN_DK;

use crate::reference::gdn_published::{
    gdn_train_published_bwd_f64, gdn_train_published_fwd_f64, Inputs, Shape,
};
use crate::reference::{npy, sha256};

/// One corpus case's files, as bytes compiled into the binary.
pub struct CorpusFiles {
    pub name: &'static str,
    pub q: &'static [u8],
    pub k: &'static [u8],
    pub v: &'static [u8],
    pub alpha: &'static [u8],
    pub beta: &'static [u8],
    pub golden: &'static [u8],
}

macro_rules! corpus_file {
    ($name:literal, $suffix:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/gdn/gdn_published_",
            $name,
            "_",
            $suffix,
            ".npy"
        ))
    };
}

macro_rules! corpus_case {
    ($name:literal) => {
        CorpusFiles {
            name: $name,
            q: corpus_file!($name, "q"),
            k: corpus_file!($name, "k"),
            v: corpus_file!($name, "v"),
            alpha: corpus_file!($name, "alpha"),
            beta: corpus_file!($name, "beta"),
            golden: corpus_file!($name, "y_seq_f64"),
        }
    };
}

/// tessl's 13 published cases (`tests/fixtures/gdn/`, copied byte-identical
/// by L-cuda-oracle): T = 1, 63, 64, 65, 127 (three chunk sizes), 129, 1000,
/// 8191, two grouped-head cases and the tiny-alpha clamp probe.
pub const PUBLISHED_CORPUS: [CorpusFiles; 13] = [
    corpus_case!("L1"),
    corpus_case!("L63"),
    corpus_case!("L64"),
    corpus_case!("L64_gqa_kv2"),
    corpus_case!("L65"),
    corpus_case!("L65_tinyalpha"),
    corpus_case!("L127"),
    corpus_case!("L127c16"),
    corpus_case!("L127c64"),
    corpus_case!("L129"),
    corpus_case!("L129_gqa_kv3"),
    corpus_case!("L1000"),
    corpus_case!("L8191"),
];

/// tessl's pinned sums for the corpus, embedded with it.
pub const PUBLISHED_SHA256SUMS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gdn/gdn_published_SHA256SUMS"
));

/// Every embedded byte string hashes to tessl's pinned sum for its file, and
/// every case the sums list (by its golden) is embedded: the binary carries
/// exactly tessl's corpus. Returns the number of files checked.
pub fn verify_embedded_corpus() -> usize {
    let sums: Vec<(&str, &str)> = PUBLISHED_SHA256SUMS
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_once("  ")
                .unwrap_or_else(|| panic!("malformed SHA256SUMS line {l:?}"))
        })
        .collect();
    let mut checked = 0;
    for c in &PUBLISHED_CORPUS {
        for (suffix, bytes) in [
            ("q", c.q),
            ("k", c.k),
            ("v", c.v),
            ("alpha", c.alpha),
            ("beta", c.beta),
            ("y_seq_f64", c.golden),
        ] {
            let file = format!("gdn_published_{}_{suffix}.npy", c.name);
            let want = sums
                .iter()
                .find(|(_, f)| *f == file)
                .unwrap_or_else(|| panic!("{file} is not in tessl's SHA256SUMS"))
                .0;
            assert_eq!(
                sha256::hex(bytes),
                want,
                "{file}: embedded bytes differ from tessl's pinned sum"
            );
            assert!(file.contains("published"), "rule 9: {file}");
            checked += 1;
        }
    }
    let goldens = sums
        .iter()
        .filter(|(_, f)| f.ends_with("_y_seq_f64.npy"))
        .count();
    assert_eq!(
        goldens,
        PUBLISHED_CORPUS.len(),
        "the sums list {goldens} cases, {} embedded",
        PUBLISHED_CORPUS.len()
    );
    checked
}

/// A corpus case in the kernels' shape, with its golden.
pub struct CorpusCase {
    pub case: GdnPublishedCase,
    /// `y_seq_f64` as `[B, T, H, D]`.
    pub golden: Vec<f64>,
    /// The corpus head dim (8).
    pub d: usize,
}

/// Parse and embed one corpus case (`GdnPublishedCase::from_published_corpus`).
pub fn load_corpus(files: &CorpusFiles, seed: u64) -> CorpusCase {
    let name = files.name;
    let parse = |what: &str, bytes: &[u8]| {
        npy::parse(bytes).unwrap_or_else(|e| panic!("{name} {what}: {e}"))
    };
    let q = parse("q", files.q);
    assert_eq!(
        q.shape.len(),
        4,
        "{name}: q must be [B, H, L, D], got {:?}",
        q.shape
    );
    let (b, h, l, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let (k, v, alpha, beta, golden) = (
        parse("k", files.k),
        parse("v", files.v),
        parse("alpha", files.alpha),
        parse("beta", files.beta),
        parse("y_seq_f64", files.golden),
    );
    for (what, a) in [("k", &k), ("v", &v), ("y_seq_f64", &golden)] {
        assert_eq!(a.shape, q.shape, "{name}: {what} shape");
    }
    for (what, a) in [("alpha", &alpha), ("beta", &beta)] {
        assert_eq!(a.shape, vec![b, h, l], "{name}: {what} shape");
    }
    let f32s = |what: &str, a: &npy::Npy| {
        a.f32s()
            .unwrap_or_else(|e| panic!("{name} {what}: {e}"))
            .to_vec()
    };
    let case = GdnPublishedCase::from_published_corpus(
        name,
        (b, h, l, d),
        &f32s("q", &q),
        &f32s("k", &k),
        &f32s("v", &v),
        &f32s("alpha", &alpha),
        &f32s("beta", &beta),
        seed,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    // [B, H, L, D] -> [B, L, H, D].
    let g = golden
        .f64s()
        .unwrap_or_else(|e| panic!("{name} golden: {e}"));
    let mut blhd = vec![0.0; g.len()];
    for bi in 0..b {
        for hi in 0..h {
            for t in 0..l {
                let src = ((bi * h + hi) * l + t) * d;
                let dst = ((bi * l + t) * h + hi) * d;
                blhd[dst..dst + d].copy_from_slice(&g[src..src + d]);
            }
        }
    }
    CorpusCase {
        case,
        golden: blhd,
        d,
    }
}

/// The float64 reference's outputs for a case, in the plan's layout.
#[derive(Clone, Debug)]
pub struct Expected {
    pub o: Vec<f64>,
    pub s_fin: Vec<f64>,
    pub ckpt: Vec<f64>,
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    pub dg: Vec<f64>,
    pub dbeta: Vec<f64>,
    /// Present exactly when the case has an initial state.
    pub ds0: Option<Vec<f64>>,
}

fn wide(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// L-cuda-oracle's `gdn_train_published_{fwd,bwd}_f64` on the case's f32
/// operands widened exactly; the backward consumes the reference's own
/// checkpoints. A dense batch is one reference call at its `B` (tessl's
/// shape); a variable-length batch is one `B = 1` call per sequence,
/// concatenated (the plan's layout).
pub fn reference_outputs(case: &GdnPublishedCase) -> Expected {
    let p = &case.plan;
    match p.dense_seq() {
        Some(t) => {
            let s = Shape {
                b: p.batch(),
                t,
                h: p.heads(),
                dk: GDN_DK,
                dv: p.v_dim(),
            };
            let (q, k, v, g, beta) = (
                wide(&case.q),
                wide(&case.k),
                wide(&case.v),
                wide(&case.g),
                wide(&case.beta),
            );
            let s0 = case.s0.as_deref().map(wide);
            let inp = Inputs {
                s,
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
                s0: s0.as_deref(),
            };
            let f = gdn_train_published_fwd_f64(&inp);
            let d_fin = case.d_fin.as_deref().map(wide);
            let gr = gdn_train_published_bwd_f64(&inp, &f.ckpt, &wide(&case.d_o), d_fin.as_deref());
            Expected {
                o: f.o,
                s_fin: f.fin,
                ckpt: f.ckpt,
                dq: gr.dq,
                dk: gr.dk,
                dv: gr.dv,
                dg: gr.dg,
                dbeta: gr.dbeta,
                ds0: case.s0.as_ref().map(|_| gr.ds0),
            }
        }
        None => {
            let parts: Vec<Expected> = (0..p.batch())
                .map(|b| reference_outputs(&case.sequence(b).unwrap_or_else(|e| panic!("{e}"))))
                .collect();
            let cat = |f: &dyn Fn(&Expected) -> &Vec<f64>| {
                parts
                    .iter()
                    .flat_map(|e| f(e).clone())
                    .collect::<Vec<f64>>()
            };
            Expected {
                o: cat(&|e| &e.o),
                s_fin: cat(&|e| &e.s_fin),
                ckpt: cat(&|e| &e.ckpt),
                dq: cat(&|e| &e.dq),
                dk: cat(&|e| &e.dk),
                dv: cat(&|e| &e.dv),
                dg: cat(&|e| &e.dg),
                dbeta: cat(&|e| &e.dbeta),
                ds0: case.s0.as_ref().map(|_| {
                    parts
                        .iter()
                        .flat_map(|e| e.ds0.clone().expect("ds0"))
                        .collect()
                }),
            }
        }
    }
}

/// Every output within `published_bounds::REL_OF_MAX` of the reference
/// (sentinel-checked when given). Returns the worst ratio; panics naming the
/// tensor otherwise.
pub fn judge(
    label: &str,
    got: &GdnPublishedOutputs,
    want: &Expected,
    sentinel: Option<f32>,
) -> f64 {
    assert_eq!(
        got.ds0.is_some(),
        want.ds0.is_some(),
        "{label}: ds0 present on one side only"
    );
    let mut pairs: Vec<(&str, &[f32], &[f64])> = vec![
        ("o", &got.o, &want.o),
        ("s_fin", &got.s_fin, &want.s_fin),
        ("ckpt", &got.ckpt, &want.ckpt),
        ("dq", &got.dq, &want.dq),
        ("dk", &got.dk, &want.dk),
        ("dv", &got.dv, &want.dv),
        ("dg", &got.dg, &want.dg),
        ("dbeta", &got.dbeta, &want.dbeta),
    ];
    if let (Some(g), Some(w)) = (&got.ds0, &want.ds0) {
        pairs.push(("ds0", g, w));
    }
    let mut worst = 0.0f64;
    for (name, g, w) in pairs {
        let (rel, r) = rel_of_max(name, g, w, sentinel).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(
            rel <= published_bounds::REL_OF_MAX,
            "{label} {name}: max err {:.3e} at {} is {rel:.3e} of max|ref| {:.3e}, over the bound {:.0e} (tessl tests/gdn_train.rs:240-241)",
            r.max_abs_err,
            r.argmax,
            r.max_abs_ref,
            published_bounds::REL_OF_MAX
        );
        worst = worst.max(rel);
    }
    eprintln!(
        "{label}: worst {worst:.3e} of max|ref| over every output (bound {:.0e})",
        published_bounds::REL_OF_MAX
    );
    worst
}

/// The corpus golden check: `o` (f32, the kernels' layout) unscaled to the
/// corpus's operator against `y_seq_f64`, within
/// `published_bounds::CORPUS_GOLDEN_REL_OF_MAX`, and the zero-padded value
/// columns exactly zero. Also reports the reference's own offset from the
/// golden (the k re-normalization, in float64). Returns `(kernel, offset)`.
pub fn judge_corpus_golden(cc: &CorpusCase, o: &[f32], want: &Expected) -> (f64, f64) {
    let label = &cc.case.label;
    let dv = cc.case.plan.v_dim();
    for r in 0..cc.case.plan.rows() {
        assert!(
            o[r * dv + cc.d..(r + 1) * dv].iter().all(|&x| x == 0.0),
            "{label}: padded value columns of row {r} are not zero"
        );
    }
    let got = cc
        .case
        .corpus_unscaled_output(o, cc.d)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let rel = |x: &[f64]| {
        let mag = cc.golden.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(mag > 0.0, "{label}: golden is all zeros");
        x.iter()
            .zip(&cc.golden)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
            / mag
    };
    let kernel = rel(&got);
    // The same unscaling on the float64 reference: the operator offset alone.
    let scale = 1.0 / (GDN_DK as f64).sqrt();
    let mut ref_unscaled = Vec::with_capacity(got.len());
    for r in 0..cc.case.plan.rows() {
        let sumsq: f64 = cc.case.q[r * GDN_DK..(r + 1) * GDN_DK]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        let rq = 1.0 / (sumsq + 1e-6).sqrt();
        ref_unscaled.extend(
            want.o[r * dv..r * dv + cc.d]
                .iter()
                .map(|&x| x / (rq * scale)),
        );
    }
    let offset = rel(&ref_unscaled);
    eprintln!(
        "{label}: vs corpus golden {kernel:.3e} of max (bound {:.0e}); float64 seam reference vs golden {offset:.3e}",
        published_bounds::CORPUS_GOLDEN_REL_OF_MAX
    );
    assert!(
        kernel <= published_bounds::CORPUS_GOLDEN_REL_OF_MAX,
        "{label}: {kernel:.3e} of max|golden| from the corpus golden, over {:.0e}",
        published_bounds::CORPUS_GOLDEN_REL_OF_MAX
    );
    (kernel, offset)
}
