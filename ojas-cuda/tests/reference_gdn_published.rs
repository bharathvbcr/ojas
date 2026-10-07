//! Self-validation of the K2 host reference, **GDN at the published rule**
//! (`tests/reference/gdn_published.rs`), before any CUDA kernel is judged by it.
//!
//! 1. The copied tessl corpus is byte-identical to tessl's pinned
//!    `gdn_published_SHA256SUMS`, and every file and the manifest name the
//!    `published` rule (rule 9).
//! 2. The reference's recurrence reproduces the corpus's independent f64
//!    goldens on all 13 cases (T = 1, 63, 64, 65, 127, 129, 1000, 8191, the
//!    grouped and tiny-alpha cases) within `1e-12` of the golden's magnitude
//!    (tessl `tests/gdn_fixtures.rs:63`).
//! 3. The repo rule does not: it diverges by more than `1e-3` wherever the
//!    rules can differ, and agrees exactly at T = 1 where they cannot (tessl
//!    `tests/gdn_fixtures.rs:322-370`).
//! 4. Central differences of the forward match the checkpointed backward at
//!    T = 1, 63, 64, 65, 130 (tessl's method and bound, `tests/gdn_train.rs:119-177`).
//! 5. Checkpoint `c` is exactly the state a forward over the first `64c`
//!    tokens ends in.
//!
//! tessl's own host reference, run in process at the same T, is
//! `tests/reference_gdn_published_vs_tessl.rs`; the transformers-anchored
//! torch golden at the seam is `tests/reference_goldens_published.rs`.

mod reference;

use std::path::{Path, PathBuf};

use reference::gdn_published::{
    gdn_train_published_bwd_f64, gdn_train_published_fwd_f64, recurrence_f64, Inputs, Rule, Shape,
    CKPT,
};
use reference::{fd, npy, rng, sha256};

/// tessl `tests/gdn_fixtures.rs:63`: this crate's f64 reference against the
/// Python f64 golden. tessl measured 0.0 on all 13 cases with its own
/// reference; this one decays the state before the read rather than scaling
/// the read, a reassociation, so it is held to the same 1e-12 rather than to 0.
const REF_REL_BOUND: f64 = 1e-12;

/// The clamps every manifest declares (`alpha_clamp [1e-4, 1]`, `beta_clamp
/// [0, 1]`) and that the golden was generated with (tessl `tests/common/gdn.rs:59-73`).
const ALPHA_CLAMP: (f64, f64) = (1e-4, 1.0);
const BETA_CLAMP: (f64, f64) = (0.0, 1.0);

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gdn")
}

fn corpus_cases() -> Vec<String> {
    const PREFIX: &str = "gdn_published_";
    const SUFFIX: &str = "_y_seq_f64.npy";
    let mut names: Vec<String> = std::fs::read_dir(corpus_dir())
        .expect("tests/fixtures/gdn must exist")
        .map(|e| e.expect("readable dir entry").file_name())
        .filter_map(|n| {
            let n = n.to_str()?;
            Some(n.strip_prefix(PREFIX)?.strip_suffix(SUFFIX)?.to_string())
        })
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "no published GDN cases found; a suite over them must not pass vacuously"
    );
    names
}

/// One corpus case, mapped onto the seam's layout.
struct Case {
    name: String,
    s: Shape,
    /// `[B, T, H, D]`, as stored (raw q; k pre-normalized by the generator).
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    /// `ln(clamp(alpha))` and `clamp(beta)`, `[B, T, H]`.
    g: Vec<f64>,
    beta: Vec<f64>,
    /// `[B, T, H, D]`.
    golden: Vec<f64>,
}

/// `[B, H, L, D]` to `[B, L, H, D]`.
fn bhld_to_blhd(x: &[f64], b: usize, h: usize, l: usize, d: usize) -> Vec<f64> {
    assert_eq!(x.len(), b * h * l * d);
    let mut out = vec![0.0; x.len()];
    for bi in 0..b {
        for hi in 0..h {
            for t in 0..l {
                let src = ((bi * h + hi) * l + t) * d;
                let dst = ((bi * l + t) * h + hi) * d;
                out[dst..dst + d].copy_from_slice(&x[src..src + d]);
            }
        }
    }
    out
}

fn load_case(name: &str) -> Case {
    let load = |suffix: &str| {
        npy::read(&corpus_dir().join(format!("gdn_published_{name}_{suffix}.npy")))
            .unwrap_or_else(|e| panic!("load {name}_{suffix}: {e}"))
    };
    let q = load("q");
    assert_eq!(
        q.shape.len(),
        4,
        "{name}: q must be [B, H, L, D], got {:?}",
        q.shape
    );
    let (b, h, l, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let (k, v, alpha, beta, golden) = (
        load("k"),
        load("v"),
        load("alpha"),
        load("beta"),
        load("y_seq_f64"),
    );
    for (what, arr) in [("k", &k), ("v", &v), ("y_seq_f64", &golden)] {
        assert_eq!(arr.shape, q.shape, "{name}: {what} shape");
    }
    for (what, arr) in [("alpha", &alpha), ("beta", &beta)] {
        assert_eq!(arr.shape, vec![b, h, l], "{name}: {what} shape");
    }
    // f32 operands widen exactly; the golden must stay f64.
    let wide = |a: &npy::Npy| {
        a.f32s()
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .iter()
            .map(|&x| f64::from(x))
            .collect::<Vec<_>>()
    };
    let gate = |a: &npy::Npy, clamp: (f64, f64), map: &dyn Fn(f64) -> f64| {
        bhld_to_blhd(
            &wide(a)
                .iter()
                .map(|&x| map(x.clamp(clamp.0, clamp.1)))
                .collect::<Vec<_>>(),
            b,
            h,
            l,
            1,
        )
    };
    Case {
        name: name.to_string(),
        s: Shape {
            b,
            t: l,
            h,
            dk: d,
            dv: d,
        },
        q: bhld_to_blhd(&wide(&q), b, h, l, d),
        k: bhld_to_blhd(&wide(&k), b, h, l, d),
        v: bhld_to_blhd(&wide(&v), b, h, l, d),
        g: gate(&alpha, ALPHA_CLAMP, &f64::ln),
        beta: gate(&beta, BETA_CLAMP, &|x| x),
        golden: bhld_to_blhd(golden.f64s().expect("the golden is <f8"), b, h, l, d),
    }
}

impl Case {
    fn run(&self, rule: Rule) -> Vec<f64> {
        recurrence_f64(
            self.s, &self.q, &self.k, &self.v, &self.g, &self.beta, None, 1.0, rule,
        )
        .o
    }
}

/// `(max |got - want|, max |want|)`.
fn worst_abs(got: &[f64], want: &[f64]) -> (f64, f64) {
    assert_eq!(got.len(), want.len(), "length");
    let mag = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    (err, mag)
}

#[test]
fn published_corpus_is_tessls_bytes_and_every_file_names_the_published_rule() {
    let dir = corpus_dir();
    let sums = std::fs::read_to_string(dir.join("gdn_published_SHA256SUMS")).expect("SHA256SUMS");
    let mut listed = 0usize;
    for line in sums.lines().filter(|l| !l.trim().is_empty()) {
        let (want, file) = line
            .split_once("  ")
            .unwrap_or_else(|| panic!("malformed SHA256SUMS line {line:?}"));
        let got = sha256::file_hex(&dir.join(file)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            got, want,
            "{file}: sha256 differs from tessl's pinned SHA256SUMS"
        );
        listed += 1;
    }
    let entries: Vec<String> = std::fs::read_dir(&dir)
        .expect("fixture dir")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    // Every file but the sums file itself is listed, so nothing unpinned rides along.
    assert_eq!(
        listed + 1,
        entries.len(),
        "files on disk {entries:?} are not exactly the {listed} pinned ones"
    );
    for name in &entries {
        assert!(
            name.contains("published"),
            "rule 9: {name} does not name its rule; nanolab's default rule=\"repo\" is a different operator"
        );
    }
    let manifest =
        std::fs::read_to_string(dir.join("gdn_published_MANIFEST.json")).expect("manifest");
    assert!(
        manifest.contains("\"rule\": \"published\""),
        "rule 9: the manifest must declare rule=published"
    );
    assert!(
        !manifest.contains("\"rule\": \"repo\""),
        "rule 9: the manifest declares a repo-rule case"
    );
    eprintln!(
        "{listed} corpus files match tessl's SHA256SUMS; all {} names say published",
        entries.len()
    );
}

#[test]
fn published_recurrence_reproduces_the_published_corpus() {
    let cases = corpus_cases();
    for required in ["L1", "L63", "L64", "L65"] {
        assert!(
            cases.iter().any(|c| c == required),
            "the corpus lost its {required} case: {cases:?}"
        );
    }
    assert_eq!(
        cases.len(),
        13,
        "expected tessl's 13 published cases, found {cases:?}"
    );
    let mut worst = (String::new(), 0.0f64);
    for name in &cases {
        let c = load_case(name);
        let (err, mag) = worst_abs(&c.run(Rule::Published), &c.golden);
        assert!(mag > 0.0, "{name}: golden is all zeros, nothing is tested");
        let rel = err / mag;
        eprintln!(
            "published {name} (T={}): max err {err:.3e}, {rel:.3e} of max|golden| {mag:.4}",
            c.s.t
        );
        assert!(
            rel <= REF_REL_BOUND,
            "{name}: published reference differs from the golden by {rel:.3e} of its magnitude, over {REF_REL_BOUND:.0e}"
        );
        if rel >= worst.1 {
            worst = (c.name.clone(), rel);
        }
    }
    eprintln!(
        "published corpus: worst {:.3e} on {} over {} cases (bound {REF_REL_BOUND:.0e})",
        worst.1,
        worst.0,
        cases.len()
    );
}

#[test]
fn the_repo_rule_does_not_reproduce_the_published_corpus() {
    let mut smallest = f64::INFINITY;
    let mut single_step_seen = false;
    for name in corpus_cases() {
        let c = load_case(&name);
        let (err, mag) = worst_abs(&c.run(Rule::Repo), &c.golden);
        if c.s.t == 1 {
            // At t = 0 the state is zero and the two reads coincide exactly.
            single_step_seen = true;
            let published = c.run(Rule::Published);
            let repo = c.run(Rule::Repo);
            assert!(
                published
                    .iter()
                    .zip(&repo)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: with T=1 the rules must agree bit for bit"
            );
            continue;
        }
        let rel = err / mag;
        assert!(
            rel > 1e-3,
            "{name}: the repo rule came within {rel:.3e} of the published golden"
        );
        smallest = smallest.min(rel);
    }
    assert!(
        single_step_seen,
        "the T=1 case is missing; the coincide-at-zero-state branch is not exercised"
    );
    eprintln!("smallest repo-vs-published divergence (T>1): {smallest:.3e}");
}

/// Operands as tessl's `Owned::random` draws them (`tests/gdn_train.rs:43-61`):
/// log decays in [-1.5, 0), betas in [0.05, 0.95].
struct Owned {
    s: Shape,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    s0: Option<Vec<f64>>,
}

impl Owned {
    fn random(s: Shape, seed: u64, with_state: bool) -> Self {
        let n = s.rows();
        Self {
            s,
            q: rng::random_f64(n * s.dk, seed),
            k: rng::random_f64(n * s.dk, seed + 1),
            v: rng::random_f64(n * s.dv, seed + 2),
            g: rng::random_f64(n, seed + 3)
                .iter()
                .map(|&x| -0.75 * (x + 1.0) - 1e-3)
                .collect(),
            beta: rng::random_f64(n, seed + 4)
                .iter()
                .map(|&x| 0.5 + 0.45 * x)
                .collect(),
            s0: with_state.then(|| rng::random_f64(s.state_len(), seed + 5)),
        }
    }

    fn inputs(&self) -> Inputs<'_> {
        Inputs {
            s: self.s,
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: self.s0.as_deref(),
        }
    }
}

/// The chunk edges the kernels are tested at (tessl `tests/gdn_train.rs:365-371`).
const EDGES: [usize; 5] = [1, 63, 64, 65, 130];

#[test]
fn published_backward_matches_central_differences_across_checkpoints() {
    for t in EDGES {
        let s = Shape {
            b: 1,
            t,
            h: 2,
            dk: 4,
            dv: 2,
        };
        let base = Owned::random(s, 300 + u64::try_from(t).expect("T fits u64"), true);
        let w_o = rng::random_f64(s.rows() * s.dv, 90 + u64::try_from(t).expect("T fits u64"));
        let w_s = rng::random_f64(s.state_len(), 91 + u64::try_from(t).expect("T fits u64"));
        let fwd = gdn_train_published_fwd_f64(&base.inputs());
        let g = gdn_train_published_bwd_f64(&base.inputs(), &fwd.ckpt, &w_o, Some(&w_s));
        let loss = |p: &Owned| {
            let f = gdn_train_published_fwd_f64(&p.inputs());
            fd::dot(&f.o, &w_o) + fd::dot(&f.fin, &w_s)
        };
        let with = |edit: &dyn Fn(&mut Owned, &[f64]), x: &[f64]| {
            let mut p = Owned {
                s,
                q: base.q.clone(),
                k: base.k.clone(),
                v: base.v.clone(),
                g: base.g.clone(),
                beta: base.beta.clone(),
                s0: base.s0.clone(),
            };
            edit(&mut p, x);
            loss(&p)
        };
        let label = |n: &str| format!("published T={t} {n}");
        fd::check(&label("dq"), &base.q, &g.dq, &|x| {
            with(&|p, x| p.q = x.to_vec(), x)
        });
        fd::check(&label("dk"), &base.k, &g.dk, &|x| {
            with(&|p, x| p.k = x.to_vec(), x)
        });
        fd::check(&label("dv"), &base.v, &g.dv, &|x| {
            with(&|p, x| p.v = x.to_vec(), x)
        });
        fd::check(&label("dg"), &base.g, &g.dg, &|x| {
            with(&|p, x| p.g = x.to_vec(), x)
        });
        fd::check(&label("dbeta"), &base.beta, &g.dbeta, &|x| {
            with(&|p, x| p.beta = x.to_vec(), x)
        });
        let s0 = base.s0.clone().expect("built with a state");
        fd::check(&label("ds0"), &s0, &g.ds0, &|x| {
            with(&|p, x| p.s0 = Some(x.to_vec()), x)
        });
    }
}

/// The first `tp` tokens of a `[B, T, H, d]` tensor.
fn prefix(x: &[f64], s: Shape, d: usize, tp: usize) -> Vec<f64> {
    let row = s.h * d;
    (0..s.b)
        .flat_map(|bi| x[bi * s.t * row..bi * s.t * row + tp * row].to_vec())
        .collect()
}

#[test]
fn published_checkpoint_c_is_the_state_after_the_first_64c_tokens() {
    let s = Shape {
        b: 2,
        t: 130,
        h: 3,
        dk: 8,
        dv: 4,
    };
    let p = Owned::random(s, 77, true);
    let fwd = gdn_train_published_fwd_f64(&p.inputs());
    let nc = s.checkpoints();
    assert_eq!(nc, 3);
    let per = s.dk * s.dv;
    for c in 0..nc {
        let tp = c * CKPT;
        let want: Vec<f64> = if tp == 0 {
            p.s0.clone().expect("state")
        } else {
            let ps = Shape { t: tp, ..s };
            let (q, k, v) = (
                prefix(&p.q, s, s.dk, tp),
                prefix(&p.k, s, s.dk, tp),
                prefix(&p.v, s, s.dv, tp),
            );
            let (g, beta) = (prefix(&p.g, s, 1, tp), prefix(&p.beta, s, 1, tp));
            let inp = Inputs {
                s: ps,
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
                s0: p.s0.as_deref(),
            };
            gdn_train_published_fwd_f64(&inp).fin
        };
        for bh in 0..s.b * s.h {
            let got = &fwd.ckpt[(bh * nc + c) * per..(bh * nc + c + 1) * per];
            let w = &want[bh * per..(bh + 1) * per];
            assert!(
                got.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()),
                "checkpoint {c} of head-row {bh} is not the state after {tp} tokens"
            );
        }
    }
}

#[test]
#[should_panic(expected = "every dimension must be non-zero")]
fn published_reference_refuses_an_empty_sequence() {
    let s = Shape {
        b: 1,
        t: 0,
        h: 1,
        dk: 4,
        dv: 2,
    };
    gdn_train_published_fwd_f64(&Inputs {
        s,
        q: &[],
        k: &[],
        v: &[],
        g: &[],
        beta: &[],
        s0: None,
    });
}

#[test]
#[should_panic(expected = "g has 3 elements")]
fn published_reference_refuses_a_misshapen_gate() {
    let s = Shape {
        b: 1,
        t: 2,
        h: 1,
        dk: 2,
        dv: 2,
    };
    let x = [0.5; 4];
    gdn_train_published_fwd_f64(&Inputs {
        s,
        q: &x,
        k: &x,
        v: &x,
        g: &[0.0; 3],
        beta: &[0.5; 2],
        s0: None,
    });
}

#[test]
#[should_panic(expected = "is not finite")]
fn published_reference_refuses_a_non_finite_operand() {
    let s = Shape {
        b: 1,
        t: 2,
        h: 1,
        dk: 2,
        dv: 2,
    };
    let x = [0.5; 4];
    gdn_train_published_fwd_f64(&Inputs {
        s,
        q: &x,
        k: &x,
        v: &[0.5, f64::NAN, 0.5, 0.5],
        g: &[0.0; 2],
        beta: &[0.5; 2],
        s0: None,
    });
}
