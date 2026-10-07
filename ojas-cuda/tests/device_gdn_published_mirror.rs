//! K2(i) at the **published** rule, on the host: the bitwise mirror of the
//! device kernels (`ojas_cuda::gdn_host`) against L-cuda-oracle's
//! float64 reference (`tests/reference/gdn_published.rs`), at the bounds the
//! device tests use (`gdn_host::published_bounds`, written before any run).
//!
//! The mirror reproduces the kernels' every rounding, reduction tree and
//! index, so this suite is the device algorithm run on the Mac. On the box,
//! `tests/device_gdn_published.rs` checks the device bit for bit against the
//! mirror and against the same float64 bounds.
//!
//! Cases: tessl's five chunk edges with its flags (`tessl/tests/gdn_train.rs:
//! 365-371`) at `B = 2` and `B = 1`; tessl's 2B-heads case (`:378-382`); a
//! variable-length batch over the edges; tessl's 13-case published corpus;
//! and the repo-rule contrast that shows the bound separates the two rules.

mod device_gdn_published_common;
mod reference;

use device_gdn_published_common::{
    judge, judge_corpus_golden, load_corpus, reference_outputs, verify_embedded_corpus,
    PUBLISHED_CORPUS,
};
use ojas_cuda::gdn_host::{
    gdn_published_mirror, published_bounds, smoke_cases, tessl_random_f32, GdnPublishedCase,
    TESSL_EDGES,
};
use ojas_cuda::gdn_plan::GDN_DK;
use reference::gdn_published::{l2norm, recurrence_f64, Rule, Shape};

#[test]
fn published_mirror_draws_tessls_test_operands() {
    for seed in [0u64, 7, 100, 163, 4242] {
        let ours = tessl_random_f32(1000, seed);
        let oracle = reference::rng::random_f32(1000, seed);
        assert!(
            ours.iter()
                .zip(&oracle)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "seed {seed}: gdn_host's tessl generator is not the oracle's port of tessl's"
        );
    }
}

fn mirror_against_reference(case: &GdnPublishedCase) -> f64 {
    let got = gdn_published_mirror(case).unwrap_or_else(|e| panic!("{}: {e}", case.label));
    judge(
        &format!("mirror {}", case.label),
        &got,
        &reference_outputs(case),
        None,
    )
}

#[test]
fn published_mirror_matches_the_f64_reference_across_chunk_edges_b2() {
    for (t, s0, dfin) in TESSL_EDGES {
        mirror_against_reference(
            &GdnPublishedCase::tessl(2, t, 3, 32, 100 + t as u64, s0, dfin).unwrap(),
        );
    }
}

#[test]
fn published_mirror_matches_the_f64_reference_across_chunk_edges_b1() {
    for (t, s0, dfin) in TESSL_EDGES {
        mirror_against_reference(
            &GdnPublishedCase::tessl(1, t, 3, 32, 200 + t as u64, s0, dfin).unwrap(),
        );
    }
}

#[test]
fn published_mirror_matches_the_f64_reference_at_qwen35_2b_heads() {
    mirror_against_reference(&GdnPublishedCase::tessl(1, 200, 16, 128, 7, false, false).unwrap());
}

/// The cases `gdn_smoke::gdn_published_checks` runs on the device, so the
/// device's bit equality with the mirror carries this float64 judgment there.
#[test]
fn published_mirror_matches_the_f64_reference_on_every_smoke_case() {
    let cases = smoke_cases().unwrap();
    assert_eq!(cases.len(), 6);
    for c in &cases {
        mirror_against_reference(c);
    }
}

#[test]
fn published_mirror_varlen_batch_matches_per_sequence_references() {
    let c = GdnPublishedCase::varlen(&[130, 1, 65, 64, 63], 2, 48, 77, true, true).unwrap();
    assert_eq!(c.plan.dense_seq(), None);
    mirror_against_reference(&c);
}

#[test]
fn published_mirror_matches_the_published_corpus() {
    let files = verify_embedded_corpus();
    assert_eq!(files, 13 * 6);
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    for (i, files) in PUBLISHED_CORPUS.iter().enumerate() {
        let cc = load_corpus(files, 9000 + i as u64);
        let want = reference_outputs(&cc.case);
        let got =
            gdn_published_mirror(&cc.case).unwrap_or_else(|e| panic!("{}: {e}", cc.case.label));
        let seam = judge(&format!("mirror {}", cc.case.label), &got, &want, None);
        let (golden, offset) = judge_corpus_golden(&cc, &got.o, &want);
        worst = (worst.0.max(seam), worst.1.max(golden), worst.2.max(offset));
    }
    eprintln!(
        "published corpus, 13 cases: worst vs seam reference {:.3e}, vs golden {:.3e}, reference-vs-golden offset {:.3e}",
        worst.0, worst.1, worst.2
    );
}

/// The bound has teeth: the float64 recurrence at nanolab's default
/// `rule="repo"` (the undecayed read) lands far outside it from the mirror,
/// on tessl's T = 130 edge, while the published reference is inside.
#[test]
fn published_mirror_is_outside_the_bound_from_the_repo_rule() {
    let c = GdnPublishedCase::tessl(2, 130, 3, 32, 230, true, true).unwrap();
    let got = gdn_published_mirror(&c).unwrap();
    let rows = |x: &[f32]| -> Vec<f64> {
        x.as_chunks::<GDN_DK>()
            .0
            .iter()
            .flat_map(|r| l2norm(&r.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()).0)
            .collect()
    };
    let (qn, kn) = (rows(&c.q), rows(&c.k));
    let wide = |x: &[f32]| x.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>();
    let s0 = c.s0.as_deref().map(wide);
    let s = Shape {
        b: 2,
        t: 130,
        h: 3,
        dk: GDN_DK,
        dv: 32,
    };
    let scale = 1.0 / (GDN_DK as f64).sqrt();
    let rule_o = |rule| {
        recurrence_f64(
            s,
            &qn,
            &kn,
            &wide(&c.v),
            &wide(&c.g),
            &wide(&c.beta),
            s0.as_deref(),
            scale,
            rule,
        )
        .o
    };
    let rel = |want: &[f64]| {
        let mag = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        got.o
            .iter()
            .zip(want)
            .fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - w).abs()))
            / mag
    };
    let (published, repo) = (rel(&rule_o(Rule::Published)), rel(&rule_o(Rule::Repo)));
    eprintln!("mirror o vs published reference {published:.3e}, vs repo rule {repo:.3e}");
    assert!(published <= published_bounds::REL_OF_MAX);
    assert!(
        repo > 100.0 * published_bounds::REL_OF_MAX,
        "the repo rule is within {repo:.3e}"
    );
}
