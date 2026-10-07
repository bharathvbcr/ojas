//! The float64 torch goldens (`tests/fixtures/goldens/`, written by
//! `tests/fixtures/gen_goldens.py`): their pins, rule 9 over them, and the K2
//! reference (**GDN, published rule**) against the transformers-anchored
//! golden at the training seam, forward and backward.
//!
//! The GDN golden is transformers 5.12.1's `torch_recurrent_gated_delta_rule`
//! with `use_qk_l2norm_in_kernel=True` (`modeling_qwen3_5.py:327-368`),
//! transcribed at float64 and run under torch autograd; the generator checked
//! the transcription against transformers' own float32 function on the same
//! inputs (worst 2.2e-7 relative, recorded in `manifest.json`) and recorded
//! that the repo rule diverges from it (1.8e-2 to 3.6e-2). It is the only
//! independent golden for the GDN **backward**: tessl's is validated by finite
//! differences alone.

mod reference;

use reference::gdn_published::{
    gdn_train_published_bwd_f64, gdn_train_published_fwd_f64, Inputs, Shape,
};
use reference::goldens::{self, assert_rel};
use reference::sha256;

/// Bound for the GDN seam golden, relative to each output's largest magnitude,
/// written before the first run. Worst-case f64 error grows linearly in the
/// accumulations along a dependency chain: per token a `Dk`-term read, a
/// `Dk`-term output and the rank-1 update, over `T` tokens forward and again
/// backward, about `2 T (Dk + Dv) u = 2 * 130 * 144 * 1.1e-16 = 4.2e-12` at the
/// largest case; the two sides also differ in association (torch reduces with
/// `sum(dim=-2)` over a broadcast product, the reference in ascending index
/// order). 1e-10 is ~25x that worst case, and a defect (a wrong rule, a
/// dropped gate, a wrong scale) is 1e-2 or more.
const GDN_BOUND: f64 = 1e-10;

#[test]
fn sha256_matches_the_fips_180_4_vectors() {
    assert_eq!(
        sha256::hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256::hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256::hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    // 55, 56 and 64 bytes straddle the padding boundary.
    assert_eq!(
        sha256::hex(&[b'a'; 55]),
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
    );
    assert_eq!(
        sha256::hex(&[b'a'; 56]),
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
    );
    assert_eq!(
        sha256::hex(&[b'a'; 64]),
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
    );
}

#[test]
fn every_golden_is_pinned_by_the_manifest_and_gdn_and_gate_goldens_name_the_published_rule() {
    let n = goldens::verify();
    let text = goldens::manifest_text();
    assert!(
        text.contains("\"rule\": \"published\""),
        "rule 9: the goldens manifest must declare rule=published"
    );
    assert!(
        !text.contains("\"rule\": \"repo\""),
        "rule 9: the goldens manifest declares a repo-rule case"
    );
    assert!(
        text.contains("\"dtype\": \"float64\""),
        "the goldens must be generated at float64"
    );
    assert!(
        text.contains("\"deterministic_algorithms\": true"),
        "the goldens must be generated deterministically"
    );
    // The GDN operator's goldens: the recurrence (`gdn_*`) and its gates (`gates_*`).
    let gdn: Vec<_> = goldens::manifest_files(&text)
        .into_iter()
        .filter(|(f, _)| f.starts_with("gdn") || f.starts_with("gates"))
        .collect();
    let gates = gdn.iter().filter(|(f, _)| f.starts_with("gates")).count();
    assert!(
        gates > 0 && gates < gdn.len(),
        "the GDN goldens must include both the recurrence and the gates"
    );
    for (f, _) in &gdn {
        assert!(
            f.contains("published"),
            "rule 9: GDN golden {f} does not name its rule"
        );
    }
    eprintln!(
        "{n} goldens match manifest.json sha256s; {} GDN files ({gates} of them the gates) all say published",
        gdn.len()
    );
}

#[test]
fn published_reference_matches_the_transformers_golden_at_the_seam() {
    let mut worst = 0.0f64;
    for t in [1usize, 63, 64, 65, 130] {
        let name = format!("gdn_published_train_T{t}");
        let (qs, q) = goldens::f64s_shaped(&format!("{name}_q"));
        assert_eq!(qs.len(), 4, "{name}: q must be [B, T, H, Dk]");
        let (b, tt, h, dk) = (qs[0], qs[1], qs[2], qs[3]);
        assert_eq!(tt, t, "{name}: the golden's T");
        let (vs, v) = goldens::f64s_shaped(&format!("{name}_v"));
        let dv = vs[3];
        let s = Shape { b, t, h, dk, dv };
        let get = |n: &str| goldens::f64s(&format!("{name}_{n}"));
        let (k, g, beta, s0, d_o, dfin) = (
            get("k"),
            get("g"),
            get("beta"),
            get("s0"),
            get("d_o"),
            get("dfin"),
        );
        let inp = Inputs {
            s,
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            s0: Some(&s0),
        };
        let fwd = gdn_train_published_fwd_f64(&inp);
        let grads = gdn_train_published_bwd_f64(&inp, &fwd.ckpt, &d_o, Some(&dfin));
        let label = |n: &str| format!("published T={t} {n}");
        for (n, got) in [
            ("o", &fwd.o),
            ("fin", &fwd.fin),
            ("dq", &grads.dq),
            ("dk", &grads.dk),
            ("dv", &grads.dv),
            ("dg", &grads.dg),
            ("dbeta", &grads.dbeta),
            ("ds0", &grads.ds0),
        ] {
            worst = worst.max(assert_rel(&label(n), got, &get(n), GDN_BOUND));
        }
    }
    eprintln!("published vs transformers golden: worst {worst:.3e} (bound {GDN_BOUND:.0e})");
}
