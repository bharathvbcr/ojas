//! The K2 reference (**GDN, published rule**) against tessl's own host
//! reference, run in this process.
//!
//! tessl's `tests/common/gdn_train.rs` is pure f64 Rust with no imports, so it
//! is compiled in here unmodified through `#[path]` (nothing in tessl is
//! edited). It is the reference tessl's Metal GDN training kernels are judged
//! by (`tessl/tests/gdn_train.rs:251-360`); tessl checked its forward against
//! the transformers-anchored recurrence and its backward against central
//! differences (`:76-177`). This test runs it and the port on identical inputs
//! at tessl's chunk-edge cases, T = 1, 63, 64, 65, 130 with tessl's
//! initial-state and final-state-gradient flags (`:365-371`), at the kernels'
//! `Dk = 128`, and requires agreement within `1e-12` of each output's
//! magnitude; it prints whether the agreement is bit for bit.
//!
//! **What is compiled is an in-crate copy**, `tests/fixtures/tessl/gdn_train_published.rs`:
//! tessl's file byte for byte, renamed so that it names its rule (rule 9). It
//! used to be included from the sibling checkout `research/tessl` through
//! `#[path]`, and `cargo fmt` follows `#[path]` modules, so formatting this
//! crate would have rewritten a tessl file (rule 6;
//! `GAP-L-CUDA-M0-CARGO-FMT-WOULD-EDIT-TESSL-2026-10-01`). Now no `#[path]` in
//! this crate leaves it (`tests/fmt_boundary.rs` holds that), and the copy's
//! module carries `#[rustfmt::skip]`, so rustfmt does not visit the copy either.
//!
//! Three tests keep the copy honest:
//! - its sha256 is the pin, so an edited or formatted copy fails;
//! - tessl's live file, read at run time, equals the copy byte for byte and
//!   the pin. It fails loudly when the tessl checkout is absent: it is never
//!   reported as passed without having compared;
//! - the comparison below runs the copy against the port.

mod reference;

#[allow(dead_code, clippy::all)]
#[rustfmt::skip]
#[path = "fixtures/tessl/gdn_train_published.rs"]
mod tessl_gdn_train_published;

use reference::gdn_published::{
    gdn_train_published_bwd_f64, gdn_train_published_fwd_f64, Inputs, Shape, KERNEL_DK,
};
use reference::rng;
use tessl_gdn_train_published as tessl_gdn_train;

const BOUND: f64 = 1e-12;

/// tessl's `tests/common/gdn_train.rs` as it was when the reference was ported
/// and found bit-identical to it.
const TESSL_GDN_TRAIN_SHA256: &str =
    "dc455ac993c29e8b34f4300c427aca55ff33ce437f9b67f78a454930eca98656";

/// The in-crate copy, the bytes this target compiles.
const COPY: &[u8] = include_bytes!("fixtures/tessl/gdn_train_published.rs");

/// tessl's live file, relative to this crate's manifest directory.
const TESSL_LIVE: &str = "../../tessl/tests/common/gdn_train.rs";

#[test]
fn published_comparison_target_is_the_pinned_copy_of_tessls_file() {
    assert_eq!(
        reference::sha256::hex(COPY),
        TESSL_GDN_TRAIN_SHA256,
        "tests/fixtures/tessl/gdn_train_published.rs is no longer tessl's pinned file: \
         it was edited or formatted. Restore it from tessl; never format it"
    );
}

#[test]
fn the_copy_equals_tessls_live_file_and_the_pin() {
    let live = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(TESSL_LIVE);
    let bytes = std::fs::read(&live).unwrap_or_else(|e| {
        panic!(
            "NOT COMPARED: tessl's live file {} is unreadable ({e}). This test needs the \
             sibling tessl checkout; it does not pass without comparing",
            live.display()
        )
    });
    assert_eq!(
        reference::sha256::hex(&bytes),
        TESSL_GDN_TRAIN_SHA256,
        "tessl's tests/common/gdn_train.rs changed since the port: re-verify the bit-identity, \
         then update the pin and the copy together"
    );
    assert!(
        bytes == COPY,
        "the in-crate copy differs from tessl's live file"
    );
    eprintln!(
        "tessl live file {} == in-crate copy == pin {TESSL_GDN_TRAIN_SHA256}",
        live.display()
    );
}

/// `(max |got - want| / max |want|, bit-identical?)`.
fn compare(label: &str, got: &[f64], want: &[f64]) -> (f64, bool) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mag = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    let same = got
        .iter()
        .zip(want)
        .all(|(g, w)| g.to_bits() == w.to_bits());
    if mag == 0.0 {
        // Legitimately zero (e.g. dg at T=1 with no initial state: the decayed
        // state is zero). Then nothing but exact zeros is acceptable.
        assert!(
            err == 0.0,
            "{label}: tessl's output is all zeros but the port's is not (max {err:.3e})"
        );
        return (0.0, same);
    }
    let rel = err / mag;
    assert!(
        rel <= BOUND,
        "{label}: port differs from tessl's reference by {rel:.3e} of max|ref|, over {BOUND:.0e}"
    );
    (rel, same)
}

#[test]
fn published_reference_matches_tessls_gdn_train_f64_at_the_chunk_edges() {
    let mut all_bitwise = true;
    let mut worst = 0.0f64;
    for (t, with_s0, with_dfin) in [
        (1, false, false),
        (63, true, false),
        (64, false, true),
        (65, true, true),
        (130, true, true),
    ] {
        let (b, h, dv) = (2usize, 3usize, 32usize);
        let s = Shape {
            b,
            t,
            h,
            dk: KERNEL_DK,
            dv,
        };
        let seed = 100 + u64::try_from(t).expect("T fits u64");
        let n = s.rows();
        let q = rng::random_f64(n * KERNEL_DK, seed);
        let k = rng::random_f64(n * KERNEL_DK, seed + 1);
        let v = rng::random_f64(n * dv, seed + 2);
        let g: Vec<f64> = rng::random_f64(n, seed + 3)
            .iter()
            .map(|&x| -0.75 * (x + 1.0) - 1e-3)
            .collect();
        let beta: Vec<f64> = rng::random_f64(n, seed + 4)
            .iter()
            .map(|&x| 0.5 + 0.45 * x)
            .collect();
        let s0 = with_s0.then(|| rng::random_f64(s.state_len(), seed + 5));
        let d_o = rng::random_f64(n * dv, seed + 6);
        let dfin = with_dfin.then(|| rng::random_f64(s.state_len(), seed + 7));

        let ours = Inputs {
            s,
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            s0: s0.as_deref(),
        };
        let theirs = tessl_gdn_train::Inputs {
            s: tessl_gdn_train::Shape {
                b,
                t,
                h,
                dk: KERNEL_DK,
                dv,
            },
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            s0: s0.as_deref(),
        };
        let fwd = gdn_train_published_fwd_f64(&ours);
        let (want_o, want_fin) = tessl_gdn_train::gdn_train_f64(&theirs);
        let grads = gdn_train_published_bwd_f64(&ours, &fwd.ckpt, &d_o, dfin.as_deref());
        let want = tessl_gdn_train::gdn_train_bwd_f64(&theirs, &d_o, dfin.as_deref());

        let label = format!("published T={t} s0={with_s0} dfin={with_dfin}");
        let pairs: [(&str, &[f64], &[f64]); 8] = [
            ("o", &fwd.o, &want_o),
            ("final state", &fwd.fin, &want_fin),
            ("dq", &grads.dq, &want.dq),
            ("dk", &grads.dk, &want.dk),
            ("dv", &grads.dv, &want.dv),
            ("dg", &grads.dg, &want.dg),
            ("dbeta", &grads.dbeta, &want.dbeta),
            ("ds0", &grads.ds0, &want.ds0),
        ];
        for (name, got, w) in pairs {
            let (rel, same) = compare(&format!("{label} {name}"), got, w);
            eprintln!("{label} {name}: {rel:.3e} of max|tessl|, bit-identical: {same}");
            all_bitwise &= same;
            worst = worst.max(rel);
        }
    }
    eprintln!("published vs tessl in process: worst {worst:.3e} (bound {BOUND:.0e}), all bit-identical: {all_bitwise}");
}
