//! Seeded randomized sweep over shapes, strides, layouts and `accumulate`.
//! Each case is checked against the f64 reference, and all backends must agree
//! bit for bit. Each case is also shrunk by one element per operand, which must
//! be refused.

mod common;

use common::*;
use ojas_simd::{sgemm_tile_with, SimdError};

const SEED: u64 = 0x0_7A5_51D;

fn iterations() -> usize {
    if cfg!(miri) {
        25
    } else {
        3000
    }
}

fn random_kind(rng: &mut Rng) -> Kind {
    match rng.below(6) {
        0 => Kind::RowMajor,
        1 => Kind::ColMajor,
        2 => Kind::PaddedRow(1 + rng.below(7)),
        3 => Kind::PaddedCol(1 + rng.below(7)),
        4 => Kind::Strided,
        _ => Kind::BroadcastRows,
    }
}

fn random_dim(rng: &mut Rng) -> usize {
    match rng.below(10) {
        0 => 0,
        1 => 1,
        2..=6 => 1 + rng.below(24),
        _ => 1 + rng.below(if cfg!(miri) { 24 } else { 80 }),
    }
}

#[test]
fn random_shapes_and_strides() {
    let mut rng = Rng::new(SEED);
    let mut checked = 0usize;
    for it in 0..iterations() {
        let (m, n, k) = (
            random_dim(&mut rng),
            random_dim(&mut rng),
            random_dim(&mut rng),
        );
        let (ka, kb) = (random_kind(&mut rng), random_kind(&mut rng));
        let a = operand(&mut rng, m, k, ka);
        let b = operand(&mut rng, k, n, kb);
        let pad = rng.below(4);
        let (c0, c_rs) = output(&mut rng, m, n, pad);
        let accumulate = rng.coin();
        let case = Case {
            m,
            n,
            k,
            a: &a,
            b: &b,
            c0: &c0,
            c_rs,
            accumulate,
        };
        case.run_all(&format!(
            "iter {it}: {m}x{n}x{k} A={ka:?} B={kb:?} pad={pad} acc={accumulate}"
        ));
        checked += 1;

        if m == 0 || n == 0 {
            continue;
        }
        // One element short on any read operand must be an error, not UB.
        for bk in available_backends() {
            let mut c = c0.clone();
            if !a.buf.is_empty() {
                let r = sgemm_tile_with(
                    bk,
                    m,
                    n,
                    k,
                    &a.buf[..a.buf.len() - 1],
                    a.rs,
                    a.cs,
                    &b.buf,
                    b.rs,
                    b.cs,
                    &mut c,
                    c_rs,
                    accumulate,
                );
                assert!(
                    matches!(r, Err(SimdError::BufferTooShort { .. })),
                    "iter {it} A short: {r:?}"
                );
            }
            if !b.buf.is_empty() {
                let r = sgemm_tile_with(
                    bk,
                    m,
                    n,
                    k,
                    &a.buf,
                    a.rs,
                    a.cs,
                    &b.buf[..b.buf.len() - 1],
                    b.rs,
                    b.cs,
                    &mut c,
                    c_rs,
                    accumulate,
                );
                assert!(
                    matches!(r, Err(SimdError::BufferTooShort { .. })),
                    "iter {it} B short: {r:?}"
                );
            }
            let r = sgemm_tile_with(
                bk,
                m,
                n,
                k,
                &a.buf,
                a.rs,
                a.cs,
                &b.buf,
                b.rs,
                b.cs,
                &mut c[..c0.len() - 1],
                c_rs,
                accumulate,
            );
            assert!(
                matches!(r, Err(SimdError::BufferTooShort { .. })),
                "iter {it} C short: {r:?}"
            );
            assert_bits_eq(&c, &c0, &format!("iter {it}: refused call wrote C"));
        }
    }
    eprintln!("fuzz: {checked} random cases checked (seed {SEED:#x})");
}
