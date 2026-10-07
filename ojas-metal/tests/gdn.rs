//! `MetalBackend::chunked_gdn_forward` / `_backward` (tessl's `gdn_train`
//! kernels) against the CPU backend, plus the refusals and the deferred
//! non-finite fault. The CPU itself is checked against transformers'
//! float64 goldens in `ojas-oracle/tests/gdn.rs`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{
    Autocast, AutocastMode, Backend, GdnForward, GdnGrad, GdnInputs, OjasError, Tensor,
    METAL_GDN_KEY_DIM,
};

/// `max |metal - cpu| / max |cpu|` per tensor. tessl holds its kernels to
/// 1e-4 of an f64 reference (`tests/gdn_train.rs`); the CPU is within 1e-5
/// of the f64 goldens.
const BOUND: f32 = 2e-4;

fn rel_close(what: &str, got: &Tensor, want: &Tensor) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (g, w) = (down(got), down(want));
    let peak = w.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if peak == 0.0 {
        // An exact zero (dg at T 1 from a zero state: <dS^, 0>) must stay
        // exactly zero on the device.
        assert!(
            g.iter().all(|&a| a == 0.0),
            "{what}: reference is zero, device is not"
        );
        return;
    }
    let mut worst = (0.0f32, 0usize);
    for (i, (a, b)) in g.iter().zip(&w).enumerate() {
        assert!(a.is_finite(), "{what}[{i}]: non-finite {a}");
        if (a - b).abs() > worst.0 {
            worst = ((a - b).abs(), i);
        }
    }
    assert!(
        worst.0 / peak <= BOUND,
        "{what}: {:.3e} of peak {peak:.3e} at {} (metal {} cpu {})",
        worst.0 / peak,
        worst.1,
        g[worst.1],
        w[worst.1]
    );
}

/// Host operands: `g` a log decay in `[-1, -0.05)`, `beta` in `[0.1, 0.9)`.
struct Host {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Tensor,
}

fn affine(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    values(n, seed, 1.0)
        .into_iter()
        .map(|x| lo + (hi - lo) * (x + 1.0) * 0.5)
        .collect()
}

fn problem(b: usize, t: usize, h: usize, dk: usize, dv: usize, seed: u64) -> Host {
    let rows = b * t * h;
    Host {
        q: rand(&[b, t, h, dk], seed, 1.0),
        k: rand(&[b, t, h, dk], seed + 1, 1.0),
        v: rand(&[b, t, h, dv], seed + 2, 1.0),
        g: host(&affine(rows, seed + 3, -1.0, -0.05), &[b, t, h]),
        beta: host(&affine(rows, seed + 4, 0.1, 0.9), &[b, t, h]),
        s0: rand(&[b, h, dk, dv], seed + 5, 0.5),
    }
}

/// The same operands, placed on `m` (or the host ones for the CPU).
struct Placed {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Tensor,
}

impl Placed {
    fn on(m: Option<&ojas_metal::MetalBackend>, x: &Host) -> Self {
        let p = |t: &Tensor| m.map_or_else(|| t.clone(), |m| up(m, t));
        Self {
            q: p(&x.q),
            k: p(&x.k),
            v: p(&x.v),
            g: p(&x.g),
            beta: p(&x.beta),
            s0: p(&x.s0),
        }
    }

    fn inputs(&self, with_s0: bool) -> GdnInputs<'_> {
        GdnInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            initial_state: with_s0.then_some(&self.s0),
        }
    }
}

fn run<B: Backend>(
    be: &B,
    x: GdnInputs<'_>,
    d_o: &Tensor,
    d_fin: Option<&Tensor>,
) -> (GdnForward, GdnGrad) {
    let fwd = ok("forward", be.chunked_gdn_forward(x));
    let grad = ok(
        "backward",
        be.chunked_gdn_backward(x, &fwd.checkpoints, d_o, d_fin),
    );
    (fwd, grad)
}

#[test]
fn gdn_matches_cpu_across_checkpoints_slices_and_initial_state() {
    let (m, c) = (metal(), cpu());
    let dk = METAL_GDN_KEY_DIM;
    // T 1, either side of the first checkpoint, and three chunks; one and
    // two 16-column value slices; with and without an initial state and a
    // final-state gradient.
    for (i, &(b, t, h, dv)) in [
        (1, 1, 2, 16),
        (1, 64, 1, 16),
        (2, 65, 2, 32),
        (1, 130, 3, 16),
    ]
    .iter()
    .enumerate()
    {
        let x = problem(b, t, h, dk, dv, 40 + 10 * i as u64);
        let d_o = rand(&[b, t, h, dv], 90 + i as u64, 1.0);
        let d_fin = rand(&[b, h, dk, dv], 95 + i as u64, 1.0);
        let (hc, hm) = (Placed::on(None, &x), Placed::on(Some(&m), &x));
        let (dd_o, dd_fin) = (up(&m, &d_o), up(&m, &d_fin));
        for with_s0 in [false, true] {
            let label = format!("B{b} T{t} H{h} Dv{dv} s0 {with_s0}");
            let df = with_s0.then_some(&d_fin);
            let ddf = with_s0.then_some(&dd_fin);
            let (cf, cg) = run(&c, hc.inputs(with_s0), &d_o, df);
            let (mf, mg) = run(&m, hm.inputs(with_s0), &dd_o, ddf);
            ok(&label, m.sync());
            rel_close(&format!("{label} o"), &mf.output, &cf.output);
            rel_close(&format!("{label} fin"), &mf.final_state, &cf.final_state);
            if with_s0 || t > 64 {
                rel_close(&format!("{label} ckpt"), &mf.checkpoints, &cf.checkpoints);
            }
            for (name, got, want) in [
                ("dq", &mg.q, &cg.q),
                ("dk", &mg.k, &cg.k),
                ("dv", &mg.v, &cg.v),
                ("dg", &mg.g, &cg.g),
                ("dbeta", &mg.beta, &cg.beta),
            ] {
                rel_close(&format!("{label} {name}"), got, want);
            }
            assert_eq!(mg.initial_state.is_some(), with_s0, "{label}");
            if let (Some(a), Some(b)) = (&mg.initial_state, &cg.initial_state) {
                rel_close(&format!("{label} ds0"), a, b);
            }
        }
    }
}

/// `t`'s values inside a larger device buffer, starting `lead` elements in:
/// a contiguous view whose byte offset is neither zero nor 16-aligned.
fn offset_view(m: &ojas_metal::MetalBackend, t: &Tensor, lead: usize, seed: u64) -> Tensor {
    let body = down(t);
    let mut padded = values(lead, seed, 1.0);
    padded.extend_from_slice(&body);
    padded.extend(values(5, seed + 1, 1.0));
    let big = up(m, &host(&padded, &[padded.len()]));
    assert!(
        ok("contiguous", t.is_contiguous()),
        "host operands are contiguous"
    );
    ok("narrow", big.narrow(lead * 4, t.shape(), t.strides()))
}

#[test]
fn operands_at_unaligned_byte_offsets_give_the_same_bits() {
    let m = metal();
    let (b, t, h, dv) = (1, 65, 2, 16);
    let x = problem(b, t, h, METAL_GDN_KEY_DIM, dv, 77);
    let d_o = up(&m, &rand(&[b, t, h, dv], 78, 1.0));
    let d_fin = up(&m, &rand(&[b, h, METAL_GDN_KEY_DIM, dv], 79, 1.0));
    let plain = Placed::on(Some(&m), &x);
    // A different odd lead per operand, so no two share an alignment.
    let shifted = Placed {
        q: offset_view(&m, &x.q, 3, 1),
        k: offset_view(&m, &x.k, 5, 3),
        v: offset_view(&m, &x.v, 7, 5),
        g: offset_view(&m, &x.g, 1, 7),
        beta: offset_view(&m, &x.beta, 9, 9),
        s0: offset_view(&m, &x.s0, 11, 11),
    };
    let (pf, pg) = run(&m, plain.inputs(true), &d_o, Some(&d_fin));
    let sf = ok("forward", m.chunked_gdn_forward(shifted.inputs(true)));
    // The backward's own operands shifted too: checkpoints, d_o, d_fin.
    let (s_ckpt, s_d_o, s_d_fin) = (
        offset_view(&m, &sf.checkpoints, 13, 13),
        offset_view(&m, &d_o, 15, 15),
        offset_view(&m, &d_fin, 17, 17),
    );
    for (name, a) in [
        ("q", &shifted.q),
        ("k", &shifted.k),
        ("v", &shifted.v),
        ("g", &shifted.g),
        ("beta", &shifted.beta),
        ("s0", &shifted.s0),
        ("ckpt", &s_ckpt),
        ("d_o", &s_d_o),
        ("d_fin", &s_d_fin),
    ] {
        assert!(a.byte_offset() % 16 != 0, "{name}: offset not exercised");
    }
    let sg = ok(
        "backward",
        m.chunked_gdn_backward(shifted.inputs(true), &s_ckpt, &s_d_o, Some(&s_d_fin)),
    );
    ok("sync", m.sync());
    let bits = |t: &Tensor| down(t).iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (name, got, want) in [
        ("o", &sf.output, &pf.output),
        ("fin", &sf.final_state, &pf.final_state),
        ("ckpt", &sf.checkpoints, &pf.checkpoints),
        ("dq", &sg.q, &pg.q),
        ("dk", &sg.k, &pg.k),
        ("dv", &sg.v, &pg.v),
        ("dg", &sg.g, &pg.g),
        ("dbeta", &sg.beta, &pg.beta),
    ] {
        assert!(
            bits(got) == bits(want),
            "{name}: offset operands changed the bits"
        );
    }
    let (a, b) = (sg.initial_state.as_ref(), pg.initial_state.as_ref());
    assert!(bits(a.expect("ds0")) == bits(b.expect("ds0")), "ds0");
}

#[test]
fn dims_tessl_is_not_compiled_for_are_refused_and_record_nothing() {
    let m = metal();
    for (dk, dv) in [(64, 16), (METAL_GDN_KEY_DIM, 24), (METAL_GDN_KEY_DIM, 8)] {
        let x = problem(1, 3, 1, dk, dv, 7);
        let p = Placed::on(Some(&m), &x);
        let r = m.chunked_gdn_forward(p.inputs(false));
        assert!(
            matches!(
                r,
                Err(OjasError::Unsupported {
                    op: "chunked_gdn",
                    ..
                })
            ),
            "dk {dk} dv {dv}: {r:?}"
        );
        assert!(m.sync().is_ok(), "a refusal records nothing");
    }
}

#[test]
fn the_shape_check_comes_before_placement() {
    let m = metal();
    // Host tensors would be a Placement error; the misshapen beta wins.
    let x = problem(1, 3, 1, METAL_GDN_KEY_DIM, 16, 8);
    let wrong = host(&[0.5; 4], &[1, 4, 1]);
    let r = m.chunked_gdn_forward(GdnInputs {
        q: &x.q,
        k: &x.k,
        v: &x.v,
        g: &x.g,
        beta: &wrong,
        initial_state: None,
    });
    assert!(
        matches!(
            r,
            Err(OjasError::Shape {
                op: "chunked_gdn_forward",
                ..
            })
        ),
        "{r:?}"
    );
    let r = m.chunked_gdn_forward(GdnInputs {
        q: &x.q,
        k: &x.k,
        v: &x.v,
        g: &x.g,
        beta: &x.beta,
        initial_state: None,
    });
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_naming_the_op() {
    let m = metal();
    let (b, t, h, dk, dv) = (1, 70, 1, METAL_GDN_KEY_DIM, 16);
    let mut x = problem(b, t, h, dk, dv, 9);
    let mut g = down(&x.g);
    g[t - 1] = f32::NAN;
    x.g = host(&g, &[b, t, h]);
    let p = Placed::on(Some(&m), &x);
    let fwd = deferred(
        &m,
        "nan g",
        m.chunked_gdn_forward(p.inputs(false)),
        "chunked_gdn_forward",
    );

    // A clean forward, then a NaN in the output gradient.
    let clean = Placed::on(Some(&m), &problem(b, t, h, dk, dv, 10));
    let fwd2 = ok("clean forward", m.chunked_gdn_forward(clean.inputs(false)));
    ok("clean sync", m.sync());
    let mut d_o = values(b * t * h * dv, 11, 1.0);
    d_o[0] = f32::INFINITY;
    let d_o = up(&m, &host(&d_o, &[b, t, h, dv]));
    deferred(
        &m,
        "inf d_o",
        m.chunked_gdn_backward(clean.inputs(false), &fwd2.checkpoints, &d_o, None),
        "chunked_gdn_backward",
    );
    drop(fwd);
}

/// Inside a bf16 autocast region the gated delta rule stays f32: the
/// wrapper passes it through, so its bits are the bare backend's.
#[test]
fn autocast_leaves_the_gated_delta_rule_in_f32() {
    let m = metal();
    let a = Autocast::new(m.clone());
    let x = problem(1, 66, 2, METAL_GDN_KEY_DIM, 16, 12);
    let p = Placed::on(Some(&m), &x);
    let bare = ok("bare", m.chunked_gdn_forward(p.inputs(true)));
    let _region = ok("region", a.autocast_region(AutocastMode::Bf16));
    let wrapped = ok("wrapped", a.chunked_gdn_forward(p.inputs(true)));
    let bits = |t: &Tensor| down(t).iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&wrapped.output), bits(&bare.output));
    assert_eq!(bits(&wrapped.final_state), bits(&bare.final_state));
}
