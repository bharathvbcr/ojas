//! Adversarial tests for Metal `linear_backward_input` (card ft-6fb212ce):
//! dX bit-identical to `linear_backward`'s at awkward shapes, a bf16 weight,
//! budget accounting (no dW), repeatability, and agreement with the CPU.
//! Needs a Metal device; run under the mac_heavy lock.
#![cfg(target_os = "macos")]

mod common;

use common::{close, cpu, down, host, metal, ok, rand, up, values};
use ojas_core::{Backend, Budget, Tensor};

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// Shapes that miss every tile edge, with rows = 1 and the 1 x 1 corner.
const SHAPES: [(usize, usize, usize); 11] = [
    (1, 1, 1),
    (1, 1, 7),
    (1, 7, 1),
    (2, 1, 1),
    (1, 513, 1),
    (1, 2048, 2048),
    (3, 31, 5),
    (17, 129, 63),
    (65, 257, 33),
    (255, 257, 129),
    (512, 1024, 2048),
];

#[test]
fn metal_dx_only_is_bit_identical_to_the_full_path_at_awkward_shapes() {
    let m = metal();
    for (rows, kin, nout) in SHAPES {
        let x = up(&m, &rand(&[rows, kin], 1, 1.0));
        let w = up(&m, &rand(&[nout, kin], 2, 1.0));
        let g = up(&m, &rand(&[rows, nout], 3, 1.0));
        let (full, _dw) = ok("full", m.linear_backward(&x, &w, &g));
        let only = ok("dx-only", m.linear_backward_input(&x, &w, &g));
        ok("sync", m.sync());
        assert_eq!(only.shape(), full.shape());
        assert_eq!(bits(&only), bits(&full), "{rows}x{kin}x{nout}");
    }
}

/// The result is repeatable run to run (no scheduling-dependent bits).
#[test]
fn metal_dx_only_repeats_bit_for_bit() {
    let m = metal();
    let (rows, kin, nout) = (129, 257, 65);
    let x = up(&m, &rand(&[rows, kin], 1, 1.0));
    let w = up(&m, &rand(&[nout, kin], 2, 1.0));
    let g = up(&m, &rand(&[rows, nout], 3, 1.0));
    let first = bits(&ok("a", m.linear_backward_input(&x, &w, &g)));
    for _ in 0..4 {
        assert_eq!(bits(&ok("b", m.linear_backward_input(&x, &w, &g))), first);
    }
}

/// A resident bf16 weight on Metal: dX-only equals the full path with the
/// same bf16 weight, and equals the f32 path with the widened weight.
#[test]
fn metal_dx_only_takes_a_bf16_weight_and_matches_the_full_path() {
    let m = metal();
    for (rows, kin, nout) in [(1, 1, 1), (1, 513, 7), (17, 129, 63), (255, 257, 129)] {
        let x = up(&m, &rand(&[rows, kin], 1, 1.0));
        let w32 = up(&m, &rand(&[nout, kin], 2, 1.0));
        let g = up(&m, &rand(&[rows, nout], 3, 1.0));
        let wb = ok("to_bf16", m.to_bf16(&w32));
        let (full, _) = ok("full bf16", m.linear_backward(&x, &wb, &g));
        let only = ok("dx-only bf16", m.linear_backward_input(&x, &wb, &g));
        assert_eq!(bits(&only), bits(&full), "bf16 {rows}x{kin}x{nout}");
        let w_wide = ok("to_f32", m.to_f32(&wb));
        let wide = ok("dx-only widened", m.linear_backward_input(&x, &w_wide, &g));
        assert_eq!(
            bits(&only),
            bits(&wide),
            "bf16 vs widened {rows}x{kin}x{nout}"
        );
    }
}

/// bf16 gradient output (an Autocast region hands bf16 g to the backend).
#[test]
fn metal_dx_only_takes_a_bf16_grad_output_like_the_full_path() {
    let m = metal();
    let (rows, kin, nout) = (17, 129, 63);
    let x = up(&m, &rand(&[rows, kin], 1, 1.0));
    let w = up(&m, &rand(&[nout, kin], 2, 1.0));
    let g = ok("g bf16", m.to_bf16(&up(&m, &rand(&[rows, nout], 3, 1.0))));
    let full = m.linear_backward(&x, &w, &g);
    let only = m.linear_backward_input(&x, &w, &g);
    match (full, only) {
        (Ok((f, _)), Ok(o)) => assert_eq!(bits(&o), bits(&f)),
        (Err(_), Err(_)) => {}
        (f, o) => panic!(
            "verdicts differ: full ok={} only ok={}",
            f.is_ok(),
            o.is_ok()
        ),
    }
}

/// No dW on the device budget: the live bytes dX-only leaves are dX, and
/// its peak sits at least dW below the full path's.
#[test]
fn metal_dx_only_charges_no_weight_gradient() {
    let m = metal();
    let (rows, kin, nout) = (64usize, 512usize, 1024usize);
    let x = up(&m, &rand(&[rows, kin], 1, 1.0));
    let w = up(&m, &rand(&[nout, kin], 2, 1.0));
    let g = up(&m, &rand(&[rows, nout], 3, 1.0));
    ok("sync", m.sync());
    let dx = (rows * kin * 4) as u64;
    let dw = (nout * kin * 4) as u64;

    let before = ok("live", m.budget().live_bytes());
    m.budget().reset_peak();
    let (full, full_dw) = ok("full", m.linear_backward(&x, &w, &g));
    ok("sync", m.sync());
    let full_peak = m.budget().peak_bytes() - before;
    let full_kept = ok("live", m.budget().live_bytes()) - before;
    drop((full, full_dw));

    let before = ok("live", m.budget().live_bytes());
    m.budget().reset_peak();
    let only = ok("only", m.linear_backward_input(&x, &w, &g));
    ok("sync", m.sync());
    let peak = m.budget().peak_bytes() - before;
    let kept = ok("live", m.budget().live_bytes()) - before;
    assert_eq!(kept, dx, "only dX stays live");
    assert_eq!(full_kept, dx + dw);
    assert!(
        peak + dw <= full_peak,
        "dX-only peak {peak} + dW {dw} must not exceed the full peak {full_peak}"
    );
    drop(only);
}

/// A budget that holds dX-only but not the full path: the frozen path must
/// run, and the full path must refuse.
#[test]
fn metal_dx_only_runs_where_the_full_path_would_exceed_the_budget() {
    let (rows, kin, nout) = (16usize, 512usize, 1024usize);
    let probe = ok("probe", ojas_metal::MetalBackend::new(Budget::new(1 << 30)));
    let x = up(&probe, &rand(&[rows, kin], 1, 1.0));
    let w = up(&probe, &rand(&[nout, kin], 2, 1.0));
    let g = up(&probe, &rand(&[rows, nout], 3, 1.0));
    ok("sync", probe.sync());
    let before = ok("live", probe.budget().live_bytes());
    probe.budget().reset_peak();
    let only = ok("only", probe.linear_backward_input(&x, &w, &g));
    ok("sync", probe.sync());
    let peak = probe.budget().peak_bytes() - before;
    drop(only);
    // Same shapes on a backend whose cap is the operands plus that peak.
    let tight = ok(
        "tight",
        ojas_metal::MetalBackend::new(Budget::new(before + peak)),
    );
    let (x, w, g) = (
        up(&tight, &rand(&[rows, kin], 1, 1.0)),
        up(&tight, &rand(&[nout, kin], 2, 1.0)),
        up(&tight, &rand(&[rows, nout], 3, 1.0)),
    );
    let live = ok("live", tight.budget().live_bytes());
    assert!(live <= before, "operands {live} vs probe {before}");
    let only = tight.linear_backward_input(&x, &w, &g);
    assert!(only.is_ok(), "dX-only in the tight budget: {only:?}");
    drop(only);
    assert!(
        tight.linear_backward(&x, &w, &g).is_err(),
        "the full path must not fit (dW is {} B)",
        nout * kin * 4
    );
}

/// Metal dX-only against the CPU's Exact dX at awkward shapes.
#[test]
fn metal_dx_only_agrees_with_the_cpu_at_awkward_shapes() {
    let m = metal();
    let c = cpu();
    for (rows, kin, nout) in SHAPES {
        let xh = rand(&[rows, kin], 1, 1.0);
        let wh = rand(&[nout, kin], 2, 1.0);
        let gh = rand(&[rows, nout], 3, 1.0);
        let want = down(&ok("cpu", c.linear_backward_input(&xh, &wh, &gh)));
        let got = down(&ok(
            "metal",
            m.linear_backward_input(&up(&m, &xh), &up(&m, &wh), &up(&m, &gh)),
        ));
        close(
            &format!("{rows}x{kin}x{nout}"),
            &got,
            &want,
            1e-4 * (nout as f32).sqrt(),
            1e-4,
        );
    }
}

/// A NaN in `g` or a non-finite `w`: the full path's verdict and dX-only's
/// agree (error or equal bits).
#[test]
fn metal_dx_only_and_the_full_path_agree_on_non_finite_operands() {
    let m = metal();
    let (rows, kin, nout) = (4, 6, 9);
    let x = up(&m, &rand(&[rows, kin], 1, 1.0));
    let mut ws = values(nout * kin, 2, 1.0);
    ws[7] = f32::INFINITY;
    let mut gs = values(rows * nout, 3, 1.0);
    gs[5] = f32::NAN;
    for (w, g) in [
        (host(&ws, &[nout, kin]), rand(&[rows, nout], 3, 1.0)),
        (rand(&[nout, kin], 2, 1.0), host(&gs, &[rows, nout])),
    ] {
        let (w, g) = (up(&m, &w), up(&m, &g));
        let full = m.linear_backward(&x, &w, &g).and_then(|r| {
            m.sync()?;
            Ok(r)
        });
        let only = m.linear_backward_input(&x, &w, &g).and_then(|r| {
            m.sync()?;
            Ok(r)
        });
        match (full, only) {
            (Ok((f, _)), Ok(o)) => assert_eq!(bits(&o), bits(&f)),
            (Err(_), Err(_)) => {}
            (f, o) => panic!(
                "verdicts differ: full ok={} only ok={}",
                f.is_ok(),
                o.is_ok()
            ),
        }
    }
}
