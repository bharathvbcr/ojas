//! Qwen3.5's hybrid-layer ops on Metal: `causal_conv1d_silu` and
//! `gated_rms_norm` on tessl's `qwen35` kernels, `rope_partial` on
//! `ojas_rope`, each forward and backward against the CPU backend, plus the
//! refusals, unaligned operands and the deferred non-finite fault. The CPU
//! itself is checked against f64 references in `ojas-cpu/tests/hybrid.rs`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{mrope_text_tables, Backend, Budget, MropeSection, OjasError, Tensor};

/// `max |metal - cpu| / max |cpu|` per tensor.
const BOUND: f32 = 1e-5;
const EPS: f32 = 1e-6;

fn rel_close(what: &str, got: &Tensor, want: &Tensor) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (g, w) = (down(got), ok(what, want.to_f32_vec()));
    let peak = w.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(peak > 0.0, "{what}: an all-zero reference checks nothing");
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

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|v| v.to_bits()).collect()
}

/// `t`'s values inside a larger device buffer, starting `lead` elements in:
/// a contiguous view whose byte offset is neither zero nor 16-aligned.
fn offset_view(m: &ojas_metal::MetalBackend, t: &Tensor, lead: usize, seed: u64) -> Tensor {
    let mut padded = values(lead, seed, 1.0);
    padded.extend_from_slice(&ok("host", t.to_f32_vec()));
    padded.extend(values(5, seed + 1, 1.0));
    let big = up(m, &host(&padded, &[padded.len()]));
    let v = ok("narrow", big.narrow(lead * 4, t.shape(), t.strides()));
    assert!(!v.byte_offset().is_multiple_of(16), "offset not exercised");
    v
}

/// Qwen3.5's text-only MRoPE tables: interleaved sections, 64 of 256 turn.
fn qwen_tables(time: usize, rotary: usize) -> (Tensor, Tensor) {
    let half = rotary / 2;
    let section = MropeSection {
        section: [half - 2 * (half / 3), half / 3, half / 3],
        interleaved: true,
    };
    ok(
        "tables",
        mrope_text_tables(5, time, section, rotary, 1e7, &Budget::new(1 << 30)),
    )
}

// ---- causal conv1d + SiLU --------------------------------------------------

#[test]
fn conv1d_matches_cpu_forward_and_backward() {
    let (m, c) = (metal(), cpu());
    // T below, at and above the width; Qwen3.5's width 4; both width limits;
    // more rows than one weight-gradient block (256).
    for (i, &(b, t, ch, k)) in [
        (1, 1, 3, 4),
        (1, 3, 5, 4),
        (2, 17, 5, 4),
        (2, 3, 7, 2),
        (1, 9, 4, 8),
        (2, 300, 64, 4),
    ]
    .iter()
    .enumerate()
    {
        let seed = 10 * i as u64;
        let x = rand(&[b, t, ch], seed, 2.0);
        let w = rand(&[ch, k], seed + 1, 1.0);
        let gy = rand(&[b, t, ch], seed + 2, 1.0);
        let what = format!("[{b}, {t}, {ch}] k {k}");
        let want = ok("cpu fwd", c.causal_conv1d_silu_forward(&x, &w));
        let (wx, ww) = ok("cpu bwd", c.causal_conv1d_silu_backward(&x, &w, &gy));
        let (xm, wm, gm) = (up(&m, &x), up(&m, &w), up(&m, &gy));
        let got = ok("fwd", m.causal_conv1d_silu_forward(&xm, &wm));
        let (gx, gw) = ok("bwd", m.causal_conv1d_silu_backward(&xm, &wm, &gm));
        ok("sync", m.sync());
        rel_close(&format!("{what} y"), &got, &want);
        rel_close(&format!("{what} dx"), &gx, &wx);
        rel_close(&format!("{what} dw"), &gw, &ww);
    }
}

#[test]
fn conv1d_widths_the_kernels_are_not_built_for_are_refused_and_record_nothing() {
    let m = metal();
    for k in [1, 9] {
        let x = up(&m, &rand(&[1, 5, 3], 1, 1.0));
        let w = up(&m, &rand(&[3, k], 2, 1.0));
        let r = m.causal_conv1d_silu_forward(&x, &w);
        assert!(
            matches!(
                r,
                Err(OjasError::Unsupported {
                    op: "causal_conv1d_silu_forward",
                    ..
                })
            ),
            "k {k}: {r:?}"
        );
        let r = m.causal_conv1d_silu_backward(&x, &w, &x);
        assert!(
            matches!(
                r,
                Err(OjasError::Unsupported {
                    op: "causal_conv1d_silu_backward",
                    ..
                })
            ),
            "k {k}: {r:?}"
        );
        assert!(m.sync().is_ok(), "a refusal records nothing");
    }
}

// ---- gated RMSNorm ---------------------------------------------------------

#[test]
fn gated_rms_norm_matches_cpu_forward_and_backward() {
    let (m, c) = (metal(), cpu());
    // Qwen3.5's head of 128, an odd row, a leading-axis stack, more rows than
    // one weight-gradient block (64), and the backward's widest row.
    for (i, shape) in [
        vec![3, 128],
        vec![4, 7],
        vec![2, 5, 4, 64],
        vec![70, 128],
        vec![2, 512],
    ]
    .iter()
    .enumerate()
    {
        let seed = 100 + 10 * i as u64;
        let dim = shape[shape.len() - 1];
        let x = rand(shape, seed, 2.0);
        let z = rand(shape, seed + 1, 2.0);
        let w = host(
            &values(dim, seed + 2, 0.5)
                .iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>(),
            &[dim],
        );
        let gy = rand(shape, seed + 3, 1.0);
        let what = format!("{shape:?}");
        let want = ok("cpu fwd", c.gated_rms_norm_forward(&x, &z, &w, EPS));
        let wg = ok("cpu bwd", c.gated_rms_norm_backward(&x, &z, &w, &gy, EPS));
        let (xm, zm, wm, gm) = (up(&m, &x), up(&m, &z), up(&m, &w), up(&m, &gy));
        let got = ok("fwd", m.gated_rms_norm_forward(&xm, &zm, &wm, EPS));
        let g = ok("bwd", m.gated_rms_norm_backward(&xm, &zm, &wm, &gm, EPS));
        ok("sync", m.sync());
        rel_close(&format!("{what} y"), &got, &want);
        rel_close(&format!("{what} dx"), &g.input, &wg.input);
        rel_close(&format!("{what} dz"), &g.gate, &wg.gate);
        rel_close(&format!("{what} dw"), &g.weight, &wg.weight);
    }
}

#[test]
fn a_gated_rms_backward_row_wider_than_the_kernel_is_refused() {
    let m = metal();
    let shape = [2, 513];
    let t = up(&m, &rand(&shape, 1, 1.0));
    let w = up(&m, &rand(&[513], 2, 1.0));
    ok(
        "forward takes any row",
        m.gated_rms_norm_forward(&t, &t, &w, EPS),
    );
    ok("sync", m.sync());
    let r = m.gated_rms_norm_backward(&t, &t, &w, &t, EPS);
    assert!(
        matches!(
            r,
            Err(OjasError::Unsupported {
                op: "gated_rms_norm_backward",
                ..
            })
        ),
        "{r:?}"
    );
    assert!(m.sync().is_ok(), "a refusal records nothing");
}

// ---- partial RoPE ----------------------------------------------------------

#[test]
fn rope_partial_matches_cpu_both_ways() {
    let (m, c) = (metal(), cpu());
    // Qwen3.5's 64 of 256, a two-value turn, and the whole head.
    for (i, &(b, t, h, d, r)) in [(2, 5, 3, 256, 64), (1, 4, 2, 6, 2), (2, 3, 2, 8, 8)]
        .iter()
        .enumerate()
    {
        let seed = 200 + 10 * i as u64;
        let x = rand(&[b, t, h, d], seed, 2.0);
        let (cos, sin) = qwen_tables(t, r);
        let what = format!("[{b}, {t}, {h}, {d}] r {r}");
        let want = ok("cpu fwd", c.rope_partial_forward(&x, &cos, &sin));
        let wantb = ok("cpu bwd", c.rope_partial_backward(&x, &cos, &sin));
        let (xm, cm, sm) = (up(&m, &x), up(&m, &cos), up(&m, &sin));
        let got = ok("fwd", m.rope_partial_forward(&xm, &cm, &sm));
        let gotb = ok("bwd", m.rope_partial_backward(&xm, &cm, &sm));
        ok("sync", m.sync());
        same_tensor(&format!("{what} fwd"), &got, &want, 1e-6, 1e-6);
        same_tensor(&format!("{what} bwd"), &gotb, &wantb, 1e-6, 1e-6);
        // The copied tail is the input's bits.
        let (xv, gv) = (down(&x), down(&got));
        for row in 0..b * t * h {
            let tail = row * d + r..(row + 1) * d;
            let (a, e) = (&gv[tail.clone()], &xv[tail]);
            assert!(
                a.iter().zip(e).all(|(a, e)| a.to_bits() == e.to_bits()),
                "{what}: row {row} tail is not the input"
            );
        }
    }
}

/// Over the whole head the op is Metal's `rope_half_split`, bit for bit.
#[test]
fn rope_partial_over_the_whole_head_is_rope_half_split() {
    let m = metal();
    let (b, t, h, d) = (2, 5, 3, 16);
    let x = up(&m, &rand(&[b, t, h, d], 7, 2.0));
    let (cos, sin) = qwen_tables(t, d);
    let (cos, sin) = (up(&m, &cos), up(&m, &sin));
    let a = ok("partial", m.rope_partial_forward(&x, &cos, &sin));
    let e = ok("half split", m.rope_half_split_forward(&x, &cos, &sin));
    let ab = ok("partial bwd", m.rope_partial_backward(&x, &cos, &sin));
    let eb = ok("half split bwd", m.rope_half_split_backward(&x, &cos, &sin));
    ok("sync", m.sync());
    assert_eq!(bits(&a), bits(&e), "forward");
    assert_eq!(bits(&ab), bits(&eb), "backward");
}

// ---- every op ----------------------------------------------------------------

#[test]
fn operands_at_unaligned_byte_offsets_give_the_same_bits() {
    let m = metal();
    let (b, t, ch, k) = (2, 9, 8, 4);
    let x = rand(&[b, t, ch], 1, 2.0);
    let w = rand(&[ch, k], 2, 1.0);
    let z = rand(&[b, t, ch], 3, 2.0);
    let nw = rand(&[ch], 4, 1.0);
    let gy = rand(&[b, t, ch], 5, 1.0);
    let (cos, sin) = qwen_tables(t, 4);
    let x4 = rand(&[b, t, 2, ch], 6, 2.0);

    let p = |t: &Tensor| up(&m, t);
    let y = ok("conv", m.causal_conv1d_silu_forward(&p(&x), &p(&w)));
    let (dx, dw) = ok(
        "conv bwd",
        m.causal_conv1d_silu_backward(&p(&x), &p(&w), &p(&gy)),
    );
    let n = ok(
        "norm",
        m.gated_rms_norm_forward(&p(&x), &p(&z), &p(&nw), EPS),
    );
    let ng = ok(
        "norm bwd",
        m.gated_rms_norm_backward(&p(&x), &p(&z), &p(&nw), &p(&gy), EPS),
    );
    let r = ok("rope", m.rope_partial_forward(&p(&x4), &p(&cos), &p(&sin)));
    let rb = ok(
        "rope bwd",
        m.rope_partial_backward(&p(&x4), &p(&cos), &p(&sin)),
    );

    // A different odd lead per operand, so no two share an alignment.
    let s = |t: &Tensor, lead: usize| offset_view(&m, t, lead, lead as u64);
    let sy = ok("conv", m.causal_conv1d_silu_forward(&s(&x, 3), &s(&w, 5)));
    let (sdx, sdw) = ok(
        "conv bwd",
        m.causal_conv1d_silu_backward(&s(&x, 7), &s(&w, 9), &s(&gy, 11)),
    );
    let sn = ok(
        "norm",
        m.gated_rms_norm_forward(&s(&x, 13), &s(&z, 15), &s(&nw, 17), EPS),
    );
    let sng = ok(
        "norm bwd",
        m.gated_rms_norm_backward(&s(&x, 19), &s(&z, 21), &s(&nw, 23), &s(&gy, 25), EPS),
    );
    let sr = ok(
        "rope",
        m.rope_partial_forward(&s(&x4, 27), &s(&cos, 29), &s(&sin, 31)),
    );
    let srb = ok(
        "rope bwd",
        m.rope_partial_backward(&s(&x4, 33), &s(&cos, 35), &s(&sin, 37)),
    );
    ok("sync", m.sync());
    for (name, got, want) in [
        ("conv y", &sy, &y),
        ("conv dx", &sdx, &dx),
        ("conv dw", &sdw, &dw),
        ("norm y", &sn, &n),
        ("norm dx", &sng.input, &ng.input),
        ("norm dz", &sng.gate, &ng.gate),
        ("norm dw", &sng.weight, &ng.weight),
        ("rope fwd", &sr, &r),
        ("rope bwd", &srb, &rb),
    ] {
        assert!(
            bits(got) == bits(want),
            "{name}: offset operands changed the bits"
        );
    }
}

#[test]
fn the_shape_check_comes_before_placement() {
    let m = metal();
    // Host tensors would be a Placement error; the misshapen operand wins.
    let x = rand(&[1, 4, 3], 1, 1.0);
    let w = rand(&[3, 4], 2, 1.0);
    let r = m.causal_conv1d_silu_forward(&x, &rand(&[2, 4], 3, 1.0));
    assert!(
        matches!(
            r,
            Err(OjasError::Shape {
                op: "causal_conv1d_silu_forward",
                ..
            })
        ),
        "{r:?}"
    );
    let r = m.causal_conv1d_silu_forward(&x, &w);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");

    let r = m.gated_rms_norm_forward(&x, &x, &rand(&[4], 4, 1.0), EPS);
    assert!(
        matches!(
            r,
            Err(OjasError::Shape {
                op: "gated_rms_norm_forward",
                ..
            })
        ),
        "{r:?}"
    );
    let r = m.gated_rms_norm_forward(&x, &x, &rand(&[3], 4, 1.0), EPS);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");

    let x4 = rand(&[1, 4, 2, 8], 5, 1.0);
    let odd = rand(&[4, 3], 6, 1.0);
    let r = m.rope_partial_forward(&x4, &odd, &odd);
    assert!(
        matches!(
            r,
            Err(OjasError::Shape {
                op: "rope_partial_forward",
                ..
            })
        ),
        "{r:?}"
    );
    let (cos, sin) = qwen_tables(4, 4);
    let r = m.rope_partial_forward(&x4, &cos, &sin);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    assert!(m.sync().is_ok(), "a refusal records nothing");
}

#[test]
fn a_non_finite_operand_is_a_deferred_fault_naming_the_op() {
    let m = metal();
    let poisoned = |shape: &[usize], seed: u64, at: usize, v: f32| {
        let mut d = values(shape.iter().product(), seed, 1.0);
        d[at] = v;
        up(&m, &host(&d, shape))
    };
    let (b, t, ch, k) = (1, 6, 4, 4);
    let w = up(&m, &rand(&[ch, k], 1, 1.0));
    let x = up(&m, &rand(&[b, t, ch], 2, 1.0));
    let nan_x = poisoned(&[b, t, ch], 3, 5, f32::NAN);
    deferred(
        &m,
        "nan conv x",
        m.causal_conv1d_silu_forward(&nan_x, &w),
        "causal_conv1d_silu_forward",
    );
    let inf_gy = poisoned(&[b, t, ch], 4, 0, f32::INFINITY);
    deferred(
        &m,
        "inf conv gy",
        m.causal_conv1d_silu_backward(&x, &w, &inf_gy),
        "causal_conv1d_silu_backward",
    );
    let nw = up(&m, &rand(&[ch], 5, 1.0));
    deferred(
        &m,
        "nan norm gate",
        m.gated_rms_norm_forward(&x, &nan_x, &nw, EPS),
        "gated_rms_norm_forward",
    );
    deferred(
        &m,
        "inf norm gy",
        m.gated_rms_norm_backward(&x, &x, &nw, &inf_gy, EPS),
        "gated_rms_norm_backward",
    );
    let (cos, sin) = qwen_tables(t, 4);
    let (cos, sin) = (up(&m, &cos), up(&m, &sin));
    let nan_x4 = poisoned(&[b, t, 1, ch], 6, 7, f32::NAN);
    deferred(
        &m,
        "nan rope x",
        m.rope_partial_forward(&nan_x4, &cos, &sin),
        "rope_partial_forward",
    );
    deferred(
        &m,
        "nan rope gy",
        m.rope_partial_backward(&nan_x4, &cos, &sin),
        "rope_partial_backward",
    );
}
