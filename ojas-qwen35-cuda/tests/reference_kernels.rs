//! Self-validation of the K3, K4, K6, K7, K9 and K10 host references against
//! their float64 torch goldens (`tests/fixtures/goldens/`, pinned by
//! `tests/reference_goldens_published.rs`), against tessl's transformers RoPE
//! fixture, and against central differences of their own forwards.
//!
//! Bounds, written before the first run, relative to each output's largest
//! magnitude:
//!
//! - [`ELEMENTWISE`] `1e-13`: each output is a few correctly rounded operations
//!   and libm calls (`exp`, `ln_1p`), so the reference and torch differ by a
//!   few ulps per element (~1e-15); the per-head reductions over rows (`dA_log`,
//!   `ddt_bias`, 40 rows) add `n u = 4.4e-15` of the sum of magnitudes, which
//!   cancellation can lift by the ratio of that sum to the result (~6 for
//!   random signs). 1e-13 is ~4x that.
//! - [`REDUCTION`] `1e-12`: reductions over up to 2048 terms (the norms' row
//!   sums, CE's logit dot products and its 300-term log-sum-exp, the weight
//!   gradients over rows): `n u = 2048 * 1.1e-16 = 2.3e-13`, ~4x margin.
//!
//! A wrong formula (a missing `1 +` on a norm weight, a sign, a swapped RoPE
//! pair) is off by 1e-2 or more.

mod reference;

use std::path::Path;

use reference::attn_pieces::{
    norm_rope_bwd_f64, norm_rope_fwd_f64, output_gate_bwd_f64, output_gate_fwd_f64, rope_angle_f32,
    rope_cos_sin,
};
use reference::ce::{cross_entropy_rows_f64, Reduction};
use reference::conv1d::{conv1d_silu_bwd_f64, conv1d_silu_fwd_f64, ConvShape};
use reference::embed::{embed_rows_bwd_f64, embed_rows_fwd_f64};
use reference::gates_published::{gdn_gates_published_bwd_f64, gdn_gates_published_fwd_f64};
use reference::gdn_published::exact_f64;
use reference::goldens::{self, assert_rel};
use reference::norms::{
    gated_rms_norm_bwd_f64, gated_rms_norm_fwd_f64, rms_norm_bwd_f64, rms_norm_fwd_f64,
};
use reference::{fd, npy, rng};

const ELEMENTWISE: f64 = 1e-13;
const REDUCTION: f64 = 1e-12;
const EPS: f64 = 1e-6;

// ------------------------------------------------------------------- K3 ---

#[test]
fn k3_gates_published_match_the_torch_golden() {
    let get = |n: &str| goldens::f64s(&format!("gates_published_{n}"));
    let (shape, a) = goldens::f64s_shaped("gates_published_a");
    let (rows, heads) = (shape[0], shape[1]);
    let (b, a_log, dt_bias, dg, dbeta) = (
        get("b"),
        get("a_log"),
        get("dt_bias"),
        get("dg"),
        get("dbeta"),
    );
    assert!(
        (0..rows * heads).any(|o| a[o] + dt_bias[o % heads] > 20.0),
        "the golden must reach softplus' linear branch"
    );
    let (g, beta) = gdn_gates_published_fwd_f64(&a, &b, &a_log, &dt_bias, rows, heads);
    assert_rel("K3 published g", &g, &get("g"), ELEMENTWISE);
    assert_rel("K3 published beta", &beta, &get("beta"), ELEMENTWISE);
    let gr = gdn_gates_published_bwd_f64(&a, &b, &a_log, &dt_bias, &dg, &dbeta, rows, heads);
    assert_rel("K3 published da", &gr.da, &get("da"), ELEMENTWISE);
    assert_rel("K3 published db", &gr.db, &get("db"), ELEMENTWISE);
    assert_rel(
        "K3 published dA_log",
        &gr.da_log,
        &get("da_log"),
        ELEMENTWISE,
    );
    assert_rel(
        "K3 published ddt_bias",
        &gr.ddt_bias,
        &get("ddt_bias"),
        ELEMENTWISE,
    );
}

#[test]
fn k3_gates_published_backward_is_the_derivative_away_from_the_threshold() {
    let (rows, heads) = (7, 3);
    let a: Vec<f64> = rng::random_f64(rows * heads, 1)
        .iter()
        .map(|x| 6.0 * x)
        .collect();
    let b: Vec<f64> = rng::random_f64(rows * heads, 2)
        .iter()
        .map(|x| 4.0 * x)
        .collect();
    let a_log = rng::random_f64(heads, 3);
    let dt: Vec<f64> = rng::random_f64(heads, 4).iter().map(|x| x - 2.0).collect();
    let (wg, wb) = (
        rng::random_f64(rows * heads, 5),
        rng::random_f64(rows * heads, 6),
    );
    let gr = gdn_gates_published_bwd_f64(&a, &b, &a_log, &dt, &wg, &wb, rows, heads);
    let loss = |a: &[f64], b: &[f64], al: &[f64], dt: &[f64]| {
        let (g, beta) = gdn_gates_published_fwd_f64(a, b, al, dt, rows, heads);
        fd::dot(&g, &wg) + fd::dot(&beta, &wb)
    };
    fd::check("K3 published da", &a, &gr.da, &|x| loss(x, &b, &a_log, &dt));
    fd::check("K3 published db", &b, &gr.db, &|x| loss(&a, x, &a_log, &dt));
    fd::check("K3 published dA_log", &a_log, &gr.da_log, &|x| {
        loss(&a, &b, x, &dt)
    });
    fd::check("K3 published ddt_bias", &dt, &gr.ddt_bias, &|x| {
        loss(&a, &b, &a_log, x)
    });
}

// ------------------------------------------------------------------- K4 ---

#[test]
fn k4_conv1d_silu_matches_the_torch_golden() {
    for c in 0..3 {
        let name = format!("conv1d_silu_c{c}");
        let (xs, x) = goldens::f64s_shaped(&format!("{name}_x"));
        let (ws, w) = goldens::f64s_shaped(&format!("{name}_w"));
        let s = ConvShape {
            b: xs[0],
            t: xs[1],
            c: xs[2],
            k: ws[1],
        };
        let get = |n: &str| goldens::f64s(&format!("{name}_{n}"));
        let label = |n: &str| format!("K4 {name} {s:?} {n}");
        assert_rel(
            &label("y"),
            &conv1d_silu_fwd_f64(s, &x, &w),
            &get("y"),
            REDUCTION,
        );
        let (dx, dw) = conv1d_silu_bwd_f64(s, &x, &w, &get("dy"));
        assert_rel(&label("dx"), &dx, &get("dx"), REDUCTION);
        assert_rel(&label("dw"), &dw, &get("dw"), REDUCTION);
    }
}

#[test]
fn k4_conv1d_silu_backward_is_the_derivative() {
    // T = 2 < K - 1 with B = 2: a batch row's gradient must not reach its neighbour.
    for (b, t, c, k) in [(2, 2, 2, 4), (2, 5, 3, 4), (1, 4, 2, 2), (2, 3, 1, 8)] {
        let s = ConvShape { b, t, c, k };
        let x = rng::random_f64(b * t * c, 41);
        let w = rng::random_f64(c * k, 42);
        let dy = rng::random_f64(b * t * c, 43);
        let (dx, dw) = conv1d_silu_bwd_f64(s, &x, &w, &dy);
        fd::check(&format!("K4 {s:?} dx"), &x, &dx, &|v| {
            fd::dot(&conv1d_silu_fwd_f64(s, v, &w), &dy)
        });
        fd::check(&format!("K4 {s:?} dw"), &w, &dw, &|v| {
            fd::dot(&conv1d_silu_fwd_f64(s, &x, v), &dy)
        });
    }
}

// ------------------------------------------------------------------- K6 ---

/// The six `qwen35_rope_*` files are tessl's bytes (copied 2026-10-01, `cmp`
/// identical to `tessl/tests/fixtures/qwen35/`), pinned by `qwen35_rope_SHA256SUMS`,
/// and nothing unpinned rides along.
#[test]
fn k6_transformers_rope_fixture_is_pinned() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35");
    let sums = std::fs::read_to_string(dir.join("qwen35_rope_SHA256SUMS"))
        .expect("qwen35_rope_SHA256SUMS");
    let mut listed = 0usize;
    for line in sums.lines().filter(|l| !l.trim().is_empty()) {
        let (want, file) = line
            .split_once("  ")
            .unwrap_or_else(|| panic!("malformed SHA256SUMS line {line:?}"));
        let got = reference::sha256::file_hex(&dir.join(file)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            got, want,
            "{file}: sha256 differs from the pinned copy of tessl's fixture"
        );
        listed += 1;
    }
    let on_disk = std::fs::read_dir(&dir).expect("fixture dir").count();
    assert_eq!(
        listed, 6,
        "expected the six qwen35_rope_* files to be pinned"
    );
    assert_eq!(
        on_disk,
        listed + 1,
        "tests/fixtures/qwen35 holds files the SHA256SUMS does not pin"
    );
}

/// tessl's transformers fixture: `Qwen3_5Attention`'s q/k norm + partial RoPE
/// at positions 20000.. (`tessl/tests/qwen35_kernels.rs:192-212`, its bound).
#[test]
fn k6_norm_rope_matches_tessls_transformers_fixture_at_position_20000() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35");
    let load = |n: &str| {
        npy::read(&dir.join(format!("qwen35_rope_{n}.npy"))).unwrap_or_else(|e| panic!("{e}"))
    };
    let q_in = load("q_in");
    let (t, hq, d) = (q_in.shape[1], q_in.shape[2], q_in.shape[3]);
    let q = q_in.to_f64().expect("q_in");
    let k = load("k_in").to_f64().expect("k_in");
    let (qw, kw) = (
        load("q_norm_w").to_f64().expect("qw"),
        load("k_norm_w").to_f64().expect("kw"),
    );
    let (qo, ko) = (
        load("q_out").to_f64().expect("q_out"),
        load("k_out").to_f64().expect("k_out"),
    );
    let mut worst = 0.0f64;
    for ti in 0..t {
        let cs = rope_cos_sin(64, 20_000 + ti, 1e7);
        for h in 0..hq {
            let row = (ti * hq + h) * d;
            let got = norm_rope_fwd_f64(&q[row..row + d], &qw, &cs, EPS);
            for i in 0..d {
                let e = (got[i] - qo[row + i]).abs();
                assert!(e < 5e-6, "q t{ti} h{h} [{i}]: {e:.3e}");
                worst = worst.max(e);
            }
        }
        let got = norm_rope_fwd_f64(&k[ti * d..(ti + 1) * d], &kw, &cs, EPS);
        for i in 0..d {
            let e = (got[i] - ko[ti * d + i]).abs();
            assert!(e < 5e-6, "k t{ti} [{i}]: {e:.3e}");
            worst = worst.max(e);
        }
    }
    eprintln!(
        "K6 vs transformers fixture (T={t}, Hq={hq}, D={d}): worst abs {worst:.3e} (bound 5e-6)"
    );
}

#[test]
fn k6_rope_angle_is_transformers_f32_angle() {
    let pos = goldens::indices("qk_norm_rope_positions");
    let (shape, want) = goldens::f64s_shaped("qk_norm_rope_angle_f32");
    let half = shape[1];
    let (mut exact, mut worst_ulps) = (0usize, 0.0f64);
    for (r, &p) in pos.iter().enumerate() {
        for j in 0..half {
            let got = f64::from(rope_angle_f32(j, 2 * half, p, 1e7));
            let w = want[r * half + j];
            if got.to_bits() == w.to_bits() {
                exact += 1;
            } else {
                // f32 ulps at |w|: inv_freq and the product are each one f32 rounding,
                // and libm pow differs between implementations by up to an ulp.
                let ulp = f64::from(f32::EPSILON) * w.abs();
                worst_ulps = worst_ulps.max((got - w).abs() / ulp);
            }
        }
    }
    eprintln!(
        "K6 angle: {exact}/{} bit-identical to transformers' f32 angle, worst {worst_ulps:.2} ulp",
        pos.len() * half
    );
    assert!(
        worst_ulps <= 4.0,
        "the f32 angle differs from transformers' by {worst_ulps:.2} ulp"
    );
}

#[test]
fn k6_norm_rope_matches_the_torch_golden_given_the_angle() {
    let (shape, x) = goldens::f64s_shaped("qk_norm_rope_x");
    let (rows, d) = (shape[0], shape[1]);
    let get = |n: &str| goldens::f64s(&format!("qk_norm_rope_{n}"));
    let (w, dy, angle) = (get("w"), get("dy"), get("angle_f32"));
    let half = angle.len() / rows;
    let (mut y, mut dx, mut dw) = (Vec::new(), Vec::new(), vec![0.0; d]);
    for r in 0..rows {
        let cs: Vec<(f64, f64)> = angle[r * half..(r + 1) * half]
            .iter()
            .map(|a| (a.cos(), a.sin()))
            .collect();
        let xr = &x[r * d..(r + 1) * d];
        y.extend(norm_rope_fwd_f64(xr, &w, &cs, EPS));
        let (dxr, dwr) = norm_rope_bwd_f64(xr, &w, &dy[r * d..(r + 1) * d], &cs, EPS);
        dx.extend(dxr);
        dw.iter_mut().zip(dwr).for_each(|(a, v)| *a += v);
    }
    assert_rel("K6 norm_rope y", &y, &get("y"), REDUCTION);
    assert_rel("K6 norm_rope dx", &dx, &get("dx"), REDUCTION);
    assert_rel("K6 norm_rope dw", &dw, &get("dw"), REDUCTION);
}

#[test]
fn k6_norm_rope_backward_is_the_derivative() {
    for (d, rot, pos) in [
        (8, 8, 3),
        (10, 4, 7),
        (6, 0, 1),
        (12, 6, 40),
        (256, 64, 1234),
    ] {
        let x = rng::random_f64(d, 51);
        let w = rng::random_f64(d, 52);
        let g = rng::random_f64(d, 53);
        let cs = rope_cos_sin(rot, pos, 1e4);
        let (dx, dw) = norm_rope_bwd_f64(&x, &w, &g, &cs, EPS);
        let label = format!("K6 d={d} rot={rot} pos={pos}");
        fd::check(&format!("{label} dx"), &x, &dx, &|v| {
            fd::dot(&norm_rope_fwd_f64(v, &w, &cs, EPS), &g)
        });
        fd::check(&format!("{label} dw"), &w, &dw, &|v| {
            fd::dot(&norm_rope_fwd_f64(&x, v, &cs, EPS), &g)
        });
    }
}

#[test]
fn k6_output_gate_matches_the_torch_golden_and_its_derivative() {
    let get = |n: &str| goldens::f64s(&format!("attn_output_gate_{n}"));
    let (o, gate, dy) = (get("o"), get("gate"), get("dy"));
    assert_rel(
        "K6 gate y",
        &output_gate_fwd_f64(&o, &gate),
        &get("y"),
        ELEMENTWISE,
    );
    let (d_o, dgate) = output_gate_bwd_f64(&o, &gate, &dy);
    assert_rel("K6 gate do", &d_o, &get("do"), ELEMENTWISE);
    assert_rel("K6 gate dgate", &dgate, &get("dgate"), ELEMENTWISE);
    let (o2, g2, w2) = (
        rng::random_f64(40, 61),
        rng::random_f64(40, 62),
        rng::random_f64(40, 63),
    );
    let (d2, dg2) = output_gate_bwd_f64(&o2, &g2, &w2);
    fd::check("K6 gate do", &o2, &d2, &|v| {
        fd::dot(&output_gate_fwd_f64(v, &g2), &w2)
    });
    fd::check("K6 gate dgate", &g2, &dg2, &|v| {
        fd::dot(&output_gate_fwd_f64(&o2, v), &w2)
    });
}

// ------------------------------------------------------------------- K7 ---

#[test]
fn k7_rms_norms_match_the_torch_golden() {
    for c in 0..2 {
        let name = format!("rms_norm_c{c}");
        let (xs, x) = goldens::f64s_shaped(&format!("{name}_x"));
        let d = xs[1];
        let get = |n: &str| goldens::f64s(&format!("{name}_{n}"));
        let w = get("w");
        assert_rel(
            &format!("K7 {name} y"),
            &rms_norm_fwd_f64(&x, &w, d, EPS),
            &get("y"),
            REDUCTION,
        );
        let (dx, dw) = rms_norm_bwd_f64(&x, &w, &get("dy"), d, EPS);
        assert_rel(&format!("K7 {name} dx"), &dx, &get("dx"), REDUCTION);
        assert_rel(&format!("K7 {name} dw"), &dw, &get("dw"), REDUCTION);
    }
    for c in 0..2 {
        let name = format!("gated_rms_norm_c{c}");
        let (xs, x) = goldens::f64s_shaped(&format!("{name}_x"));
        let d = xs[1];
        let get = |n: &str| goldens::f64s(&format!("{name}_{n}"));
        let (z, w) = (get("z"), get("w"));
        assert_rel(
            &format!("K7 {name} y"),
            &gated_rms_norm_fwd_f64(&x, &z, &w, d, EPS),
            &get("y"),
            REDUCTION,
        );
        let (dx, dz, dw) = gated_rms_norm_bwd_f64(&x, &z, &w, &get("dy"), d, EPS);
        assert_rel(&format!("K7 {name} dx"), &dx, &get("dx"), REDUCTION);
        assert_rel(&format!("K7 {name} dz"), &dz, &get("dz"), REDUCTION);
        assert_rel(&format!("K7 {name} dw"), &dw, &get("dw"), REDUCTION);
    }
}

#[test]
fn k7_rms_norm_backwards_are_the_derivative() {
    let (rows, d) = (3, 7);
    let x = rng::random_f64(rows * d, 1);
    let w = rng::random_f64(d, 2);
    let dy = rng::random_f64(rows * d, 3);
    let (dx, dw) = rms_norm_bwd_f64(&x, &w, &dy, d, EPS);
    fd::check("K7 rms dx", &x, &dx, &|v| {
        fd::dot(&rms_norm_fwd_f64(v, &w, d, EPS), &dy)
    });
    fd::check("K7 rms dw", &w, &dw, &|v| {
        fd::dot(&rms_norm_fwd_f64(&x, v, d, EPS), &dy)
    });
    let (units, d) = (3, 6);
    let x = rng::random_f64(units * d, 11);
    let z: Vec<f64> = rng::random_f64(units * d, 12)
        .iter()
        .map(|v| 3.0 * v)
        .collect();
    let w = rng::random_f64(d, 13);
    let dy = rng::random_f64(units * d, 14);
    let (dx, dz, dw) = gated_rms_norm_bwd_f64(&x, &z, &w, &dy, d, EPS);
    let f =
        |x: &[f64], z: &[f64], w: &[f64]| fd::dot(&gated_rms_norm_fwd_f64(x, z, w, d, EPS), &dy);
    fd::check("K7 gated dx", &x, &dx, &|v| f(v, &z, &w));
    fd::check("K7 gated dz", &z, &dz, &|v| f(&x, v, &w));
    fd::check("K7 gated dw", &w, &dw, &|v| f(&x, &z, v));
}

// ------------------------------------------------------------------- K9 ---

#[test]
fn k9_embedding_matches_the_torch_golden() {
    let ids = goldens::indices("embed_ids");
    let (ts, table) = goldens::f64s_shaped("embed_table");
    let (vocab, hidden) = (ts[0], ts[1]);
    assert!(
        ids.windows(2).any(|w| w[0] > w[1]),
        "the golden's ids must be out of order"
    );
    assert!(
        (0..ids.len()).any(|i| ids[i + 1..].contains(&ids[i])),
        "the golden's ids must repeat"
    );
    let y = embed_rows_fwd_f64(&table, &ids, vocab, hidden);
    assert_rel("K9 y", &y, &goldens::f64s("embed_y"), 0.0);
    let dt = embed_rows_bwd_f64(&ids, &goldens::f64s("embed_dy"), vocab, hidden, None);
    assert_rel(
        "K9 dtable",
        &dt,
        &goldens::f64s("embed_dtable"),
        ELEMENTWISE,
    );
}

#[test]
#[should_panic(expected = "is not below vocab")]
fn k9_embedding_refuses_an_out_of_range_id() {
    embed_rows_fwd_f64(&[0.0; 8], &[0, 4], 4, 2);
}

// ------------------------------------------------------------------ K10 ---

#[test]
fn k10_cross_entropy_rows_match_the_torch_golden() {
    for (c, reduction) in [
        (0, Reduction::Mean),
        (1, Reduction::Sum),
        (2, Reduction::Mean),
    ] {
        let name = format!("ce_rows_c{c}");
        let get = |n: &str| goldens::f64s(&format!("{name}_{n}"));
        let (hs, h) = goldens::f64s_shaped(&format!("{name}_h"));
        let (ws, w) = goldens::f64s_shaped(&format!("{name}_w"));
        let (hidden, vocab) = (hs[1], ws[0]);
        let rows = goldens::indices(&format!("{name}_rows"));
        let targets = goldens::indices(&format!("{name}_targets"));
        let scale = get("scale")[0];
        let per_row_golden = get("per_row");
        let total: f64 = per_row_golden.iter().sum();
        let implied = if reduction == Reduction::Mean {
            total / exact_f64(per_row_golden.len())
        } else {
            total
        };
        assert!(
            (implied - get("loss")[0]).abs() <= 1e-12 * implied.abs(),
            "{name}: reduction label disagrees with the golden"
        );
        let out = cross_entropy_rows_f64(&h, &w, &rows, &targets, hidden, vocab, reduction, scale);
        let label = |n: &str| format!("K10 {name} {reduction:?} scale={scale} {n}");
        assert_rel(&label("per_row"), &out.per_row, &per_row_golden, REDUCTION);
        assert_rel(&label("loss"), &[out.loss], &get("loss"), REDUCTION);
        assert_rel(&label("dh"), &out.dh, &get("dh"), REDUCTION);
        assert_rel(&label("dW"), &out.dw, &get("dw"), REDUCTION);
    }
}

#[test]
fn k10_cross_entropy_rows_gradients_are_the_derivative() {
    let (t_total, hidden, vocab) = (5, 4, 9);
    let h = rng::random_f64(t_total * hidden, 71);
    let w = rng::random_f64(vocab * hidden, 72);
    let (rows, targets) = ([3usize, 0, 3], [8usize, 0, 4]);
    for (reduction, scale) in [(Reduction::Mean, 1.0), (Reduction::Sum, 0.7)] {
        let out = cross_entropy_rows_f64(&h, &w, &rows, &targets, hidden, vocab, reduction, scale);
        // dh is per supplied row: perturb the gathered rows, not h.
        let gathered: Vec<f64> = rows
            .iter()
            .flat_map(|&r| h[r * hidden..(r + 1) * hidden].to_vec())
            .collect();
        let dense: Vec<usize> = (0..rows.len()).collect();
        let loss_g = |g: &[f64]| {
            scale
                * cross_entropy_rows_f64(g, &w, &dense, &targets, hidden, vocab, reduction, 1.0)
                    .loss
        };
        let loss_w = |v: &[f64]| {
            scale
                * cross_entropy_rows_f64(&h, v, &rows, &targets, hidden, vocab, reduction, 1.0).loss
        };
        fd::check(
            &format!("K10 {reduction:?} dh"),
            &gathered,
            &out.dh,
            &loss_g,
        );
        fd::check(&format!("K10 {reduction:?} dW"), &w, &out.dw, &loss_w);
    }
}
