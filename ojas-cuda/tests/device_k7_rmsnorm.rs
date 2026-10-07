//! K7: `Qwen3_5RMSNorm` (`x * rstd * (1 + w)`) and `Qwen3_5RMSNormGated`
//! (`w * (x * rstd) * silu(z)`), forward and backward.
//!
//! - **Host** (run everywhere): the host mirror (`rmsnorm::*_mirror`, the
//!   device's operation sequence) against L-cuda-oracle's float64 reference on
//!   the torch goldens' f32-rounded inputs, and at the 2B's shapes in fused
//!   windows.
//! - **Device** (`--features cuda`, `#[ignore]`: an sm_90 GPU; on the Mac they
//!   compile and are NOT RUN): the kernels against the same reference, bit for
//!   bit against the mirror and against a repeat run, and rung a's checks.
//!
//! Bounds, written before any run (`tests/device_small_common`): forwards
//! `abs 1e-6 + rel 1e-5` (`tessl/tests/qwen35_kernels.rs:1118,1174`),
//! backwards `1e-4 * max|ref|` (`tessl/tests/qwen35_bwd.rs:5,43-66`).

mod device_small_common;
mod reference;

use device_small_common::{
    eps64, f32s, golden, wide, Goldens, BWD_PEAK_REL, BWD_PEAK_SOURCE, EPS, NORM_FWD_SOURCE,
    NORM_FWD_TOL,
};
use ojas_cuda::check::Check;
use ojas_cuda::inputs::splitmix_f32;
use ojas_cuda::rmsnorm::{
    gated_rms_norm_bwd_mirror, gated_rms_norm_fwd_mirror, rms_norm_bwd_mirror, rms_norm_fwd_mirror,
    GatedGradWindows, GatedRmsNormPlan, RmsNormPlan,
};
use ojas_cuda::small_common::{elementwise_check, peak_check, Window};
use reference::norms::{
    gated_rms_norm_bwd_f64, gated_rms_norm_fwd_f64, rms_norm_bwd_f64, rms_norm_fwd_f64,
};

const GOLDENS: Goldens = Goldens(&[
    golden!("rms_norm_c0_x"),
    golden!("rms_norm_c0_w"),
    golden!("rms_norm_c0_dy"),
    golden!("rms_norm_c1_x"),
    golden!("rms_norm_c1_w"),
    golden!("rms_norm_c1_dy"),
    golden!("gated_rms_norm_c0_x"),
    golden!("gated_rms_norm_c0_z"),
    golden!("gated_rms_norm_c0_w"),
    golden!("gated_rms_norm_c0_dy"),
    golden!("gated_rms_norm_c1_x"),
    golden!("gated_rms_norm_c1_z"),
    golden!("gated_rms_norm_c1_w"),
    golden!("gated_rms_norm_c1_dy"),
]);

/// One `rms_norm` case: f32 inputs and the reference on them.
struct RmsCase {
    plan: RmsNormPlan,
    x: Vec<f32>,
    w: Vec<f32>,
    dy: Vec<f32>,
    y_ref: Vec<f64>,
    dx_ref: Vec<f64>,
    dw_ref: Vec<f64>,
}

impl RmsCase {
    fn new(rows: usize, d: usize, (x, w, dy): (Vec<f32>, Vec<f32>, Vec<f32>)) -> Self {
        let plan = RmsNormPlan::new(rows as u64, d as u64, EPS).expect("plan");
        let y_ref = rms_norm_fwd_f64(&wide(&x), &wide(&w), d, eps64());
        let (dx_ref, dw_ref) = rms_norm_bwd_f64(&wide(&x), &wide(&w), &wide(&dy), d, eps64());
        RmsCase {
            plan,
            x,
            w,
            dy,
            y_ref,
            dx_ref,
            dw_ref,
        }
    }

    fn golden(c: usize) -> Self {
        let name = format!("rms_norm_c{c}");
        let (xs, x) = GOLDENS.shaped(&format!("{name}_x"));
        let get = |n: &str| f32s(&GOLDENS.f64s(&format!("{name}_{n}")));
        RmsCase::new(xs[0], xs[1], (f32s(&x), get("w"), get("dy")))
    }

    fn judge(&self, label: &str, (y, dx, dw): (&[f32], &[f32], &[f32])) -> Vec<Check> {
        vec![
            elementwise_check(
                &format!("{label} y"),
                y,
                &self.y_ref,
                NORM_FWD_TOL,
                NORM_FWD_SOURCE,
            ),
            peak_check(
                &format!("{label} dx"),
                dx,
                &self.dx_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dw"),
                dw,
                &self.dw_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
        ]
    }

    fn mirror(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let y = rms_norm_fwd_mirror(&self.plan, &self.x, &self.w).expect("fwd mirror");
        let (dx, dw) =
            rms_norm_bwd_mirror(&self.plan, &self.x, &self.w, &self.dy, None).expect("bwd mirror");
        (y, dx, dw)
    }
}

/// One gated case over `rows` rows of `heads` heads, `z` and `dz` in a
/// fused-projection window, `out` in a padded buffer.
struct GatedCase {
    plan: GatedRmsNormPlan,
    x: Vec<f32>,
    z: Vec<f32>,
    w: Vec<f32>,
    dy: Vec<f32>,
    out_win: Window,
    win: GatedGradWindows,
    y_ref: Vec<f64>,
    dx_ref: Vec<f64>,
    dz_ref: Vec<f64>,
    dw_ref: Vec<f64>,
}

impl GatedCase {
    /// `x`, `z`, `dy` dense `[rows * heads, d]` (the reference's layout).
    fn new(
        (rows, heads, d): (usize, usize, usize),
        (x, z_dense, w, dy): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
        (z_ld, z_off): (usize, usize),
    ) -> Self {
        let width = heads * d;
        let zw = Window {
            ld: z_ld as u64,
            off: z_off as u64,
        };
        let dense = Window::dense(width as u64);
        let plan = GatedRmsNormPlan::new((rows as u64, heads as u64, d as u64), EPS, dense, zw)
            .expect("plan");
        let mut z = splitmix_f32(0x7c1, rows * z_ld, 9.0);
        for r in 0..rows {
            z[r * z_ld + z_off..][..width].copy_from_slice(&z_dense[r * width..][..width]);
        }
        let y_ref = gated_rms_norm_fwd_f64(&wide(&x), &wide(&z_dense), &wide(&w), d, eps64());
        let (dx_ref, dz_ref, dw_ref) = gated_rms_norm_bwd_f64(
            &wide(&x),
            &wide(&z_dense),
            &wide(&w),
            &wide(&dy),
            d,
            eps64(),
        );
        GatedCase {
            plan,
            x,
            z,
            w,
            dy,
            out_win: Window {
                ld: width as u64 + 6,
                off: 3,
            },
            win: GatedGradWindows {
                dy: dense,
                dx: dense,
                dz: zw,
            },
            y_ref,
            dx_ref,
            dz_ref,
            dw_ref,
        }
    }

    fn golden(c: usize, (rows, heads): (usize, usize)) -> Self {
        let name = format!("gated_rms_norm_c{c}");
        let (xs, x) = GOLDENS.shaped(&format!("{name}_x"));
        let (units, d) = (xs[0], xs[1]);
        assert_eq!(rows * heads, units, "{name}: {units} units");
        let get = |n: &str| f32s(&GOLDENS.f64s(&format!("{name}_{n}")));
        let width = heads * d;
        GatedCase::new(
            (rows, heads, d),
            (f32s(&x), get("z"), get("w"), get("dy")),
            (2 * width + 7, width + 5),
        )
    }

    fn rows(&self) -> usize {
        self.plan.rows as usize
    }

    fn width(&self) -> usize {
        self.plan.width() as usize
    }

    /// Dense `[rows, width]` from a window of a buffer.
    fn dense_of(&self, win: Window, buf: &[f32]) -> Vec<f32> {
        (0..self.rows())
            .flat_map(|r| (0..self.width()).map(move |c| buf[win.at(r, c)]))
            .collect()
    }

    fn judge(
        &self,
        label: &str,
        (out, dx, dz, dw): (&[f32], &[f32], &[f32], &[f32]),
    ) -> Vec<Check> {
        let y = self.dense_of(self.out_win, out);
        let dz = self.dense_of(self.win.dz, dz);
        vec![
            elementwise_check(
                &format!("{label} y"),
                &y,
                &self.y_ref,
                NORM_FWD_TOL,
                NORM_FWD_SOURCE,
            ),
            peak_check(
                &format!("{label} dx"),
                dx,
                &self.dx_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dz"),
                &dz,
                &self.dz_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dw"),
                dw,
                &self.dw_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
        ]
    }

    /// The buffers the outputs start from (out and dz are written in part).
    fn starts(&self) -> (Vec<f32>, Vec<f32>) {
        let r = self.rows();
        (
            splitmix_f32(0x7c2, r * self.out_win.ld as usize, 1.0),
            splitmix_f32(0x7c3, r * self.win.dz.ld as usize, 1.0),
        )
    }

    fn mirror(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut out, mut dz) = self.starts();
        gated_rms_norm_fwd_mirror(
            &self.plan,
            (&self.x, &self.z, &self.w),
            self.out_win,
            &mut out,
        )
        .expect("fwd mirror");
        let mut dx = vec![f32::NAN; self.x.len()];
        let dw = gated_rms_norm_bwd_mirror(
            &self.plan,
            (&self.x, &self.z, &self.w, &self.dy),
            self.win,
            &mut dx,
            &mut dz,
        )
        .expect("bwd mirror");
        (out, dx, dz, dw)
    }
}

fn rms_2b() -> RmsCase {
    let (rows, d) = (70, 2048);
    RmsCase::new(
        rows,
        d,
        (
            splitmix_f32(0x7d1, rows * d, 3.0),
            splitmix_f32(0x7d2, d, 0.5),
            splitmix_f32(0x7d3, rows * d, 1.0),
        ),
    )
}

fn gated_2b() -> GatedCase {
    let (rows, heads, d) = (9, 16, 128);
    let n = rows * heads * d;
    GatedCase::new(
        (rows, heads, d),
        (
            splitmix_f32(0x7d4, n, 2.0),
            splitmix_f32(0x7d5, n, 4.0),
            splitmix_f32(0x7d6, d, 1.0),
            splitmix_f32(0x7d7, n, 1.0),
        ),
        (2 * heads * d + 104, 1000),
    )
}

#[test]
fn the_embedded_goldens_are_l_cuda_oracles_pinned_bytes() {
    assert_eq!(GOLDENS.verify(), 14);
}

#[test]
fn the_mirrors_match_the_reference_on_the_torch_goldens() {
    let mut checks = Vec::new();
    for c in 0..2 {
        let case = RmsCase::golden(c);
        let (y, dx, dw) = case.mirror();
        checks.extend(case.judge(&format!("mirror rms_norm_c{c}"), (&y, &dx, &dw)));
    }
    for (c, shape) in [(0, (3, 4)), (1, (2, 2))] {
        let case = GatedCase::golden(c, shape);
        let (out, dx, dz, dw) = case.mirror();
        checks.extend(case.judge(
            &format!("mirror gated_rms_norm_c{c}"),
            (&out, &dx, &dz, &dw),
        ));
    }
    device_small_common::assert_pass(&checks);
}

#[test]
fn the_mirrors_match_the_reference_at_the_2b_shapes_in_fused_windows() {
    let rms = rms_2b();
    let (y, dx, dw) = rms.mirror();
    let mut checks = rms.judge("mirror rms_norm 70x2048", (&y, &dx, &dw));
    let gated = gated_2b();
    let (out, dx, dz, dw) = gated.mirror();
    checks.extend(gated.judge("mirror gated 9x16x128", (&out, &dx, &dz, &dw)));
    device_small_common::assert_pass(&checks);
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use device_small_common::{assert_bits, assert_pass, runtime};
    use ojas_cuda::rmsnorm_cuda::{gated_rms_norm, gated_rms_norm_bwd, rms_norm, rms_norm_bwd};
    use ojas_cuda::runtime::CudaRuntime;
    use ojas_cuda::small_smoke::k7_checks;

    fn nan(rt: &CudaRuntime, n: usize) -> ojas_cuda::buffer::CudaBuffer<f32> {
        rt.upload(&vec![f32::NAN; n], "k7 sentinel")
            .expect("sentinel")
    }

    fn run_rms(rt: &CudaRuntime, c: &RmsCase) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let p = &c.plan;
        let n = c.x.len();
        let x = rt.upload(&c.x, "x").expect("x");
        let w = rt.upload(&c.w, "w").expect("w");
        let dy = rt.upload(&c.dy, "dy").expect("dy");
        let mut y = nan(rt, n);
        rms_norm(rt, p, &x, &w, &mut y).expect("rms_norm");
        let mut dx = nan(rt, n);
        let mut part = nan(rt, p.part_len() as usize);
        let mut dw = nan(rt, c.w.len());
        rms_norm_bwd(rt, p, (&x, &w, &dy), &mut dx, false, &mut part, &mut dw).expect("bwd");
        (
            rt.download(&y).expect("y"),
            rt.download(&dx).expect("dx"),
            rt.download(&dw).expect("dw"),
        )
    }

    fn run_gated(rt: &CudaRuntime, c: &GatedCase) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let p = &c.plan;
        let (out0, dz0) = c.starts();
        let x = rt.upload(&c.x, "x").expect("x");
        let z = rt.upload(&c.z, "z").expect("z");
        let w = rt.upload(&c.w, "w").expect("w");
        let dy = rt.upload(&c.dy, "dy").expect("dy");
        let mut out = rt.upload(&out0, "out").expect("out");
        gated_rms_norm(rt, p, (&x, &z, &w), c.out_win, &mut out).expect("gated fwd");
        let mut dx = nan(rt, c.x.len());
        let mut dz = rt.upload(&dz0, "dz").expect("dz");
        let mut part = nan(rt, p.part_len() as usize);
        let mut dw = nan(rt, c.w.len());
        gated_rms_norm_bwd(
            rt,
            p,
            (&x, &z, &w, &dy),
            c.win,
            (&mut dx, &mut dz),
            &mut part,
            &mut dw,
        )
        .expect("gated bwd");
        (
            rt.download(&out).expect("out"),
            rt.download(&dx).expect("dx"),
            rt.download(&dz).expect("dz"),
            rt.download(&dw).expect("dw"),
        )
    }

    fn rms_case(rt: &CudaRuntime, label: &str, c: &RmsCase) -> Vec<Check> {
        let got = run_rms(rt, c);
        let again = run_rms(rt, c);
        let want = c.mirror();
        assert_bits(&format!("{label} y vs mirror"), &got.0, &want.0);
        assert_bits(&format!("{label} dx vs mirror"), &got.1, &want.1);
        assert_bits(&format!("{label} dw vs mirror"), &got.2, &want.2);
        assert_bits(&format!("{label} y repeat"), &again.0, &got.0);
        assert_bits(&format!("{label} dx repeat"), &again.1, &got.1);
        assert_bits(&format!("{label} dw repeat"), &again.2, &got.2);
        c.judge(label, (&got.0, &got.1, &got.2))
    }

    fn gated_case(rt: &CudaRuntime, label: &str, c: &GatedCase) -> Vec<Check> {
        let got = run_gated(rt, c);
        let again = run_gated(rt, c);
        let want = c.mirror();
        for (name, g, a, w) in [
            ("out", &got.0, &again.0, &want.0),
            ("dx", &got.1, &again.1, &want.1),
            ("dz", &got.2, &again.2, &want.2),
            ("dw", &got.3, &again.3, &want.3),
        ] {
            assert_bits(&format!("{label} {name} vs mirror"), g, w);
            assert_bits(&format!("{label} {name} repeat"), a, g);
        }
        c.judge(label, (&got.0, &got.1, &got.2, &got.3))
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_matches_the_reference_and_the_mirror_bitwise_on_the_goldens() {
        let rt = runtime();
        let mut checks = Vec::new();
        for c in 0..2 {
            checks.extend(rms_case(
                &rt,
                &format!("device rms_norm_c{c}"),
                &RmsCase::golden(c),
            ));
        }
        for (c, shape) in [(0, (3, 4)), (1, (2, 2))] {
            let case = GatedCase::golden(c, shape);
            checks.extend(gated_case(
                &rt,
                &format!("device gated_rms_norm_c{c}"),
                &case,
            ));
        }
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_matches_the_reference_and_the_mirror_bitwise_at_the_2b_shapes() {
        let rt = runtime();
        let mut checks = rms_case(&rt, "device rms_norm 70x2048", &rms_2b());
        checks.extend(gated_case(&rt, "device gated 9x16x128", &gated_2b()));
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn rung_a_k7_checks_pass() {
        assert_pass(&k7_checks(&runtime()));
    }
}
