//! K4: the causal depthwise conv1d + SiLU of the GDN mixer (zero initial
//! state, kernel width `K`), forward and backward.
//!
//! - **Host** (run everywhere): the host mirror (`conv1d::*_mirror`) against
//!   L-cuda-oracle's float64 reference on the three torch goldens'
//!   f32-rounded inputs (including `T = 2 < K - 1` over three batch rows), at
//!   the 2B's 6,144 channels in a fused-projection window over two
//!   weight-gradient blocks, and at kernel width 2.
//! - **Device** (`--features cuda`, `#[ignore]`: an sm_90 GPU; on the Mac they
//!   compile and are NOT RUN): the kernels against the same reference, bit for
//!   bit against the mirror and a repeat run, and rung a's checks.
//!
//! Bounds, written before any run: `y` within `abs 1e-6 + rel 1e-5`
//! (`tessl/tests/qwen35_kernels.rs:1071`); `dx`, `dw` within `1e-4 *
//! max|ref|` (`tessl/tests/qwen35_bwd.rs:5,43-66`).

mod device_small_common;
mod reference;

use device_small_common::{
    f32s, golden, wide, Goldens, BWD_PEAK_REL, BWD_PEAK_SOURCE, NORM_FWD_SOURCE, NORM_FWD_TOL,
};
use ojas_cuda::check::Check;
use ojas_cuda::conv1d::{conv1d_silu_bwd_mirror, conv1d_silu_fwd_mirror, Conv1dPlan};
use ojas_cuda::inputs::splitmix_f32;
use ojas_cuda::small_common::{elementwise_check, peak_check, Window};
use reference::conv1d::{conv1d_silu_bwd_f64, conv1d_silu_fwd_f64, ConvShape};

const GOLDENS: Goldens = Goldens(&[
    golden!("conv1d_silu_c0_x"),
    golden!("conv1d_silu_c0_w"),
    golden!("conv1d_silu_c0_dy"),
    golden!("conv1d_silu_c1_x"),
    golden!("conv1d_silu_c1_w"),
    golden!("conv1d_silu_c1_dy"),
    golden!("conv1d_silu_c2_x"),
    golden!("conv1d_silu_c2_w"),
    golden!("conv1d_silu_c2_dy"),
]);

/// One conv case: `x` (and `dx`) in a window of a projection, f32 inputs,
/// the reference on them.
struct ConvCase {
    plan: Conv1dPlan,
    xw: Window,
    x: Vec<f32>,
    w: Vec<f32>,
    dy: Vec<f32>,
    dx0: Vec<f32>,
    want: [Vec<f64>; 3],
}

impl ConvCase {
    /// `x` and `dy` dense `[b, t, c]`, `w` `[c, k]`.
    fn new(
        s: ConvShape,
        (x, w, dy): (Vec<f32>, Vec<f32>, Vec<f32>),
        (ld, off): (usize, usize),
    ) -> Self {
        let xw = Window {
            ld: ld as u64,
            off: off as u64,
        };
        let plan =
            Conv1dPlan::new(s.b as u64, s.t as u64, s.c as u64, s.k as u32, xw).expect("plan");
        let rows = s.b * s.t;
        let mut xp = splitmix_f32(0x4c1, rows * ld, 3.0);
        for r in 0..rows {
            xp[r * ld + off..][..s.c].copy_from_slice(&x[r * s.c..][..s.c]);
        }
        let y = conv1d_silu_fwd_f64(s, &wide(&x), &wide(&w));
        let (dx, dw) = conv1d_silu_bwd_f64(s, &wide(&x), &wide(&w), &wide(&dy));
        ConvCase {
            plan,
            xw,
            x: xp,
            w,
            dy,
            dx0: splitmix_f32(0x4c2, rows * ld, 1.0),
            want: [y, dx, dw],
        }
    }

    fn golden(c: usize) -> Self {
        let name = format!("conv1d_silu_c{c}");
        let (xs, x) = GOLDENS.shaped(&format!("{name}_x"));
        let (ws, w) = GOLDENS.shaped(&format!("{name}_w"));
        let s = ConvShape {
            b: xs[0],
            t: xs[1],
            c: xs[2],
            k: ws[1],
        };
        let dy = f32s(&GOLDENS.f64s(&format!("{name}_dy")));
        ConvCase::new(s, (f32s(&x), f32s(&w), dy), (s.c + 9, 4))
    }

    fn random(s: ConvShape, (ld, off): (usize, usize)) -> Self {
        let n = s.b * s.t * s.c;
        ConvCase::new(
            s,
            (
                splitmix_f32(0x4d1, n, 2.0),
                splitmix_f32(0x4d2, s.c * s.k, 0.5),
                splitmix_f32(0x4d3, n, 1.0),
            ),
            (ld, off),
        )
    }

    /// Dense `[rows, c]` from `dx`'s window.
    fn dense_dx(&self, dx: &[f32]) -> Vec<f32> {
        let (rows, c) = (self.plan.rows() as usize, self.plan.channels as usize);
        (0..rows)
            .flat_map(|r| (0..c).map(move |j| dx[self.xw.at(r, j)]))
            .collect()
    }

    fn judge(&self, label: &str, (y, dx, dw): (&[f32], &[f32], &[f32])) -> Vec<Check> {
        let dx = self.dense_dx(dx);
        vec![
            elementwise_check(
                &format!("{label} y"),
                y,
                &self.want[0],
                NORM_FWD_TOL,
                NORM_FWD_SOURCE,
            ),
            peak_check(
                &format!("{label} dx"),
                &dx,
                &self.want[1],
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dw"),
                dw,
                &self.want[2],
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
        ]
    }

    fn mirror(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let y = conv1d_silu_fwd_mirror(&self.plan, &self.x, &self.w).expect("fwd mirror");
        let mut dx = self.dx0.clone();
        let dy_win = Window::dense(self.plan.channels);
        let dw = conv1d_silu_bwd_mirror(
            &self.plan,
            (&self.x, &self.w),
            (dy_win, &self.dy),
            (self.xw, &mut dx),
        )
        .expect("bwd mirror");
        (y, dx, dw)
    }
}

/// The 2B's conv: 6,144 channels (q + k + v) at `K = 4`, 300 tokens (two
/// 256-row weight-gradient blocks) in a projection window; and `K = 2` over
/// three short batch rows.
fn shaped_cases() -> [(&'static str, ConvCase); 2] {
    [
        (
            "b1_t300_c6144_k4",
            ConvCase::random(
                ConvShape {
                    b: 1,
                    t: 300,
                    c: 6144,
                    k: 4,
                },
                (6200, 40),
            ),
        ),
        (
            "b3_t37_c40_k2",
            ConvCase::random(
                ConvShape {
                    b: 3,
                    t: 37,
                    c: 40,
                    k: 2,
                },
                (40, 0),
            ),
        ),
    ]
}

#[test]
fn the_embedded_goldens_are_l_cuda_oracles_pinned_bytes() {
    assert_eq!(GOLDENS.verify(), 9);
}

#[test]
fn the_mirror_matches_the_reference_on_the_torch_goldens() {
    let mut checks = Vec::new();
    for c in 0..3 {
        let case = ConvCase::golden(c);
        let (y, dx, dw) = case.mirror();
        checks.extend(case.judge(&format!("mirror conv1d_silu_c{c}"), (&y, &dx, &dw)));
    }
    device_small_common::assert_pass(&checks);
}

#[test]
fn the_mirror_matches_the_reference_at_the_2b_channels_and_kernel_width_2() {
    let mut checks = Vec::new();
    for (tag, case) in shaped_cases() {
        let (y, dx, dw) = case.mirror();
        checks.extend(case.judge(&format!("mirror {tag}"), (&y, &dx, &dw)));
    }
    device_small_common::assert_pass(&checks);
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use device_small_common::{assert_bits, assert_pass, runtime};
    use ojas_cuda::conv1d_cuda::{conv1d_silu, conv1d_silu_bwd};
    use ojas_cuda::runtime::CudaRuntime;
    use ojas_cuda::small_smoke::k4_checks;

    fn run(rt: &CudaRuntime, c: &ConvCase) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let nan = |len: usize| {
            rt.upload(&vec![f32::NAN; len], "k4 sentinel")
                .expect("sentinel")
        };
        let x = rt.upload(&c.x, "x").expect("x");
        let w = rt.upload(&c.w, "w").expect("w");
        let mut y = nan(c.plan.y_len() as usize);
        conv1d_silu(rt, &c.plan, &x, &w, &mut y).expect("conv fwd");
        let dy = rt.upload(&c.dy, "dy").expect("dy");
        let mut dx = rt.upload(&c.dx0, "dx").expect("dx");
        let mut part = nan(c.plan.part_len() as usize);
        let mut dw = nan(c.w.len());
        let dy_win = Window::dense(c.plan.channels);
        conv1d_silu_bwd(
            rt,
            &c.plan,
            (&x, &w),
            (dy_win, &dy),
            (c.xw, &mut dx),
            &mut part,
            &mut dw,
        )
        .expect("conv bwd");
        (
            rt.download(&y).expect("y"),
            rt.download(&dx).expect("dx"),
            rt.download(&dw).expect("dw"),
        )
    }

    fn check(rt: &CudaRuntime, label: &str, c: &ConvCase) -> Vec<Check> {
        let got = run(rt, c);
        let again = run(rt, c);
        let want = c.mirror();
        for (name, g, a, w) in [
            ("y", &got.0, &again.0, &want.0),
            ("dx", &got.1, &again.1, &want.1),
            ("dw", &got.2, &again.2, &want.2),
        ] {
            assert_bits(&format!("{label} {name} vs mirror"), g, w);
            assert_bits(&format!("{label} {name} repeat"), a, g);
        }
        c.judge(label, (&got.0, &got.1, &got.2))
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_conv_matches_the_reference_and_the_mirror_bitwise() {
        let rt = runtime();
        let mut checks = Vec::new();
        for c in 0..3 {
            checks.extend(check(
                &rt,
                &format!("device conv1d_silu_c{c}"),
                &ConvCase::golden(c),
            ));
        }
        for (tag, case) in shaped_cases() {
            checks.extend(check(&rt, &format!("device {tag}"), &case));
        }
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn rung_a_k4_checks_pass() {
        assert_pass(&k4_checks(&runtime()));
    }
}
