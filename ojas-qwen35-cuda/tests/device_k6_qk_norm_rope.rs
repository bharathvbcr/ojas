//! K6: the attention pieces around the core, read from Qwen3.5's fused
//! projection (each query head `2D` wide, its gate after it): the q/k
//! `(1 + w)` norm with partial RoPE (`v` copied), and the output gate
//! `o * sigmoid(gate)`, forward and backward.
//!
//! - **Host** (run everywhere): the host mirrors (`qk_norm_rope::*_mirror`)
//!   against L-cuda-oracle's float64 reference on the torch goldens'
//!   f32-rounded inputs, the q/k golden's rows placed at their positions
//!   (0 .. 20001) of a 20,002-token sequence, and at the 2B's head layout.
//! - **Device** (`--features cuda`, `#[ignore]`: an sm_90 GPU; on the Mac they
//!   compile and are NOT RUN): the kernels against the same reference, bit for
//!   bit against the mirror where it is bitwise (the gate, the `v` copy) and
//!   against a repeat run, and rung a's checks.
//!
//! **The angle** is the host's `rope_inv_freq` times the position, one f32
//! multiply: bit-identical to the reference's `rope_angle_f32` (the same
//! arithmetic), so the reference's `(cos, sin)` are of the kernel's own angle.
//! (`rope_angle_f32` against transformers' angle is L-cuda-oracle's
//! `k6_rope_angle_is_transformers_f32_angle`.)
//!
//! Bounds, written before any run: q/k forward within `2e-5` absolute
//! (`tessl/tests/qwen35_kernels.rs:1556,1562`); gate forward within `abs 1e-7
//! + rel 1e-5` (`:1617`); every backward within `1e-4 * max|ref|`
//! (`tessl/tests/qwen35_bwd.rs:5,43-66`).

mod device_small_common;
mod reference;

use device_small_common::{
    eps64, f32s, golden, wide, Goldens, BWD_PEAK_REL, BWD_PEAK_SOURCE, EPS, GATE_FWD_SOURCE,
    GATE_FWD_TOL, QK_FWD_ABS, QK_FWD_SOURCE,
};
use ojas_qwen35_cuda::check::Check;
use ojas_qwen35_cuda::inputs::splitmix_f32;
use ojas_qwen35_cuda::qk_norm_rope::{
    output_gate_bwd_mirror, output_gate_fwd_mirror, qk_norm_rope_bwd_mirror,
    qk_norm_rope_fwd_mirror, rope_inv_freq, OutputGatePlan, QkNormRopePlan, Qkv,
};
use ojas_qwen35_cuda::small_common::{elementwise_check, peak_check, Window};
use reference::attn_pieces::{
    norm_rope_bwd_f64, norm_rope_fwd_f64, output_gate_bwd_f64, output_gate_fwd_f64, rope_cos_sin,
};

/// Qwen3.5-2B's `rope_theta` (and the golden's).
const THETA: f64 = 1e7;

const GOLDENS: Goldens = Goldens(&[
    golden!("qk_norm_rope_x"),
    golden!("qk_norm_rope_w"),
    golden!("qk_norm_rope_dy"),
    golden!("qk_norm_rope_positions"),
    golden!("attn_output_gate_o"),
    golden!("attn_output_gate_gate"),
    golden!("attn_output_gate_dy"),
]);

/// What the q/k kernels produce: `(q, k, v, dp, dq_norm_w, dk_norm_w)`.
type QkOut = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// One q/k case over a fused projection, with dense per-head reference
/// outputs `[rows, heads, dim]`.
struct QkCase {
    plan: QkNormRopePlan,
    p: Vec<f32>,
    qw: Vec<f32>,
    kw: Vec<f32>,
    invf: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dp0: Vec<f32>,
    q_ref: Vec<f64>,
    k_ref: Vec<f64>,
    dxq_ref: Vec<f64>,
    dxk_ref: Vec<f64>,
    dqw_ref: Vec<f64>,
    dkw_ref: Vec<f64>,
}

impl QkCase {
    /// `p` already holds the inputs; the reference reads them from it.
    fn new(
        plan: QkNormRopePlan,
        p: Vec<f32>,
        (qw, kw): (Vec<f32>, Vec<f32>),
        (dq, dk, dv): (Vec<f32>, Vec<f32>, Vec<f32>),
    ) -> Self {
        let (seq, hq, hkv, d, rot) = (
            plan.seq as usize,
            plan.hq as usize,
            plan.hkv as usize,
            plan.dim as usize,
            plan.rot as usize,
        );
        let ld = plan.ld_p as usize;
        let rows = plan.rows() as usize;
        let invf = rope_inv_freq(plan.rot, THETA).expect("inv_freq");
        let (mut q_ref, mut k_ref, mut dxq_ref, mut dxk_ref) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut dqw_ref, mut dkw_ref) = (vec![0.0f64; d], vec![0.0f64; d]);
        let (wq, wk) = (wide(&qw), wide(&kw));
        for r in 0..rows {
            let cs = rope_cos_sin(rot, r % seq, THETA);
            for (heads, col0, stride, w, g, out, dx_out, dw_out) in [
                (
                    hq,
                    plan.q_off as usize,
                    2 * d,
                    &wq,
                    &dq,
                    &mut q_ref,
                    &mut dxq_ref,
                    &mut dqw_ref,
                ),
                (
                    hkv,
                    plan.k_off as usize,
                    d,
                    &wk,
                    &dk,
                    &mut k_ref,
                    &mut dxk_ref,
                    &mut dkw_ref,
                ),
            ] {
                for h in 0..heads {
                    let x = wide(&p[r * ld + col0 + h * stride..][..d]);
                    out.extend(norm_rope_fwd_f64(&x, w, &cs, eps64()));
                    let (dx, dw) = norm_rope_bwd_f64(
                        &x,
                        w,
                        &wide(&g[(r * heads + h) * d..][..d]),
                        &cs,
                        eps64(),
                    );
                    dx_out.extend(dx);
                    dw_out.iter_mut().zip(dw).for_each(|(a, v)| *a += v);
                }
            }
        }
        QkCase {
            dp0: splitmix_f32(0x6c9, p.len(), 1.0),
            plan,
            p,
            qw,
            kw,
            invf,
            dq,
            dk,
            dv,
            q_ref,
            k_ref,
            dxq_ref,
            dxk_ref,
            dqw_ref,
            dkw_ref,
        }
    }

    /// The golden's 10 rows at their positions (0 .. 20001) of one sequence:
    /// one query head and one key head, both the golden row with the golden
    /// weight; every other token's q and k are zero, with zero gradient.
    fn golden() -> Self {
        let (shape, x) = GOLDENS.shaped("qk_norm_rope_x");
        let d = shape[1];
        let pos = GOLDENS.indices("qk_norm_rope_positions");
        let (x, w, dy) = (
            f32s(&x),
            f32s(&GOLDENS.f64s("qk_norm_rope_w")),
            f32s(&GOLDENS.f64s("qk_norm_rope_dy")),
        );
        let seq = pos.iter().max().expect("positions") + 1;
        let (q_off, k_off, v_off) = (2usize, 2 + 2 * d, 2 + 3 * d);
        let ld = v_off + d + 4;
        let plan = QkNormRopePlan::new(
            (1, seq as u64, 1, 1, d as u64),
            (64, EPS),
            (ld as u64, q_off as u64, k_off as u64, v_off as u64),
        )
        .expect("plan");
        let mut p = vec![0.0f32; seq * ld];
        let (mut dq, mut dk) = (vec![0.0f32; seq * d], vec![0.0f32; seq * d]);
        for (r, &t) in pos.iter().enumerate() {
            p[t * ld + q_off..][..d].copy_from_slice(&x[r * d..][..d]);
            p[t * ld + k_off..][..d].copy_from_slice(&x[r * d..][..d]);
            dq[t * d..][..d].copy_from_slice(&dy[r * d..][..d]);
            dk[t * d..][..d].copy_from_slice(&dy[r * d..][..d]);
        }
        let v = splitmix_f32(0x6c1, seq * d, 1.0);
        for t in 0..seq {
            p[t * ld + v_off..][..d].copy_from_slice(&v[t * d..][..d]);
        }
        QkCase::new(
            plan,
            p,
            (w.clone(), w),
            (dq, dk, splitmix_f32(0x6c2, seq * d, 1.0)),
        )
    }

    /// The 2B's attention heads (8 query with gates, 2 key/value, D = 256,
    /// 64 rotated) over two sequences of 40 tokens.
    fn two_b() -> Self {
        let (batch, seq, hq, hkv, d) = (2usize, 40usize, 8usize, 2usize, 256usize);
        let (q_off, k_off, v_off) = (3, 3 + 2 * hq * d, 3 + 2 * hq * d + hkv * d);
        let ld = v_off + hkv * d + 5;
        let plan = QkNormRopePlan::new(
            (batch as u64, seq as u64, hq as u64, hkv as u64, d as u64),
            (64, EPS),
            (ld as u64, q_off as u64, k_off as u64, v_off as u64),
        )
        .expect("plan");
        let rows = batch * seq;
        QkCase::new(
            plan,
            splitmix_f32(0x6d1, rows * ld, 2.0),
            (splitmix_f32(0x6d2, d, 0.5), splitmix_f32(0x6d3, d, 0.5)),
            (
                splitmix_f32(0x6d4, rows * hq * d, 1.0),
                splitmix_f32(0x6d5, rows * hkv * d, 1.0),
                splitmix_f32(0x6d6, rows * hkv * d, 1.0),
            ),
        )
    }

    /// Dense `[rows, heads, dim]` of a region of `dp`.
    fn region(&self, dp: &[f32], col0: u64, heads: u64, stride: u64) -> Vec<f32> {
        let (ld, d) = (self.plan.ld_p as usize, self.plan.dim as usize);
        let mut out = Vec::new();
        for r in 0..self.plan.rows() as usize {
            for h in 0..heads as usize {
                out.extend_from_slice(&dp[r * ld + col0 as usize + h * stride as usize..][..d]);
            }
        }
        out
    }

    fn judge(&self, label: &str, out: &QkOut) -> Vec<Check> {
        let (q, k, v, dp, dqw, dkw) = out;
        let p = &self.plan;
        let fwd = (0.0, QK_FWD_ABS);
        // v is copied: its bits are the projection's.
        let v_want = self.region(&self.p, p.v_off, p.hkv, p.dim);
        device_small_common::assert_bits(&format!("{label} v"), v, &v_want);
        let dv_got = self.region(dp, p.v_off, p.hkv, p.dim);
        device_small_common::assert_bits(&format!("{label} dv into dp"), &dv_got, &self.dv);
        let dxq = self.region(dp, p.q_off, p.hq, 2 * p.dim);
        let dxk = self.region(dp, p.k_off, p.hkv, p.dim);
        vec![
            elementwise_check(&format!("{label} q"), q, &self.q_ref, fwd, QK_FWD_SOURCE),
            elementwise_check(&format!("{label} k"), k, &self.k_ref, fwd, QK_FWD_SOURCE),
            peak_check(
                &format!("{label} dq into dp"),
                &dxq,
                &self.dxq_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dk into dp"),
                &dxk,
                &self.dxk_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dq_norm_w"),
                dqw,
                &self.dqw_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dk_norm_w"),
                dkw,
                &self.dkw_ref,
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
        ]
    }

    fn mirror(&self) -> (Qkv, Vec<f32>, Vec<f32>, Vec<f32>) {
        let qkv = qk_norm_rope_fwd_mirror(&self.plan, &self.p, (&self.qw, &self.kw), &self.invf)
            .expect("fwd mirror");
        let mut dp = self.dp0.clone();
        let (dqw, dkw) = qk_norm_rope_bwd_mirror(
            &self.plan,
            &self.p,
            (&self.qw, &self.kw),
            &self.invf,
            (&self.dq, &self.dk, &self.dv),
            &mut dp,
        )
        .expect("bwd mirror");
        (qkv, dp, dqw, dkw)
    }
}

/// The output gate over the golden's `[9, 96]` as 3 query heads of 32 in a
/// fused projection; `out` in a padded buffer, `dy` in a window.
struct GateCase {
    plan: OutputGatePlan,
    p: Vec<f32>,
    attn: Vec<f32>,
    out_win: Window,
    dy_win: Window,
    dy: Vec<f32>,
    out0: Vec<f32>,
    dp0: Vec<f32>,
    want: [Vec<f64>; 3],
}

impl GateCase {
    fn golden() -> Self {
        let (shape, o) = GOLDENS.shaped("attn_output_gate_o");
        let (rows, width) = (shape[0], shape[1]);
        let (hq, d) = (3usize, width / 3);
        let get = |n: &str| f32s(&GOLDENS.f64s(&format!("attn_output_gate_{n}")));
        let (o, gate, dy_dense) = (f32s(&o), get("gate"), get("dy"));
        let (q_off, ld) = (1usize, 1 + 2 * width + 2);
        let plan = OutputGatePlan::new(rows as u64, hq as u64, d as u64, (ld as u64, q_off as u64))
            .expect("plan");
        let mut p = splitmix_f32(0x6e1, rows * ld, 3.0);
        for r in 0..rows {
            for h in 0..hq {
                p[r * ld + q_off + h * 2 * d + d..][..d]
                    .copy_from_slice(&gate[r * width + h * d..][..d]);
            }
        }
        let dy_win = Window {
            ld: width as u64 + 3,
            off: 2,
        };
        let mut dy = splitmix_f32(0x6e2, rows * (width + 3), 1.0);
        for r in 0..rows {
            dy[r * (width + 3) + 2..][..width].copy_from_slice(&dy_dense[r * width..][..width]);
        }
        let y = output_gate_fwd_f64(&wide(&o), &wide(&gate));
        let (d_o, dgate) = output_gate_bwd_f64(&wide(&o), &wide(&gate), &wide(&dy_dense));
        GateCase {
            plan,
            out0: splitmix_f32(0x6e3, rows * (width + 5), 1.0),
            dp0: splitmix_f32(0x6e4, rows * ld, 1.0),
            p,
            attn: o,
            out_win: Window {
                ld: width as u64 + 5,
                off: 4,
            },
            dy_win,
            dy,
            want: [y, d_o, dgate],
        }
    }

    /// Dense `[rows, hq * dim]` of the gate columns of `dp`.
    fn gate_cols(&self, dp: &[f32]) -> Vec<f32> {
        let (rows, hq, d, ld, q_off) = (
            self.plan.rows as usize,
            self.plan.hq as usize,
            self.plan.dim as usize,
            self.plan.ld_p as usize,
            self.plan.q_off as usize,
        );
        (0..rows)
            .flat_map(|r| {
                (0..hq).flat_map(move |h| dp[r * ld + q_off + h * 2 * d + d..][..d].to_vec())
            })
            .collect()
    }

    fn judge(&self, label: &str, (out, d_attn, dp): (&[f32], &[f32], &[f32])) -> Vec<Check> {
        let width = self.plan.width() as usize;
        let y: Vec<f32> = (0..self.plan.rows as usize)
            .flat_map(|r| (0..width).map(move |c| out[self.out_win.at(r, c)]))
            .collect();
        vec![
            elementwise_check(
                &format!("{label} y"),
                &y,
                &self.want[0],
                GATE_FWD_TOL,
                GATE_FWD_SOURCE,
            ),
            peak_check(
                &format!("{label} do"),
                d_attn,
                &self.want[1],
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
            peak_check(
                &format!("{label} dgate"),
                &self.gate_cols(dp),
                &self.want[2],
                BWD_PEAK_REL,
                BWD_PEAK_SOURCE,
            ),
        ]
    }

    fn mirror(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut out = self.out0.clone();
        output_gate_fwd_mirror(&self.plan, &self.attn, &self.p, self.out_win, &mut out)
            .expect("fwd mirror");
        let mut dp = self.dp0.clone();
        let d_attn = output_gate_bwd_mirror(
            &self.plan,
            (&self.attn, &self.p),
            (self.dy_win, &self.dy),
            &mut dp,
        )
        .expect("bwd mirror");
        (out, d_attn, dp)
    }
}

fn qk_mirror_checks(label: &str, c: &QkCase) -> Vec<Check> {
    let ((q, k, v), dp, dqw, dkw) = c.mirror();
    c.judge(label, &(q, k, v, dp, dqw, dkw))
}

#[test]
fn the_embedded_goldens_are_l_cuda_oracles_pinned_bytes() {
    assert_eq!(GOLDENS.verify(), 7);
}

#[test]
fn the_qk_mirror_matches_the_reference_on_the_golden_rows_at_positions_up_to_20001() {
    device_small_common::assert_pass(&qk_mirror_checks(
        "mirror qk_norm_rope golden",
        &QkCase::golden(),
    ));
}

#[test]
fn the_qk_mirror_matches_the_reference_at_the_2b_head_layout() {
    device_small_common::assert_pass(&qk_mirror_checks(
        "mirror qk_norm_rope 2x40",
        &QkCase::two_b(),
    ));
}

#[test]
fn the_gate_mirror_matches_the_reference_on_the_torch_golden() {
    let c = GateCase::golden();
    let (out, d_attn, dp) = c.mirror();
    device_small_common::assert_pass(&c.judge("mirror output gate golden", (&out, &d_attn, &dp)));
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use device_small_common::{assert_bits, assert_pass, runtime};
    use ojas_qwen35_cuda::qk_norm_rope_cuda::{
        output_gate, output_gate_bwd, qk_norm_rope, qk_norm_rope_bwd,
    };
    use ojas_qwen35_cuda::runtime::CudaRuntime;
    use ojas_qwen35_cuda::small_smoke::k6_checks;

    fn run_qk(rt: &CudaRuntime, c: &QkCase) -> QkOut {
        let pl = &c.plan;
        let nan = |len: usize| {
            rt.upload(&vec![f32::NAN; len], "k6 sentinel")
                .expect("sentinel")
        };
        let p = rt.upload(&c.p, "p").expect("p");
        let qw = rt.upload(&c.qw, "q_norm_w").expect("qw");
        let kw = rt.upload(&c.kw, "k_norm_w").expect("kw");
        let invf = rt.upload(&c.invf, "inv_freq").expect("inv_freq");
        let (mut q, mut k, mut v) = (
            nan(pl.q_len() as usize),
            nan(pl.kv_len() as usize),
            nan(pl.kv_len() as usize),
        );
        qk_norm_rope(rt, pl, (&p, &qw, &kw, &invf), (&mut q, &mut k, &mut v)).expect("qk fwd");
        let dq = rt.upload(&c.dq, "dq").expect("dq");
        let dk = rt.upload(&c.dk, "dk").expect("dk");
        let dv = rt.upload(&c.dv, "dv").expect("dv");
        let mut dp = rt.upload(&c.dp0, "dp").expect("dp");
        let mut part = nan(pl.part_len() as usize);
        let (mut dqw, mut dkw) = (nan(pl.dim as usize), nan(pl.dim as usize));
        qk_norm_rope_bwd(
            rt,
            pl,
            (&p, &qw, &kw, &invf),
            (&dq, &dk, &dv),
            &mut dp,
            &mut part,
            (&mut dqw, &mut dkw),
        )
        .expect("qk bwd");
        (
            rt.download(&q).expect("q"),
            rt.download(&k).expect("k"),
            rt.download(&v).expect("v"),
            rt.download(&dp).expect("dp"),
            rt.download(&dqw).expect("dq_norm_w"),
            rt.download(&dkw).expect("dk_norm_w"),
        )
    }

    fn qk(rt: &CudaRuntime, label: &str, c: &QkCase) -> Vec<Check> {
        let got = run_qk(rt, c);
        let again = run_qk(rt, c);
        for (name, g, a) in [
            ("q", &got.0, &again.0),
            ("k", &got.1, &again.1),
            ("v", &got.2, &again.2),
            ("dp", &got.3, &again.3),
            ("dq_norm_w", &got.4, &again.4),
            ("dk_norm_w", &got.5, &again.5),
        ] {
            assert_bits(&format!("{label} {name} repeat"), a, g);
        }
        // Not bitwise against the mirror (libdevice cosf/sinf against Rust's):
        // the mirror is held to the same reference by the host tests.
        c.judge(label, &got)
    }

    fn run_gate(rt: &CudaRuntime, c: &GateCase) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let p = rt.upload(&c.p, "p").expect("p");
        let attn = rt.upload(&c.attn, "attn").expect("attn");
        let mut out = rt.upload(&c.out0, "out").expect("out");
        output_gate(rt, &c.plan, (&attn, &p), c.out_win, &mut out).expect("gate fwd");
        let dy = rt.upload(&c.dy, "dy").expect("dy");
        let mut d_attn = rt
            .upload(&vec![f32::NAN; c.attn.len()], "d_attn")
            .expect("d_attn");
        let mut dp = rt.upload(&c.dp0, "dp").expect("dp");
        output_gate_bwd(
            rt,
            &c.plan,
            (&attn, &p),
            (c.dy_win, &dy),
            &mut d_attn,
            &mut dp,
        )
        .expect("gate bwd");
        (
            rt.download(&out).expect("out"),
            rt.download(&d_attn).expect("d_attn"),
            rt.download(&dp).expect("dp"),
        )
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_qk_norm_rope_matches_the_reference_up_to_position_20001() {
        let rt = runtime();
        let mut checks = qk(&rt, "device qk_norm_rope golden", &QkCase::golden());
        checks.extend(qk(&rt, "device qk_norm_rope 2x40", &QkCase::two_b()));
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_output_gate_matches_the_reference_and_the_mirror_bitwise() {
        let rt = runtime();
        let c = GateCase::golden();
        let got = run_gate(&rt, &c);
        let again = run_gate(&rt, &c);
        let want = c.mirror();
        for (name, g, a, w) in [
            ("out", &got.0, &again.0, &want.0),
            ("d_attn", &got.1, &again.1, &want.1),
            ("dp", &got.2, &again.2, &want.2),
        ] {
            assert_bits(&format!("gate {name} vs mirror"), g, w);
            assert_bits(&format!("gate {name} repeat"), a, g);
        }
        assert_pass(&c.judge("device output gate golden", (&got.0, &got.1, &got.2)));
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn rung_a_k6_checks_pass() {
        assert_pass(&k6_checks(&runtime()));
    }
}
