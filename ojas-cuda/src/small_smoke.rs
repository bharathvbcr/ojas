//! Rung (a)'s checks for lane L-cuda-small: every kernel of K3, K4, K6, K7,
//! K9 and K10 on the device, against this crate's host mirrors, each run
//! twice from scratch (the second run bit-identical to the first).
//!
//! These are self-contained (no fixture files), so `runga` can call them on
//! the box. Device results are compared to the **mirrors**:
//!
//! - **bitwise** wherever the mirror is bitwise (K3, K4, K7, K9, the K6
//!   output gate, and K10 on the FFMA engine): one rounding per `+`/`*` under
//!   `--fmad=false`, the fixed reductions of [`crate::small_common`], and
//!   `k8_act`'s exp/log/sigmoid/SiLU with their bit-identical host twins;
//! - within tessl's bounds where it is not: the K6 q/k norm + RoPE (libdevice
//!   `cosf`/`sinf` against Rust's `cos`/`sin`) and K10 on cuBLAS's bf16 engine
//!   (its own summation order). Each bound is a constant below, with its
//!   source, written before any run.
//!
//! Comparisons against L-cuda-oracle's float64 references and goldens are
//! the device tests' (`tests/device_k*.rs`). Output and partial buffers start
//! as NaN sentinels, so an element no thread wrote fails the bitwise check;
//! buffers a kernel writes only in part start with seeded values the mirror
//! starts from too, so a stray write fails it as well.

use crate::buffer::CudaBuffer;
use crate::ce_rows::{ce_rows_mirror, CeOutput, CeRowsPlan, Reduction, CE_CHUNK};
use crate::ce_rows_cuda::{ce_rows, CeGrads, CeWorkspace};
use crate::check::{bitwise_check, diff_bits_f32, Check};
use crate::conv1d::{conv1d_silu_bwd_mirror, conv1d_silu_fwd_mirror, Conv1dPlan};
use crate::conv1d_cuda::{conv1d_silu, conv1d_silu_bwd};
use crate::embed::{embed_rows_bwd_mirror, embed_rows_fwd_mirror, EmbedPlan};
use crate::embed_cuda::{embed_rows, embed_rows_bwd};
use crate::error::CudaError;
use crate::gates_published::{
    gates_published_bwd_mirror, gates_published_fwd_mirror, GatesPublishedPlan,
};
use crate::gates_published_cuda::{gates_published, gates_published_bwd};
use crate::gemm::Bf16Engine;
use crate::host_ref::Operands;
use crate::inputs::{splitmix_bits, splitmix_f32};
use crate::k0::GatherSource;
use crate::qk_norm_rope::{
    output_gate_bwd_mirror, output_gate_fwd_mirror, qk_norm_rope_bwd_mirror,
    qk_norm_rope_fwd_mirror, rope_inv_freq, OutputGatePlan, QkNormRopePlan,
};
use crate::qk_norm_rope_cuda::{output_gate, output_gate_bwd, qk_norm_rope, qk_norm_rope_bwd};
use crate::rmsnorm::{
    gated_rms_norm_bwd_mirror, gated_rms_norm_fwd_mirror, rms_norm_bwd_mirror, rms_norm_fwd_mirror,
    GatedGradWindows, GatedRmsNormPlan, RmsNormPlan,
};
use crate::rmsnorm_cuda::{gated_rms_norm, gated_rms_norm_bwd, rms_norm, rms_norm_bwd};
use crate::runtime::CudaRuntime;
use crate::small_common::{
    elementwise_check, f64s, peak_check, to_usize, Window, BWD_PEAK_REL, BWD_PEAK_SOURCE,
    CE_BF16_GRAD_PEAK_REL, CE_LOSS_TOL, CE_SOURCE, QK_FWD_ABS, QK_FWD_SOURCE,
};
use crate::small_common_cuda::compile_checks;
use crate::smoke::guarded;

const EPS: f32 = 1e-6;

/// What one output is compared to.
enum Want {
    /// The bitwise mirror.
    Bits(Vec<f32>),
    /// The mirror, elementwise within `(rel, abs)`.
    Close(Vec<f32>, (f64, f64), &'static str),
    /// The mirror, within `rel` of its peak.
    Peak(Vec<f32>, f64, &'static str),
}

fn compare(name: &str, got: &[f32], want: &Want) -> Check {
    match want {
        Want::Bits(w) => bitwise_check(name, diff_bits_f32(got, w), w.len()),
        Want::Close(w, tol, source) => elementwise_check(name, got, &f64s(w), *tol, source),
        Want::Peak(w, rel, source) => peak_check(name, got, &f64s(w), *rel, source),
    }
}

/// Two device runs from scratch: each output against its [`Want`], and the
/// second run against the first, bitwise.
fn run_twice(
    name: &str,
    wants: &[(&str, Want)],
    run: impl Fn() -> Result<Vec<Vec<f32>>, CudaError>,
) -> Vec<Check> {
    let first = match run() {
        Ok(v) => v,
        Err(e) => return vec![Check::from_error(name, &e)],
    };
    let second = match run() {
        Ok(v) => v,
        Err(e) => return vec![Check::from_error(&format!("{name}.repeat"), &e)],
    };
    if first.len() != wants.len() || second.len() != wants.len() {
        return vec![Check::fail(
            name,
            format!(
                "{} and {} outputs for {} expected",
                first.len(),
                second.len(),
                wants.len()
            ),
        )];
    }
    let mut out = Vec::with_capacity(2 * wants.len());
    for (((label, want), a), b) in wants.iter().zip(&first).zip(&second) {
        out.push(compare(&format!("{name}.{label}.vs_mirror"), a, want));
        out.push(bitwise_check(
            &format!("{name}.{label}.repeat"),
            diff_bits_f32(b, a),
            a.len(),
        ));
    }
    out
}

/// One case: build the mirror's answers (which may refuse), then run.
fn case(name: &str, f: impl FnOnce() -> Result<Vec<Check>, CudaError>) -> Vec<Check> {
    guarded(name, || match f() {
        Ok(checks) => checks,
        Err(e) => vec![Check::from_error(name, &e)],
    })
}

fn len(v: u64) -> Result<usize, CudaError> {
    to_usize(v, "small_smoke")
}

/// A buffer of NaNs: an element no kernel writes stays NaN.
fn sentinel(rt: &CudaRuntime, n: usize, label: &str) -> Result<CudaBuffer<f32>, CudaError> {
    rt.upload(&vec![f32::NAN; n], label)
}

/// Every check of this lane: the compile check, then K3, K4, K6, K7, K9, K10.
pub fn small_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = compile_checks(rt);
    out.extend(k3_checks(rt));
    out.extend(k4_checks(rt));
    out.extend(k6_checks(rt));
    out.extend(k7_checks(rt));
    out.extend(k9_checks(rt));
    out.extend(k10_checks(rt));
    out
}

// ---------------------------------------------------------------------- K3 ---

/// K3: the `published` GDN gates, forward and backward, bitwise.
pub fn k3_checks(rt: &CudaRuntime) -> Vec<Check> {
    let name = "k3.gates_published.300x16";
    case(name, || {
        // The fused projection row: [other | b (16) | a (16) | other], as tessl's
        // run_gates (tessl/tests/qwen35_bwd.rs:1221-1236).
        let (rows, heads) = (300u64, 16u64);
        let (b_off, a_off) = (5u64, 21u64);
        let ld = a_off + heads + 3;
        let plan = GatesPublishedPlan::new(rows, heads, ld, (a_off, b_off))?;
        let (r, h, l) = (len(rows)?, len(heads)?, len(ld)?);
        let a = len(a_off)?;
        let mut p = splitmix_f32(0x3a1, r * l, 12.0);
        for row in (0..r).step_by(3) {
            p[row * l + a] = 26.0; // past softplus's threshold of 20
        }
        for row in (1..r).step_by(5) {
            p[row * l + a + h - 1] = -24.0;
        }
        p[(r - 1) * l + a] = 120.0; // e^x would overflow f32
        let a_log = splitmix_f32(0x3a2, h, 0.8);
        let dt: Vec<f32> = splitmix_f32(0x3a3, h, 1.0)
            .iter()
            .map(|v| v - 3.0)
            .collect();
        let dg = splitmix_f32(0x3a4, r * h, 1.0);
        let dbeta = splitmix_f32(0x3a5, r * h, 1.0);
        let dp0 = splitmix_f32(0x3a6, r * l, 1.0);
        let (g, beta) = gates_published_fwd_mirror(&plan, &p, &a_log, &dt)?;
        let mut dp = dp0.clone();
        let grads = gates_published_bwd_mirror(&plan, (&p, &a_log, &dt), (&dg, &dbeta), &mut dp)?;
        let wants = [
            ("g", Want::Bits(g)),
            ("beta", Want::Bits(beta)),
            ("dp", Want::Bits(dp)),
            ("dA_log", Want::Bits(grads.da_log)),
            ("ddt_bias", Want::Bits(grads.ddt_bias)),
        ];
        Ok(run_twice(name, &wants, || {
            let pb = rt.upload(&p, "k3 p")?;
            let alb = rt.upload(&a_log, "k3 A_log")?;
            let dtb = rt.upload(&dt, "k3 dt_bias")?;
            let mut gb = sentinel(rt, r * h, "k3 g")?;
            let mut betab = sentinel(rt, r * h, "k3 beta")?;
            gates_published(rt, &plan, (&pb, &alb, &dtb), &mut gb, &mut betab)?;
            let dgb = rt.upload(&dg, "k3 dg")?;
            let dbb = rt.upload(&dbeta, "k3 dbeta")?;
            let mut dpb = rt.upload(&dp0, "k3 dp")?;
            let mut part = sentinel(rt, len(plan.part_len())?, "k3 part")?;
            let mut dal = sentinel(rt, h, "k3 dA_log")?;
            let mut ddt = sentinel(rt, h, "k3 ddt_bias")?;
            gates_published_bwd(
                rt,
                &plan,
                (&pb, &alb, &dtb),
                (&dgb, &dbb),
                &mut dpb,
                &mut part,
                (&mut dal, &mut ddt),
            )?;
            Ok(vec![
                rt.download(&gb)?,
                rt.download(&betab)?,
                rt.download(&dpb)?,
                rt.download(&dal)?,
                rt.download(&ddt)?,
            ])
        }))
    })
}

// ---------------------------------------------------------------------- K4 ---

/// K4: the causal conv + SiLU, forward and backward, bitwise, at the 2B's
/// 6,144 channels in a fused-projection window and at a small multi-batch
/// shape with kernel width 2.
pub fn k4_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    for (batch, seq, channels, kw, ld, off) in [
        (1u64, 300u64, 6144u64, 4u32, 6200u64, 40u64),
        (3, 37, 40, 2, 40, 0),
    ] {
        let name = format!("k4.conv1d_silu.b{batch}_t{seq}_c{channels}_k{kw}");
        out.extend(case(&name, || {
            let xw = Window { ld, off };
            let plan = Conv1dPlan::new(batch, seq, channels, kw, xw)?;
            let rows = len(plan.rows())?;
            let x = splitmix_f32(0x4a1, rows * len(ld)?, 2.0);
            let w = splitmix_f32(0x4a2, len(plan.w_len())?, 0.5);
            let dy = splitmix_f32(0x4a3, len(plan.y_len())?, 1.0);
            let dx0 = splitmix_f32(0x4a4, rows * len(ld)?, 1.0);
            let y = conv1d_silu_fwd_mirror(&plan, &x, &w)?;
            let dy_win = Window::dense(channels);
            let mut dx = dx0.clone();
            let dw = conv1d_silu_bwd_mirror(&plan, (&x, &w), (dy_win, &dy), (xw, &mut dx))?;
            let wants = [
                ("y", Want::Bits(y)),
                ("dx", Want::Bits(dx)),
                ("dw", Want::Bits(dw)),
            ];
            Ok(run_twice(&name, &wants, || {
                let xb = rt.upload(&x, "k4 x")?;
                let wb = rt.upload(&w, "k4 w")?;
                let mut yb = sentinel(rt, len(plan.y_len())?, "k4 y")?;
                conv1d_silu(rt, &plan, &xb, &wb, &mut yb)?;
                let dyb = rt.upload(&dy, "k4 dy")?;
                let mut dxb = rt.upload(&dx0, "k4 dx")?;
                let mut part = sentinel(rt, len(plan.part_len())?, "k4 part")?;
                let mut dwb = sentinel(rt, len(plan.w_len())?, "k4 dw")?;
                conv1d_silu_bwd(
                    rt,
                    &plan,
                    (&xb, &wb),
                    (dy_win, &dyb),
                    (xw, &mut dxb),
                    &mut part,
                    &mut dwb,
                )?;
                Ok(vec![
                    rt.download(&yb)?,
                    rt.download(&dxb)?,
                    rt.download(&dwb)?,
                ])
            }))
        }));
    }
    out
}

// ---------------------------------------------------------------------- K6 ---

/// K6: the q/k norm + partial RoPE (within tessl's bounds of the mirror:
/// libdevice `cosf`/`sinf`) and the output gate (bitwise), forward and
/// backward, at the 2B's attention shape in the fused layout.
pub fn k6_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    // 8 query heads (each 2D wide with its gate), 2 kv heads, D = 256, 64
    // rotated dims, theta 1e7; positions up to 549.
    let (batch, seq, hq, hkv, dim, rot) = (2u64, 550u64, 8u64, 2u64, 256u64, 64u64);
    let (q_off, k_off, v_off) = (3u64, 3 + 2 * hq * dim, 3 + 2 * hq * dim + hkv * dim);
    let ld_p = v_off + hkv * dim + 5;
    let name = "k6.qk_norm_rope.b2_t550";
    out.extend(case(name, || {
        let plan = QkNormRopePlan::new(
            (batch, seq, hq, hkv, dim),
            (rot, EPS),
            (ld_p, q_off, k_off, v_off),
        )?;
        let rows = len(plan.rows())?;
        let p = splitmix_f32(0x6a1, rows * len(ld_p)?, 2.0);
        let qw = splitmix_f32(0x6a2, len(dim)?, 0.5);
        let kw = splitmix_f32(0x6a3, len(dim)?, 0.5);
        let invf = rope_inv_freq(rot, 1e7)?;
        let (ql, kvl) = (len(plan.q_len())?, len(plan.kv_len())?);
        let dq = splitmix_f32(0x6a4, ql, 1.0);
        let dk = splitmix_f32(0x6a5, kvl, 1.0);
        let dv = splitmix_f32(0x6a6, kvl, 1.0);
        let dp0 = splitmix_f32(0x6a7, rows * len(ld_p)?, 1.0);
        let (q, k, v) = qk_norm_rope_fwd_mirror(&plan, &p, (&qw, &kw), &invf)?;
        let mut dp = dp0.clone();
        let (dqw, dkw) =
            qk_norm_rope_bwd_mirror(&plan, &p, (&qw, &kw), &invf, (&dq, &dk, &dv), &mut dp)?;
        let fwd = (0.0, QK_FWD_ABS);
        let wants = [
            ("q", Want::Close(q, fwd, QK_FWD_SOURCE)),
            ("k", Want::Close(k, fwd, QK_FWD_SOURCE)),
            ("v", Want::Bits(v)),
            ("dp", Want::Peak(dp, BWD_PEAK_REL, BWD_PEAK_SOURCE)),
            ("dq_norm_w", Want::Peak(dqw, BWD_PEAK_REL, BWD_PEAK_SOURCE)),
            ("dk_norm_w", Want::Peak(dkw, BWD_PEAK_REL, BWD_PEAK_SOURCE)),
        ];
        Ok(run_twice(name, &wants, || {
            let pb = rt.upload(&p, "k6 p")?;
            let qwb = rt.upload(&qw, "k6 q_norm_w")?;
            let kwb = rt.upload(&kw, "k6 k_norm_w")?;
            let ifb = rt.upload(&invf, "k6 inv_freq")?;
            let mut qb = sentinel(rt, ql, "k6 q")?;
            let mut kb = sentinel(rt, kvl, "k6 k")?;
            let mut vb = sentinel(rt, kvl, "k6 v")?;
            qk_norm_rope(
                rt,
                &plan,
                (&pb, &qwb, &kwb, &ifb),
                (&mut qb, &mut kb, &mut vb),
            )?;
            let dqb = rt.upload(&dq, "k6 dq")?;
            let dkb = rt.upload(&dk, "k6 dk")?;
            let dvb = rt.upload(&dv, "k6 dv")?;
            let mut dpb = rt.upload(&dp0, "k6 dp")?;
            let mut part = sentinel(rt, len(plan.part_len())?, "k6 part")?;
            let mut dqwb = sentinel(rt, len(dim)?, "k6 dq_norm_w")?;
            let mut dkwb = sentinel(rt, len(dim)?, "k6 dk_norm_w")?;
            qk_norm_rope_bwd(
                rt,
                &plan,
                (&pb, &qwb, &kwb, &ifb),
                (&dqb, &dkb, &dvb),
                &mut dpb,
                &mut part,
                (&mut dqwb, &mut dkwb),
            )?;
            Ok(vec![
                rt.download(&qb)?,
                rt.download(&kb)?,
                rt.download(&vb)?,
                rt.download(&dpb)?,
                rt.download(&dqwb)?,
                rt.download(&dkwb)?,
            ])
        }))
    }));

    let name = "k6.output_gate.t1100";
    out.extend(case(name, || {
        let rows = batch * seq;
        let plan = OutputGatePlan::new(rows, hq, dim, (ld_p, q_off))?;
        let (r, width) = (len(rows)?, len(plan.width())?);
        let out_win = Window {
            ld: plan.width() + 8,
            off: 4,
        };
        let dy_win = Window {
            ld: plan.width() + 3,
            off: 1,
        };
        let p = splitmix_f32(0x6b1, r * len(ld_p)?, 4.0);
        let attn = splitmix_f32(0x6b2, r * width, 1.0);
        let out0 = splitmix_f32(0x6b3, r * len(out_win.ld)?, 1.0);
        let dy = splitmix_f32(0x6b4, r * len(dy_win.ld)?, 1.0);
        let dp0 = splitmix_f32(0x6b5, r * len(ld_p)?, 1.0);
        let mut y = out0.clone();
        output_gate_fwd_mirror(&plan, &attn, &p, out_win, &mut y)?;
        let mut dp = dp0.clone();
        let d_attn = output_gate_bwd_mirror(&plan, (&attn, &p), (dy_win, &dy), &mut dp)?;
        let wants = [
            ("out", Want::Bits(y)),
            ("d_attn", Want::Bits(d_attn)),
            ("dp", Want::Bits(dp)),
        ];
        Ok(run_twice(name, &wants, || {
            let pb = rt.upload(&p, "k6 gate p")?;
            let ab = rt.upload(&attn, "k6 gate attn")?;
            let mut ob = rt.upload(&out0, "k6 gate out")?;
            output_gate(rt, &plan, (&ab, &pb), out_win, &mut ob)?;
            let dyb = rt.upload(&dy, "k6 gate dy")?;
            let mut dab = sentinel(rt, r * width, "k6 gate d_attn")?;
            let mut dpb = rt.upload(&dp0, "k6 gate dp")?;
            output_gate_bwd(rt, &plan, (&ab, &pb), (dy_win, &dyb), &mut dab, &mut dpb)?;
            Ok(vec![
                rt.download(&ob)?,
                rt.download(&dab)?,
                rt.download(&dpb)?,
            ])
        }))
    }));
    out
}

// ---------------------------------------------------------------------- K7 ---

/// K7: `rms_norm` (with `1 + w`) at the 2B's hidden size and at a ragged
/// width, and the gated norm at the 2B's 16 heads of 128 in fused windows;
/// forward and backward (with and without accumulation), bitwise.
pub fn k7_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    for (rows, dim) in [(70u64, 2048u64), (5, 300)] {
        let name = format!("k7.rms_norm.{rows}x{dim}");
        out.extend(case(&name, || {
            let plan = RmsNormPlan::new(rows, dim, EPS)?;
            let (n, d) = (len(plan.len())?, len(dim)?);
            let x = splitmix_f32(0x7a1, n, 3.0);
            let w = splitmix_f32(0x7a2, d, 0.5);
            let dy = splitmix_f32(0x7a3, n, 1.0);
            let dx0 = splitmix_f32(0x7a4, n, 1.0);
            let y = rms_norm_fwd_mirror(&plan, &x, &w)?;
            let (dx, dw) = rms_norm_bwd_mirror(&plan, &x, &w, &dy, None)?;
            let (dx_acc, _) = rms_norm_bwd_mirror(&plan, &x, &w, &dy, Some(&dx0))?;
            let wants = [
                ("y", Want::Bits(y)),
                ("dx", Want::Bits(dx)),
                ("dw", Want::Bits(dw)),
                ("dx_accumulate", Want::Bits(dx_acc)),
            ];
            Ok(run_twice(&name, &wants, || {
                let xb = rt.upload(&x, "k7 x")?;
                let wb = rt.upload(&w, "k7 w")?;
                let dyb = rt.upload(&dy, "k7 dy")?;
                let mut yb = sentinel(rt, n, "k7 y")?;
                rms_norm(rt, &plan, &xb, &wb, &mut yb)?;
                let mut part = sentinel(rt, len(plan.part_len())?, "k7 part")?;
                let mut dxb = sentinel(rt, n, "k7 dx")?;
                let mut dwb = sentinel(rt, d, "k7 dw")?;
                rms_norm_bwd(
                    rt,
                    &plan,
                    (&xb, &wb, &dyb),
                    &mut dxb,
                    false,
                    &mut part,
                    &mut dwb,
                )?;
                let mut dxab = rt.upload(&dx0, "k7 dx accumulate")?;
                let mut dw2 = sentinel(rt, d, "k7 dw accumulate")?;
                rms_norm_bwd(
                    rt,
                    &plan,
                    (&xb, &wb, &dyb),
                    &mut dxab,
                    true,
                    &mut part,
                    &mut dw2,
                )?;
                Ok(vec![
                    rt.download(&yb)?,
                    rt.download(&dxb)?,
                    rt.download(&dwb)?,
                    rt.download(&dxab)?,
                ])
            }))
        }));
    }

    let name = "k7.gated_rms_norm.9x16x128";
    out.extend(case(name, || {
        let (rows, heads, dim) = (9u64, 16u64, 128u64);
        let width = heads * dim;
        // z and dz in the fused GDN projection (and its gradient); out in a
        // padded buffer; x, dy, dx dense.
        let proj = Window {
            ld: 2 * width + 104,
            off: 1000,
        };
        let out_win = Window {
            ld: width + 16,
            off: 8,
        };
        let dense = Window::dense(width);
        let plan = GatedRmsNormPlan::new((rows, heads, dim), EPS, dense, proj)?;
        let (r, wd) = (len(rows)?, len(width)?);
        let x = splitmix_f32(0x7b1, r * wd, 2.0);
        let z = splitmix_f32(0x7b2, r * len(proj.ld)?, 4.0);
        let w = splitmix_f32(0x7b3, len(dim)?, 1.0);
        let out0 = splitmix_f32(0x7b4, r * len(out_win.ld)?, 1.0);
        let dy = splitmix_f32(0x7b5, r * wd, 1.0);
        let dz0 = splitmix_f32(0x7b6, r * len(proj.ld)?, 1.0);
        let mut y = out0.clone();
        gated_rms_norm_fwd_mirror(&plan, (&x, &z, &w), out_win, &mut y)?;
        let win = GatedGradWindows {
            dy: dense,
            dx: dense,
            dz: proj,
        };
        let mut dx = vec![f32::NAN; r * wd];
        let mut dz = dz0.clone();
        let dw = gated_rms_norm_bwd_mirror(&plan, (&x, &z, &w, &dy), win, &mut dx, &mut dz)?;
        let wants = [
            ("out", Want::Bits(y)),
            ("dx", Want::Bits(dx)),
            ("dz", Want::Bits(dz)),
            ("dw", Want::Bits(dw)),
        ];
        Ok(run_twice(name, &wants, || {
            let xb = rt.upload(&x, "k7 gated x")?;
            let zb = rt.upload(&z, "k7 gated z")?;
            let wb = rt.upload(&w, "k7 gated w")?;
            let mut ob = rt.upload(&out0, "k7 gated out")?;
            gated_rms_norm(rt, &plan, (&xb, &zb, &wb), out_win, &mut ob)?;
            let dyb = rt.upload(&dy, "k7 gated dy")?;
            let mut dxb = sentinel(rt, r * wd, "k7 gated dx")?;
            let mut dzb = rt.upload(&dz0, "k7 gated dz")?;
            let mut part = sentinel(rt, len(plan.part_len())?, "k7 gated part")?;
            let mut dwb = sentinel(rt, len(dim)?, "k7 gated dw")?;
            gated_rms_norm_bwd(
                rt,
                &plan,
                (&xb, &zb, &wb, &dyb),
                win,
                (&mut dxb, &mut dzb),
                &mut part,
                &mut dwb,
            )?;
            Ok(vec![
                rt.download(&ob)?,
                rt.download(&dxb)?,
                rt.download(&dzb)?,
                rt.download(&dwb)?,
            ])
        }))
    }));
    out
}

// ---------------------------------------------------------------------- K9 ---

/// K9: the embedding gather (bits, a NaN payload included) and its
/// run-ordered backward added onto an existing `dW`, at the 2B's hidden size.
pub fn k9_checks(rt: &CudaRuntime) -> Vec<Check> {
    let name = "k9.embed_rows.600x1000x2048";
    case(name, || {
        let (n, vocab, hidden) = (600u64, 1000u64, 2048u64);
        let plan = EmbedPlan::new(n, vocab, hidden)?;
        let (nu, vu, hu) = (len(n)?, len(vocab)?, len(hidden)?);
        // Repeated, out-of-order ids, and both ends of the table.
        let mut ids: Vec<u32> = splitmix_bits(0x9a1, nu).iter().map(|b| b % 1000).collect();
        ids[0] = 999;
        ids[1] = 0;
        let mut table = splitmix_f32(0x9a2, vu * hu, 1.0);
        table[ids[2] as usize * hu + 5] = f32::from_bits(0x7fa0_0001);
        let dh = splitmix_f32(0x9a3, nu * hu, 1.0);
        let dw0 = splitmix_f32(0x9a4, vu * hu, 0.01);
        let y = embed_rows_fwd_mirror(&plan, &ids, &table)?;
        let mut dw = dw0.clone();
        embed_rows_bwd_mirror(&plan, &ids, &dh, &mut dw)?;
        let wants = [("out", Want::Bits(y)), ("dw", Want::Bits(dw))];
        Ok(run_twice(name, &wants, || {
            let tb = rt.upload(&table, "k9 table")?;
            let mut ob = sentinel(rt, nu * hu, "k9 out")?;
            embed_rows(rt, &plan, &ids, &tb, &mut ob)?;
            let dhb = rt.upload(&dh, "k9 dh")?;
            let mut dwb = rt.upload(&dw0, "k9 dw")?;
            embed_rows_bwd(rt, &plan, &ids, &dhb, &mut dwb)?;
            Ok(vec![rt.download(&ob)?, rt.download(&dwb)?])
        }))
    })
}

// --------------------------------------------------------------------- K10 ---

/// One cross-entropy case.
struct CeCase {
    tag: &'static str,
    n: u64,
    hidden: u64,
    vocab: u64,
    chunk: u64,
    operands: Operands,
    engine: Bf16Engine,
}

/// The device walk once: `(output, dh, dw)`.
fn ce_device(
    rt: &CudaRuntime,
    plan: CeRowsPlan,
    (h, w): (&[f32], &[f32]),
    (rows, targets): (&[u32], &[u32]),
    (operands, engine): (Operands, Bf16Engine),
) -> Result<(CeOutput, Vec<f32>, Vec<f32>), CudaError> {
    let mut ws = CeWorkspace::new(rt, plan)?;
    let hb = rt.upload(h, "k10 h")?;
    let wb = rt.upload(w, "k10 w")?;
    let mut dh = sentinel(rt, len(plan.n * plan.hidden)?, "k10 dh")?;
    let mut dw = sentinel(rt, len(plan.vocab * plan.hidden)?, "k10 dW")?;
    let grads = CeGrads {
        scale: 1.0,
        dh: &mut dh,
        dw: &mut dw,
    };
    let out = ce_rows(
        rt,
        &mut ws,
        GatherSource::F32(&hb),
        &wb,
        (rows, targets),
        (operands, engine),
        Some(grads),
    )?;
    Ok((out, rt.download(&dh)?, rt.download(&dw)?))
}

fn loss_bits_check(name: &str, got: &CeOutput, want: &CeOutput) -> Check {
    let mut g: Vec<u64> = got.per_row.iter().map(|v| v.to_bits()).collect();
    g.push(got.loss.to_bits());
    let mut w: Vec<u64> = want.per_row.iter().map(|v| v.to_bits()).collect();
    w.push(want.loss.to_bits());
    if g == w {
        Check::pass(
            name,
            format!(
                "{} per-row losses and the loss bit-identical",
                want.per_row.len()
            ),
        )
    } else {
        Check::fail(
            name,
            format!(
                "losses differ: got {:?} / {}, want {:?} / {}",
                got.per_row, got.loss, want.per_row, want.loss
            ),
        )
    }
}

fn loss_close_check(name: &str, got: &CeOutput, want: &CeOutput) -> Check {
    let (abs, rel) = CE_LOSS_TOL;
    let mut worst = 0.0f64;
    for (g, w) in got
        .per_row
        .iter()
        .chain([&got.loss])
        .zip(want.per_row.iter().chain([&want.loss]))
    {
        worst = worst.max((g - w).abs() / (abs + rel * w.abs()));
    }
    let same_len = got.per_row.len() == want.per_row.len();
    let detail = format!("worst |err| / (1e-5 + 1e-5|ref|) = {worst:.3e} ({CE_SOURCE})");
    if same_len && worst <= 1.0 {
        Check::pass(name, detail)
    } else {
        Check::fail(name, detail)
    }
    .with("worst_ratio", worst)
}

/// K10: the chunked cross-entropy and its gradients. On the FFMA engine the
/// device equals the mirror bitwise (ExactF32 and bf16 operands); on
/// cuBLAS's bf16 engine it is held to tessl's bounds of the same mirror.
/// Shapes: a chunk of 64 over 300 columns (a 44-wide tail) with a dominant
/// logit in the last chunk, 16,384 columns in two production chunks, and the
/// 2B's 248,320 columns (30 chunks of 8,192 and a 2,560 tail).
pub fn k10_checks(rt: &CudaRuntime) -> Vec<Check> {
    let cases = [
        CeCase {
            tag: "v300_c64_exact",
            n: 5,
            hidden: 64,
            vocab: 300,
            chunk: 64,
            operands: Operands::ExactF32,
            engine: Bf16Engine::Ffma,
        },
        CeCase {
            tag: "v300_c64_bf16_ffma",
            n: 5,
            hidden: 64,
            vocab: 300,
            chunk: 64,
            operands: Operands::Bf16,
            engine: Bf16Engine::Ffma,
        },
        CeCase {
            tag: "v300_c64_bf16_cublas",
            n: 5,
            hidden: 64,
            vocab: 300,
            chunk: 64,
            operands: Operands::Bf16,
            engine: Bf16Engine::Cublas,
        },
        CeCase {
            tag: "v16384_exact",
            n: 4,
            hidden: 32,
            vocab: 16_384,
            chunk: CE_CHUNK,
            operands: Operands::ExactF32,
            engine: Bf16Engine::Ffma,
        },
        CeCase {
            tag: "v248320_exact",
            n: 3,
            hidden: 16,
            vocab: 248_320,
            chunk: CE_CHUNK,
            operands: Operands::ExactF32,
            engine: Bf16Engine::Ffma,
        },
    ];
    let mut out = Vec::new();
    for c in cases {
        let name = format!("k10.ce_rows.{}", c.tag);
        out.extend(case(&name, || {
            let t_rows = c.n + 4;
            let h_win = Window {
                ld: c.hidden + 16,
                off: 8,
            };
            let plan = CeRowsPlan::new(
                (c.n, c.hidden, c.vocab),
                c.chunk,
                (t_rows, h_win),
                Reduction::Mean,
            )?;
            let (hu, vu) = (len(c.hidden)?, len(c.vocab)?);
            let h = splitmix_f32(0x10a1, len(t_rows)? * len(h_win.ld)?, 1.0);
            let mut w = splitmix_f32(0x10a2, vu * hu, 0.25);
            // Rows repeat; one target sits in the last column, whose head row
            // is planted to dominate its hidden row's logits there.
            let rows: Vec<u32> = (0..c.n).map(|i| ((i * 3) % t_rows) as u32).collect();
            let mut targets: Vec<u32> = (0..c.n)
                .map(|i| ((i * 7919 + 11) % c.vocab) as u32)
                .collect();
            targets[0] = (c.vocab - 1) as u32;
            let src = rows[0] as usize * len(h_win.ld)? + len(h_win.off)?;
            for k in 0..hu {
                w[(vu - 1) * hu + k] = 4.0 * h[src + k];
            }
            let mirror = ce_rows_mirror(&plan, &h, &w, (&rows, &targets), c.operands, Some(1.0))?;
            let want_dh = mirror.dh.clone().unwrap_or_default();
            let want_dw = mirror.dw.clone().unwrap_or_default();
            let first = ce_device(
                rt,
                plan,
                (&h, &w),
                (&rows, &targets),
                (c.operands, c.engine),
            )?;
            let second = ce_device(
                rt,
                plan,
                (&h, &w),
                (&rows, &targets),
                (c.operands, c.engine),
            )?;
            let bitwise = c.engine == Bf16Engine::Ffma || c.operands == Operands::ExactF32;
            let mut checks = Vec::new();
            if bitwise {
                checks.push(loss_bits_check(
                    &format!("{name}.loss.vs_mirror"),
                    &first.0,
                    &mirror.out,
                ));
                checks.push(bitwise_check(
                    &format!("{name}.dh.vs_mirror"),
                    diff_bits_f32(&first.1, &want_dh),
                    want_dh.len(),
                ));
                checks.push(bitwise_check(
                    &format!("{name}.dw.vs_mirror"),
                    diff_bits_f32(&first.2, &want_dw),
                    want_dw.len(),
                ));
            } else {
                checks.push(loss_close_check(
                    &format!("{name}.loss.vs_mirror"),
                    &first.0,
                    &mirror.out,
                ));
                checks.push(peak_check(
                    &format!("{name}.dh.vs_mirror"),
                    &first.1,
                    &f64s(&want_dh),
                    CE_BF16_GRAD_PEAK_REL,
                    CE_SOURCE,
                ));
                checks.push(peak_check(
                    &format!("{name}.dw.vs_mirror"),
                    &first.2,
                    &f64s(&want_dw),
                    CE_BF16_GRAD_PEAK_REL,
                    CE_SOURCE,
                ));
            }
            checks.push(loss_bits_check(
                &format!("{name}.loss.repeat"),
                &second.0,
                &first.0,
            ));
            checks.push(bitwise_check(
                &format!("{name}.dh.repeat"),
                diff_bits_f32(&second.1, &first.1),
                first.1.len(),
            ));
            checks.push(bitwise_check(
                &format!("{name}.dw.repeat"),
                diff_bits_f32(&second.2, &first.2),
                first.2.len(),
            ));
            Ok(checks)
        }));
    }
    out
}
