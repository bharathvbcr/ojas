//! K3: the `published` GDN gates (rule 9), `g = -exp(A_log) softplus(a +
//! dt_bias)` and `beta = sigmoid(b)` read from the fused projection, forward
//! and backward.
//!
//! - **Host** (run everywhere): the host mirror (`gates_published::*_mirror`)
//!   against L-cuda-oracle's float64 `published` reference on the torch
//!   golden's f32-rounded inputs, and at the 2B's 16 heads over two row
//!   blocks with softplus' linear branch and an `a` of 120 (where `e^x`
//!   overflows f32).
//! - **Device** (`--features cuda`, `#[ignore]`: an sm_90 GPU; on the Mac they
//!   compile and are NOT RUN): the kernels against the same reference, bit for
//!   bit against the mirror and a repeat run, and rung a's checks.
//!
//! Bound, written before any run: every output, forward and backward, within
//! `1e-4 * max|ref|`, tessl's bound for its gates kernels
//! (`tessl/tests/qwen35_bwd.rs:43-66`, applied to g, beta, da, db, dA_log and
//! ddt_bias at `:1296-1307`).

mod device_small_common;
mod reference;

use device_small_common::{f32s, golden, wide, Goldens, BWD_PEAK_REL, BWD_PEAK_SOURCE};
use ojas_qwen35_cuda::check::Check;
use ojas_qwen35_cuda::gates_published::{
    gates_published_bwd_mirror, gates_published_fwd_mirror, GatesPublishedPlan,
};
use ojas_qwen35_cuda::inputs::splitmix_f32;
use ojas_qwen35_cuda::small_common::peak_check;
use reference::gates_published::{gdn_gates_published_bwd_f64, gdn_gates_published_fwd_f64};

const GOLDENS: Goldens = Goldens(&[
    golden!("gates_published_a"),
    golden!("gates_published_b"),
    golden!("gates_published_a_log"),
    golden!("gates_published_dt_bias"),
    golden!("gates_published_dg"),
    golden!("gates_published_dbeta"),
]);

/// One published-gates case: the fused projection `p` (`[other | b | other |
/// a | other]`), f32 inputs, and the reference on them.
struct GatesCase {
    plan: GatesPublishedPlan,
    p: Vec<f32>,
    a_log: Vec<f32>,
    dt: Vec<f32>,
    dg: Vec<f32>,
    dbeta: Vec<f32>,
    dp0: Vec<f32>,
    want: [Vec<f64>; 6],
}

impl GatesCase {
    /// `a`, `b`, `dg`, `dbeta` are `[rows, heads]`.
    fn new(rows: usize, heads: usize, (a, b): (Vec<f32>, Vec<f32>), rest: [Vec<f32>; 4]) -> Self {
        let [a_log, dt, dg, dbeta] = rest;
        let (b_off, a_off) = (2, heads + 5);
        let ld = a_off + heads + 3;
        let plan = GatesPublishedPlan::new(
            rows as u64,
            heads as u64,
            ld as u64,
            (a_off as u64, b_off as u64),
        )
        .expect("plan");
        let mut p = splitmix_f32(0x3c1, rows * ld, 5.0);
        for r in 0..rows {
            p[r * ld + a_off..][..heads].copy_from_slice(&a[r * heads..][..heads]);
            p[r * ld + b_off..][..heads].copy_from_slice(&b[r * heads..][..heads]);
        }
        let (g, beta) = gdn_gates_published_fwd_f64(
            &wide(&a),
            &wide(&b),
            &wide(&a_log),
            &wide(&dt),
            rows,
            heads,
        );
        let gr = gdn_gates_published_bwd_f64(
            &wide(&a),
            &wide(&b),
            &wide(&a_log),
            &wide(&dt),
            &wide(&dg),
            &wide(&dbeta),
            rows,
            heads,
        );
        GatesCase {
            plan,
            p,
            a_log,
            dt,
            dg,
            dbeta,
            dp0: splitmix_f32(0x3c2, rows * ld, 1.0),
            want: [g, beta, gr.da, gr.db, gr.da_log, gr.ddt_bias],
        }
    }

    fn golden() -> Self {
        let (shape, a) = GOLDENS.shaped("gates_published_a");
        let get = |n: &str| f32s(&GOLDENS.f64s(&format!("gates_published_{n}")));
        GatesCase::new(
            shape[0],
            shape[1],
            (f32s(&a), get("b")),
            [get("a_log"), get("dt_bias"), get("dg"), get("dbeta")],
        )
    }

    /// The 2B's 16 value heads over 300 rows (two 256-row weight-gradient
    /// blocks), with tessl's run_gates extremes (`tessl/tests/qwen35_bwd.rs:1221-1233`).
    fn two_b() -> Self {
        let (rows, heads) = (300, 16);
        let mut a = splitmix_f32(0x3d1, rows * heads, 12.0);
        for r in (0..rows).step_by(3) {
            a[r * heads] = 26.0;
        }
        for r in (1..rows).step_by(5) {
            a[r * heads + heads - 1] = -24.0;
        }
        a[(rows - 1) * heads] = 120.0;
        let dt: Vec<f32> = splitmix_f32(0x3d3, heads, 1.0)
            .iter()
            .map(|v| v - 3.0)
            .collect();
        GatesCase::new(
            rows,
            heads,
            (a, splitmix_f32(0x3d2, rows * heads, 12.0)),
            [
                splitmix_f32(0x3d4, heads, 0.8),
                dt,
                splitmix_f32(0x3d5, rows * heads, 1.0),
                splitmix_f32(0x3d6, rows * heads, 1.0),
            ],
        )
    }

    /// `[rows, heads]` from the a or b columns of `dp`.
    fn cols(&self, dp: &[f32], off: u64) -> Vec<f32> {
        let (rows, heads, ld) = (
            self.plan.rows as usize,
            self.plan.heads as usize,
            self.plan.ld as usize,
        );
        (0..rows)
            .flat_map(|r| dp[r * ld + off as usize..][..heads].to_vec())
            .collect()
    }

    /// `(g, beta, dp, dA_log, ddt_bias)` judged against the reference.
    fn judge(&self, label: &str, out: &[Vec<f32>; 5]) -> Vec<Check> {
        let [g, beta, dp, dal, ddt] = out;
        let da = self.cols(dp, self.plan.a_off);
        let db = self.cols(dp, self.plan.b_off);
        let got: [(&str, &[f32]); 6] = [
            ("g", g),
            ("beta", beta),
            ("da", &da),
            ("db", &db),
            ("dA_log", dal),
            ("ddt_bias", ddt),
        ];
        got.iter()
            .zip(&self.want)
            .map(|((n, g), w)| {
                peak_check(&format!("{label} {n}"), g, w, BWD_PEAK_REL, BWD_PEAK_SOURCE)
            })
            .collect()
    }

    fn mirror(&self) -> [Vec<f32>; 5] {
        let (g, beta) = gates_published_fwd_mirror(&self.plan, &self.p, &self.a_log, &self.dt)
            .expect("fwd mirror");
        let mut dp = self.dp0.clone();
        let gr = gates_published_bwd_mirror(
            &self.plan,
            (&self.p, &self.a_log, &self.dt),
            (&self.dg, &self.dbeta),
            &mut dp,
        )
        .expect("bwd mirror");
        [g, beta, dp, gr.da_log, gr.ddt_bias]
    }
}

#[test]
fn the_embedded_published_goldens_are_l_cuda_oracles_pinned_bytes() {
    assert_eq!(GOLDENS.verify(), 6);
}

#[test]
fn the_published_mirror_matches_the_reference_on_the_torch_golden() {
    let case = GatesCase::golden();
    let out = case.mirror();
    // The mirror leaves dp's other columns alone.
    let (ld, heads) = (case.plan.ld as usize, case.plan.heads as usize);
    for (i, (&got, &was)) in out[2].iter().zip(&case.dp0).enumerate() {
        let c = i % ld;
        let in_a = (case.plan.a_off as usize..case.plan.a_off as usize + heads).contains(&c);
        let in_b = (case.plan.b_off as usize..case.plan.b_off as usize + heads).contains(&c);
        if !(in_a || in_b) {
            assert_eq!(
                got.to_bits(),
                was.to_bits(),
                "dp[{i}] outside a and b changed"
            );
        }
    }
    device_small_common::assert_pass(&case.judge("mirror published golden", &out));
}

#[test]
fn the_published_mirror_matches_the_reference_at_the_2b_heads_with_extremes() {
    let case = GatesCase::two_b();
    device_small_common::assert_pass(&case.judge("mirror published 300x16", &case.mirror()));
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use device_small_common::{assert_bits, assert_pass, runtime};
    use ojas_qwen35_cuda::gates_published_cuda::{gates_published, gates_published_bwd};
    use ojas_qwen35_cuda::runtime::CudaRuntime;
    use ojas_qwen35_cuda::small_smoke::k3_checks;

    fn run(rt: &CudaRuntime, c: &GatesCase) -> [Vec<f32>; 5] {
        let n = c.dg.len();
        let heads = c.a_log.len();
        let nan = |len: usize| {
            rt.upload(&vec![f32::NAN; len], "k3 sentinel")
                .expect("sentinel")
        };
        let p = rt.upload(&c.p, "p").expect("p");
        let al = rt.upload(&c.a_log, "A_log").expect("A_log");
        let dt = rt.upload(&c.dt, "dt_bias").expect("dt_bias");
        let (mut g, mut beta) = (nan(n), nan(n));
        gates_published(rt, &c.plan, (&p, &al, &dt), &mut g, &mut beta).expect("published fwd");
        let dg = rt.upload(&c.dg, "dg").expect("dg");
        let db = rt.upload(&c.dbeta, "dbeta").expect("dbeta");
        let mut dp = rt.upload(&c.dp0, "dp").expect("dp");
        let mut part = nan(c.plan.part_len() as usize);
        let (mut dal, mut ddt) = (nan(heads), nan(heads));
        gates_published_bwd(
            rt,
            &c.plan,
            (&p, &al, &dt),
            (&dg, &db),
            &mut dp,
            &mut part,
            (&mut dal, &mut ddt),
        )
        .expect("published bwd");
        [
            rt.download(&g).expect("g"),
            rt.download(&beta).expect("beta"),
            rt.download(&dp).expect("dp"),
            rt.download(&dal).expect("dA_log"),
            rt.download(&ddt).expect("ddt_bias"),
        ]
    }

    fn check(rt: &CudaRuntime, label: &str, c: &GatesCase) -> Vec<Check> {
        let got = run(rt, c);
        let again = run(rt, c);
        let want = c.mirror();
        for (i, name) in ["g", "beta", "dp", "dA_log", "ddt_bias"].iter().enumerate() {
            assert_bits(&format!("{label} {name} vs mirror"), &got[i], &want[i]);
            assert_bits(&format!("{label} {name} repeat"), &again[i], &got[i]);
        }
        c.judge(label, &got)
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_published_device_gates_match_the_reference_and_the_mirror_bitwise() {
        let rt = runtime();
        let mut checks = check(&rt, "device published golden", &GatesCase::golden());
        checks.extend(check(&rt, "device published 300x16", &GatesCase::two_b()));
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn rung_a_k3_published_checks_pass() {
        assert_pass(&k3_checks(&runtime()));
    }
}
