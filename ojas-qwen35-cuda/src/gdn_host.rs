//! K2(i) on the host: a **bitwise mirror** of the published-rule GDN kernels
//! (`crate::gdn_kernels`), the test cases they run on, and the bounds they
//! are judged by.
//!
//! The kernels round every float operation of the scan explicitly, and their
//! decay `exp(g)` is `qd_exp_nonpos` from the crate's activation prelude,
//! whose host twin is [`crate::k8_act::exp_nonpos_f32`]; so this file
//! reproduces their bits with `f32` arithmetic (Rust never contracts
//! `a * b + c`, and `f32::mul_add` is one fused rounding:
//! `host_ref::tests::mul_add_is_fused_on_this_host`). It
//! follows the kernels' structure exactly: per (sequence, head, value slice)
//! block, 128 rows of state, the warp-butterfly column sums
//! ([`warp_sum`], [`block_colsum`]), the checkpoint every 64 tokens, the
//! chunked reverse recurrence, and the finish kernel's slice-order sums. Two
//! consequences:
//! - the whole algorithm (indexing, variable lengths, checkpoints, the
//!   backward) is tested on the Mac against L-cuda-oracle's float64
//!   reference, at the device tests' bounds, before any GPU runs
//!   (`tests/device_gdn_published_mirror.rs`);
//! - on the device, every output is checked **bit for bit** against this
//!   mirror, in addition to the float64 bound (`tests/device_gdn_published.rs`,
//!   `crate::gdn_smoke`).
//!
//! NaN payloads are not mirrored: CUDA's float add returns a canonical NaN,
//! the host's propagates an operand's. The cases here are finite.

use crate::check::{tolerance_vs_f64, TolReport};
use crate::error::CudaError;
use crate::gdn_plan::{GdnPublishedPlan, GDN_BV, GDN_CKPT, GDN_DK, GDN_L2_EPS};
use crate::inputs::splitmix_f32;
use crate::k8_act::exp_nonpos_f32;

const DK: usize = GDN_DK;
const BV: usize = GDN_BV;
const WARP: usize = 32;

/// The bounds every K2(i) output is judged by, on the host mirror and on the
/// device. Written before either ran.
pub mod published_bounds {
    /// Every f32 output (o, the final state, the checkpoints, dq, dk, dv, dg,
    /// dbeta, ds0) against L-cuda-oracle's float64 reference on the same
    /// inputs: `max |got - want| <= 1e-4 * max |want|`, per tensor.
    ///
    /// - **Source:** tessl's own Metal-kernel-vs-float64 bound,
    ///   `tessl/tests/gdn_train.rs:227-249` (`rel <= 1e-4` at `:240-241`),
    ///   over the same five chunk edges (`:365-371`) and the 2B-heads case
    ///   (`:378-382`). tessl's kernel is this algorithm in f32.
    /// - **Against the f32 accumulation width:** u = 2^-24 = 5.96e-8. The
    ///   widest single reduction is a 128-row column sum, here a fixed tree of
    ///   depth 7 (5 butterfly levels + 2), so its rounding error is at most
    ///   `7u ≈ 4.2e-7` of the sum of magnitudes (pairwise summation's bound);
    ///   the backward's partials add a 16-term sequential sum per row and a
    ///   slice-order sum of `Dv/16 <= 8` terms (`(16 + 8) u ≈ 1.4e-6`). Over
    ///   T tokens the recurrence is a contraction (`exp(g) <= 1`; `I - beta
    ///   k^ k^T` has eigenvalues in `[1 - beta, 1]` for unit `k^`), so
    ///   per-token errors do not grow; their worst-case linear sum
    ///   `gamma_T = T u` reaches 4.9e-4 at T = 8191, above this bound, but the
    ///   observed f32 error of the same operator on tessl's published corpus
    ///   is 11 u = 6.6e-7 at T = 8191 and 24.7 u = 1.5e-6 at worst
    ///   (`tessl/tests/gdn_fixtures.rs:36-47`): a 1e-4 bound has over 60x
    ///   headroom on the observed error while still failing the published vs
    ///   repo rule difference (1.8e-2 to 3.6e-2 on the oracle's goldens,
    ///   2.0e-1 on the corpus) by over two orders of magnitude.
    pub const REL_OF_MAX: f64 = 1e-4;

    /// The forward output against the corpus's own float64 golden
    /// (`gdn_published_*_y_seq_f64.npy`), after undoing the seam's q
    /// normalization in float64, on the same `1e-4 * max |golden|`.
    ///
    /// The corpus's operator differs from the seam's by one known term: its
    /// `k` is stored already l2-normalized, and the seam normalizes again,
    /// `k^ = k / sqrt(|k|^2 + 1e-6)`, a factor `1 - 5e-7` (to within the f32
    /// storage's `|k|^2 = 1 +- 16u`). That is a systematic relative change of
    /// at most ~1.1e-6 in `beta k^ k^T` and in the write `k^ delta^T`: a
    /// perturbation of the coefficients of a contraction, so the output moves
    /// by the same order (~1e-6), not by `T` times it. 1e-4 leaves two orders
    /// of magnitude over that offset plus the f32 error above.
    pub const CORPUS_GOLDEN_REL_OF_MAX: f64 = 1e-4;

    /// The decay's exponential against `f64::exp`, in ulps of the f32
    /// result: at most 2, the documented maximum of CUDA's own `expf`
    /// (CUDA C Programming Guide, single-precision `expf`: 2 ulp), so the
    /// IEEE-only replacement does not loosen the kernel's exponential.
    /// Pre-registered by this lane for its own exp; the decay now uses
    /// `crate::k8_act::exp_nonpos_f32` (the crate's one exp, by the lead's
    /// ruling), whose test `exp_on_the_nonpositive_range_measured_for_the_gdn_decay`
    /// asserts this same 2 ulp on `[-104, 0]` (L-cuda-M1 measured 0.918).
    pub const EXP_MAX_ULP: f64 = 2.0;
}

/// `1 / sqrt(x + 1e-6)`: the device's `qd_gdn_rnorm`.
pub fn rnorm(sumsq: f32) -> f32 {
    1.0 / (sumsq + GDN_L2_EPS).sqrt()
}

/// `1 / sqrt(128)`, as the kernels form it.
pub fn q_scale() -> f32 {
    1.0 / (DK as f32).sqrt()
}

/// The value every lane holds after the device's butterfly
/// `v += shfl_xor(v, off)` for `off = 16, 8, 4, 2, 1`. Lane `l < off` adds
/// lane `l + off`'s value at each level; the other lanes hold the same bits
/// (IEEE addition is commutative), so lane 0's tree is every lane's.
pub fn warp_sum(lanes: &[f32; WARP]) -> f32 {
    let mut s = *lanes;
    for off in [16usize, 8, 4, 2, 1] {
        for l in 0..off {
            s[l] += s[l + off];
        }
    }
    s[0]
}

/// The device's `qd_gdn_colsum<N>`: each column over 128 threads, as four
/// warp butterflies then `(w0 + w1) + (w2 + w3)`.
pub fn block_colsum<const N: usize>(p: &[[f32; N]; DK]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for (n, slot) in out.iter_mut().enumerate() {
        let mut w = [0.0f32; 4];
        for (wi, wv) in w.iter_mut().enumerate() {
            let mut lanes = [0.0f32; WARP];
            for (l, lane) in lanes.iter_mut().enumerate() {
                *lane = p[wi * WARP + l][n];
            }
            *wv = warp_sum(&lanes);
        }
        *slot = (w[0] + w[1]) + (w[2] + w[3]);
    }
    out
}

/// The operands, in [`crate::gdn_plan`]'s packed layout.
#[derive(Clone, Copy, Debug)]
pub struct GdnPublishedInputs<'a> {
    /// `[N, H, 128]`.
    pub q: &'a [f32],
    /// `[N, H, 128]`.
    pub k: &'a [f32],
    /// `[N, H, Dv]`.
    pub v: &'a [f32],
    /// The log decay, `[N, H]`.
    pub g: &'a [f32],
    /// `[N, H]`.
    pub beta: &'a [f32],
    /// The initial state `[B, H, 128, Dv]`; zeros when `None`.
    pub s0: Option<&'a [f32]>,
}

fn want_len(op: &str, name: &str, got: usize, want: usize) -> Result<(), CudaError> {
    if got == want {
        Ok(())
    } else {
        Err(CudaError::invalid(
            op,
            format!("{name} has {got} elements, the plan needs {want}"),
        ))
    }
}

impl GdnPublishedInputs<'_> {
    /// Every length against the plan.
    pub fn check(&self, plan: &GdnPublishedPlan, op: &str) -> Result<(), CudaError> {
        want_len(op, "q", self.q.len(), plan.qk_len())?;
        want_len(op, "k", self.k.len(), plan.qk_len())?;
        want_len(op, "v", self.v.len(), plan.v_len())?;
        want_len(op, "g", self.g.len(), plan.gate_len())?;
        want_len(op, "beta", self.beta.len(), plan.gate_len())?;
        if let Some(s0) = self.s0 {
            want_len(op, "s0", s0.len(), plan.state_len())?;
        }
        Ok(())
    }
}

/// Every output of a forward and backward, in the plan's layouts. The same
/// type holds the device's results, so the two compare field by field.
#[derive(Clone, Debug, PartialEq)]
pub struct GdnPublishedOutputs {
    /// `[N, H, Dv]`.
    pub o: Vec<f32>,
    /// The final state `[B, H, 128, Dv]`.
    pub s_fin: Vec<f32>,
    /// `[sum NC_b, H, 128, Dv]` in the plan's per-sequence blocks.
    pub ckpt: Vec<f32>,
    /// `[N, H, 128]`.
    pub dq: Vec<f32>,
    /// `[N, H, 128]`.
    pub dk: Vec<f32>,
    /// `[N, H, Dv]`.
    pub dv: Vec<f32>,
    /// `[N, H]`.
    pub dg: Vec<f32>,
    /// `[N, H]`.
    pub dbeta: Vec<f32>,
    /// `[B, H, 128, Dv]`, exactly when the forward had `s0`.
    pub ds0: Option<Vec<f32>>,
}

impl GdnPublishedOutputs {
    /// `(name, values)` for every tensor present, in a fixed order.
    pub fn named(&self) -> Vec<(&'static str, &[f32])> {
        let mut v: Vec<(&'static str, &[f32])> = vec![
            ("o", &self.o),
            ("s_fin", &self.s_fin),
            ("ckpt", &self.ckpt),
            ("dq", &self.dq),
            ("dk", &self.dk),
            ("dv", &self.dv),
            ("dg", &self.dg),
            ("dbeta", &self.dbeta),
        ];
        if let Some(d) = &self.ds0 {
            v.push(("ds0", d));
        }
        v
    }
}

/// What the forward produces.
#[derive(Clone, Debug, PartialEq)]
pub struct MirrorForward {
    /// `[N, H, Dv]`.
    pub o: Vec<f32>,
    /// `[B, H, 128, Dv]`.
    pub s_fin: Vec<f32>,
    /// The checkpoints.
    pub ckpt: Vec<f32>,
}

/// The forward kernel, bit for bit.
pub fn gdn_published_fwd_mirror(
    plan: &GdnPublishedPlan,
    x: &GdnPublishedInputs<'_>,
) -> Result<MirrorForward, CudaError> {
    const OP: &str = "gdn_published_fwd_mirror";
    x.check(plan, OP)?;
    let (h_n, dv) = (plan.heads(), plan.v_dim());
    let scale = q_scale();
    let mut o = vec![0.0f32; plan.v_len()];
    let mut s_fin = vec![0.0f32; plan.state_len()];
    let mut ckpt = vec![0.0f32; plan.ckpt_len()];
    let mut p = [[0.0f32; BV + 2]; DK];
    let mut po = [[0.0f32; BV]; DK];
    for b in 0..plan.batch() {
        for h in 0..h_n {
            for sl in 0..plan.slices() {
                let j0 = sl * BV;
                let srow = plan.state_offset(b, h);
                let mut s = [[0.0f32; BV]; DK];
                if let Some(s0) = x.s0 {
                    for (i, row) in s.iter_mut().enumerate() {
                        row.copy_from_slice(&s0[srow + i * dv + j0..srow + i * dv + j0 + BV]);
                    }
                }
                for t in 0..plan.lens()[b] {
                    if t % GDN_CKPT == 0 {
                        let c = plan.ckpt_offset(b, h, t / GDN_CKPT);
                        for (i, row) in s.iter().enumerate() {
                            ckpt[c + i * dv + j0..c + i * dv + j0 + BV].copy_from_slice(row);
                        }
                    }
                    let r = plan.row(b, t, h);
                    let a = exp_nonpos_f32(x.g[r]);
                    let bt = x.beta[r];
                    for i in 0..DK {
                        let (qi, ki) = (x.q[r * DK + i], x.k[r * DK + i]);
                        for j in 0..BV {
                            s[i][j] *= a;
                            p[i][j] = s[i][j] * ki;
                        }
                        p[i][BV] = ki * ki;
                        p[i][BV + 1] = qi * qi;
                    }
                    let tot = block_colsum(&p);
                    let rk = rnorm(tot[BV]);
                    let rq = rnorm(tot[BV + 1]);
                    for i in 0..DK {
                        let kh = x.k[r * DK + i] * rk;
                        let qh = (x.q[r * DK + i] * rq) * scale;
                        for j in 0..BV {
                            let u = x.v[r * dv + j0 + j] - rk * tot[j];
                            s[i][j] += kh * (bt * u);
                            po[i][j] = s[i][j] * qh;
                        }
                    }
                    let out = block_colsum(&po);
                    o[r * dv + j0..r * dv + j0 + BV].copy_from_slice(&out);
                }
                for (i, row) in s.iter().enumerate() {
                    s_fin[srow + i * dv + j0..srow + i * dv + j0 + BV].copy_from_slice(row);
                }
            }
        }
    }
    Ok(MirrorForward { o, s_fin, ckpt })
}

/// What the backward produces.
#[derive(Clone, Debug, PartialEq)]
pub struct MirrorGrads {
    /// `[N, H, 128]`.
    pub dq: Vec<f32>,
    /// `[N, H, 128]`.
    pub dk: Vec<f32>,
    /// `[N, H, Dv]`.
    pub dv: Vec<f32>,
    /// `[N, H]`.
    pub dg: Vec<f32>,
    /// `[N, H]`.
    pub dbeta: Vec<f32>,
    /// `[B, H, 128, Dv]`, exactly when `s0` was given.
    pub ds0: Option<Vec<f32>>,
}

/// The backward and finish kernels, bit for bit, from the forward's
/// checkpoints, `d_o`, and `d_fin` (the final state's gradient) when given.
pub fn gdn_published_bwd_mirror(
    plan: &GdnPublishedPlan,
    x: &GdnPublishedInputs<'_>,
    ckpt: &[f32],
    d_o: &[f32],
    d_fin: Option<&[f32]>,
) -> Result<MirrorGrads, CudaError> {
    const OP: &str = "gdn_published_bwd_mirror";
    x.check(plan, OP)?;
    want_len(OP, "ckpt", ckpt.len(), plan.ckpt_len())?;
    want_len(OP, "d_o", d_o.len(), plan.v_len())?;
    if let Some(d) = d_fin {
        want_len(OP, "d_fin", d.len(), plan.state_len())?;
    }
    let (h_n, dv, rows, ns) = (plan.heads(), plan.v_dim(), plan.rows(), plan.slices());
    let scale = q_scale();
    let mut gdv = vec![0.0f32; plan.v_len()];
    let mut dq_part = vec![0.0f32; plan.qk_part_len()];
    let mut dk_part = vec![0.0f32; plan.qk_part_len()];
    let mut dg_part = vec![0.0f32; plan.gate_part_len()];
    let mut dbeta_part = vec![0.0f32; plan.gate_part_len()];
    let mut ds0 = x.s0.map(|_| vec![0.0f32; plan.state_len()]);
    let mut mine = vec![[[0.0f32; BV]; DK]; GDN_CKPT];
    let mut p18 = [[0.0f32; BV + 2]; DK];
    let mut p16 = [[0.0f32; BV]; DK];
    let mut p1 = [[0.0f32; 1]; DK];
    for b in 0..plan.batch() {
        let t_len = plan.lens()[b];
        let nc = plan.seq_checkpoints(b);
        for h in 0..h_n {
            for sl in 0..ns {
                let j0 = sl * BV;
                let srow = plan.state_offset(b, h);
                let mut ds = [[0.0f32; BV]; DK];
                if let Some(df) = d_fin {
                    for (i, row) in ds.iter_mut().enumerate() {
                        row.copy_from_slice(&df[srow + i * dv + j0..srow + i * dv + j0 + BV]);
                    }
                }
                for cc in (0..nc).rev() {
                    let t0 = cc * GDN_CKPT;
                    let t1 = t_len.min(t0 + GDN_CKPT);
                    let c = plan.ckpt_offset(b, h, cc);
                    let mut s = [[0.0f32; BV]; DK];
                    for (i, row) in s.iter_mut().enumerate() {
                        row.copy_from_slice(&ckpt[c + i * dv + j0..c + i * dv + j0 + BV]);
                    }
                    let mut u_c = [[0.0f32; BV]; GDN_CKPT];
                    let mut rk_c = [0.0f32; GDN_CKPT];
                    let mut rq_c = [0.0f32; GDN_CKPT];
                    // Recompute the chunk's states from its checkpoint.
                    for t in t0..t1 {
                        let lt = t - t0;
                        mine[lt] = s;
                        let r = plan.row(b, t, h);
                        let a = exp_nonpos_f32(x.g[r]);
                        for i in 0..DK {
                            let (qi, ki) = (x.q[r * DK + i], x.k[r * DK + i]);
                            for j in 0..BV {
                                s[i][j] *= a;
                                p18[i][j] = s[i][j] * ki;
                            }
                            p18[i][BV] = ki * ki;
                            p18[i][BV + 1] = qi * qi;
                        }
                        let tot = block_colsum(&p18);
                        let rk = rnorm(tot[BV]);
                        let bt = x.beta[r];
                        for j in 0..BV {
                            u_c[lt][j] = x.v[r * dv + j0 + j] - rk * tot[j];
                        }
                        for (i, row) in s.iter_mut().enumerate() {
                            let kh = x.k[r * DK + i] * rk;
                            for (sij, u) in row.iter_mut().zip(&u_c[lt]) {
                                *sij += kh * (bt * u);
                            }
                        }
                        rk_c[lt] = rk;
                        rq_c[lt] = rnorm(tot[BV + 1]);
                    }
                    // Reverse over the chunk.
                    for t in (t0..t1).rev() {
                        let lt = t - t0;
                        let r = plan.row(b, t, h);
                        let a = exp_nonpos_f32(x.g[r]);
                        let bt = x.beta[r];
                        let mut delta = [0.0f32; BV];
                        let mut dorow = [0.0f32; BV];
                        for j in 0..BV {
                            delta[j] = bt * u_c[lt][j];
                            dorow[j] = d_o[r * dv + j0 + j];
                        }
                        let mut dqh = [0.0f32; DK];
                        let mut kh = [0.0f32; DK];
                        for i in 0..DK {
                            kh[i] = x.k[r * DK + i] * rk_c[lt];
                            let qh = (x.q[r * DK + i] * rq_c[lt]) * scale;
                            for j in 0..BV {
                                let sh = a * mine[lt][i][j];
                                dqh[i] += (sh + kh[i] * delta[j]) * dorow[j];
                            }
                            for j in 0..BV {
                                ds[i][j] += qh * dorow[j];
                                p16[i][j] = ds[i][j] * kh[i];
                            }
                        }
                        let ddelta = block_colsum(&p16);
                        let mut dkh = [0.0f32; DK];
                        for i in 0..DK {
                            let mut dgp = 0.0f32;
                            for j in 0..BV {
                                let sh = a * mine[lt][i][j];
                                dkh[i] += ds[i][j] * delta[j] - (bt * sh) * ddelta[j];
                                let dsh = ds[i][j] - (bt * kh[i]) * ddelta[j];
                                dgp += dsh * sh;
                                ds[i][j] = a * dsh;
                            }
                            p1[i][0] = dgp;
                        }
                        let dgt = block_colsum(&p1);
                        for j in 0..BV {
                            gdv[r * dv + j0 + j] = bt * ddelta[j];
                        }
                        let part = sl * rows + r;
                        dq_part[part * DK..(part + 1) * DK].copy_from_slice(&dqh);
                        dk_part[part * DK..(part + 1) * DK].copy_from_slice(&dkh);
                        let mut db = 0.0f32;
                        for j in 0..BV {
                            db += ddelta[j] * u_c[lt][j];
                        }
                        dbeta_part[part] = db;
                        dg_part[part] = dgt[0];
                    }
                }
                if let Some(d) = ds0.as_mut() {
                    for (i, row) in ds.iter().enumerate() {
                        d[srow + i * dv + j0..srow + i * dv + j0 + BV].copy_from_slice(row);
                    }
                }
            }
        }
    }

    // The finish kernel, one 128-thread block per row.
    let mut gdq = vec![0.0f32; plan.qk_len()];
    let mut gdk = vec![0.0f32; plan.qk_len()];
    let mut gdg = vec![0.0f32; plan.gate_len()];
    let mut gdbeta = vec![0.0f32; plan.gate_len()];
    let mut pq = [[0.0f32; 2]; DK];
    let mut pd = [[0.0f32; 2]; DK];
    let mut sums = [[0.0f32; 2]; DK];
    for r in 0..rows {
        for (i, sum) in sums.iter_mut().enumerate() {
            let at = r * DK + i;
            let (mut dqh, mut dkh) = (0.0f32, 0.0f32);
            for s in 0..ns {
                dqh += dq_part[s * rows * DK + at];
                dkh += dk_part[s * rows * DK + at];
            }
            *sum = [dqh, dkh];
            let (qi, ki) = (x.q[at], x.k[at]);
            pq[i] = [qi * qi, ki * ki];
        }
        let tot = block_colsum(&pq);
        let (rq, rk) = (rnorm(tot[0]), rnorm(tot[1]));
        for i in 0..DK {
            let at = r * DK + i;
            let (yq, yk) = (x.q[at] * rq, x.k[at] * rk);
            let dyq = sums[i][0] * scale;
            pd[i] = [yq * dyq, yk * sums[i][1]];
        }
        let dots = block_colsum(&pd);
        for (i, sum) in sums.iter().enumerate() {
            let at = r * DK + i;
            let (yq, yk) = (x.q[at] * rq, x.k[at] * rk);
            let dyq = sum[0] * scale;
            gdq[at] = rq * (dyq - yq * dots[0]);
            gdk[at] = rk * (sum[1] - yk * dots[1]);
        }
        let (mut sa, mut sb) = (0.0f32, 0.0f32);
        for s in 0..ns {
            sa += dg_part[s * rows + r];
            sb += dbeta_part[s * rows + r];
        }
        gdg[r] = sa;
        gdbeta[r] = sb;
    }
    Ok(MirrorGrads {
        dq: gdq,
        dk: gdk,
        dv: gdv,
        dg: gdg,
        dbeta: gdbeta,
        ds0,
    })
}

/// tessl's test generator `random_f32(n, seed)` (`tessl/tests/common/mod.rs`):
/// ojas's splitmix ([`splitmix_f32`]) after tessl's seed whitening, the same
/// values bit for bit (L-cuda-oracle's port, `tests/reference/rng.rs`, is
/// checked against this in `tests/device_gdn_published_mirror.rs`).
pub fn tessl_random_f32(n: usize, seed: u64) -> Vec<f32> {
    splitmix_f32(seed ^ 0x9e37_79b9_7f4a_7c15, n, 1.0)
}

/// One problem: the plan, its operands, and the backward's incoming
/// gradients.
#[derive(Clone, Debug)]
pub struct GdnPublishedCase {
    /// For reports.
    pub label: String,
    /// The validated shape.
    pub plan: GdnPublishedPlan,
    /// `[N, H, 128]`.
    pub q: Vec<f32>,
    /// `[N, H, 128]`.
    pub k: Vec<f32>,
    /// `[N, H, Dv]`.
    pub v: Vec<f32>,
    /// `[N, H]`.
    pub g: Vec<f32>,
    /// `[N, H]`.
    pub beta: Vec<f32>,
    /// `[B, H, 128, Dv]`.
    pub s0: Option<Vec<f32>>,
    /// `[N, H, Dv]`.
    pub d_o: Vec<f32>,
    /// `[B, H, 128, Dv]`.
    pub d_fin: Option<Vec<f32>>,
}

impl GdnPublishedCase {
    /// tessl's GPU test case `Case::new(b, t, h, dv, seed, with_s0,
    /// with_dfin)` (`tessl/tests/gdn_train.rs:197-214`), bit for bit: log
    /// decays `-0.75 (x + 1) - 1e-3` in [-1.5, 0), betas `0.5 + 0.45 x`,
    /// formed in f32 as tessl does.
    #[allow(clippy::too_many_arguments)]
    pub fn tessl(
        b: usize,
        t: usize,
        h: usize,
        dv: usize,
        seed: u64,
        with_s0: bool,
        with_dfin: bool,
    ) -> Result<Self, CudaError> {
        let plan = GdnPublishedPlan::dense(b, t, h, DK, dv)?;
        let n = b * t * h;
        let state = plan.state_len();
        Ok(GdnPublishedCase {
            label: format!("published B={b} T={t} H={h} Dv={dv} s0={with_s0} dfin={with_dfin}"),
            q: tessl_random_f32(n * DK, seed),
            k: tessl_random_f32(n * DK, seed + 1),
            v: tessl_random_f32(n * dv, seed + 2),
            g: tessl_random_f32(n, seed + 3)
                .iter()
                .map(|&x| -0.75 * (x + 1.0) - 1e-3)
                .collect(),
            beta: tessl_random_f32(n, seed + 4)
                .iter()
                .map(|&x| 0.5 + 0.45 * x)
                .collect(),
            s0: with_s0.then(|| tessl_random_f32(state, seed + 5)),
            d_o: tessl_random_f32(n * dv, seed + 6),
            d_fin: with_dfin.then(|| tessl_random_f32(state, seed + 7)),
            plan,
        })
    }

    /// Sequences of the given lengths in one batch: sequence `i` is tessl's
    /// `B = 1` case at seed `seed + 1000 i`, and the batch is their
    /// concatenation (the plan's layout), so each sequence's `B = 1` results
    /// are exactly a slice of the batch's.
    pub fn varlen(
        lens: &[usize],
        h: usize,
        dv: usize,
        seed: u64,
        with_s0: bool,
        with_dfin: bool,
    ) -> Result<Self, CudaError> {
        let plan = GdnPublishedPlan::new(lens, h, DK, dv)?;
        let mut parts = Vec::with_capacity(lens.len());
        for (i, &t) in lens.iter().enumerate() {
            parts.push(Self::tessl(
                1,
                t,
                h,
                dv,
                seed + 1000 * i as u64,
                with_s0,
                with_dfin,
            )?);
        }
        let cat = |f: &dyn Fn(&GdnPublishedCase) -> &[f32]| {
            parts
                .iter()
                .flat_map(|c| f(c).to_vec())
                .collect::<Vec<f32>>()
        };
        let cat_opt = |f: &dyn Fn(&GdnPublishedCase) -> Option<&[f32]>| {
            parts
                .iter()
                .map(f)
                .collect::<Option<Vec<&[f32]>>>()
                .map(|v| v.concat())
        };
        Ok(GdnPublishedCase {
            label: format!("published varlen {lens:?} H={h} Dv={dv} s0={with_s0} dfin={with_dfin}"),
            q: cat(&|c| &c.q),
            k: cat(&|c| &c.k),
            v: cat(&|c| &c.v),
            g: cat(&|c| &c.g),
            beta: cat(&|c| &c.beta),
            s0: cat_opt(&|c| c.s0.as_deref()),
            d_o: cat(&|c| &c.d_o),
            d_fin: cat_opt(&|c| c.d_fin.as_deref()),
            plan,
        })
    }

    /// Sequence `b` alone, as a `B = 1` case: the slices of this batch.
    pub fn sequence(&self, b: usize) -> Result<Self, CudaError> {
        let p = &self.plan;
        let one = p.sequence(b)?;
        let (dv, dk) = (p.v_dim(), DK);
        Ok(GdnPublishedCase {
            label: format!("{} [sequence {b}]", self.label),
            q: p.split_tokens(&self.q, dk)?[b].to_vec(),
            k: p.split_tokens(&self.k, dk)?[b].to_vec(),
            v: p.split_tokens(&self.v, dv)?[b].to_vec(),
            g: p.split_tokens(&self.g, 1)?[b].to_vec(),
            beta: p.split_tokens(&self.beta, 1)?[b].to_vec(),
            s0: match &self.s0 {
                Some(s) => Some(p.split_states(s)?[b].to_vec()),
                None => None,
            },
            d_o: p.split_tokens(&self.d_o, dv)?[b].to_vec(),
            d_fin: match &self.d_fin {
                Some(s) => Some(p.split_states(s)?[b].to_vec()),
                None => None,
            },
            plan: one,
        })
    }

    /// The forward's operands.
    pub fn inputs(&self) -> GdnPublishedInputs<'_> {
        GdnPublishedInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            s0: self.s0.as_deref(),
        }
    }

    /// One of tessl's published-corpus cases (`tests/fixtures/gdn/`),
    /// embedded in the kernels' shape: `q`, `k` `[B, H, L, D]` zero-padded to
    /// 128 key columns, `v` zero-padded to 16 value columns, transposed to
    /// `[B, L, H, *]`; `g = ln(clamp(alpha, 1e-4, 1))` formed in f64 and
    /// rounded to f32 (the kernel's input); `beta` clamped to `[0, 1]`
    /// (the clamps every corpus manifest declares, tessl
    /// `tests/gdn_fixtures.rs:399-408`). `d_o` is tessl's generator at
    /// `seed` over all 16 value columns; no initial state, no `d_fin`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_published_corpus(
        name: &str,
        bhld: (usize, usize, usize, usize),
        q: &[f32],
        k: &[f32],
        v: &[f32],
        alpha: &[f32],
        beta: &[f32],
        seed: u64,
    ) -> Result<Self, CudaError> {
        const OP: &str = "gdn_published_corpus";
        let (b, h, l, d) = bhld;
        if d == 0 || d > BV {
            return Err(CudaError::invalid(
                OP,
                format!("{name}: corpus head dim {d} does not embed in {BV} value columns"),
            ));
        }
        let n = b * h * l;
        for (what, got, want) in [
            ("q", q.len(), n * d),
            ("k", k.len(), n * d),
            ("v", v.len(), n * d),
            ("alpha", alpha.len(), n),
            ("beta", beta.len(), n),
        ] {
            want_len(OP, what, got, want)?;
        }
        let plan = GdnPublishedPlan::dense(b, l, h, DK, BV)?;
        // [B, H, L, D] -> [B, L, H, width], zero-padded.
        let embed = |x: &[f32], width: usize| {
            let mut out = vec![0.0f32; n * width];
            for bi in 0..b {
                for hi in 0..h {
                    for t in 0..l {
                        let src = ((bi * h + hi) * l + t) * d;
                        let dst = ((bi * l + t) * h + hi) * width;
                        out[dst..dst + d].copy_from_slice(&x[src..src + d]);
                    }
                }
            }
            out
        };
        let gate = |x: &[f32], f: &dyn Fn(f32) -> f32| {
            let mut out = vec![0.0f32; n];
            for bi in 0..b {
                for hi in 0..h {
                    for t in 0..l {
                        out[(bi * l + t) * h + hi] = f(x[(bi * h + hi) * l + t]);
                    }
                }
            }
            out
        };
        Ok(GdnPublishedCase {
            label: format!("published corpus {name} (B={b} H={h} T={l} D={d})"),
            q: embed(q, DK),
            k: embed(k, DK),
            v: embed(v, BV),
            g: gate(alpha, &|a| f64::from(a.clamp(1e-4, 1.0)).ln() as f32),
            beta: gate(beta, &|x| x.clamp(0.0, 1.0)),
            s0: None,
            d_o: tessl_random_f32(n * BV, seed),
            d_fin: None,
            plan,
        })
    }

    /// The forward output as the corpus's golden states it: the first `d`
    /// value columns, `[B, T, H, d]`, divided by the seam's q scaling
    /// `rq * 1/sqrt(128)` (formed in f64 from this case's f32 `q`), because
    /// the corpus's output is `S^T q` with `q` as stored.
    pub fn corpus_unscaled_output(&self, o: &[f32], d: usize) -> Result<Vec<f64>, CudaError> {
        let p = &self.plan;
        want_len("corpus_unscaled_output", "o", o.len(), p.v_len())?;
        let scale = 1.0 / (DK as f64).sqrt();
        let mut out = Vec::with_capacity(p.rows() * d);
        for r in 0..p.rows() {
            let sumsq: f64 = self.q[r * DK..(r + 1) * DK]
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum();
            let rq = 1.0 / (sumsq + f64::from(GDN_L2_EPS)).sqrt();
            let dv = p.v_dim();
            out.extend(
                o[r * dv..r * dv + d]
                    .iter()
                    .map(|&x| f64::from(x) / (rq * scale)),
            );
        }
        Ok(out)
    }
}

/// The forward and backward mirrors on a case, with the final state always
/// kept: the device path's results in the same shape.
pub fn gdn_published_mirror(case: &GdnPublishedCase) -> Result<GdnPublishedOutputs, CudaError> {
    let x = case.inputs();
    let f = gdn_published_fwd_mirror(&case.plan, &x)?;
    let g = gdn_published_bwd_mirror(&case.plan, &x, &f.ckpt, &case.d_o, case.d_fin.as_deref())?;
    Ok(GdnPublishedOutputs {
        o: f.o,
        s_fin: f.s_fin,
        ckpt: f.ckpt,
        dq: g.dq,
        dk: g.dk,
        dv: g.dv,
        dg: g.dg,
        dbeta: g.dbeta,
        ds0: g.ds0,
    })
}

/// The cases `crate::gdn_smoke` runs on the device (bitwise against the
/// mirror), and which `tests/device_gdn_published_mirror.rs` holds to the
/// float64 bounds on the host: tessl's five chunk edges at its shape and
/// flags (`tessl/tests/gdn_train.rs:365-371`), and one variable-length batch
/// over the same edges with an initial state and a final-state gradient.
pub fn smoke_cases() -> Result<Vec<GdnPublishedCase>, CudaError> {
    let mut out = Vec::new();
    for (t, s0, dfin) in TESSL_EDGES {
        out.push(GdnPublishedCase::tessl(
            2,
            t,
            3,
            32,
            100 + t as u64,
            s0,
            dfin,
        )?);
    }
    out.push(GdnPublishedCase::varlen(
        &[1, 63, 64, 65, 130],
        3,
        32,
        500,
        true,
        true,
    )?);
    Ok(out)
}

/// tessl's chunk-edge cases `(T, with s0, with d_fin)`
/// (`tessl/tests/gdn_train.rs:365-371`).
pub const TESSL_EDGES: [(usize, bool, bool); 5] = [
    (1, false, false),
    (63, true, false),
    (64, false, true),
    (65, true, true),
    (130, true, true),
];

/// One tensor against its float64 reference, as a fraction of the
/// reference's largest magnitude. Fails on a length mismatch, a non-finite
/// value, or `sentinel` left in place (an element never written).
pub fn rel_of_max(
    name: &str,
    got: &[f32],
    want: &[f64],
    sentinel: Option<f32>,
) -> Result<(f64, TolReport), String> {
    if let Some(s) = sentinel {
        if let Some(i) = got.iter().position(|&g| g.to_bits() == s.to_bits()) {
            return Err(format!("{name}[{i}]: never written (sentinel {s:e})"));
        }
    }
    let r = tolerance_vs_f64(got, want).map_err(|e| format!("{name}: {e}"))?;
    if r.nonfinite > 0 {
        return Err(format!("{name}: {} non-finite outputs", r.nonfinite));
    }
    Ok((r.max_abs_err / r.max_abs_ref.max(1e-30), r))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ulps of `got` from the correctly rounded `exact`.
    fn ulps(got: f32, exact: f64) -> f64 {
        let nearest = exact as f32;
        let ulp = if nearest == 0.0 || nearest.is_subnormal() {
            f64::from(f32::from_bits(1))
        } else {
            let up = f32::from_bits(nearest.abs().to_bits() + 1);
            f64::from(up) - f64::from(nearest.abs())
        };
        (f64::from(got) - exact).abs() / ulp
    }

    /// The pre-registered 2 ulp, checked on the exponential the decay
    /// actually calls, over the log decays it sees: every 1021st f32 in
    /// [-104, 0] and a dense sweep of [-30, 0].
    #[test]
    fn the_published_decay_exp_is_within_its_pre_registered_ulps() {
        let mut worst = (0.0f64, 0.0f32);
        let lo = (-104.0f32).to_bits();
        let mut sweep: Vec<f32> = (0..(lo - 0x8000_0000) / 1021)
            .map(|i| f32::from_bits(lo - i * 1021))
            .collect();
        sweep.extend((0..300_000u32).map(|i| -30.0 * (i as f32) / 300_000.0));
        let count = sweep.len();
        for x in sweep {
            let e = ulps(exp_nonpos_f32(x), f64::from(x).exp());
            if e > worst.0 {
                worst = (e, x);
            }
        }
        eprintln!(
            "decay exp: worst {:.3} ulp at x = {:e} over {count} points (bound {})",
            worst.0,
            worst.1,
            published_bounds::EXP_MAX_ULP
        );
        assert!(worst.0 <= published_bounds::EXP_MAX_ULP, "{worst:?}");
    }

    /// `g <= 0` is the operator's precondition. A positive log decay is NaN
    /// in the decay and so in every output it reaches: loud, not a silently
    /// wrong number. The tokens before it are untouched.
    #[test]
    fn a_positive_log_decay_is_loud_in_the_published_mirror() {
        let mut c = GdnPublishedCase::tessl(1, 5, 1, 16, 3, false, false).unwrap();
        c.g[2] = 0.25;
        let f = gdn_published_fwd_mirror(&c.plan, &c.inputs()).unwrap();
        let dv = c.plan.v_dim();
        assert!(f.o[..2 * dv].iter().all(|x| x.is_finite()));
        assert!(
            f.o[2 * dv..].iter().all(|x| x.is_nan()),
            "{:?}",
            &f.o[2 * dv..3 * dv]
        );
        assert!(f.s_fin.iter().all(|x| x.is_nan()));
        c.g[2] = f32::INFINITY;
        let f = gdn_published_fwd_mirror(&c.plan, &c.inputs()).unwrap();
        assert!(f.o[2 * dv..].iter().all(|x| x.is_nan()));
        // g = 0 (no decay) is in the domain.
        c.g[2] = 0.0;
        let f = gdn_published_fwd_mirror(&c.plan, &c.inputs()).unwrap();
        assert!(f.o.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn published_warp_sum_is_the_butterfly_tree_not_a_sequential_sum() {
        // Values whose sum depends on the order: 2^24 + 1 + ... .
        let mut lanes = [1.0f32; 32];
        lanes[0] = 16_777_216.0;
        // The butterfly pairs lane 0 with lane 16 first: 2^24 + 1 rounds to
        // 2^24; the other 30 ones sum exactly to 30 in the tree; 2^24 + 30.
        assert_eq!(warp_sum(&lanes), 16_777_246.0);
        // A sequential sum from lane 0 loses every 1: 2^24.
        assert_eq!(lanes.iter().fold(0.0f32, |a, &x| a + x), 16_777_216.0);
        let p: [[f32; 1]; 128] = std::array::from_fn(|i| [i as f32]);
        assert_eq!(block_colsum(&p), [8128.0]);
    }

    #[test]
    fn published_case_generator_is_on_tessls_unit_grid() {
        let v = tessl_random_f32(1000, 7);
        assert!(v.iter().all(|&x| (-1.0..1.0).contains(&x)));
        assert!(v.iter().all(|&x| ((x + 1.0) * 8_388_608.0).fract() == 0.0));
    }

    #[test]
    fn published_mirror_refuses_misshapen_operands() {
        let c = GdnPublishedCase::tessl(1, 3, 2, 16, 1, false, false).unwrap();
        let mut x = c.inputs();
        x.g = &c.g[1..];
        let e = gdn_published_fwd_mirror(&c.plan, &x)
            .unwrap_err()
            .to_string();
        assert!(e.contains("g has 5 elements"), "{e}");
        let f = gdn_published_fwd_mirror(&c.plan, &c.inputs()).unwrap();
        let e =
            gdn_published_bwd_mirror(&c.plan, &c.inputs(), &f.ckpt[1..], &c.d_o, None).unwrap_err();
        assert!(e.to_string().contains("ckpt"), "{e}");
    }

    /// The `B = 1` slices of a variable-length batch give the same bits as the
    /// batch: the mirror is per block, as the kernels are.
    #[test]
    fn published_mirror_of_a_batch_is_the_concatenation_of_its_sequences() {
        let c = GdnPublishedCase::varlen(&[1, 64, 65], 2, 16, 9, true, true).unwrap();
        let all = gdn_published_mirror(&c).unwrap();
        let p = &c.plan;
        for b in 0..p.batch() {
            let one = gdn_published_mirror(&c.sequence(b).unwrap()).unwrap();
            let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&one.o), bits(p.split_tokens(&all.o, 16).unwrap()[b]));
            assert_eq!(
                bits(&one.dq),
                bits(p.split_tokens(&all.dq, 128).unwrap()[b])
            );
            assert_eq!(bits(&one.dg), bits(p.split_tokens(&all.dg, 1).unwrap()[b]));
            assert_eq!(bits(&one.ckpt), bits(p.split_ckpt(&all.ckpt).unwrap()[b]));
            assert_eq!(
                bits(&one.s_fin),
                bits(p.split_states(&all.s_fin).unwrap()[b])
            );
            assert_eq!(
                bits(one.ds0.as_ref().unwrap()),
                bits(p.split_states(all.ds0.as_ref().unwrap()).unwrap()[b])
            );
        }
    }

    #[test]
    fn published_rel_of_max_names_unwritten_and_non_finite_outputs() {
        let s = -7.25e27f32;
        assert!(rel_of_max("o", &[1.0, s], &[1.0, 2.0], Some(s))
            .unwrap_err()
            .contains("never written"));
        assert!(rel_of_max("o", &[f32::NAN], &[1.0], None)
            .unwrap_err()
            .contains("non-finite"));
        let (rel, _) = rel_of_max("o", &[1.0, 2.0], &[1.0, 2.5], None).unwrap();
        assert_eq!(rel, 0.2);
    }
}
