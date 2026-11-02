//! Does attention backward under a bf16 autocast region leak a nonzero softmax
//! row sum on the paths that form `Δ_i = dO_i · O_i` from the forward's
//! emitted (bf16-rounded) output? The inner kernels do when handed that
//! output (the `inner` arm measures it); `Autocast` therefore recomputes an
//! f32 output for the backward when the saved one is bf16-tagged, and this
//! test asserts that its gradients equal the f32-output floor's, bit for bit.
//!
//! `dS_ij = P_ij (dP_ij - Δ_i)` sums to zero over `j` only when `Δ_i` equals
//! `Σ_j P_ij dP_ij`. That holds for the f32 output `O = P V`, but the tape
//! records the output `Autocast` emits (`round_bf16(O)`) and hands it to the
//! backward (tape.rs `Rec::Sdpa` / `Rec::Attn`), so on the dO·O paths
//! (`attn.rs` `expected = dot_up(gy_row, row(out, ..))`, and the same in
//! `attn/general.rs`) the row sum picks up `dO_i · (O_i - round(O_i))`, about
//! `2^-9 |dO| |O|`, while the true `dS` shrinks like `1 - P_max` in sharp heads.
//! The blocked flash backward (Fast, T > 256) forms `Δ` from its own `P` and is
//! the control.
//!
//! Every arm sees the same bf16-valued operands, so one f64 oracle serves all
//! of them:
//! - `ac`: the real `Autocast<CpuBackend>` forward and backward in a bf16
//!   region (its gradients are emitted, so they are bf16-rounded too);
//! - `inner`: the inner backend's backward on the emitted O, without the
//!   gradient emit, which isolates the leak (what `Autocast` computed before
//!   it recomputed the output);
//! - `floor`: the inner backward on the f32 O (the noise floor);
//! - `floor_emit`: `floor` rounded to bf16, the fair partner of `ac`;
//! - `recompute`: `Autocast`'s `causal_sdpa_backward_recompute` (causal only).
//!
//! Per (path, T, P_max target, seed, arm) it prints the dQ error against the
//! oracle, the row sum of dS in ulps of the row's largest summand (both
//! predicted from the O the kernel consumed and measured from the kernel's own
//! dQ under a common key shift `k -> k + c`), and the dQ drift under that
//! shift. The keys are multiples of 1/8 in [-1, 1) and `c = 16`, so the shifted
//! keys stay exact in bf16 and the shift is the only change.
//!
//! It asserts: the autocast forward ran (its O is the bf16 emit of the inner
//! forward's), its dQ is the bf16 emit of `floor` (no leak: this fails against
//! an `Autocast` that hands the emitted O to the backward), the f32-O floor is
//! accurate, and the flash path does not read O. The leak numbers of `inner`
//! are printed, not asserted.
//!
//! Run: `cargo test -p ojas-cpu --release --test attention_rowsum_autocast --
//! --test-threads=1 --nocapture`.

mod common;

use common::SplitMix64;
use ojas_core::{
    round_f32_to_bf16, AttentionOpts, Autocast, AutocastMode, Backend, Budget, Numerics, Tensor,
};
use ojas_cpu::CpuBackend;

const HEADS: usize = 4;
const DIM: usize = 64;
const SHIFT: f32 = 16.0;
const SEEDS: u64 = 5;
const TARGETS: [f64; 5] = [0.5, 0.9, 0.99, 0.999, 0.9999];
/// f32 bound on the floor's relative dQ error: the row sum's rounding is about
/// `2^-24 sqrt(T) |dP|` against a true `|dS|` of about `(1 - P_max) |dP|`, so
/// about 5e-3 at T = 260 and P_max = 0.9999. The bound leaves 10x headroom.
const FLOOR_BOUND: f64 = 0.05;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Causal,
    General,
}

#[derive(Clone, Copy)]
struct Path {
    name: &'static str,
    kind: Kind,
    numerics: Numerics,
    time: usize,
    /// The backward forms Δ from the saved output (false: from its own P).
    reads_output: bool,
}

const PATHS: [Path; 4] = [
    Path {
        name: "causal_exact",
        kind: Kind::Causal,
        numerics: Numerics::Exact,
        time: 64,
        reads_output: true,
    },
    Path {
        name: "causal_fast_rows",
        kind: Kind::Causal,
        numerics: Numerics::Fast,
        time: 64,
        reads_output: true,
    },
    Path {
        name: "general_exact",
        kind: Kind::General,
        numerics: Numerics::Exact,
        time: 64,
        reads_output: true,
    },
    Path {
        name: "causal_fast_flash",
        kind: Kind::Causal,
        numerics: Numerics::Fast,
        time: 260,
        reads_output: false,
    },
];

fn cpu(numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 28), 2)
        .unwrap()
        .with_numerics(numerics)
}

fn bf16(values: &[f32]) -> Vec<f32> {
    values.iter().copied().map(round_f32_to_bf16).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().copied().map(f32::to_bits).collect()
}

fn causal_opts() -> AttentionOpts<'static> {
    AttentionOpts {
        causal: true,
        ..Default::default()
    }
}

fn scale() -> f64 {
    1.0 / (DIM as f64).sqrt()
}

fn span(time: usize, head: usize, t: usize) -> std::ops::Range<usize> {
    let start = (head * time + t) * DIM;
    start..start + DIM
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

fn median(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// One ulp of an f32 near `x`.
fn ulp(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::from(f32::MIN_POSITIVE);
    }
    2f64.powf(x.log2().floor() - 23.0)
}

/// q (before sharpening), k, v, dO for one draw. k is a multiple of 1/8 so
/// `k + SHIFT` is exact in bf16; v and dO are bf16-valued.
struct Draw {
    q0: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gy: Vec<f32>,
}

fn draw(seed: u64, time: usize) -> Draw {
    let n = HEADS * time * DIM;
    let mut rng = SplitMix64(0xE1_0000_u64 + seed * 0x1_0001 + time as u64);
    let q0 = rng.vec(n, 0.5);
    let k = (0..n).map(|_| (rng.below(16) as f32 - 8.0) / 8.0).collect();
    let v = bf16(&rng.vec(n, 0.5));
    let gy = bf16(&rng.vec(n, 0.5));
    Draw { q0, k, v, gy }
}

fn sharpened(q0: &[f32], alpha: f64) -> Vec<f32> {
    q0.iter()
        .map(|&x| round_f32_to_bf16((alpha * f64::from(x)) as f32))
        .collect()
}

/// Causal softmax row `i` of head `head` in f64.
fn row_probs(q: &[f32], k: &[f32], time: usize, head: usize, i: usize, probs: &mut Vec<f64>) {
    let qi = &q[span(time, head, i)];
    probs.clear();
    probs.extend((0..=i).map(|j| scale() * dot64(qi, &k[span(time, head, j)])));
    let m = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut z = 0.0;
    for p in probs.iter_mut() {
        *p = (*p - m).exp();
        z += *p;
    }
    for p in probs.iter_mut() {
        *p /= z;
    }
}

fn median_pmax(q: &[f32], k: &[f32], time: usize) -> f64 {
    let mut probs = Vec::new();
    let mut maxes = Vec::with_capacity(HEADS * time);
    for head in 0..HEADS {
        for i in 0..time {
            row_probs(q, k, time, head, i, &mut probs);
            maxes.push(probs.iter().copied().fold(0.0, f64::max));
        }
    }
    median(maxes)
}

/// The query scale whose median row P_max is `target`, by bisection on ln α.
fn calibrate(d: &Draw, time: usize, target: f64) -> f64 {
    let (mut lo, mut hi) = (-4.0f64, 10.0f64);
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if median_pmax(&sharpened(&d.q0, mid.exp()), &d.k, time) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi.exp()
}

struct Oracle {
    dq: Vec<f64>,
    /// `Σ_j P_ij dP_ij` per row.
    delta: Vec<f64>,
    /// The row sum's largest summand, `max(max_j P_ij |dP_ij|, |Δ_i|)`: the
    /// scale an f32 sum of the row is rounded at. (`max_j |dS_ij|` is not:
    /// it underflows in sharp rows, which makes an ulp count of it meaningless.)
    term_max: Vec<f64>,
    pmax_median: f64,
}

fn oracle(q: &[f32], k: &[f32], v: &[f32], gy: &[f32], time: usize) -> Oracle {
    let rows = HEADS * time;
    let mut dq = vec![0.0; rows * DIM];
    let mut delta = vec![0.0; rows];
    let mut term_max = vec![0.0; rows];
    let mut pmax = Vec::with_capacity(rows);
    let mut probs = Vec::new();
    let mut g = Vec::new();
    for head in 0..HEADS {
        for i in 0..time {
            let r = head * time + i;
            row_probs(q, k, time, head, i, &mut probs);
            let gyi = &gy[span(time, head, i)];
            g.clear();
            g.extend((0..=i).map(|j| dot64(gyi, &v[span(time, head, j)])));
            let d: f64 = g.iter().zip(&probs).map(|(dp, p)| p * dp).sum();
            term_max[r] = g
                .iter()
                .zip(&probs)
                .fold(d.abs(), |m: f64, (dp, p)| m.max((p * dp).abs()));
            for (gj, &p) in g.iter_mut().zip(&probs) {
                *gj = p * (*gj - d);
            }
            delta[r] = d;
            pmax.push(probs.iter().copied().fold(0.0, f64::max));
            let dqi = &mut dq[span(time, head, i)];
            for (j, &gj) in g.iter().enumerate() {
                for (slot, &kv) in dqi.iter_mut().zip(&k[span(time, head, j)]) {
                    *slot += scale() * gj * f64::from(kv);
                }
            }
        }
    }
    Oracle {
        dq,
        delta,
        term_max,
        pmax_median: median(pmax),
    }
}

struct Inputs {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gy: Tensor,
}

fn tensor(be: &CpuBackend, data: &[f32], time: usize) -> Tensor {
    Tensor::from_f32(data, &[1, HEADS, time, DIM], be.budget()).unwrap()
}

fn forward<B: Backend>(be: &B, kind: Kind, x: &Inputs) -> (Tensor, Tensor) {
    match kind {
        Kind::Causal => be.causal_sdpa_forward(&x.q, &x.k, &x.v, None).unwrap(),
        Kind::General => be
            .attention_forward_saving(&x.q, &x.k, &x.v, causal_opts())
            .unwrap(),
    }
}

fn backward_dq<B: Backend>(be: &B, kind: Kind, x: &Inputs, out: &Tensor, lse: &Tensor) -> Vec<f32> {
    match kind {
        Kind::Causal => be
            .causal_sdpa_backward(&x.q, &x.k, &x.v, out, lse, &x.gy, None)
            .unwrap()
            .0
            .to_f32_vec()
            .unwrap(),
        Kind::General => be
            .attention_backward(&x.q, &x.k, &x.v, out, lse, &x.gy, causal_opts(), false)
            .unwrap()
            .q
            .to_f32_vec()
            .unwrap(),
    }
}

/// Each arm's dQ, plus the output each O-reading arm consumed.
struct Arms {
    names: Vec<&'static str>,
    dq: Vec<Vec<f32>>,
    used_out: Vec<Option<Vec<f32>>>,
}

fn run_arms(path: Path, q: &[f32], k: &[f32], v: &[f32], gy: &[f32]) -> Arms {
    let t = path.time;
    let raw = cpu(path.numerics);
    let ac = Autocast::new(cpu(path.numerics));
    let x = Inputs {
        q: tensor(&raw, q, t),
        k: tensor(&raw, k, t),
        v: tensor(&raw, v, t),
        gy: tensor(&raw, gy, t),
    };

    let (out32, lse) = forward(&raw, path.kind, &x);
    let out32v = out32.to_f32_vec().unwrap();
    let outb = bf16(&out32v);
    let inner = backward_dq(&raw, path.kind, &x, &tensor(&raw, &outb, t), &lse);
    let floor = backward_dq(&raw, path.kind, &x, &out32, &lse);

    let region = ac.autocast_region(AutocastMode::Bf16).unwrap();
    let (out_ac, lse_ac) = forward(&ac, path.kind, &x);
    let dq_ac = backward_dq(&ac, path.kind, &x, &out_ac, &lse_ac);
    let recompute = (path.kind == Kind::Causal).then(|| {
        ac.causal_sdpa_backward_recompute(&x.q, &x.k, &x.v, &x.gy, None)
            .unwrap()
            .0
            .to_f32_vec()
            .unwrap()
    });
    drop(region);

    // The autocast path ran: it emitted round(O) and kept the lse. Its dQ is
    // the bf16 emit of the backward on the f32 O, so the rounded O the tape
    // saved never reached the row sum.
    assert_eq!(
        bits(&out_ac.to_f32_vec().unwrap()),
        bits(&outb),
        "{}: autocast O",
        path.name
    );
    assert_eq!(
        bits(&lse_ac.to_f32_vec().unwrap()),
        bits(&lse.to_f32_vec().unwrap()),
        "{}: autocast lse",
        path.name
    );
    assert_eq!(
        bits(&dq_ac),
        bits(&bf16(&floor)),
        "{}: autocast dQ is not the f32-output floor's (the rounded O leaked into the row sum)",
        path.name
    );
    if !path.reads_output {
        assert_eq!(bits(&inner), bits(&floor), "{}: flash read O", path.name);
    }

    let mut names = vec!["ac", "inner", "floor", "floor_emit"];
    let mut dq = vec![dq_ac, inner, floor.clone(), bf16(&floor)];
    // The flash backward reads no O, so there is no consumed O to predict from.
    let mut used_out = if path.reads_output {
        vec![Some(outb.clone()), Some(outb), Some(out32v), None]
    } else {
        vec![None; 4]
    };
    if let Some(r) = recompute {
        names.push("recompute");
        dq.push(r);
        used_out.push(None);
    }
    Arms {
        names,
        dq,
        used_out,
    }
}

struct Metrics {
    rel_frob: f64,
    row_med: f64,
    row_max: f64,
    /// Predicted from the O the backward consumed, in ulps; NaN if it read none.
    rsum_pred_max: f64,
    rsum_pred_med: f64,
    /// Measured from the kernel's dQ under the key shift, in ulps.
    rsum_meas_max: f64,
    drift: f64,
}

fn row_norm(x: &[f32]) -> f64 {
    x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt()
}

fn metrics(
    time: usize,
    o: &Oracle,
    gy: &[f32],
    dq: &[f32],
    dq_shift: &[f32],
    used_out: Option<&[f32]>,
) -> Metrics {
    let rows = HEADS * time;
    let ref_norms: Vec<f64> = (0..rows)
        .map(|r| {
            o.dq[r * DIM..(r + 1) * DIM]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let floor = 1e-3 * median(ref_norms.iter().copied().filter(|&n| n > 0.0).collect());
    let (mut num, mut den) = (0.0, 0.0);
    let mut row_errs = Vec::new();
    for (r, &rn) in ref_norms.iter().enumerate() {
        let got = &dq[r * DIM..(r + 1) * DIM];
        let want = &o.dq[r * DIM..(r + 1) * DIM];
        let e2: f64 = got
            .iter()
            .zip(want)
            .map(|(&a, &b)| (f64::from(a) - b).powi(2))
            .sum();
        num += e2;
        den += rn * rn;
        if rn > floor {
            row_errs.push(e2.sqrt() / rn);
        }
    }

    let mut pred = Vec::new();
    let mut meas = Vec::new();
    let (mut drift_num, mut drift_den) = (0.0f64, 0.0f64);
    for r in 0..rows {
        let base = &dq[r * DIM..(r + 1) * DIM];
        let shifted = &dq_shift[r * DIM..(r + 1) * DIM];
        let diff: Vec<f32> = shifted.iter().zip(base).map(|(a, b)| a - b).collect();
        drift_num = drift_num.max(row_norm(&diff));
        drift_den = drift_den.max(row_norm(base));
        if o.term_max[r] <= 0.0 {
            continue;
        }
        let u = ulp(o.term_max[r]);
        let summed: f64 = diff.iter().map(|&x| f64::from(x)).sum();
        meas.push((summed / (scale() * f64::from(SHIFT) * DIM as f64)).abs() / u);
        if let Some(out) = used_out {
            let used = dot64(&gy[r * DIM..(r + 1) * DIM], &out[r * DIM..(r + 1) * DIM]);
            pred.push((o.delta[r] - used).abs() / u);
        }
    }
    let max = |xs: &[f64]| xs.iter().copied().fold(f64::NAN, f64::max);
    Metrics {
        rel_frob: (num / den).sqrt(),
        row_med: median(row_errs.clone()),
        row_max: max(&row_errs),
        rsum_pred_max: max(&pred),
        rsum_pred_med: median(pred),
        rsum_meas_max: max(&meas),
        drift: drift_num / drift_den,
    }
}

/// Worst case over seeds at P_max >= 0.999 for one path. The row-sum maxima
/// start at NaN so a path that reads no O reports none.
struct Sharp {
    inner_over_floor: f64,
    ac_over_emit: f64,
    inner_pred: f64,
    inner_meas: f64,
    floor_meas: f64,
}

impl Default for Sharp {
    fn default() -> Self {
        Self {
            inner_over_floor: 0.0,
            ac_over_emit: 0.0,
            inner_pred: f64::NAN,
            inner_meas: f64::NAN,
            floor_meas: f64::NAN,
        }
    }
}

fn ok(pass: bool) -> &'static str {
    if pass {
        "within"
    } else {
        "OUTSIDE"
    }
}

#[test]
fn bf16_emitted_output_row_sum_leak_measurement() {
    eprintln!(
        "path T target pmax_med seed arm rel_frob row_med row_max rsum_pred_max_ulp rsum_pred_med_ulp rsum_meas_max_ulp shift_drift"
    );
    let mut floor_worst = 0.0f64;
    for path in PATHS {
        let t = path.time;
        let ulp_bar = 4.0 + (t as f64).sqrt();
        let mut sharp = Sharp::default();
        for target in TARGETS {
            let mut by_arm: Vec<(&'static str, Vec<f64>)> = Vec::new();
            for seed in 0..SEEDS {
                let d = draw(seed, t);
                let q = sharpened(&d.q0, calibrate(&d, t, target));
                let k_shift: Vec<f32> = d.k.iter().map(|&x| x + SHIFT).collect();
                assert_eq!(bits(&bf16(&k_shift)), bits(&k_shift), "shifted keys exact");
                let o = oracle(&q, &d.k, &d.v, &d.gy, t);
                let base = run_arms(path, &q, &d.k, &d.v, &d.gy);
                let shifted = run_arms(path, &q, &k_shift, &d.v, &d.gy);
                let mut seed_rel = Vec::new();
                for (a, name) in base.names.iter().enumerate() {
                    let m = metrics(
                        t,
                        &o,
                        &d.gy,
                        &base.dq[a],
                        &shifted.dq[a],
                        base.used_out[a].as_deref(),
                    );
                    eprintln!(
                        "{} {t} {target} {:.5} {seed} {name} {:.3e} {:.3e} {:.3e} {:.1} {:.1} {:.1} {:.3e}",
                        path.name,
                        o.pmax_median,
                        m.rel_frob,
                        m.row_med,
                        m.row_max,
                        m.rsum_pred_max,
                        m.rsum_pred_med,
                        m.rsum_meas_max,
                        m.drift
                    );
                    if *name == "floor" {
                        assert!(
                            m.rel_frob < FLOOR_BOUND,
                            "{} T={t} target={target} seed={seed}: f32-O floor dQ error {:.3e}",
                            path.name,
                            m.rel_frob
                        );
                        floor_worst = floor_worst.max(m.rel_frob);
                    }
                    seed_rel.push((*name, m.rel_frob, m.rsum_pred_max, m.rsum_meas_max));
                }
                let get = |n: &str| seed_rel.iter().find(|r| r.0 == n).copied();
                if target >= 0.999 {
                    let (_, inner_rel, inner_pred, inner_meas) = get("inner").unwrap();
                    let (_, floor_rel, _, floor_meas) = get("floor").unwrap();
                    let (_, ac_rel, _, _) = get("ac").unwrap();
                    let (_, emit_rel, _, _) = get("floor_emit").unwrap();
                    sharp.inner_over_floor = sharp.inner_over_floor.max(inner_rel / floor_rel);
                    sharp.ac_over_emit = sharp.ac_over_emit.max(ac_rel / emit_rel);
                    sharp.inner_pred = sharp.inner_pred.max(inner_pred);
                    sharp.inner_meas = sharp.inner_meas.max(inner_meas);
                    sharp.floor_meas = sharp.floor_meas.max(floor_meas);
                }
                for (name, rel, _, _) in seed_rel {
                    match by_arm.iter_mut().find(|(n, _)| *n == name) {
                        Some((_, v)) => v.push(rel),
                        None => by_arm.push((name, vec![rel])),
                    }
                }
            }
            let summary: Vec<String> = by_arm
                .into_iter()
                .map(|(n, v)| format!("{n}={:.3e}", median(v)))
                .collect();
            eprintln!(
                "SUMMARY {} T={t} target={target} median-over-seeds rel_frob: {}",
                path.name,
                summary.join(" ")
            );
        }
        let s = &sharp;
        let parts = [
            s.inner_over_floor <= 10.0,
            s.ac_over_emit <= 10.0,
            s.inner_pred.is_nan() || s.inner_pred <= ulp_bar,
            s.inner_meas <= ulp_bar,
        ];
        eprintln!(
            "VERDICT {} T={t} (P_max>=0.999, worst of {SEEDS} seeds): inner/floor dQ err {:.1}x [{}], ac/floor_emit {:.2}x [{}], inner row sum predicted {:.1} ulps [{}], measured {:.1} ulps [{}] (bar {ulp_bar:.1}; floor measured {:.1} ulps) => refuting condition {}",
            path.name,
            s.inner_over_floor,
            ok(parts[0]),
            s.ac_over_emit,
            ok(parts[1]),
            s.inner_pred,
            ok(parts[2]),
            s.inner_meas,
            ok(parts[3]),
            s.floor_meas,
            if parts.iter().all(|&p| p) { "HOLDS (no defect)" } else { "FAILS (leak measured)" }
        );
    }
    eprintln!("worst f32-O floor rel_frob {floor_worst:.3e} (bound {FLOOR_BOUND})");
}
