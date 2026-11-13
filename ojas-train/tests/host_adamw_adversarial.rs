//! ADVERSARIAL lane for ft-1185100b (b3/hadamw), carried from b3/hadamw-adv (5cd5b67, 392bfb8)
//! and adapted to the checked-token API (`adamw_check` returns a `CheckedUpdate`).
//! ADVERSARIAL lane for ft-1185100b (b3/hadamw): attacks on the claim that a refusal at any
//! entry of a host AdamW step leaves every entry, moment and step count unmoved, and that
//! `adamw_check` / `adamw_entry_check` are exactly `adamw_update` / `adamw_entry_step`'s refusal
//! rule.
//!
//! Every generator is a seeded splitmix64, so a failure names a reproducible case.

use std::path::{Path, PathBuf};

use ojas_train::lappi::adamw::{
    EntryState, Moments, adamw_check, adamw_entry_check, adamw_entry_step, adamw_update,
};
use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::recipe::{BETA1, EPS, EntryHyper, OptimizerRecipe, WEIGHT_DECAY};
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::LrSchedule;
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, CheckpointContext, HiddenGrad, ParamSpec, SequenceJob, StepError,
    StepProvider,
};
use ojas_train::lappi::trainer::{
    BatchCounts, ConsumedBatch, Hooks, HostParams, MicroBatchPlan, MicroLoss, Objective,
    SequenceWork, TrainConfig, TrainError, train,
};

// ---------------------------------------------------------------------------------------------
// Generators.
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}

/// f32 values that stress the refusal rule: zeros of both signs, subnormals, the squares-overflow
/// band around 1.8e20, the f32 extremes, infinities and NaNs with different payloads and signs.
const SPECIAL_F32: &[u32] = &[
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // smallest subnormal
    0x8000_0001,
    0x007F_FFFF, // largest subnormal
    0x0080_0000, // MIN_POSITIVE
    0x7F7F_FFFF, // MAX
    0xFF7F_FFFF, // MIN
    0x7F80_0000, // +inf
    0xFF80_0000, // -inf
    0x7FC0_0000, // quiet NaN
    0xFFC0_0000, // negative quiet NaN
    0x7FC0_0001, // payload NaN
    0x7F80_0001, // signalling NaN
    0xFFBF_FFFF, // negative signalling NaN, max payload
];

fn f32_value(r: &mut Rng) -> f32 {
    match r.below(10) {
        0 => f32::from_bits(r.pick(SPECIAL_F32)),
        1 => r.pick(&[
            1e19f32, 1.7e20, 1.8e20, 1.9e20, 1e21, 3e38, -1e21, 1e-40, -1e-40, 1e-30,
        ]),
        2 => f32::from_bits(r.next() as u32), // any bit pattern
        _ => ((r.unit() * 2.0 - 1.0) * 10f64.powi(r.below(5) as i32 - 3)) as f32,
    }
}

fn f64_edge(r: &mut Rng, ok: &[f64], bad: &[f64], p_bad: f64) -> f64 {
    if r.unit() < p_bad {
        r.pick(bad)
    } else {
        r.pick(ok)
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|f| f.to_bits()).collect()
}

fn mbits(m: &Moments) -> (Vec<u32>, Vec<u32>) {
    (bits(&m.m), bits(&m.v))
}

/// Bitwise view of an entry's state: `NaN != NaN` makes `PartialEq` useless for it.
fn sbits(s: &EntryState) -> ((Vec<u32>, Vec<u32>), u64) {
    (mbits(&s.moments), s.steps)
}

struct Case {
    param: Vec<f32>,
    grad: Vec<f32>,
    moments: Moments,
    hyper: AdamWHyper,
    lr: f64,
    wd: f64,
    step: u64,
}

fn case(r: &mut Rng) -> Case {
    let n = r.below(6); // includes 0
    let vec_of = |r: &mut Rng, n: usize| Vec::from_fn(n, |_| f32_value(r));
    // Quiet most of the time so a large share of cases is accepted, and so the scan, not the
    // scalar validation, is what the comparison exercises.
    let quiet = r.unit() < 0.6;
    let mut param = vec_of(r, n);
    let mut grad = vec_of(r, n);
    let mut m = vec_of(r, n);
    let mut v = vec_of(r, n);
    if quiet {
        for x in param.iter_mut().chain(grad.iter_mut()).chain(m.iter_mut()) {
            *x = ((r.unit() * 2.0 - 1.0) * 0.5) as f32;
        }
        for x in v.iter_mut() {
            *x = (r.unit() * 0.01) as f32;
        }
    }
    // Occasionally a length mismatch, a refusal the scan never sees.
    if r.below(40) == 0 && !grad.is_empty() {
        grad.pop();
    }
    if r.below(40) == 0 {
        m.push(0.0);
    }
    let p_bad = if quiet { 0.03 } else { 0.25 };
    let hyper = AdamWHyper {
        lr: 1.0, // not read by the update; validate() reads it, so keep it legal mostly
        beta1: f64_edge(
            r,
            &[0.0, 0.5, 0.9, 0.499_999_9, 0.5, 0.500_000_1, 1.0 - 1e-9],
            &[1.0, -0.1, f64::NAN, 1.5, f64::INFINITY],
            p_bad,
        ),
        beta2: f64_edge(
            r,
            &[0.0, 0.9, 0.999, 0.9999, 1.0 - 1e-12, 1e-9],
            &[1.0, -0.1, f64::NAN, 2.0],
            p_bad,
        ),
        eps: f64_edge(
            r,
            &[1e-8, 1e-30, 1e-45, 1e-300, 1.0, 1e10, 1e-5],
            &[0.0, -1.0, f64::NAN, f64::INFINITY],
            p_bad,
        ),
        grad_scale: f64_edge(
            r,
            &[1.0, 0.5, 1e-30, 1e-300, 1e30, 3e38, 1e300],
            &[0.0, -1.0, f64::NAN, f64::INFINITY],
            p_bad,
        ),
    };
    let hyper = AdamWHyper {
        lr: hyper_lr(r, p_bad),
        ..hyper
    };
    let lr = f64_edge(
        r,
        &[1e-3, 1e-2, 1.0, 1e-30, 1e30, 5e-324, 1e-300, 1e300],
        &[0.0, -1e-3, f64::NAN, f64::INFINITY],
        p_bad,
    );
    let wd = f64_edge(
        r,
        &[0.0, 0.01, 0.1, 1.0, 1e10, 1e-300, 1e300],
        &[-0.0 - 1e-9, f64::NAN, f64::INFINITY, -1.0],
        p_bad,
    );
    let step = r.pick(&[
        1u64,
        1,
        2,
        3,
        10,
        1000,
        1 << 20,
        (1 << 53) - 1,
        1 << 53,
        (1 << 53) + 1,
        u64::MAX - 1,
        u64::MAX,
        0,
    ]);
    Case {
        param,
        grad,
        moments: Moments { m, v },
        hyper,
        lr,
        wd,
        step: if quiet && step > 1 << 20 { 3 } else { step },
    }
}

fn hyper_lr(r: &mut Rng, p_bad: f64) -> f64 {
    f64_edge(r, &[1e-3, 1.0, 1e-30], &[0.0, f64::NAN, -1.0], p_bad * 0.2)
}

// ---------------------------------------------------------------------------------------------
// Attack 1: check / apply divergence, differential over thousands of cases.
// ---------------------------------------------------------------------------------------------

#[test]
fn attack1_adamw_check_is_err_iff_adamw_update_is_err() {
    let mut r = Rng(0xADA3_0001);
    let (mut ok, mut refused, mut scan_refused) = (0u32, 0u32, 0u32);
    const N: u32 = 60_000;
    for case_no in 0..N {
        let c = case(&mut r);
        let (p0, g0, mo0) = (bits(&c.param), bits(&c.grad), mbits(&c.moments));
        let (mut pc, mut mc) = (c.param.clone(), c.moments.clone());
        let chk = adamw_check(&mut pc, &c.grad, &mut mc, &c.hyper, c.lr, c.wd, c.step).map(|_| ());
        let (mut pc2, mut mc2) = (c.param.clone(), c.moments.clone());
        let chk2 =
            adamw_check(&mut pc2, &c.grad, &mut mc2, &c.hyper, c.lr, c.wd, c.step).map(|_| ());
        assert_eq!(
            chk.as_ref().err().map(|e| e.to_string()),
            chk2.as_ref().err().map(|e| e.to_string()),
            "case {case_no}: the check is not deterministic"
        );
        // The check (token dropped unapplied) leaves its inputs bit-identical.
        assert_eq!(bits(&pc), p0, "case {case_no}: check wrote param");
        assert_eq!(bits(&c.grad), g0, "case {case_no}: check wrote grad");
        assert_eq!(mbits(&mc), mo0, "case {case_no}: check wrote moments");

        let (mut p, mut mo) = (c.param.clone(), c.moments.clone());
        let upd = adamw_update(&mut p, &c.grad, &mut mo, &c.hyper, c.lr, c.wd, c.step);
        match (&chk, &upd) {
            (Ok(()), Ok(())) => {
                ok += 1;
                // Everything stored is finite: the invariant the scan claims to enforce.
                assert!(
                    p.iter().chain(&mo.m).chain(&mo.v).all(|x| x.is_finite()),
                    "case {case_no}: an accepted update stored a non-finite value"
                );
            }
            (Err(a), Err(b)) => {
                refused += 1;
                if a.to_string().contains("non-finite") {
                    scan_refused += 1;
                }
                assert_eq!(
                    a.to_string(),
                    b.to_string(),
                    "case {case_no}: refusal text differs"
                );
                assert_eq!(bits(&p), p0, "case {case_no}: refused update wrote param");
                assert_eq!(
                    mbits(&mo),
                    mo0,
                    "case {case_no}: refused update wrote moments"
                );
            }
            (Ok(()), Err(e)) => panic!(
                "BREAK case {case_no}: check Ok but update refuses ({e}); param={:?} grad={:?} \
                 m={:?} v={:?} hyper={:?} lr={:e} wd={:e} step={}",
                c.param, c.grad, c.moments.m, c.moments.v, c.hyper, c.lr, c.wd, c.step
            ),
            (Err(e), Ok(())) => panic!(
                "BREAK case {case_no}: check refuses ({e}) but update succeeds; param={:?} \
                 grad={:?} m={:?} v={:?} hyper={:?} lr={:e} wd={:e} step={}",
                c.param, c.grad, c.moments.m, c.moments.v, c.hyper, c.lr, c.wd, c.step
            ),
        }
    }
    eprintln!("attack1: {N} cases, {ok} accepted, {refused} refused ({scan_refused} by the scan)");
    assert!(
        ok > 2_000,
        "too few accepted cases ({ok}) for the attack to mean anything"
    );
    assert!(
        scan_refused > 2_000,
        "too few scan refusals ({scan_refused})"
    );
}

/// Entry level: `adamw_entry_check` vs `adamw_entry_step`, including the step count and `None`.
#[test]
fn attack1_entry_check_is_err_iff_entry_step_is_err() {
    let mut r = Rng(0xADA3_0002);
    let (mut ok, mut skipped, mut refused) = (0u32, 0u32, 0u32);
    for case_no in 0..40_000u32 {
        let c = case(&mut r);
        let steps = r.pick(&[
            0u64,
            0,
            1,
            2,
            100,
            (1 << 53) - 2,
            (1 << 53) - 1,
            1 << 53,
            (1 << 53) + 1,
            u64::MAX - 1,
            u64::MAX,
        ]);
        let row = EntryHyper {
            name: format!("e{case_no}"),
            lr_scale: f64_edge(
                &mut r,
                &[1.0, 0.1, 1e-30, 1e30, 2.0],
                &[0.0, -1.0, f64::NAN, f64::INFINITY],
                0.1,
            ),
            weight_decay: c.wd,
            eps: c.hyper.eps,
            beta1: c.hyper.beta1,
            beta2: c.hyper.beta2,
        };
        let state = EntryState {
            moments: c.moments.clone(),
            steps,
        };
        let grad = if r.below(8) == 0 {
            None
        } else {
            Some(c.grad.as_slice())
        };
        let grad_scale = c.hyper.grad_scale;
        let lr = c.lr;
        let st0 = state.clone();
        let p0 = bits(&c.param);

        let (mut pc, mut stc) = (c.param.clone(), state.clone());
        let chk = adamw_entry_check(&mut pc, grad, &mut stc, &row, lr, grad_scale).map(|_| ());
        assert_eq!(
            sbits(&stc),
            sbits(&st0),
            "case {case_no}: entry check moved the state"
        );
        assert_eq!(
            bits(&pc),
            p0,
            "case {case_no}: entry check wrote the parameter"
        );

        let (mut p, mut st) = (c.param.clone(), state.clone());
        let stepped = adamw_entry_step(&mut p, grad, &mut st, &row, lr, grad_scale);
        match (&chk, &stepped) {
            (Ok(()), Ok(true)) => {
                ok += 1;
                assert_eq!(st.steps, st0.steps + 1, "case {case_no}: count");
            }
            (Ok(()), Ok(false)) => {
                skipped += 1;
                assert!(grad.is_none());
                assert_eq!(sbits(&st), sbits(&st0));
                assert_eq!(bits(&p), p0);
            }
            (Err(a), Err(b)) => {
                refused += 1;
                assert_eq!(a.to_string(), b.to_string(), "case {case_no}");
                assert_eq!(
                    sbits(&st),
                    sbits(&st0),
                    "case {case_no}: refused entry step moved state/count"
                );
                assert_eq!(
                    bits(&p),
                    p0,
                    "case {case_no}: refused entry step wrote param"
                );
            }
            (a, b) => panic!(
                "BREAK case {case_no}: entry check {a:?} vs entry step {b:?}; row={row:?} \
                 steps={steps} lr={lr:e} gs={grad_scale:e} param={:?} grad={:?}",
                c.param, grad
            ),
        }
    }
    eprintln!("attack1 entry: {ok} stepped, {skipped} skipped, {refused} refused");
    assert!(ok > 1_000 && refused > 1_000 && skipped > 1_000);
}

// ---------------------------------------------------------------------------------------------
// Attack 2 + 4 + 3 (provider): ToyProvider::adamw_step atomicity, differential vs per-entry
// adamw_update, retry equivalence.
// ---------------------------------------------------------------------------------------------

const VOCAB: usize = 8;
const HIDDEN: usize = 4;
const LAYERS: usize = 2;
const LR: f64 = 1e-2;
const BETA2: f64 = 0.999;

fn toy() -> ToyProvider {
    ToyProvider::new(VOCAB, HIDDEN, LAYERS, 42).expect("toy provider")
}

fn hyper() -> AdamWHyper {
    AdamWHyper {
        lr: LR,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        grad_scale: 1.0,
    }
}

/// Values, moments and step counter, as bit patterns.
type Snapshot = (Vec<Vec<u32>>, Vec<(Vec<u32>, Vec<u32>)>, u64);

fn snapshot(p: &ToyProvider) -> Snapshot {
    (
        p.values().iter().map(|v| bits(v)).collect(),
        p.moments().iter().map(mbits).collect(),
        p.step_count(),
    )
}

fn fill_bank(p: &mut ToyProvider, r: &mut Rng, wild: bool) {
    for g in p.grads_mut().iter_mut() {
        for x in g.iter_mut() {
            *x = if wild && r.below(12) == 0 {
                f32_value(r)
            } else {
                ((r.unit() * 2.0 - 1.0) * 0.1) as f32
            };
        }
    }
}

#[test]
fn attack2_toy_provider_step_is_atomic_and_equals_per_entry_updates() {
    let mut r = Rng(0xADA3_0003);
    let n = toy().parameters().len();
    let (mut ok, mut refused) = (0u32, 0u32);
    for case_no in 0..3_000u32 {
        let mut p = toy();
        // Two steps: the first usually succeeds so moments are non-trivial; the second is the
        // one under attack.
        let mut taken = 0u64;
        for step in 1..=2u64 {
            let wild = step == 2 && r.below(3) != 0;
            fill_bank(&mut p, &mut r, wild);
            if step == 2 && r.below(4) == 0 {
                // A value (parameter) the update would turn non-finite.
                let e = r.below(n);
                let k = r.below(p.values()[e].len());
                p.values_mut()[e][k] = f32_value(&mut r);
            }
            let lr_scale: Vec<f64> = (0..n)
                .map(|_| {
                    f64_edge(
                        &mut r,
                        &[1.0, 0.1, 1e-3, 1e30],
                        &[0.0, f64::NAN, -1.0],
                        0.02,
                    )
                })
                .collect();
            let wd: Vec<f64> = (0..n)
                .map(|_| f64_edge(&mut r, &[0.0, WEIGHT_DECAY, 1.0], &[-1.0, f64::NAN], 0.02))
                .collect();
            let before = snapshot(&p);
            // Reference: per-entry adamw_update on clones, only meaningful when all accepted.
            let mut ref_vals: Vec<Vec<f32>> = p.values().to_vec();
            let mut ref_mom: Vec<Moments> = p.moments().to_vec();
            let mut ref_ok = true;
            for i in 0..n {
                let l = hyper().lr * lr_scale[i];
                if adamw_update(
                    &mut ref_vals[i],
                    &p.grads()[i],
                    &mut ref_mom[i],
                    &hyper(),
                    l,
                    wd[i],
                    step,
                )
                .is_err()
                {
                    ref_ok = false;
                    break;
                }
            }
            let res = p.adamw_step(&hyper(), step, &lr_scale, &wd);
            match res {
                Ok(()) => {
                    ok += 1;
                    assert!(
                        ref_ok,
                        "case {case_no}: provider stepped, a per-entry update refuses"
                    );
                    taken = step;
                    assert_eq!(p.step_count(), step);
                    for i in 0..n {
                        assert_eq!(
                            bits(&p.values()[i]),
                            bits(&ref_vals[i]),
                            "case {case_no} e{i}"
                        );
                        assert_eq!(
                            mbits(&p.moments()[i]),
                            mbits(&ref_mom[i]),
                            "case {case_no} e{i}"
                        );
                    }
                }
                Err(e) => {
                    refused += 1;
                    assert_eq!(
                        snapshot(&p),
                        before,
                        "BREAK case {case_no} step {step}: refusal ({e}) moved state"
                    );
                    assert_eq!(p.step_count(), taken);
                    // A provider-validation refusal may also be a per-entry refusal; if every
                    // per-entry update accepted, the provider refused for a provider-level
                    // reason (lr scale, wd) which per-entry updates do not see: fine.
                    break;
                }
            }
        }
    }
    eprintln!("attack2 toy: {ok} steps accepted, {refused} refused");
    assert!(ok > 500 && refused > 300);
}

/// Attack 3 (provider): refusal at entry k, repair, retry the same step number: bit-identical to
/// a provider that never saw the refusal. Repeated over entries, seeds and prior-step counts.
#[test]
fn attack3_retry_after_refusal_is_bit_identical_to_never_refused() {
    let n = toy().parameters().len();
    for seed in 0..40u64 {
        for entry in 0..n {
            for prior in 0..3u64 {
                let mut r = Rng(seed * 1000 + entry as u64);
                let mut clean = toy();
                let mut dirty = toy();
                let ones = vec![1.0; n];
                let wd = vec![WEIGHT_DECAY; n];
                for s in 1..=prior {
                    fill_bank(&mut clean, &mut r, false);
                    dirty.grads_mut().clone_from_slice(clean.grads());
                    clean.adamw_step(&hyper(), s, &ones, &wd).unwrap();
                    dirty.adamw_step(&hyper(), s, &ones, &wd).unwrap();
                }
                let t = prior + 1;
                fill_bank(&mut clean, &mut r, false);
                dirty.grads_mut().clone_from_slice(clean.grads());
                let k = r.below(dirty.grads()[entry].len());
                let good = dirty.grads()[entry][k];
                dirty.grads_mut()[entry][k] = 1e21;
                let before = snapshot(&dirty);
                dirty.adamw_step(&hyper(), t, &ones, &wd).unwrap_err();
                assert_eq!(
                    snapshot(&dirty),
                    before,
                    "seed {seed} entry {entry} prior {prior}"
                );
                dirty.grads_mut()[entry][k] = good;
                dirty.adamw_step(&hyper(), t, &ones, &wd).unwrap();
                clean.adamw_step(&hyper(), t, &ones, &wd).unwrap();
                assert_eq!(
                    snapshot(&dirty),
                    snapshot(&clean),
                    "seed {seed} entry {entry}"
                );
            }
        }
    }
}

/// Attack 4: refusal first / middle / last entry, with every entry NaN-variant and -0.0 grads,
/// plus zero-length trailing and leading entries.
#[test]
fn attack4_boundaries_first_middle_last_nan_variants_and_negative_zero() {
    for &nan_bits in SPECIAL_F32 {
        let bad = f32::from_bits(nan_bits);
        let finite = bad.is_finite();
        for with_empty in [false, true] {
            let mk = || {
                let p = toy();
                if with_empty {
                    p.with_trainable_entry("empty.weight", &[0]).unwrap()
                } else {
                    p
                }
            };
            let n = mk().parameters().len();
            for entry in [0, n / 2, n - 1] {
                let mut p = mk();
                for g in p.grads_mut().iter_mut() {
                    for x in g.iter_mut() {
                        *x = -0.0;
                    }
                }
                if p.grads()[entry].is_empty() {
                    continue;
                }
                p.grads_mut()[entry][0] = bad;
                let before = snapshot(&p);
                let res = p.adamw_step(&hyper(), 1, &vec![1.0; n], &vec![WEIGHT_DECAY; n]);
                // Gradients of 1e21-class or non-finite are refused; a plain finite grad must pass.
                if res.is_err() {
                    assert_eq!(snapshot(&p), before, "bits {nan_bits:#x} entry {entry}");
                } else {
                    assert!(
                        finite || bad == 0.0,
                        "BREAK: non-finite grad {nan_bits:#x} accepted at entry {entry}"
                    );
                    assert_eq!(p.step_count(), 1);
                    assert!(p.values().iter().flatten().all(|x| x.is_finite()));
                }
            }
        }
    }
    // All -0.0 gradients: accepted, finite result.
    let mut p = toy();
    for g in p.grads_mut().iter_mut() {
        g.iter_mut().for_each(|x| *x = -0.0);
    }
    let n = p.parameters().len();
    p.adamw_step(&hyper(), 1, &vec![1.0; n], &vec![WEIGHT_DECAY; n])
        .unwrap();
}

/// A single-entry (bank of one) check: lr_scale 0 on the only entry refuses, nothing moves.
#[test]
fn attack4_zero_lr_scale_first_and_last_entry() {
    let n = toy().parameters().len();
    for which in [0, n - 1] {
        let mut p = toy();
        let mut r = Rng(5);
        fill_bank(&mut p, &mut r, false);
        let before = snapshot(&p);
        let mut s = vec![1.0; n];
        s[which] = 0.0;
        p.adamw_step(&hyper(), 1, &s, &vec![WEIGHT_DECAY; n])
            .unwrap_err();
        assert_eq!(snapshot(&p), before);
    }
}

// ---------------------------------------------------------------------------------------------
// Attack 2 (trainer): host check passes, provider refuses; and host-entry refusal ordering.
// ---------------------------------------------------------------------------------------------

struct Batch(u64);

impl ConsumedBatch for Batch {
    fn index(&self) -> u64 {
        self.0
    }
    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        prefix.fold(&[&self.0.to_be_bytes()]);
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Plant {
    None,
    /// 1e21 in `(batch, host entry)`'s gradient.
    Huge(u64, usize),
    /// A non-finite value in `(batch, host entry)`'s gradient.
    Nan(u64, usize),
}

struct HostObjective {
    values: Vec<Vec<f32>>,
    grads: Vec<Option<Vec<f32>>>,
    plant: Plant,
    batch: u64,
    /// Entries given no gradient at all (torch's `grad is None`).
    skip: Vec<usize>,
    /// What `grads()` reports, against what `values_and_grads()` hands the step.
    lie: Lie,
    shadow: Vec<Option<Vec<f32>>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Lie {
    Honest,
    /// `grads()` reports the benign gradient; `values_and_grads()` the planted one.
    Benign,
    /// `grads()` drops its last entry.
    Short,
}

impl HostObjective {
    fn new(plant: Plant) -> Self {
        Self {
            values: vec![vec![0.5; 3], vec![-0.25; 4], vec![], vec![0.125; 2]],
            grads: vec![None, None, None, None],
            plant,
            batch: 0,
            skip: Vec::new(),
            lie: Lie::Honest,
            shadow: vec![None, None, None, None],
        }
    }
}

impl HostParams for HostObjective {
    fn entries(&self) -> Vec<ParamSpec> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, v)| ParamSpec::new(format!("host.{i}"), &[v.len()]))
            .collect()
    }
    fn zero_grads(&mut self) {
        self.grads.iter_mut().for_each(|g| *g = None);
    }
    fn grads(&self) -> Vec<Option<&[f32]>> {
        let src = if self.lie == Lie::Benign {
            &self.shadow
        } else {
            &self.grads
        };
        let mut g: Vec<Option<&[f32]>> = src.iter().map(|g| g.as_deref()).collect();
        if self.lie == Lie::Short {
            g.pop();
        }
        g
    }
    fn values(&self) -> Vec<&[f32]> {
        self.values.iter().map(Vec::as_slice).collect()
    }
    fn values_and_grads(&mut self) -> Vec<(&mut [f32], Option<&[f32]>)> {
        self.values
            .iter_mut()
            .zip(self.grads.iter())
            .map(|(v, g)| (v.as_mut_slice(), g.as_deref()))
            .collect()
    }
}

impl Objective for HostObjective {
    type Batch = Batch;
    fn run_kind(&self) -> &'static str {
        "ft"
    }
    fn plan(&mut self, batch: &Batch) -> Result<MicroBatchPlan, TrainError> {
        self.batch = batch.0;
        let tokens: Vec<u32> = (0..5)
            .map(|k| ((batch.0 * 3 + k * 2) % VOCAB as u64) as u32)
            .collect();
        Ok(MicroBatchPlan {
            sequences: vec![SequenceWork {
                positions: (0..4).collect(),
                targets: tokens[1..].to_vec(),
                tokens,
                hidden_at: Vec::new(),
            }],
            row_scale: 0.25,
            counts: BatchCounts::default(),
        })
    }
    fn hidden_grad(
        &mut self,
        _s: usize,
        _p: &[u32],
        _h: &[f32],
        _w: usize,
    ) -> Result<Vec<f32>, TrainError> {
        Err(TrainError::Refused("no outside loss".into()))
    }
    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        for (i, g) in self.grads.iter_mut().enumerate() {
            if self.skip.contains(&i) {
                *g = None;
                continue;
            }
            let n = self.values[i].len();
            let mut grad = vec![0.05 + 0.01 * i as f32; n];
            match self.plant {
                Plant::Huge(b, e) if b == self.batch && e == i && n > 0 => grad[0] = 1e21,
                Plant::Nan(b, e) if b == self.batch && e == i && n > 0 => {
                    grad[n - 1] = f32::from_bits(0x7FC0_0001)
                }
                _ => {}
            }
            self.shadow[i] = Some(vec![0.05 + 0.01 * i as f32; n]);
            *g = Some(grad);
        }
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * 0.25,
            channels: Vec::new(),
        })
    }
    fn host(&mut self) -> Option<&mut dyn HostParams> {
        Some(self)
    }
    fn host_ref(&self) -> Option<&dyn HostParams> {
        Some(self)
    }
}

fn config(steps: u64) -> TrainConfig {
    TrainConfig {
        schedule: LrSchedule::new(LR, steps, 1, LR / 10.0).expect("schedule"),
        cap: WallClockCap::new(3600.0).expect("cap"),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: LR,
            beta2: BETA2,
            lower_layers: None,
        },
        // Large so the clip does not scale a planted 1e21 gradient back into range.
        max_grad_norm: 1e30,
        epoch: 0,
        epochs: 1,
        seed: 7,
        checkpoint: None,
        eta: None,
    }
}

/// Delegates to a `ToyProvider`, but its `adamw_step` refuses at `refuse_step` (a provider-side
/// refusal that the host check cannot foresee) and counts how many times it was asked.
struct RefusingAt {
    inner: ToyProvider,
    refuse_step: u64,
    asked: Vec<u64>,
}

impl StepProvider for RefusingAt {
    fn parameters(&self) -> &[ParamSpec] {
        self.inner.parameters()
    }
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        self.inner.accumulate(job, mode, external)
    }
    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        self.inner.grad_sq_norm()
    }
    fn supports_lr_scale(&self) -> bool {
        self.inner.supports_lr_scale()
    }
    fn adamw_step(
        &mut self,
        hyper: &AdamWHyper,
        step: u64,
        lr_scale: &[f64],
        weight_decay: &[f64],
    ) -> Result<(), StepError> {
        self.asked.push(step);
        if step == self.refuse_step {
            return Err(StepError::Contract("provider-side refusal".into()));
        }
        self.inner.adamw_step(hyper, step, lr_scale, weight_decay)
    }
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }
    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        self.inner.read_parameters()
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn save_state(&mut self, dir: &Path) -> Result<Vec<PathBuf>, StepError> {
        self.inner.save_state(dir)
    }
    fn load_state(&mut self, dir: &Path) -> Result<(), StepError> {
        self.inner.load_state(dir)
    }
    fn save_state_with(
        &mut self,
        dir: &Path,
        context: &CheckpointContext,
    ) -> Result<Vec<PathBuf>, StepError> {
        self.inner.save_state_with(dir, context)
    }
    fn load_state_with(
        &mut self,
        dir: &Path,
        context: &CheckpointContext,
    ) -> Result<(), StepError> {
        self.inner.load_state_with(dir, context)
    }
}

fn run_on<P: StepProvider>(
    provider: &mut P,
    objective: &mut HostObjective,
    steps: u64,
    n_batches: u64,
) -> Result<(), TrainError> {
    train(
        provider,
        objective,
        (0..n_batches).map(|i| Ok::<_, TrainError>(Batch(i))),
        &config(steps),
        &MonotonicClock::new(),
        None,
        &mut Hooks::default(),
    )
    .map(|_| ())
}

fn host_bits(h: &HostObjective) -> Vec<Vec<u32>> {
    h.values.iter().map(|v| bits(v)).collect()
}

/// The host check passes, then the PROVIDER refuses: the host entries must not have stepped
/// (they step after the provider), and the provider is where the wrapper left it.
#[test]
fn attack2_provider_refusal_after_a_passing_host_check_leaves_host_unmoved() {
    for good_steps in 0..3u64 {
        // Reference: `good_steps` good steps, then stop for lack of data.
        let mut ref_p = RefusingAt {
            inner: toy(),
            refuse_step: u64::MAX,
            asked: vec![],
        };
        let mut ref_h = HostObjective::new(Plant::None);
        if good_steps > 0 {
            run_on(&mut ref_p, &mut ref_h, 4, good_steps).expect("reference");
        }
        let mut p = RefusingAt {
            inner: toy(),
            refuse_step: good_steps + 1,
            asked: vec![],
        };
        let mut h = HostObjective::new(Plant::None);
        let e = run_on(&mut p, &mut h, 4, good_steps + 2).expect_err("provider refuses");
        assert!(e.to_string().contains("provider-side refusal"), "{e}");
        assert_eq!(p.asked.last(), Some(&(good_steps + 1)));
        assert_eq!(p.step_count(), good_steps);
        assert_eq!(
            host_bits(&h),
            host_bits(&ref_h),
            "BREAK: host moved after provider refusal (good_steps {good_steps})"
        );
        for (a, b) in p.inner.moments().iter().zip(ref_p.inner.moments()) {
            assert_eq!(a, b);
        }
    }
}

/// Host entry refusal (huge / NaN gradient, first / middle / empty / last entry), after several
/// good steps: provider, provider moments, every host entry equal a run that stopped before the
/// refused step. Entries with `grad None` interleaved, so counts lag the model.
#[test]
fn attack2_host_entry_refusal_after_several_good_steps_with_lagging_entries() {
    for skip in [vec![], vec![0usize], vec![1, 3]] {
        for good in [0u64, 1, 3] {
            for entry in [0usize, 1, 2, 3] {
                for huge in [true, false] {
                    if skip.contains(&entry) {
                        continue;
                    }
                    let plant = if huge {
                        Plant::Huge(good, entry)
                    } else {
                        Plant::Nan(good, entry)
                    };
                    let mut ref_p = toy();
                    let mut ref_h = HostObjective::new(Plant::None);
                    ref_h.skip = skip.clone();
                    if good > 0 {
                        run_on(&mut ref_p, &mut ref_h, 6, good).expect("reference");
                    }
                    let mut p = toy();
                    let mut h = HostObjective::new(plant);
                    h.skip = skip.clone();
                    let res = run_on(&mut p, &mut h, 6, good + 1);
                    // Entry 2 is empty: nothing to plant there, the step succeeds.
                    if entry == 2 {
                        assert!(res.is_ok(), "empty entry cannot carry the planted value");
                        continue;
                    }
                    let tag = format!("skip {skip:?} good {good} entry {entry} huge {huge}");
                    let e = res.expect_err(&tag);
                    assert!(e.to_string().contains("non-finite"), "{tag}: {e}");
                    assert_eq!(p.step_count(), good, "{tag}: provider stepped");
                    assert_eq!(host_bits(&h), host_bits(&ref_h), "BREAK {tag}: host moved");
                    for (a, b) in p.moments().iter().zip(ref_p.moments()) {
                        assert_eq!(a, b, "{tag}: provider moments moved");
                    }
                    assert_eq!(
                        bits(&p.values().concat()),
                        bits(&ref_p.values().concat()),
                        "{tag}"
                    );
                }
            }
        }
    }
}

/// The trainer's pre-check reads `grads()` + `values()`; the step reads `values_and_grads()`.
/// `HostParams` does not enforce that they agree. If they do not, the provider steps and then a
/// host entry refuses. (Contract gap vs break is for the orchestrator: see the trait docs.)
#[test]
fn attack2_disagreeing_host_accessors_must_not_let_the_provider_step_alone() {
    let mut stepped = 0;
    for (lie, entry) in [(Lie::Benign, 1usize), (Lie::Benign, 3), (Lie::Short, 3)] {
        let mut p = toy();
        let mut h = HostObjective::new(Plant::Huge(0, entry));
        h.lie = lie;
        let res = run_on(&mut p, &mut h, 2, 1);
        assert!(
            res.is_err(),
            "the planted gradient must be refused somewhere"
        );
        eprintln!(
            "ATTACK2-LIE entry {entry}: provider step_count after refusal = {} ({})",
            p.step_count(),
            res.unwrap_err()
        );
        if p.step_count() != 0 {
            stepped += 1;
        }
    }
    assert_eq!(
        stepped, 0,
        "BREAK-or-GAP: in {stepped} of 3 cases the provider stepped and a host entry then refused"
    );
}
