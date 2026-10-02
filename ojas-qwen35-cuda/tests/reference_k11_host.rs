//! Tier 2 of K11 (see `src/k11_golden.rs`): the f32 kernel-order emulation in
//! `src/k11_host.rs`, which the device must equal bit for bit, against
//! L-oracle's decay-sensitive torch golden, and every pre-registered mutation
//! of it measured in the same kernel-order arithmetic. Plus tier 3's sanity
//! numbers on the adamw_f float64 golden.
//!
//! Every bound here is someone else's, written before this file:
//! - 1e-6 absolute on the fp32 masters, and "every mutation clears 100x": L-oracle's
//!   pre-registration (`tests/fixtures/adamw_decay_sensitive/preregistration.json`,
//!   `manifest.json` `measure`, `all_mutations_clear_100x`);
//! - [`ADAMW_F_PARAM_BOUND`] and [`ADAMW_F_NORM_BOUND`]: `src/k11_golden.rs`, derived
//!   there before the first run.
//!
//! The scalar-only mutations (D1, D3, lr_scale missing, both D7s) go through
//! the library's own `entry_scalars` and `adamw_update_f32` with altered
//! inputs. The two eps-placement mutations change the element formula, so
//! this file carries a per-element function with a variant switch; its
//! unmutated form is asserted bit-identical to `adamw_update_f32` over the
//! whole replay before any ratio is read.

use ojas_qwen35_cuda::k11_golden::{
    AdamwF, DecaySensitive, ADAMW_F_NORM_BOUND, ADAMW_F_PARAM_BOUND,
};
use ojas_qwen35_cuda::k11_host::{adamw_update_f32, entry_scalars, EntryScalars, StepScalars};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    D1TesslDefaultWeightDecay,
    D3DecayNotLrScaled,
    EpsInsideSqrt,
    EpsBeforeBiasCorrection,
    LrScaleMissing,
    D7NoGradSteppedAsZero,
    D7ModelWideStepCount,
}

const MUTATIONS: [(Mutation, &str); 7] = [
    (
        Mutation::D1TesslDefaultWeightDecay,
        "D1_tessl_default_weight_decay",
    ),
    (Mutation::D3DecayNotLrScaled, "D3_decay_not_lr_scaled"),
    (Mutation::EpsInsideSqrt, "eps_inside_sqrt"),
    (
        Mutation::EpsBeforeBiasCorrection,
        "eps_before_bias_correction",
    ),
    (Mutation::LrScaleMissing, "lr_scale_missing"),
    (
        Mutation::D7NoGradSteppedAsZero,
        "D7_no_grad_stepped_as_zero",
    ),
    (Mutation::D7ModelWideStepCount, "D7_model_wide_step_count"),
];

/// The kernel's element update with the eps placement switchable. `bc2` is
/// `f32(1 - beta2^t)`, used only by `EpsInsideSqrt`.
#[allow(clippy::too_many_arguments)]
fn element(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    e: EntryScalars,
    s: StepScalars,
    bc2: f32,
    mutation: Mutation,
) {
    for k in 0..p.len() {
        let w = p[k] * e.decay_mul;
        let gi = g[k] * s.grad_scale;
        let d = gi - m[k];
        let mi = if s.lerp_w < 0.5 {
            m[k] + s.lerp_w * d
        } else {
            gi - d * (1.0 - s.lerp_w)
        };
        let vi = v[k] * s.beta2 + (s.one_minus_beta2 * gi) * gi;
        let denom = match mutation {
            Mutation::EpsInsideSqrt => (vi / bc2 + s.eps).sqrt(),
            Mutation::EpsBeforeBiasCorrection => (vi.sqrt() + s.eps) / e.bc2_sqrt,
            _ => vi.sqrt() / e.bc2_sqrt + s.eps,
        };
        p[k] = w + (-e.step_size) * (mi / denom);
        m[k] = mi;
        v[k] = vi;
    }
}

struct Replay {
    w: Vec<Vec<f32>>,
    steps: Vec<u64>,
}

/// One replay of the golden's five steps under `mutation`. With
/// `check_against_lib`, every window update is also run through the
/// library's `adamw_update_f32` and the two are asserted bit-identical.
fn replay(ds: &DecaySensitive, mutation: Mutation, check_against_lib: bool) -> Replay {
    let n = ds.table.bank_len();
    let h = ds.hyper;
    let s = h.step_scalars();
    let (mut p, mut m, mut v) = (ds.init.clone(), vec![0.0f32; n], vec![0.0f32; n]);
    let mut count = vec![0u64; ds.entries.len()];
    let mut out = Vec::with_capacity(ds.steps);
    for (ti, (g, active)) in ds.grads.iter().enumerate() {
        let t = ti as u64 + 1;
        for (i, (e, row)) in ds.table.entries().iter().zip(&ds.entries).enumerate() {
            let on = active[i] || mutation == Mutation::D7NoGradSteppedAsZero;
            if !on {
                continue;
            }
            count[i] += 1;
            let t_i = if mutation == Mutation::D7ModelWideStepCount {
                t
            } else {
                count[i]
            };
            let scale = if mutation == Mutation::LrScaleMissing {
                1.0
            } else {
                e.lr_scale
            };
            let wd =
                if mutation == Mutation::D1TesslDefaultWeightDecay && row.tessl_default_excludes {
                    0.0
                } else {
                    e.weight_decay
                };
            let mut sc = entry_scalars(&h, t_i, scale, wd).expect("entry scalars");
            if mutation == Mutation::D3DecayNotLrScaled {
                sc.decay_mul = entry_scalars(&h, t_i, 1.0, wd).expect("scalars").decay_mul;
            }
            let bc2 = (1.0 - h.beta2.powf(t_i as f64)) as f32;
            let r = e.offset..e.offset + e.len;
            if check_against_lib {
                let (mut p2, mut m2, mut v2) = (
                    p[r.clone()].to_vec(),
                    m[r.clone()].to_vec(),
                    v[r.clone()].to_vec(),
                );
                adamw_update_f32(&mut p2, &g[r.clone()], &mut m2, &mut v2, sc, s).expect("lib");
                element(
                    &mut p[r.clone()],
                    &g[r.clone()],
                    &mut m[r.clone()],
                    &mut v[r.clone()],
                    sc,
                    s,
                    bc2,
                    mutation,
                );
                let same =
                    |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(
                    same(&p[r.clone()], &p2) && same(&m[r.clone()], &m2) && same(&v[r], &v2),
                    "step {t} {}: this file's element function is not the library's",
                    e.name
                );
            } else {
                element(
                    &mut p[r.clone()],
                    &g[r.clone()],
                    &mut m[r.clone()],
                    &mut v[r.clone()],
                    sc,
                    s,
                    bc2,
                    mutation,
                );
            }
        }
        out.push(p.clone());
    }
    Replay {
        w: out,
        steps: count,
    }
}

#[test]
fn this_files_unmutated_replay_is_the_librarys_emulation_bit_for_bit() {
    let ds = DecaySensitive::embedded().expect("golden");
    let ours = replay(&ds, Mutation::None, true);
    let (lib, lib_steps) = ds.replay_f32().expect("lib replay");
    for (t, (a, b)) in ours.w.iter().zip(&lib).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "step {}: replay differs from AdamwBank::step_f32",
            t + 1
        );
    }
    assert_eq!(ours.steps, lib_steps);
}

#[test]
fn the_emulation_meets_the_decay_sensitive_golden() {
    let ds = DecaySensitive::embedded().expect("golden");
    let (w, steps) = ds.replay_f32().expect("replay");
    let floor = ds.measure(&w).expect("measure");
    println!(
        "K11_DECAY_FLOOR max_abs={:e} at_step={} entry={:?} bound={:e} ratio={:e}",
        floor.max_abs,
        floor.at_step,
        floor.entry,
        ds.bound,
        floor.max_abs / ds.bound
    );
    assert!(
        floor.max_abs <= ds.bound,
        "the kernel-order emulation is {:e} from torch's golden at step {} {} (bound {:e})",
        floor.max_abs,
        floor.at_step,
        floor.entry,
        ds.bound
    );
    // torch's per-entry step counts, as the table read them off the optimizer.
    for (e, &got) in ds.entries.iter().zip(&steps) {
        assert_eq!(got, e.adamw_steps_taken, "{}: step count", e.name);
    }
    // The entry that never has a gradient is bit-identical to its init at
    // every step, in the emulation and in the golden.
    let never: Vec<usize> = ds
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.grad_steps.is_empty())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        never
            .iter()
            .map(|&i| ds.entries[i].name.as_str())
            .collect::<Vec<_>>(),
        ["layers.23.mlp.up_proj.weight"]
    );
    for &i in &never {
        let e = &ds.table.entries()[i];
        let r = e.offset..e.offset + e.len;
        for (t, (emulated, golden)) in w.iter().zip(&ds.golden_w).enumerate() {
            for (what, bank) in [("emulation", emulated), ("golden", golden)] {
                assert!(
                    bank[r.clone()]
                        .iter()
                        .zip(&ds.init[r.clone()])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{what}: {} moved at step {} with no gradient",
                    e.name,
                    t + 1
                );
            }
        }
    }
}

#[test]
fn every_preregistered_mutation_clears_100x_in_kernel_order_arithmetic() {
    let ds = DecaySensitive::embedded().expect("golden");
    let mut failures = Vec::new();
    for (mutation, name) in MUTATIONS {
        let r = replay(&ds, mutation, false);
        let gap = ds.measure(&r.w).expect("measure");
        let ratio = gap.max_abs / ds.bound;
        println!(
            "K11_MUTATION {name} max_abs={:e} at_step={} entry={:?} ratio_over_bound={ratio:.1}",
            gap.max_abs, gap.at_step, gap.entry
        );
        if ratio.is_nan() || ratio < 100.0 {
            failures.push(format!("{name}: {ratio:.1}x"));
        }
    }
    assert!(
        failures.is_empty(),
        "mutations under 100x the bound: {failures:?}"
    );
}

#[test]
fn the_emulation_meets_the_adamw_f_float64_golden_as_a_sanity_check() {
    let f = AdamwF::embedded().expect("adamw_f");
    let (traj, norms) = f.replay_f32().expect("replay");
    let p = f.param_rel_err(&traj).expect("params");
    let n = f.norm_rel_err(&norms).expect("norms");
    println!(
        "K11_ADAMW_F params_rel_max={p:e} (bound {ADAMW_F_PARAM_BOUND:e}) grad_sq_norm_rel={n:e} (bound {ADAMW_F_NORM_BOUND:e})"
    );
    assert!(p <= ADAMW_F_PARAM_BOUND, "params {p:e}");
    assert!(n <= ADAMW_F_NORM_BOUND, "grad_sq_norm {n:e}");
}
