//! K11's device checks: what `runga` runs and what `tests/device_k11.rs`
//! asserts. The three tiers of `src/k11_golden.rs`, per run:
//! 1. the device trajectory (parameters every step, final moments, every
//!    step's squared-norm partials, step counts) bit for bit against
//!    [`crate::k11_host::replay_f32`], and a second device run bit for bit
//!    against the first;
//! 2. on L-oracle's decay-sensitive golden, the device parameters within its
//!    pre-registered 1e-6 of torch's;
//! 3. on the adamw_f float64 golden, the sanity bounds
//!    [`ADAMW_F_PARAM_BOUND`] and [`ADAMW_F_NORM_BOUND`].
//!
//! Each run is guarded: a panic is a `panicked` check, never a pass.

use crate::check::{bitwise_check, diff_bits_f32, Check};
use crate::error::CudaError;
use crate::k11::{adamw_step, grad_sq_norm};
use crate::k11_golden::{AdamwF, DecaySensitive, ADAMW_F_NORM_BOUND, ADAMW_F_PARAM_BOUND};
use crate::k11_host::{replay_f32, synthetic_cases, AdamwBank, BankCase, Trajectory};
use crate::k11_kernels::K11;
use crate::kernels::STRICT_SM90;
use crate::runtime::CudaRuntime;
use crate::smoke::guarded;

/// One run on the device from fresh buffers: what [`replay_f32`] computes on
/// the host.
pub fn run_device(rt: &CudaRuntime, case: &BankCase) -> Result<Trajectory, CudaError> {
    let n = case.table.bank_len();
    let mut bank = AdamwBank::new(case.table.clone());
    let mut p = rt.upload(&case.init, "k11 p")?;
    let mut m = rt.alloc_zeros::<f32>(n, "k11 m")?;
    let mut v = rt.alloc_zeros::<f32>(n, "k11 v")?;
    let mut g = rt.alloc_zeros::<f32>(n, "k11 g")?;
    let mut out = Trajectory {
        w: Vec::with_capacity(case.grads.len()),
        m: Vec::new(),
        v: Vec::new(),
        partials: Vec::with_capacity(case.grads.len()),
        norms: Vec::with_capacity(case.grads.len()),
        steps: Vec::new(),
    };
    for (gh, active) in &case.grads {
        rt.write(&mut g, gh)?;
        let (norm, parts) = grad_sq_norm(rt, &case.table, &g, active)?;
        out.norms.push(norm);
        out.partials.push(parts);
        adamw_step(
            rt,
            &mut bank,
            &mut p,
            &g,
            &mut m,
            &mut v,
            active,
            &case.hyper,
        )?;
        out.w.push(rt.download(&p)?);
    }
    out.m = rt.download(&m)?;
    out.v = rt.download(&v)?;
    out.steps = bank.steps().to_vec();
    Ok(out)
}

fn flat(xs: &[Vec<f32>]) -> Vec<f32> {
    xs.iter().flatten().copied().collect()
}

/// Tier 1 for one case: two device runs, each against the emulation, and
/// the second against the first. Returns the first run for the caller's
/// tier-2/3 judgments.
fn tier1(rt: &CudaRuntime, name: &str, case: &BankCase) -> (Vec<Check>, Option<Trajectory>) {
    let want = match replay_f32(&case.table, &case.hyper, &case.init, &case.grads) {
        Ok(t) => t,
        Err(e) => {
            return (
                vec![Check::from_error(&format!("{name}.emulation"), &e)],
                None,
            )
        }
    };
    let first = match run_device(rt, case) {
        Ok(t) => t,
        Err(e) => return (vec![Check::from_error(name, &e)], None),
    };
    let second = match run_device(rt, case) {
        Ok(t) => t,
        Err(e) => return (vec![Check::from_error(&format!("{name}.repeat"), &e)], None),
    };
    let mut out = Vec::new();
    for (what, got, exp) in [
        ("w", flat(&first.w), flat(&want.w)),
        ("m", first.m.clone(), want.m.clone()),
        ("v", first.v.clone(), want.v.clone()),
        ("sq_partials", flat(&first.partials), flat(&want.partials)),
    ] {
        out.push(bitwise_check(
            &format!("{name}.{what}.vs_emulation"),
            diff_bits_f32(&got, &exp),
            exp.len(),
        ));
    }
    out.push(if first.steps == want.steps {
        Check::pass(&format!("{name}.step_counts"), format!("{:?}", first.steps))
    } else {
        Check::fail(
            &format!("{name}.step_counts"),
            format!("device {:?}, emulation {:?}", first.steps, want.steps),
        )
    });
    let repeat_w = flat(&first.w);
    out.push(bitwise_check(
        &format!("{name}.repeat"),
        diff_bits_f32(&flat(&second.w), &repeat_w),
        repeat_w.len(),
    ));
    let repeat_p = flat(&first.partials);
    out.push(bitwise_check(
        &format!("{name}.repeat_sq_partials"),
        diff_bits_f32(&flat(&second.partials), &repeat_p),
        repeat_p.len(),
    ));
    (out, Some(first))
}

/// K11's whole device check list.
pub fn k11_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    let name = format!("nvrtc.{}", K11.name);
    out.extend(guarded(&name, || {
        for entry in K11.entries {
            if let Err(e) = rt.function(&K11, &STRICT_SM90, entry) {
                return vec![Check::from_error(&name, &e)];
            }
        }
        vec![Check::pass(
            &name,
            format!("{} entries compiled and loaded", K11.entries.len()),
        )]
    }));

    out.extend(guarded("k11.decay_sensitive", || {
        let ds = match DecaySensitive::embedded() {
            Ok(d) => d,
            Err(e) => return vec![Check::from_error("k11.decay_sensitive.golden", &e)],
        };
        let case = BankCase {
            label: "decay_sensitive".to_string(),
            table: ds.table.clone(),
            hyper: ds.hyper,
            init: ds.init.clone(),
            grads: ds.grads.clone(),
        };
        let (mut checks, run) = tier1(rt, "k11.decay_sensitive", &case);
        if let Some(run) = run {
            let n = "k11.decay_sensitive.vs_torch_golden";
            checks.push(match ds.measure(&run.w) {
                Ok(gap) => {
                    let c = if gap.max_abs <= ds.bound {
                        Check::pass(
                            n,
                            format!("max |w - torch| {:e} <= {:e}", gap.max_abs, ds.bound),
                        )
                    } else {
                        Check::fail(
                            n,
                            format!("max |w - torch| {:e} > {:e}", gap.max_abs, ds.bound),
                        )
                    };
                    c.with("max_abs", gap.max_abs)
                        .with("bound", ds.bound)
                        .with("ratio_over_bound", gap.max_abs / ds.bound)
                        .with("at_step", gap.at_step)
                        .with("entry", gap.entry.as_str())
                }
                Err(e) => Check::from_error(n, &e),
            });
            let want: Vec<u64> = ds.entries.iter().map(|e| e.adamw_steps_taken).collect();
            let n = "k11.decay_sensitive.torch_step_counts";
            checks.push(if run.steps == want {
                Check::pass(n, format!("{want:?}"))
            } else {
                Check::fail(n, format!("device {:?}, torch {want:?}", run.steps))
            });
        }
        checks
    }));

    out.extend(guarded("k11.adamw_f", || {
        let f = match AdamwF::embedded() {
            Ok(f) => f,
            Err(e) => return vec![Check::from_error("k11.adamw_f.golden", &e)],
        };
        let case = BankCase {
            label: "adamw_f".to_string(),
            table: f.table.clone(),
            hyper: f.hyper,
            init: f.p0_f32(),
            grads: f.grads_with_flags(),
        };
        let (mut checks, run) = tier1(rt, "k11.adamw_f", &case);
        if let Some(run) = run {
            for (n, got, bound) in [
                (
                    "k11.adamw_f.params_vs_f64_sanity",
                    f.param_rel_err(&run.w),
                    ADAMW_F_PARAM_BOUND,
                ),
                (
                    "k11.adamw_f.grad_sq_norm_vs_f64_sanity",
                    f.norm_rel_err(&run.norms),
                    ADAMW_F_NORM_BOUND,
                ),
            ] {
                checks.push(
                    match got {
                        Ok(x) if x <= bound => {
                            Check::pass(n, format!("{x:e} <= {bound:e} of max |ref|"))
                        }
                        Ok(x) => Check::fail(n, format!("{x:e} > {bound:e} of max |ref|")),
                        Err(e) => Check::from_error(n, &e),
                    }
                    .with("bound", bound),
                );
            }
        }
        checks
    }));

    match synthetic_cases() {
        Ok(cases) => {
            for case in &cases {
                let name = format!("k11.synthetic.{}", case.label);
                out.extend(guarded(&name, || tier1(rt, &name, case).0));
            }
        }
        Err(e) => out.push(Check::from_error("k11.synthetic", &e)),
    }
    out
}
