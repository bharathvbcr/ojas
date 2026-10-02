//! K8's device checks, for `runga` and `tests/device_k8.rs`: every kernel
//! bit for bit against its [`crate::k8_plan`] host reference, from buffers
//! prefilled with a sentinel (so a write outside a window shows), and a
//! second run bit for bit against the first. Plus the activation sweep:
//! each of the crate's activation functions on the device against its host
//! emulation over [`act_sweep_inputs`].

use crate::check::{bitwise_check, diff_bits_f32, diff_bits_u16, Check};
use crate::error::CudaError;
use crate::k8::{
    act_sweep, residual_add, swiglu_bf16, swiglu_bwd_separate, swiglu_bwd_shared, swiglu_f32,
};
use crate::k8_act::act_sweep_inputs;
use crate::k8_kernels::K8;
use crate::k8_plan::{act_sweep_host, k8_cases, BwdOut, K8Case, ACT_SWEEP};
use crate::kernels::STRICT_SM90;
use crate::runtime::CudaRuntime;
use crate::smoke::{guarded, twice};

/// Random inputs the device sweep adds to the edge values.
pub const SWEEP_RANDOM: usize = 1 << 20;

/// What output buffers start as.
pub const SENTINEL: f32 = -7.5e27;

/// One case's five kernels.
pub fn case_checks(rt: &CudaRuntime, c: &K8Case) -> Vec<Check> {
    let mut out = Vec::new();
    let name = format!("k8.{}", c.label);
    let dense = (c.rows * c.width) as usize;
    let plan = match c.swiglu_plan() {
        Ok(p) => p,
        Err(e) => return vec![Check::from_error(&name, &e)],
    };

    let mut want = vec![SENTINEL; dense];
    if let Err(e) = plan.host_f32(&c.fused, &c.fused, &mut want) {
        return vec![Check::from_error(&name, &e)];
    }
    out.extend(twice(
        &format!("{name}.swiglu_f32"),
        &want,
        diff_bits_f32,
        || {
            let x = rt.upload(&c.fused, "k8 fused")?;
            let mut o = rt.upload(&vec![SENTINEL; dense], "k8 out")?;
            swiglu_f32(rt, &plan, &x, &x, &mut o)?;
            rt.download(&o)
        },
    ));

    let sentinel16 = crate::bf16::f32_to_bf16_bits(SENTINEL);
    let mut want16 = vec![sentinel16; dense];
    if let Err(e) = plan.host_bf16(&c.fused, &c.fused, &mut want16) {
        return vec![Check::from_error(&name, &e)];
    }
    out.extend(twice(
        &format!("{name}.swiglu_bf16"),
        &want16,
        diff_bits_u16,
        || {
            let x = rt.upload(&c.fused, "k8 fused")?;
            let mut o = rt.upload(&vec![sentinel16; dense], "k8 out bf16")?;
            swiglu_bf16(rt, &plan, &x, &x, &mut o)?;
            rt.download(&o)
        },
    ));

    match c.bwd_plan(BwdOut::Shared) {
        Ok(bp) => {
            let mut want = vec![SENTINEL; c.fused.len()];
            match bp.host_shared((&c.fused, &c.fused, &c.dy), &mut want) {
                Ok(()) => out.extend(twice(
                    &format!("{name}.swiglu_bwd_shared"),
                    &want,
                    diff_bits_f32,
                    || {
                        let x = rt.upload(&c.fused, "k8 fused")?;
                        let dy = rt.upload(&c.dy, "k8 dy")?;
                        let mut dgu = rt.upload(&vec![SENTINEL; c.fused.len()], "k8 dgu")?;
                        swiglu_bwd_shared(rt, &bp, (&x, &x, &dy), &mut dgu)?;
                        rt.download(&dgu)
                    },
                )),
                Err(e) => out.push(Check::from_error(&format!("{name}.swiglu_bwd_shared"), &e)),
            }
        }
        Err(e) => out.push(Check::from_error(&format!("{name}.swiglu_bwd_shared"), &e)),
    }

    match c.bwd_plan(BwdOut::Separate) {
        Ok(bp) => {
            let (mut wg, mut wu) = (vec![SENTINEL; dense], vec![SENTINEL; dense]);
            match bp.host_separate((&c.fused, &c.fused, &c.dy), &mut wg, &mut wu) {
                Ok(()) => {
                    let mut want = wg;
                    want.extend_from_slice(&wu);
                    out.extend(twice(
                        &format!("{name}.swiglu_bwd_separate"),
                        &want,
                        diff_bits_f32,
                        || {
                            let x = rt.upload(&c.fused, "k8 fused")?;
                            let dy = rt.upload(&c.dy, "k8 dy")?;
                            let mut dg = rt.upload(&vec![SENTINEL; dense], "k8 dgate")?;
                            let mut du = rt.upload(&vec![SENTINEL; dense], "k8 dup")?;
                            swiglu_bwd_separate(rt, &bp, (&x, &x, &dy), &mut dg, &mut du)?;
                            let mut both = rt.download(&dg)?;
                            both.extend(rt.download(&du)?);
                            Ok(both)
                        },
                    ));
                }
                Err(e) => out.push(Check::from_error(
                    &format!("{name}.swiglu_bwd_separate"),
                    &e,
                )),
            }
        }
        Err(e) => out.push(Check::from_error(
            &format!("{name}.swiglu_bwd_separate"),
            &e,
        )),
    }

    match c.residual_plan() {
        Ok(rp) => {
            let mut want = c.resid.clone();
            match rp.host(&c.y, &mut want) {
                Ok(()) => out.extend(twice(
                    &format!("{name}.residual_add"),
                    &want,
                    diff_bits_f32,
                    || {
                        let y = rt.upload(&c.y, "k8 y")?;
                        let mut r = rt.upload(&c.resid, "k8 resid")?;
                        residual_add(rt, &rp, &y, &mut r)?;
                        rt.download(&r)
                    },
                )),
                Err(e) => out.push(Check::from_error(&format!("{name}.residual_add"), &e)),
            }
        }
        Err(e) => out.push(Check::from_error(&format!("{name}.residual_add"), &e)),
    }
    out
}

/// The activation sweep: one check per function, plus the repeat.
pub fn act_sweep_checks(rt: &CudaRuntime) -> Vec<Check> {
    let x = act_sweep_inputs(SWEEP_RANDOM);
    let n = x.len();
    let want = act_sweep_host(&x);
    let run = || -> Result<Vec<f32>, CudaError> {
        let xb = rt.upload(&x, "k8 sweep x")?;
        let mut ob = rt.alloc_zeros::<f32>(want.len(), "k8 sweep out")?;
        act_sweep(rt, &xb, &mut ob)?;
        rt.download(&ob)
    };
    let first = match run() {
        Ok(v) => v,
        Err(e) => return vec![Check::from_error("k8.act_sweep", &e)],
    };
    let second = match run() {
        Ok(v) => v,
        Err(e) => return vec![Check::from_error("k8.act_sweep.repeat", &e)],
    };
    let mut out: Vec<Check> = ACT_SWEEP
        .iter()
        .enumerate()
        .map(|(k, (fname, _))| {
            let r = k * n..(k + 1) * n;
            bitwise_check(
                &format!("k8.act_sweep.{fname}.vs_host"),
                diff_bits_f32(&first[r.clone()], &want[r]),
                n,
            )
        })
        .collect();
    out.push(bitwise_check(
        "k8.act_sweep.repeat",
        diff_bits_f32(&second, &first),
        first.len(),
    ));
    out
}

/// K8's whole device check list.
pub fn k8_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    let name = format!("nvrtc.{}", K8.name);
    out.extend(guarded(&name, || {
        for entry in K8.entries {
            if let Err(e) = rt.function(&K8, &STRICT_SM90, entry) {
                return vec![Check::from_error(&name, &e)];
            }
        }
        vec![Check::pass(
            &name,
            format!("{} entries compiled and loaded", K8.entries.len()),
        )]
    }));
    out.extend(guarded("k8.act_sweep", || act_sweep_checks(rt)));
    for c in k8_cases() {
        let name = format!("k8.{}", c.label);
        out.extend(guarded(&name, || case_checks(rt, &c)));
    }
    out
}
