//! The device checks: what rung 0 runs and what the `#[ignore]` device tests
//! assert. Every check builds its inputs, runs on the device from fresh
//! buffers, compares with [`crate::host_ref`], and runs again to show the
//! second run is bit-identical to the first (repeat-run equality).
//!
//! Each case runs under `catch_unwind`, so a panic inside cudarc (a missing
//! symbol panics, `cudarc/src/cublas/sys/mod.rs:11-13`) becomes a
//! [`Status::Panicked`](crate::check::Status) entry and the remaining checks
//! still run.
//!
//! Bounds, all cited:
//! - K0 and the FFMA GEMM: bitwise against the host.
//! - ExactF32 vs float64: `1e-4 + 1e-7 * k` absolute (`tessl/src/gemm.rs:2286`).
//! - Bf16 vs float64 on the same bf16-rounded operands: `2e-3` absolute
//!   (`tessl/src/gemm.rs:2286`; the ragged-shape test, `:2336-2365`).
//! - The rung-0 cuBLAS probe: `2^-8 * max|ref|` (`cuda-backend-scoping.md`
//!   §5.3, rung 0).
//! - Shapes: tessl's ragged list (`tessl/src/gemm.rs:2342-2348`); accumulate
//!   prefill 0.25 (`tessl/src/gemm.rs:2218`); 25-repetition determinism
//!   (`tessl/src/gemm.rs:2371-2425`).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use crate::bf16::f32_slice_to_bf16;
use crate::buffer::Element;
use crate::check::{
    bitwise_check, diff_bits_f32, diff_bits_u16, panic_text, tolerance_check, tolerance_vs_f64,
    BitDiff, Check,
};
use crate::error::CudaError;
use crate::gemm::{gemm, gemm_bf16_cublas, Accumulate, Bf16Engine, GemmSpec, Operands};
use crate::gemm_plan::{GemmLayout, GemmShape};
use crate::host_ref;
use crate::inputs::{
    bf16_cast_edge_cases, splitmix_bits, splitmix_f32, tessl_ragged_a, tessl_ragged_b,
};
use crate::k0::{self, GatherSource};
use crate::k0_plan::{
    CopyColsPlan, DeliverMode, DeliverPlan, GatherRowsPlan, ScatterAddRowsPlan, ZeroPlan,
};
use crate::kernels::{ALL_MODULES, GEMM_FFMA, STRICT_SM90, STRICT_SM90A};
use crate::runtime::CudaRuntime;

/// tessl's ragged GEMM shapes `(m, n, k)` (`tessl/src/gemm.rs:2342-2348`).
pub const TESSL_RAGGED_SHAPES: [(usize, usize, usize); 5] = [
    (65, 33, 128),
    (63, 31, 129),
    (1, 1, 130),
    (130, 70, 260),
    (127, 95, 2049),
];

/// The gap the cuBLAS probe falsifies.
pub const CUBLAS_GAP: &str = "GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01";

/// Run `f` and turn a panic into one [`Check`]; add the elapsed time.
pub fn guarded(name: &str, f: impl FnOnce() -> Vec<Check>) -> Vec<Check> {
    let start = Instant::now();
    let out = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(checks) => checks,
        Err(payload) => vec![Check::panicked(name, panic_text(payload.as_ref()))],
    };
    let ms = start.elapsed().as_secs_f64() * 1e3;
    out.into_iter().map(|c| c.with("group_ms", ms)).collect()
}

/// Two device runs from scratch, each compared to `want` bitwise, and to
/// each other. Shared by every kernel family's checks (K0, K8).
pub fn twice<T: Element>(
    name: &str,
    want: &[T],
    diff: fn(&[T], &[T]) -> BitDiff,
    run: impl Fn() -> Result<Vec<T>, CudaError>,
) -> Vec<Check> {
    let first = match run() {
        Ok(v) => v,
        Err(e) => return vec![Check::from_error(name, &e)],
    };
    let second = match run() {
        Ok(v) => v,
        Err(e) => return vec![Check::from_error(&format!("{name}.repeat"), &e)],
    };
    vec![
        bitwise_check(&format!("{name}.vs_host"), diff(&first, want), want.len()),
        bitwise_check(
            &format!("{name}.repeat"),
            diff(&second, &first),
            first.len(),
        ),
    ]
}

/// Every module and entry compiles through NVRTC and loads, plus one module
/// at `compute_90a` (the later WGMMA tier's target).
pub fn compile_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    for module in ALL_MODULES {
        let name = format!("nvrtc.{}", module.name);
        out.extend(guarded(&name, || {
            let start = Instant::now();
            for entry in module.entries {
                if let Err(e) = rt.function(module, &STRICT_SM90, entry) {
                    return vec![Check::from_error(&name, &e)];
                }
            }
            vec![Check::pass(
                &name,
                format!(
                    "{} entries compiled ({:?}) and loaded",
                    module.entries.len(),
                    STRICT_SM90.options()
                ),
            )
            .with("entries", module.entries.len())
            .with("compile_and_load_ms", start.elapsed().as_secs_f64() * 1e3)]
        }));
    }
    let name = "nvrtc.compute_90a";
    out.extend(guarded(name, || {
        match rt.function(&GEMM_FFMA, &STRICT_SM90A, GEMM_FFMA.entries[0]) {
            Ok(_) => vec![Check::pass(
                name,
                format!("{:?} compiled and loaded", STRICT_SM90A.options()),
            )],
            Err(e) => vec![Check::from_error(name, &e)],
        }
    }));
    out
}

/// Every K0 kernel against its host reference, twice.
pub fn k0_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();

    out.extend(guarded("k0.cast_f32_to_bf16", || {
        let mut input = bf16_cast_edge_cases();
        input.extend(
            splitmix_bits(0xca57, 200_003)
                .into_iter()
                .map(f32::from_bits),
        );
        let want = host_ref::cast_f32_to_bf16(&input);
        twice("k0.cast_f32_to_bf16", &want, diff_bits_u16, || {
            let src = rt.upload(&input, "cast src")?;
            let mut dst = rt.alloc_zeros::<u16>(input.len(), "cast dst")?;
            k0::cast_f32_to_bf16(rt, &src, &mut dst)?;
            rt.download(&dst)
        })
    }));

    out.extend(guarded("k0.cast_bf16_to_f32", || {
        let input: Vec<u16> = (0..=u16::MAX).collect();
        let want = host_ref::cast_bf16_to_f32(&input);
        twice("k0.cast_bf16_to_f32", &want, diff_bits_f32, || {
            let src = rt.upload(&input, "widen src")?;
            let mut dst = rt.alloc_zeros::<f32>(input.len(), "widen dst")?;
            k0::cast_bf16_to_f32(rt, &src, &mut dst)?;
            rt.download(&dst)
        })
    }));

    out.extend(guarded("k0.copy_cols", || {
        let (rows, width) = (37u64, 17u64);
        let src = splitmix_f32(0xc01, 37 * 29, 2.0);
        let dst0 = splitmix_f32(0xc02, 37 * 23, 2.0);
        let plan = match CopyColsPlan::new(rows, width, (29, 5, src.len()), (23, 3, dst0.len())) {
            Ok(p) => p,
            Err(e) => return vec![Check::from_error("k0.copy_cols", &e)],
        };
        let mut want = dst0.clone();
        host_ref::copy_cols(&plan, &src, &mut want);
        twice("k0.copy_cols", &want, diff_bits_f32, || {
            let s = rt.upload(&src, "copy_cols src")?;
            let mut d = rt.upload(&dst0, "copy_cols dst")?;
            k0::copy_cols(rt, &plan, &s, &mut d)?;
            rt.download(&d)
        })
    }));

    for (mode, tag) in [(DeliverMode::Copy, "copy"), (DeliverMode::Add, "add")] {
        let name = format!("k0.deliver_{tag}");
        out.extend(guarded(&name, || {
            let src = splitmix_f32(0xde1, 1000, 3.0);
            let dst0 = splitmix_f32(0xde2, 900, 3.0);
            let plan = match DeliverPlan::new((7, src.len()), (13, dst0.len()), 777, mode) {
                Ok(p) => p,
                Err(e) => return vec![Check::from_error(&name, &e)],
            };
            let mut want = dst0.clone();
            host_ref::deliver(&plan, &src, &mut want);
            twice(&name, &want, diff_bits_f32, || {
                let s = rt.upload(&src, "deliver src")?;
                let mut d = rt.upload(&dst0, "deliver dst")?;
                k0::deliver(rt, &plan, &s, &mut d)?;
                rt.download(&d)
            })
        }));
    }

    out.extend(guarded("k0.zero", || {
        let dst0 = splitmix_f32(0x2e0, 600, 1.0);
        let plan = match ZeroPlan::new(11, 500, dst0.len()) {
            Ok(p) => p,
            Err(e) => return vec![Check::from_error("k0.zero", &e)],
        };
        let mut want = dst0.clone();
        host_ref::zero(&plan, &mut want);
        twice("k0.zero", &want, diff_bits_f32, || {
            let mut d = rt.upload(&dst0, "zero dst")?;
            k0::zero(rt, &plan, &mut d)?;
            rt.download(&d)
        })
    }));

    out.extend(guarded("k0.scatter_add_rows", || {
        let (dst_rows, width) = (50u64, 33u64);
        // (7i + 3) mod 50 for i < 17: distinct because gcd(7, 50) = 1.
        let pos: Vec<u32> = (0..17u32).map(|i| (7 * i + 3) % 50).collect();
        let src = splitmix_f32(0x5ca, pos.len() * 33, 1.0);
        let dst0 = splitmix_f32(0x5cb, 50 * 33, 1.0);
        let plan = match ScatterAddRowsPlan::new(&pos, width, src.len(), dst_rows, dst0.len()) {
            Ok(p) => p,
            Err(e) => return vec![Check::from_error("k0.scatter_add_rows", &e)],
        };
        let mut want = dst0.clone();
        host_ref::scatter_add_rows(&plan, &src, &pos, &mut want);
        let mut checks = twice("k0.scatter_add_rows", &want, diff_bits_f32, || {
            let s = rt.upload(&src, "scatter src")?;
            let mut d = rt.upload(&dst0, "scatter dst")?;
            k0::scatter_add_rows(rt, &s, &pos, width, dst_rows, &mut d)?;
            rt.download(&d)
        });
        // A repeated row is refused before anything is queued.
        let dup_name = "k0.scatter_add_rows.refuses_duplicates";
        let dup = (|| -> Result<Check, CudaError> {
            let s = rt.upload(&src[..66], "scatter dup src")?;
            let mut d = rt.upload(&dst0, "scatter dup dst")?;
            Ok(
                match k0::scatter_add_rows(rt, &s, &[4, 4], width, dst_rows, &mut d) {
                    Err(CudaError::Invalid { detail, .. }) => Check::pass(dup_name, detail),
                    Err(other) => {
                        Check::fail(dup_name, format!("refused with the wrong error: {other}"))
                    }
                    Ok(()) => Check::fail(dup_name, "a duplicated row was accepted"),
                },
            )
        })();
        checks.push(dup.unwrap_or_else(|e| Check::from_error(dup_name, &e)));
        checks
    }));

    for bf16 in [false, true] {
        let name = if bf16 {
            "k0.ce_gather_rows_bf16"
        } else {
            "k0.ce_gather_rows_f32"
        };
        out.extend(guarded(name, || {
            let (ld, off, hidden) = (37u64, 4u64, 30u64);
            let rows = [3u32, 39, 3, 0, 17];
            let h = splitmix_f32(0x9a7, 40 * 37, 4.0);
            let h16 = f32_slice_to_bf16(&h);
            let mut want = vec![0.0f32; rows.len() * 30];
            let plan = match GatherRowsPlan::new(&rows, hidden, (ld, off, h.len()), want.len()) {
                Ok(p) => p,
                Err(e) => return vec![Check::from_error(name, &e)],
            };
            if bf16 {
                host_ref::ce_gather_rows_bf16(&plan, &h16, &rows, &mut want);
            } else {
                host_ref::ce_gather_rows_f32(&plan, &h, &rows, &mut want);
            }
            twice(name, &want, diff_bits_f32, || {
                let mut out = rt.alloc_zeros::<f32>(want.len(), "gather out")?;
                if bf16 {
                    let hd = rt.upload(&h16, "gather h bf16")?;
                    k0::ce_gather_rows(
                        rt,
                        GatherSource::Bf16(&hd),
                        &rows,
                        hidden,
                        (ld, off),
                        &mut out,
                    )?;
                } else {
                    let hd = rt.upload(&h, "gather h")?;
                    k0::ce_gather_rows(
                        rt,
                        GatherSource::F32(&hd),
                        &rows,
                        hidden,
                        (ld, off),
                        &mut out,
                    )?;
                }
                rt.download(&out)
            })
        }));
    }
    out
}

/// The falsifier for [`CUBLAS_GAP`]: one 64x64x64 `cublasGemmEx` with bf16
/// A/B and an f32 C. Its status is data: `CUBLAS_STATUS_NOT_SUPPORTED` closes
/// the gap negatively; success closes it positively if the result is within
/// `2^-8 * max|ref|` of the float64 product of the same bf16 values.
pub fn cublas_probe(rt: &CudaRuntime) -> Vec<Check> {
    let name = "cublas.gemm_ex_bf16_f32_probe";
    guarded(name, || {
        let shape = match GemmShape::new(64, 64, 64) {
            Ok(s) => s,
            Err(e) => return vec![Check::from_error(name, &e)],
        };
        let a = splitmix_f32(0xb1, shape.a_len(), 1.0);
        let b = splitmix_f32(0xb2, shape.b_len(), 1.0);
        let want = host_ref::gemm_f64(GemmLayout::Nn, shape, Operands::Bf16, &a, &b, None);
        let run = || -> Result<Vec<f32>, CudaError> {
            let a16 = rt.upload(&f32_slice_to_bf16(&a), "probe A")?;
            let b16 = rt.upload(&f32_slice_to_bf16(&b), "probe B")?;
            let mut c = rt.alloc_zeros::<f32>(shape.c_len(), "probe C")?;
            gemm_bf16_cublas(
                rt,
                GemmLayout::Nn,
                shape,
                &a16,
                &b16,
                &mut c,
                Accumulate::Overwrite,
            )?;
            rt.download(&c)
        };
        let check = match run() {
            Ok(got) => match tolerance_vs_f64(&got, &want) {
                Ok(r) => {
                    let bound = 2f64.powi(-8) * r.max_abs_ref;
                    tolerance_check(
                        name,
                        r,
                        bound,
                        "2^-8 * max|ref|, cuda-backend-scoping.md 5.3 rung 0",
                    )
                    .with("cublas_status", "CUBLAS_STATUS_SUCCESS")
                }
                Err(e) => Check::from_error(name, &e),
            },
            Err(e @ CudaError::Cublas { .. }) => {
                let status = match &e {
                    CudaError::Cublas { status, .. } => status.clone(),
                    _ => String::new(),
                };
                Check::fail(
                    name,
                    format!("{e}; the bf16 tier falls back to the FFMA kernel"),
                )
                .with("cublas_status", status)
            }
            Err(e) => Check::from_error(name, &e),
        };
        vec![check.with("gap", CUBLAS_GAP)]
    })
}

/// One GEMM configuration: f32 operands built from tessl's ragged pattern.
#[derive(Clone, Copy, Debug)]
pub struct GemmCase {
    /// Tier, layout, shape, accumulate.
    pub spec: GemmSpec,
    /// Which engine computes the bf16 tier.
    pub engine: Bf16Engine,
}

impl GemmCase {
    /// A stable name, e.g. `k1.ffma.exact_f32.nn.130x70x260`.
    pub fn name(&self) -> String {
        let engine = match (self.spec.operands, self.engine) {
            (Operands::ExactF32, _) | (Operands::Bf16, Bf16Engine::Ffma) => "ffma",
            (Operands::Bf16, Bf16Engine::Cublas) => "cublas",
        };
        let acc = match self.spec.acc {
            Accumulate::Overwrite => "",
            Accumulate::Add => ".acc",
        };
        let s = self.spec.shape;
        format!(
            "k1.{engine}.{}.{}.{}x{}x{}{acc}",
            self.spec.operands.name(),
            self.spec.layout.name(),
            s.m,
            s.n,
            s.k
        )
    }

    fn is_ffma(&self) -> bool {
        self.spec.operands == Operands::ExactF32 || self.engine == Bf16Engine::Ffma
    }
}

/// The cases rung 0 and the device tests sweep: every layout, every tessl
/// ragged shape, ExactF32 FFMA, bf16 FFMA and bf16 cuBLAS, overwrite; plus
/// the accumulate form of each engine at `130x70x260`.
pub fn gemm_cases() -> Result<Vec<GemmCase>, CudaError> {
    let engines = [
        (Operands::ExactF32, Bf16Engine::Ffma),
        (Operands::Bf16, Bf16Engine::Ffma),
        (Operands::Bf16, Bf16Engine::Cublas),
    ];
    let mut out = Vec::new();
    for layout in GemmLayout::ALL {
        for &(m, n, k) in &TESSL_RAGGED_SHAPES {
            let shape = GemmShape::new(m, n, k)?;
            for (operands, engine) in engines {
                for acc in [Accumulate::Overwrite, Accumulate::Add] {
                    if acc == Accumulate::Add && (m, n, k) != (130, 70, 260) {
                        continue;
                    }
                    out.push(GemmCase {
                        spec: GemmSpec {
                            operands,
                            layout,
                            shape,
                            acc,
                        },
                        engine,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// The float64 bound for a case (see the module docs for citations).
pub fn gemm_bound(case: &GemmCase) -> (f64, &'static str) {
    match case.spec.operands {
        Operands::ExactF32 => (
            1e-4 + 1e-7 * case.spec.shape.k as f64,
            "1e-4 + 1e-7*k, tessl/src/gemm.rs:2286",
        ),
        Operands::Bf16 => (
            2e-3,
            "2e-3 on bf16-rounded operands, tessl/src/gemm.rs:2286",
        ),
    }
}

/// Run one case on the device from fresh buffers.
fn run_gemm(
    rt: &CudaRuntime,
    case: &GemmCase,
    a: &[f32],
    b: &[f32],
    c0: &[f32],
) -> Result<Vec<f32>, CudaError> {
    let ad = rt.upload(a, "gemm A")?;
    let bd = rt.upload(b, "gemm B")?;
    let mut cd = rt.upload(c0, "gemm C")?;
    gemm(rt, case.spec, case.engine, &ad, &bd, &mut cd)?;
    rt.download(&cd)
}

/// One case: tolerance vs float64, bitwise vs the host FFMA emulation (FFMA
/// engines), and a bit-identical second run.
pub fn gemm_case_checks(rt: &CudaRuntime, case: &GemmCase) -> Vec<Check> {
    let name = case.name();
    guarded(&name, || {
        let s = case.spec.shape;
        let a = tessl_ragged_a(s.a_len());
        let b = tessl_ragged_b(s.b_len());
        let c0 = match case.spec.acc {
            Accumulate::Overwrite => vec![0.0f32; s.c_len()],
            Accumulate::Add => vec![0.25f32; s.c_len()],
        };
        let prev = (case.spec.acc == Accumulate::Add).then_some(c0.as_slice());
        let first = match run_gemm(rt, case, &a, &b, &c0) {
            Ok(v) => v,
            Err(e) => return vec![Check::from_error(&name, &e)],
        };
        let mut checks = Vec::new();
        let want64 = host_ref::gemm_f64(case.spec.layout, s, case.spec.operands, &a, &b, prev);
        let (bound, source) = gemm_bound(case);
        checks.push(match tolerance_vs_f64(&first, &want64) {
            Ok(r) => tolerance_check(&format!("{name}.vs_f64"), r, bound, source),
            Err(e) => Check::from_error(&name, &e),
        });
        if case.is_ffma() {
            let emul =
                host_ref::gemm_ffma_f32(case.spec.layout, s, case.spec.operands, &a, &b, prev);
            checks.push(bitwise_check(
                &format!("{name}.vs_host_ffma"),
                diff_bits_f32(&first, &emul),
                emul.len(),
            ));
        }
        checks.push(match run_gemm(rt, case, &a, &b, &c0) {
            Ok(second) => bitwise_check(
                &format!("{name}.repeat"),
                diff_bits_f32(&second, &first),
                first.len(),
            ),
            Err(e) => Check::from_error(&format!("{name}.repeat"), &e),
        });
        checks
    })
}

/// `reps` runs of one case, every one bit-identical to the first (tessl's
/// soak, `tessl/src/gemm.rs:2371-2425`, runs 25).
pub fn gemm_determinism(rt: &CudaRuntime, case: &GemmCase, reps: usize) -> Vec<Check> {
    let name = format!("{}.determinism_x{reps}", case.name());
    guarded(&name, || {
        let s = case.spec.shape;
        let a = splitmix_f32(0xd57e, s.a_len(), 1.0);
        let b = splitmix_f32(0xd57f, s.b_len(), 1.0);
        let c0 = vec![0.5f32; s.c_len()];
        let mut base: Option<Vec<f32>> = None;
        for rep in 0..reps {
            let got = match run_gemm(rt, case, &a, &b, &c0) {
                Ok(v) => v,
                Err(e) => return vec![Check::from_error(&name, &e)],
            };
            match &base {
                None => base = Some(got),
                Some(first) => {
                    let d = diff_bits_f32(&got, first);
                    if d.mismatches != 0 {
                        return vec![bitwise_check(&name, d, first.len()).with("rep", rep)];
                    }
                }
            }
        }
        vec![Check::pass(&name, format!("{reps} runs bit-identical")).with("reps", reps)]
    })
}

/// The cuBLAS and FFMA bf16 engines on the same operands agree within the
/// bf16 bound: the FFMA kernel is the on-device oracle for cuBLAS. Not
/// bitwise: cuBLAS's summation order is its own.
pub fn bf16_engines_agree(
    rt: &CudaRuntime,
    layout: GemmLayout,
    (m, n, k): (usize, usize, usize),
) -> Vec<Check> {
    let name = format!("k1.bf16.cublas_vs_ffma.{}.{m}x{n}x{k}", layout.name());
    guarded(&name, || {
        let shape = match GemmShape::new(m, n, k) {
            Ok(s) => s,
            Err(e) => return vec![Check::from_error(&name, &e)],
        };
        let spec = GemmSpec {
            operands: Operands::Bf16,
            layout,
            shape,
            acc: Accumulate::Overwrite,
        };
        let a = tessl_ragged_a(shape.a_len());
        let b = tessl_ragged_b(shape.b_len());
        let c0 = vec![0.0f32; shape.c_len()];
        let ffma = run_gemm(
            rt,
            &GemmCase {
                spec,
                engine: Bf16Engine::Ffma,
            },
            &a,
            &b,
            &c0,
        );
        let cublas = run_gemm(
            rt,
            &GemmCase {
                spec,
                engine: Bf16Engine::Cublas,
            },
            &a,
            &b,
            &c0,
        );
        match (ffma, cublas) {
            (Ok(f), Ok(c)) => {
                let f64s: Vec<f64> = f.iter().map(|&x| f64::from(x)).collect();
                match tolerance_vs_f64(&c, &f64s) {
                    Ok(r) => vec![tolerance_check(
                        &name,
                        r,
                        2e-3,
                        "2e-3, tessl/src/gemm.rs:2286",
                    )],
                    Err(e) => vec![Check::from_error(&name, &e)],
                }
            }
            (Err(e), _) | (_, Err(e)) => vec![Check::from_error(&name, &e)],
        }
    })
}

/// Milestone M0's device checks in rung 0's order, phase by phase: NVRTC,
/// K0, the cuBLAS probe, K1 on tessl's ragged shapes and the engines'
/// agreement, then K1 determinism over `reps` runs per engine. `phase` is
/// called before each phase's checks run (so a wall-clock cap names it),
/// `record` with each batch of results. `runga` runs this; `src/bin/rung0.rs`
/// still carries its own copy of the same sequence (frozen;
/// `GAP-L-CUDA-M1-RUNG0-PRIVATE-WATCHDOG-COPY-2026-10-01`).
pub fn m0_phases(
    rt: &CudaRuntime,
    reps: usize,
    phase: &mut dyn FnMut(&str),
    record: &mut dyn FnMut(Vec<Check>),
) {
    phase("nvrtc");
    record(compile_checks(rt));
    phase("k0");
    record(k0_checks(rt));
    phase("cublas_probe");
    record(cublas_probe(rt));
    phase("k1");
    match gemm_cases() {
        Ok(cases) => {
            for case in &cases {
                record(gemm_case_checks(rt, case));
            }
        }
        Err(e) => record(vec![Check::from_error("k1.cases", &e)]),
    }
    for layout in GemmLayout::ALL {
        record(bf16_engines_agree(rt, layout, (130, 70, 260)));
    }
    phase("k1_determinism");
    match GemmShape::new(130, 70, 260) {
        Ok(shape) => {
            for (operands, engine) in [
                (Operands::ExactF32, Bf16Engine::Ffma),
                (Operands::Bf16, Bf16Engine::Ffma),
                (Operands::Bf16, Bf16Engine::Cublas),
            ] {
                let case = GemmCase {
                    spec: GemmSpec {
                        operands,
                        layout: GemmLayout::Nn,
                        shape,
                        acc: Accumulate::Add,
                    },
                    engine,
                };
                record(gemm_determinism(rt, &case, reps));
            }
        }
        Err(e) => record(vec![Check::from_error("k1.determinism", &e)]),
    }
}
