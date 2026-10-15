//! CPU GDN forward follow-ups: bit identity, rejection, and the release harness.
//!
//! The ignored benches write `bench/results/2026-10-14-cpu-gdn-followups/`.
//! They do not assert on the clock.

use std::fs;
use std::time::Instant;

use ojas_core::{
    Backend, Budget, DType, GdnInputs, Numerics, OjasError, Tensor, GDN_CHECKPOINT_TOKENS,
    RMS_NORM_EPS,
};
use ojas_cpu::CpuBackend;

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// Folds of forward `(y, state, checkpoints)` and backward `(dq, dk, dv, dg, dbeta[, ds0])`
/// for the shapes in `forward_and_backward_bits_match_across_threads_and_exact`,
/// state false then true. Recorded on the unmodified kernel.
const DIGESTS: [&str; 12] = [
    "f1a293ac4e085ac8 377840788e88af66 0000000000000000 e94af006557e59d8,e5f7fa2f0756f7fc,5a82d747e814a7b0,0000000000000000,00000000bbbc03dc",
    "2cf360a37411ac09 708299ee6d45e3d6 02db25d5e8dd622c 59a36244cf35d884,ea0ed7e9455b75db,1020a534f45b5b50,000000003a10f4e6,000000003bd3c14d,3199cb6e50a07170",
    "3573e5bb08db4f9d 34772f4dc01bdec8 0000000000000000 929e7dc1593e750b,eb715ae992e81ff8,ae455ae3f7d35080,0000000000000000,c8cfbf8ee4bc870a",
    "6ecb4a7454308e7e 0eee9be1c752f419 ee3d50c5927d3b04 0034be25d4e15ee0,bdedd3b7ec3bcfaf,ea8e90dad2570de0,7eea1a4701529c4a,2c5f0bd8832a190d,ccef7aa0804b0c20",
    "afce64264d8949d6 e3eea6917af0b977 ec37ac68cd9fda8f bcb16230c7ad640a,c421940b67618db1,e1b1d0f3de042510,a8b4d5408982c0e6,ae34172988e82318",
    "1d35f284ef341be6 3a1a5667db8baaa7 463227fd154d5f54 11e231513e1571cf,12e3e84dc9ad6304,4d5aa87d0968c750,9ab38a84e6128c11,aed8cc19eccad119,7344c32777255b70",
    "4327e5e2826163e3 39debdd1718e0fa7 1a7f201ea07f08e2 7ff4e56196fe3cc2,4bbaf8f60275cc3c,24e8ef95b6fffe28,6f4688e2287e0e11,38b0e8b9e9bba491",
    "2891389e09119e61 487644b8cc983789 1702b9edc0fe8959 88eac9018ce75920,40e2128a444fda47,57da6ace502a2620,eedb5d3394a4dff7,9992c8218a0fa73a,acf46d707fbaf058",
    "6c73df9da06d295e e6674c50f7b2f4ad 0000000000000000 6e87ee16213faa52,22f2a9be7946d5c5,f884abb5579b04a0,c6b033d78829ae02,5ac3755a6c4b80c4",
    "769a9fd0d3317013 732f7e9bd76b2bde e02009750b72a34c 4606faba7bf583c7,92b49013d463ac59,3085c966e56a3bf0,7ace6148c5a40647,49add4014cbfa006,9949c53ac833b770",
    "3a1f41152a26fe3e 6a5a2805c8eb38b2 0000000000000000 87fd5badfd4e06b4,e314a21f428f3afb,953f54d54275d873,dd4e11afa9326263,8a46b6869aa23cb6",
    "57298262a2109c2c b76e40f6cde2a408 c17625457549a814 95be9cd0ab385278,0ff860c884a9eff4,9b1a1b9cf28e2f21,7c5a2e259ee023f4,ee3712563cf26019,54f1504ddf5ab98c",
];

fn fold_bits(words: &[u32]) -> u64 {
    words.iter().fold(0u64, |acc, w| {
        acc.wrapping_mul(0x9E37_79B1_85EB_CA87)
            .wrapping_add(u64::from(*w))
    })
}

fn f32_fill(budget: &Budget, n: usize, shape: &[usize], value: f32) -> Tensor {
    Tensor::from_f32(&vec![value; n], shape, budget).unwrap()
}

fn pattern(n: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 17) as f32 - 8.0) * scale + bias)
        .collect()
}

#[derive(Clone, Copy)]
struct Shape {
    b: usize,
    t: usize,
    h: usize,
    dk: usize,
    dv: usize,
}

struct Case {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Option<Tensor>,
}

fn build(budget: &Budget, s: Shape, with_state: bool) -> Case {
    let rows = s.b * s.t * s.h;
    let qn = rows * s.dk;
    let vn = rows * s.dv;
    let gn = rows;
    let case = Case {
        q: Tensor::from_f32(&pattern(qn, 0.05, 0.1), &[s.b, s.t, s.h, s.dk], budget).unwrap(),
        k: Tensor::from_f32(&pattern(qn, 0.04, -0.2), &[s.b, s.t, s.h, s.dk], budget).unwrap(),
        v: Tensor::from_f32(&pattern(vn, 0.03, 0.05), &[s.b, s.t, s.h, s.dv], budget).unwrap(),
        g: Tensor::from_f32(&pattern(gn, 0.01, -0.05), &[s.b, s.t, s.h], budget).unwrap(),
        beta: Tensor::from_f32(&pattern(gn, 0.02, 0.4), &[s.b, s.t, s.h], budget).unwrap(),
        s0: None,
    };
    if !with_state {
        return case;
    }
    let sn = s.b * s.h * s.dk * s.dv;
    Case {
        s0: Some(
            Tensor::from_f32(&pattern(sn, 0.01, 0.0), &[s.b, s.h, s.dk, s.dv], budget).unwrap(),
        ),
        ..case
    }
}

fn inputs(c: &Case) -> GdnInputs<'_> {
    GdnInputs {
        q: &c.q,
        k: &c.k,
        v: &c.v,
        g: &c.g,
        beta: &c.beta,
        initial_state: c.s0.as_ref(),
    }
}

fn forward_bits(cpu: &CpuBackend, c: &Case) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let y = cpu.chunked_gdn_forward(inputs(c)).unwrap();
    (bits(&y.output), bits(&y.final_state), bits(&y.checkpoints))
}

fn backward_bits(cpu: &CpuBackend, c: &Case, grad_state: bool) -> Vec<Vec<u32>> {
    let y = cpu.chunked_gdn_forward(inputs(c)).unwrap();
    let gy = f32_fill(
        cpu.budget(),
        y.output.num_elements().unwrap(),
        y.output.shape(),
        0.02,
    );
    let gs = if grad_state {
        Some(f32_fill(
            cpu.budget(),
            y.final_state.num_elements().unwrap(),
            y.final_state.shape(),
            0.01,
        ))
    } else {
        None
    };
    let g = cpu
        .chunked_gdn_backward(inputs(c), &y.checkpoints, &gy, gs.as_ref())
        .unwrap();
    let mut out = vec![
        bits(&g.q),
        bits(&g.k),
        bits(&g.v),
        bits(&g.g),
        bits(&g.beta),
    ];
    if let Some(ds) = g.initial_state {
        out.push(bits(&ds));
    }
    out
}

fn cpu(threads: usize, numerics: Numerics) -> CpuBackend {
    let budget = Budget::new(1 << 30);
    let cpu = if threads == 1 {
        CpuBackend::new(budget)
    } else {
        CpuBackend::with_threads(budget, threads).unwrap()
    };
    cpu.with_numerics(numerics)
}

#[test]
fn forward_and_backward_bits_match_across_threads_and_exact() {
    let shapes = [
        Shape {
            b: 1,
            t: 1,
            h: 1,
            dk: 4,
            dv: 4,
        },
        Shape {
            b: 2,
            t: 1,
            h: 1,
            dk: 4,
            dv: 8,
        },
        Shape {
            b: 1,
            t: GDN_CHECKPOINT_TOKENS + 1,
            h: 1,
            dk: 4,
            dv: 6,
        },
        Shape {
            b: 1,
            t: GDN_CHECKPOINT_TOKENS + 1,
            h: 3,
            dk: 4,
            dv: 6,
        },
        Shape {
            b: 2,
            t: 3,
            h: 4,
            dk: 8,
            dv: 4,
        },
        Shape {
            b: 2,
            t: 5,
            h: 1,
            dk: 3,
            dv: 5,
        },
    ];
    for (i, shape) in shapes.iter().enumerate() {
        for with_state in [false, true] {
            let fast = cpu(1, Numerics::Fast);
            let wide = cpu(6, Numerics::Fast);
            let exact = cpu(6, Numerics::Exact);
            let built = build(fast.budget(), Shape { ..*shape }, with_state);
            // Tensors are bound to `fast`'s budget. The other backends only
            // read them and charge their own outputs, so the values are shared
            // by cloning onto each budget.
            let on = |backend: &CpuBackend| {
                let b = backend.budget();
                Case {
                    q: Tensor::from_f32(&built.q.to_f32_vec().unwrap(), built.q.shape(), b)
                        .unwrap(),
                    k: Tensor::from_f32(&built.k.to_f32_vec().unwrap(), built.k.shape(), b)
                        .unwrap(),
                    v: Tensor::from_f32(&built.v.to_f32_vec().unwrap(), built.v.shape(), b)
                        .unwrap(),
                    g: Tensor::from_f32(&built.g.to_f32_vec().unwrap(), built.g.shape(), b)
                        .unwrap(),
                    beta: Tensor::from_f32(
                        &built.beta.to_f32_vec().unwrap(),
                        built.beta.shape(),
                        b,
                    )
                    .unwrap(),
                    s0: built
                        .s0
                        .as_ref()
                        .map(|s| Tensor::from_f32(&s.to_f32_vec().unwrap(), s.shape(), b).unwrap()),
                }
            };
            let a = on(&fast);
            let b = on(&wide);
            let e = on(&exact);
            let fa = forward_bits(&fast, &a);
            let fb = forward_bits(&wide, &b);
            let fe = forward_bits(&exact, &e);
            assert_eq!(fa, fb, "forward shape {i} state {with_state} threads");
            assert_eq!(fa, fe, "forward shape {i} state {with_state} exact");
            let ba = backward_bits(&fast, &a, with_state);
            let bb = backward_bits(&wide, &b, with_state);
            let be = backward_bits(&exact, &e, with_state);
            assert_eq!(ba, bb, "backward shape {i} state {with_state} threads");
            assert_eq!(ba, be, "backward shape {i} state {with_state} exact");
            let got = format!(
                "{:016x} {:016x} {:016x} {}",
                fold_bits(&fa.0),
                fold_bits(&fa.1),
                fold_bits(&fa.2),
                ba.iter()
                    .map(|w| format!("{:016x}", fold_bits(w)))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            assert_eq!(
                got,
                DIGESTS[i * 2 + usize::from(with_state)],
                "shape {i} state {with_state}"
            );
        }
    }
}

#[test]
fn non_finite_q_k_or_v_is_rejected_and_cancellation_leaves_the_backend_usable() {
    let budget = Budget::new(1 << 20);
    let cpu = CpuBackend::with_threads(budget.clone(), 4).unwrap();
    let shape_q = [1usize, 4, 2, 4];
    let shape_v = [1usize, 4, 2, 4];
    let shape_g = [1usize, 4, 2];
    let g = pattern(8, 0.01, -0.05);
    let beta = pattern(8, 0.02, 0.4);

    for name in ["q", "k", "v"] {
        let mut q = pattern(32, 0.05, 0.1);
        let mut k = pattern(32, 0.04, -0.2);
        let mut v = pattern(32, 0.03, 0.05);
        match name {
            "q" => q[3] = f32::NAN,
            "k" => k[3] = f32::NAN,
            "v" => v[7] = f32::NAN,
            _ => unreachable!(),
        }
        let budget = Budget::new(1 << 20);
        let cpu = CpuBackend::new(budget.clone());
        let tq = Tensor::from_f32(&q, &shape_q, &budget).unwrap();
        let tk = Tensor::from_f32(&k, &shape_q, &budget).unwrap();
        let tv = Tensor::from_f32(&v, &shape_v, &budget).unwrap();
        let tg = Tensor::from_f32(&g, &shape_g, &budget).unwrap();
        let tb = Tensor::from_f32(&beta, &shape_g, &budget).unwrap();
        let held = budget.live_bytes().unwrap();
        let err = cpu
            .chunked_gdn_forward(GdnInputs {
                q: &tq,
                k: &tk,
                v: &tv,
                g: &tg,
                beta: &tb,
                initial_state: None,
            })
            .unwrap_err();
        assert!(
            matches!(err, OjasError::NonFinite { .. }),
            "{name}: {err:?}"
        );
        assert_eq!(budget.live_bytes().unwrap(), held, "{name} leaked a charge");
    }
    let q = pattern(32, 0.05, 0.1);
    let k = pattern(32, 0.04, -0.2);
    let v = pattern(32, 0.03, 0.05);

    let tq = Tensor::from_f32(&q, &shape_q, cpu.budget()).unwrap();
    let tk = Tensor::from_f32(&k, &shape_q, cpu.budget()).unwrap();
    let tv = Tensor::from_f32(&v, &shape_v, cpu.budget()).unwrap();
    let tg = Tensor::from_f32(&g, &shape_g, cpu.budget()).unwrap();
    let tb = Tensor::from_f32(&beta, &shape_g, cpu.budget()).unwrap();
    let call = || {
        cpu.chunked_gdn_forward(GdnInputs {
            q: &tq,
            k: &tk,
            v: &tv,
            g: &tg,
            beta: &tb,
            initial_state: None,
        })
    };
    cpu.set_cancel(|| {
        Err(OjasError::Unsupported {
            op: "cancel",
            detail: "stop".to_string(),
        })
    });
    let held = cpu.budget().live_bytes().unwrap();
    let err = call().unwrap_err();
    assert!(
        matches!(err, OjasError::Unsupported { op: "cancel", .. }),
        "{err:?}"
    );
    assert_eq!(cpu.budget().live_bytes().unwrap(), held);
    // A second refusal is the same error, not a stuck backend or a success
    // that kept a partial write.
    let err = call().unwrap_err();
    assert!(
        matches!(err, OjasError::Unsupported { op: "cancel", .. }),
        "{err:?}"
    );
    cpu.set_cancel(|| Ok(()));
    let y = call().unwrap();
    assert!(y.output.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
    assert!(y
        .final_state
        .to_f32_vec()
        .unwrap()
        .iter()
        .all(|v| v.is_finite()));
}

#[test]
fn malformed_shapes_and_an_overflowing_product_are_refused_not_panicked() {
    let budget = Budget::new(1 << 20);
    let cpu = CpuBackend::new(budget.clone());
    let q = f32_fill(&budget, 4, &[1, 1, 4], 0.1);
    let k = f32_fill(&budget, 8, &[1, 1, 1, 8], 0.1);
    let v = f32_fill(&budget, 8, &[1, 1, 1, 8], 0.1);
    let g = f32_fill(&budget, 1, &[1, 1, 1], 0.0);
    let beta = f32_fill(&budget, 1, &[1, 1, 1], 0.5);
    let err = cpu
        .chunked_gdn_forward(GdnInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            initial_state: None,
        })
        .unwrap_err();
    assert!(matches!(err, OjasError::Shape { .. }), "{err:?}");

    let zero = Tensor::from_f32(&[], &[1, 0, 1, 4], &budget).unwrap();
    let err = cpu
        .chunked_gdn_forward(GdnInputs {
            q: &zero,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            initial_state: None,
        })
        .unwrap_err();
    assert!(matches!(err, OjasError::Shape { .. }), "{err:?}");

    let err = Tensor::zeros(&[usize::MAX, usize::MAX], DType::F32, &budget).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::OutOfRange { .. } | OjasError::CapacityExceeded { .. }
        ),
        "{err:?}"
    );

    let err = Tensor::from_f32(&[1.0], &[usize::MAX / 2, 8], &budget).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::OutOfRange { .. } | OjasError::CapacityExceeded { .. }
        ),
        "{err:?}"
    );
}

/// Peak bytes for a serial forward (`CpuBackend::new`, so `inflight` is 1:
/// these shapes are under `2 * TASK_WORK`).
///
/// Already live: the input tensors. `fill_outs` then charges the output,
/// the final state and the checkpoints, and `room_for` charges
/// `rows * 2 * Dk` normalized q and k, plus `rows * Dv` head-major output
/// only when `H > 1`, plus one task (`Dk * Dv + 2 * Dv`). Each slot is 4
/// bytes (`payload_bytes` / `Scratch::try_alloc`). The previous forward
/// also charged two norm slots per row.
///
/// H = 1, B = 1, T = 2, Dk = Dv = 4, no state: inputs 28 f32 = 112;
/// outputs 8 + 16 + 16 = 40 f32 = 160; scratch 16 + 0 + 24 = 40 f32 = 160;
/// cap 432. H = 2: inputs 56 f32 = 224; outputs 16 + 32 + 32 = 80 f32 = 320;
/// scratch 32 + 16 + 24 = 72 f32 = 288; cap 832.
#[test]
fn forward_budget_matches_the_scratch_that_is_allocated() {
    fn run(h: usize, cap: u64, expect_ok: bool) {
        let budget = Budget::new(cap);
        let cpu = CpuBackend::new(budget.clone());
        let case = build(
            &budget,
            Shape {
                b: 1,
                t: 2,
                h,
                dk: 4,
                dv: 4,
            },
            false,
        );
        let inputs_live = budget.live_bytes().unwrap();
        let got = cpu.chunked_gdn_forward(inputs(&case));
        if expect_ok {
            let y = got.expect("cap holds outputs plus the new scratch");
            let outs = (y.output.num_elements().unwrap()
                + y.final_state.num_elements().unwrap()
                + y.checkpoints.num_elements().unwrap()) as u64
                * 4;
            assert_eq!(budget.live_bytes().unwrap(), inputs_live + outs);
            assert!(y.output.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
        } else {
            let err = got.expect_err("one byte under the peak must refuse");
            assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
            assert_eq!(budget.live_bytes().unwrap(), inputs_live);
        }
    }
    run(1, 432, true);
    run(1, 431, false);
    run(2, 832, true);
    run(2, 831, false);
}

#[test]
fn mul_forward_bits_match_at_one_thread_and_six() {
    let n = 1024 * 1024;
    let one = CpuBackend::new(Budget::new(1 << 28));
    let six = CpuBackend::with_threads(Budget::new(1 << 28), 6).unwrap();
    let run = |cpu: &CpuBackend| {
        let a = f32_fill(cpu.budget(), n, &[n], 1.25);
        let b = Tensor::from_f32(&pattern(n, 0.01, 0.5), &[n], cpu.budget()).unwrap();
        bits(&cpu.mul_forward(&a, &b).unwrap())
    };
    assert_eq!(run(&one), run(&six));
}

fn loadavg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn min_median(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = samples.len() / 2;
    let median = if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    };
    (samples[0], median)
}

fn time_us(warmup: usize, reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    min_median(&mut samples)
}

/// Release timings of the public CPU API on the tree this binary was built
/// from. Not a pass/fail clock check.
#[test]
#[ignore = "release measurement; writes bench/results"]
fn followup_bench_public() {
    const WARMUP: usize = 3;
    const REPS: usize = 20;
    let mut lines = vec![format!(
        "public API baseline-or-after. warmup {WARMUP} reps {REPS} min/median us. threads noted per row. load {}",
        loadavg()
    )];

    let gdn_shapes = [
        (1usize, 512usize, 1usize, 128usize, 128usize),
        (1, 512, 16, 128, 128),
        (4, 512, 1, 128, 128),
    ];
    for &(b, t, h, dk, dv) in &gdn_shapes {
        for &threads in &[1usize, 6] {
            let cpu = cpu(threads, Numerics::Fast);
            let case = build(cpu.budget(), Shape { b, t, h, dk, dv }, false);
            let (mn, med) = time_us(WARMUP, REPS, || {
                let y = cpu.chunked_gdn_forward(inputs(&case)).unwrap();
                std::hint::black_box(y.output.num_elements().unwrap());
            });
            lines.push(format!(
                "gdn_fwd B={b} T={t} H={h} Dk={dk} Dv={dv} threads={threads} min_us={mn:.1} med_us={med:.1}"
            ));
            eprintln!("{}", lines.last().unwrap());
        }
    }
    // Backward spot check, same H>1 shape the forward copy still takes.
    {
        let threads = 6usize;
        let cpu = cpu(threads, Numerics::Fast);
        let case = build(
            cpu.budget(),
            Shape {
                b: 1,
                t: 512,
                h: 16,
                dk: 128,
                dv: 128,
            },
            false,
        );
        let y = cpu.chunked_gdn_forward(inputs(&case)).unwrap();
        let gy = f32_fill(
            cpu.budget(),
            y.output.num_elements().unwrap(),
            y.output.shape(),
            0.02,
        );
        let (mn, med) = time_us(1, 8, || {
            let g = cpu
                .chunked_gdn_backward(inputs(&case), &y.checkpoints, &gy, None)
                .unwrap();
            std::hint::black_box(g.q.num_elements().unwrap());
        });
        lines.push(format!(
            "gdn_bwd B=1 T=512 H=16 Dk=128 Dv=128 threads={threads} reps=8 warmup=1 min_us={mn:.1} med_us={med:.1}"
        ));
        eprintln!("{}", lines.last().unwrap());
    }

    // Public mul at the piece-rule sizes, whatever cut production uses.
    for &threads in &[1usize, 4, 6, 8] {
        let cpu = cpu(threads, Numerics::Fast);
        for &n in &[1024 * 1024, 1024 * 2048, 1024 * 4096] {
            let a = f32_fill(cpu.budget(), n, &[n], 1.25);
            let b = f32_fill(cpu.budget(), n, &[n], 0.75);
            let (mn, med) = time_us(WARMUP, REPS, || {
                let y = cpu.mul_forward(&a, &b).unwrap();
                std::hint::black_box(y.num_elements().unwrap());
            });
            lines.push(format!(
                "mul_fwd n={n} threads={threads} min_us={mn:.1} med_us={med:.1}"
            ));
            eprintln!("{}", lines.last().unwrap());
        }
    }

    // Small and large RMS, a public `chunks_into` pass (row split).
    for &threads in &[1usize, 2, 4, 6] {
        let cpu = cpu(threads, Numerics::Fast);
        for &(rows, dim, label) in &[(1536usize, 128usize, "small"), (65536, 128, "large")] {
            let n = rows * dim;
            let x = f32_fill(cpu.budget(), n, &[rows, dim], 0.5);
            let w = f32_fill(cpu.budget(), dim, &[dim], 1.0);
            let (mn, med) = time_us(WARMUP, REPS, || {
                let y = cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS).unwrap();
                std::hint::black_box(y.num_elements().unwrap());
            });
            lines.push(format!(
                "rms_fwd {label} rows={rows} dim={dim} threads={threads} min_us={mn:.1} med_us={med:.1}"
            ));
            eprintln!("{}", lines.last().unwrap());
        }
    }

    // Scoped spawn the pool pays: 8 MiB stack, 1 extra thread vs 5.
    for &extra in &[1usize, 5] {
        let (mn, med) = time_us(5, 50, || {
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(extra);
                for _ in 0..extra {
                    handles.push(
                        std::thread::Builder::new()
                            .stack_size(8 << 20)
                            .spawn_scoped(scope, || 1u64)
                            .unwrap(),
                    );
                }
                let mut acc = 0u64;
                for handle in handles {
                    acc += handle.join().unwrap();
                }
                std::hint::black_box(acc);
            });
        });
        lines.push(format!(
            "spawn_extra={extra} stack=8MiB reps=50 warmup=5 min_us={mn:.1} med_us={med:.1}"
        ));
        eprintln!("{}", lines.last().unwrap());
    }

    let dir = format!(
        "{}/../bench/results/2026-10-14-cpu-gdn-followups",
        env!("CARGO_MANIFEST_DIR")
    );
    fs::create_dir_all(&dir).unwrap();
    let tag = std::env::var("OJAS_FOLLOWUP_TAG").unwrap_or_default();
    let name = if tag.is_empty() {
        "public.txt".to_string()
    } else {
        format!("public-{tag}.txt")
    };
    let path = format!("{dir}/{name}");
    fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {path}");
}
