//! Exact-numerics outputs pinned to the bits of the pre-pool kernels.
//!
//! Each digest is FNV-1a over the output bits of one op on a fixed fixture,
//! recorded from the per-family kernels (`gemm_rows4`, `grad_x_quad`,
//! `grad_w_panel8`, the serial attention/norm/optimizer loops) before the
//! packed GEMM and worker pool replaced them. Every thread count must
//! reproduce them.
//!
//! The twelve `sdpa_*` digests and `ce_grad` were re-recorded on 2026-10-05
//! when exact softmax and cross-entropy moved from libm `f32::exp` to the
//! correctly rounded `exp_exact`; no other digest moved. Libm `expf` is not
//! the same everywhere: `expf(-2^-25)` is `0x3f800000` on glibc 2.41 and
//! `0x3f7fffff` on macOS 27.
//!
//! Every digest but `ce_loss` is now plain `f32` arithmetic, `sqrt` and
//! `exp_exact`, so it holds on every platform. `ce_loss` still takes the
//! log-sum from libm `f32::ln`; it is checked only on Apple silicon, where it
//! was recorded, and a macOS update that changes `ln` would move it.
//! Elsewhere its thread-count check still runs.

use ojas_core::{AdamWConfig, Backend, Budget, MuonNs5Config, Numerics, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{f32t, u32t, SplitMix64};

const THREADS: [usize; 6] = [1, 2, 3, 7, 16, 18];

fn fnv(values: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn backend(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 30), threads)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

/// Linear shapes `(rows, in, out)`: 1x1, primes, both sides of 64/96/128/256
/// block edges, and one product above every parallel threshold.
const LINEAR: [(usize, usize, usize); 12] = [
    (1, 1, 1),
    (3, 5, 9),
    (73, 11, 13),
    (127, 67, 131),
    (64, 128, 96),
    (70, 80, 100),
    (257, 129, 65),
    (65, 257, 129),
    (129, 65, 257),
    (97, 255, 95),
    (33, 1000, 17),
    (200, 300, 520),
];

fn digests(cpu: &CpuBackend) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for (i, &(rows, kin, nout)) in LINEAR.iter().enumerate() {
        let mut rng = SplitMix64(100 + i as u64);
        let x = f32t(cpu, &rng.vec(rows * kin, 0.5), &[rows, kin]);
        let w = f32t(cpu, &rng.vec(nout * kin, 0.5), &[nout, kin]);
        let gy = f32t(cpu, &rng.vec(rows * nout, 0.5), &[rows, nout]);
        let y = cpu.linear_forward(&x, &w).unwrap();
        let (gx, gw) = cpu.linear_backward(&x, &w, &gy).unwrap();
        out.push((format!("linear_fwd {rows}x{kin}x{nout}"), fnv(&flat(&y))));
        out.push((format!("linear_gx {rows}x{kin}x{nout}"), fnv(&flat(&gx))));
        out.push((format!("linear_gw {rows}x{kin}x{nout}"), fnv(&flat(&gw))));
    }
    for (i, &(rows, cols)) in [(64usize, 48usize), (48, 64), (130, 70), (97, 97)]
        .iter()
        .enumerate()
    {
        let mut rng = SplitMix64(200 + i as u64);
        let mut p = f32t(cpu, &rng.vec(rows * cols, 0.1), &[rows, cols]);
        let g = f32t(cpu, &rng.vec(rows * cols, 0.1), &[rows, cols]);
        let mut m = f32t(cpu, &rng.vec(rows * cols, 0.01), &[rows, cols]);
        cpu.muon_ns5_step(&mut p, &g, &mut m, MuonNs5Config::nanolab_default())
            .unwrap();
        out.push((format!("muon_p {rows}x{cols}"), fnv(&flat(&p))));
        out.push((format!("muon_m {rows}x{cols}"), fnv(&flat(&m))));
    }
    for (i, shape) in [[2usize, 3, 33, 24], [1, 2, 300, 64], [2, 2, 64, 16]]
        .iter()
        .enumerate()
    {
        let n: usize = shape.iter().product();
        let mut rng = SplitMix64(300 + i as u64);
        let q = f32t(cpu, &rng.vec(n, 0.5), shape);
        let k = f32t(cpu, &rng.vec(n, 0.5), shape);
        let v = f32t(cpu, &rng.vec(n, 0.5), shape);
        let gy = f32t(cpu, &rng.vec(n, 0.5), shape);
        let y = cpu.causal_sdpa_forward(&q, &k, &v).unwrap();
        let (gq, gk, gv) = cpu.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
        let tag = format!("{shape:?}");
        out.push((format!("sdpa_fwd {tag}"), fnv(&flat(&y))));
        out.push((format!("sdpa_gq {tag}"), fnv(&flat(&gq))));
        out.push((format!("sdpa_gk {tag}"), fnv(&flat(&gk))));
        out.push((format!("sdpa_gv {tag}"), fnv(&flat(&gv))));
    }
    {
        let (rows, dim) = (300usize, 256usize);
        let mut rng = SplitMix64(400);
        let x = f32t(cpu, &rng.vec(rows * dim, 1.0), &[rows, dim]);
        let w: Vec<f32> = rng.vec(dim, 0.1).iter().map(|v| 1.0 + v).collect();
        let w = f32t(cpu, &w, &[dim]);
        let gy = f32t(cpu, &rng.vec(rows * dim, 1.0), &[rows, dim]);
        let y = cpu.rms_norm_forward(&x, &w, 1e-6).unwrap();
        let (gx, gw) = cpu.rms_norm_backward(&x, &w, &gy, 1e-6).unwrap();
        out.push(("rms_fwd".into(), fnv(&flat(&y))));
        out.push(("rms_gx".into(), fnv(&flat(&gx))));
        out.push(("rms_gw".into(), fnv(&flat(&gw))));
    }
    {
        let (b, t, h, d) = (2usize, 64usize, 4usize, 32usize);
        let mut rng = SplitMix64(500);
        let x = f32t(cpu, &rng.vec(b * t * h * d, 1.0), &[b, t, h, d]);
        let cos = f32t(cpu, &rng.vec(t * d, 1.0), &[t, d]);
        let sin = f32t(cpu, &rng.vec(t * d, 1.0), &[t, d]);
        let y = cpu.rope_half_split_forward(&x, &cos, &sin).unwrap();
        let gx = cpu.rope_half_split_backward(&x, &cos, &sin).unwrap();
        out.push(("rope_fwd".into(), fnv(&flat(&y))));
        out.push(("rope_bwd".into(), fnv(&flat(&gx))));
    }
    {
        let (rows, vocab) = (257usize, 500usize);
        let mut rng = SplitMix64(600);
        let logits = f32t(cpu, &rng.vec(rows * vocab, 4.0), &[rows, vocab]);
        let targets: Vec<u32> = (0..rows)
            .map(|i| {
                if i % 11 == 3 {
                    7
                } else {
                    rng.below(vocab) as u32
                }
            })
            .collect();
        let targets = u32t(cpu, &targets, &[rows]);
        let loss = cpu
            .cross_entropy_mean_forward(&logits, &targets, Some(7))
            .unwrap();
        let grad = cpu
            .cross_entropy_mean_backward(&logits, &targets, Some(7))
            .unwrap();
        out.push(("ce_loss".into(), fnv(&flat(&loss))));
        out.push(("ce_grad".into(), fnv(&flat(&grad))));
    }
    {
        let n = 100_003usize;
        let mut rng = SplitMix64(700);
        let mut p = f32t(cpu, &rng.vec(n, 1.0), &[n]);
        let g = f32t(cpu, &rng.vec(n, 0.1), &[n]);
        let mut m1 = f32t(cpu, &rng.vec(n, 0.01), &[n]);
        let m2v: Vec<f32> = rng.vec(n, 0.01).iter().map(|v| v.abs()).collect();
        let mut m2 = f32t(cpu, &m2v, &[n]);
        cpu.adamw_step(
            &mut p,
            &g,
            &mut m1,
            &mut m2,
            9,
            AdamWConfig::nanolab(3e-3, 0.1),
        )
        .unwrap();
        out.push(("adamw_p".into(), fnv(&flat(&p))));
        out.push(("adamw_m1".into(), fnv(&flat(&m1))));
        out.push(("adamw_m2".into(), fnv(&flat(&m2))));
    }
    out
}

/// Digests that still depend on the platform libm (`ln`), checked only
/// where they were recorded.
const LIBM_LN: &[&str] = &["ce_loss"];
const RECORDED_HERE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

/// Recorded at one thread on Apple silicon: from the pre-pool kernels, and
/// the `sdpa_*` and `ce_grad` rows again with `exp_exact` (see the header).
const GOLDEN: &[(&str, u64)] = &[
    ("linear_fwd 1x1x1", 0x13589361bf99bd02),
    ("linear_gx 1x1x1", 0x20c93d8a4baa555a),
    ("linear_gw 1x1x1", 0x05554feafed63906),
    ("linear_fwd 3x5x9", 0x051f15f495bb4c8f),
    ("linear_gx 3x5x9", 0x3deaaf5f87fbc938),
    ("linear_gw 3x5x9", 0x554a2a73c8b6b96c),
    ("linear_fwd 73x11x13", 0x064c1e2793b06b86),
    ("linear_gx 73x11x13", 0x038773853eeb7a14),
    ("linear_gw 73x11x13", 0x8287fc31225e37e0),
    ("linear_fwd 127x67x131", 0xb4f93375d1ba94c0),
    ("linear_gx 127x67x131", 0x7bfaf16f0316af8a),
    ("linear_gw 127x67x131", 0x45ed4681de3fe65f),
    ("linear_fwd 64x128x96", 0x11f5ae443161bdcd),
    ("linear_gx 64x128x96", 0x7625a93b16ed40ac),
    ("linear_gw 64x128x96", 0xcd5abe07b40bc69c),
    ("linear_fwd 70x80x100", 0x41492c6f8061de58),
    ("linear_gx 70x80x100", 0x9b3c58322b0104cd),
    ("linear_gw 70x80x100", 0x3effce559f004041),
    ("linear_fwd 257x129x65", 0x856addfcd3fe0025),
    ("linear_gx 257x129x65", 0x8a8cf211d61a65c5),
    ("linear_gw 257x129x65", 0xba5bf57f4c6c2e3f),
    ("linear_fwd 65x257x129", 0x927d9beba11360bb),
    ("linear_gx 65x257x129", 0x4f4a2c8f748d08b3),
    ("linear_gw 65x257x129", 0xdb94a20c7fc84254),
    ("linear_fwd 129x65x257", 0xd112c28a10a816fd),
    ("linear_gx 129x65x257", 0x68c65518236c1da3),
    ("linear_gw 129x65x257", 0x134a01141b62f748),
    ("linear_fwd 97x255x95", 0x3083d06745550c49),
    ("linear_gx 97x255x95", 0xb348bdd547869bac),
    ("linear_gw 97x255x95", 0x7b92eb4033da1c10),
    ("linear_fwd 33x1000x17", 0x26d2723873223074),
    ("linear_gx 33x1000x17", 0xbcda32e0f817b439),
    ("linear_gw 33x1000x17", 0x57bf4be2c33228a1),
    ("linear_fwd 200x300x520", 0xfb50ad4ec9a0e7c5),
    ("linear_gx 200x300x520", 0x0c34fdf44903b2c6),
    ("linear_gw 200x300x520", 0x8c870b66cbc1e789),
    ("muon_p 64x48", 0x65cb51fe53eb8a32),
    ("muon_m 64x48", 0xbbe854ecdd0d7f33),
    ("muon_p 48x64", 0xbaff4c833efceb4f),
    ("muon_m 48x64", 0x9ca3993e33c2c47c),
    ("muon_p 130x70", 0x2c43f77936251e31),
    ("muon_m 130x70", 0xf6c65cf8d4c4b1de),
    ("muon_p 97x97", 0xdb6ad320000effe0),
    ("muon_m 97x97", 0x79f473d9606dbbc2),
    ("sdpa_fwd [2, 3, 33, 24]", 0xe99bd019cac5e4ba),
    ("sdpa_gq [2, 3, 33, 24]", 0x2ec8021bbfc4c690),
    ("sdpa_gk [2, 3, 33, 24]", 0x2e7d56a3a5239844),
    ("sdpa_gv [2, 3, 33, 24]", 0xdc4e2d81ca41df77),
    ("sdpa_fwd [1, 2, 300, 64]", 0x4eea62887d667b58),
    ("sdpa_gq [1, 2, 300, 64]", 0x61f643902ee5a8f8),
    ("sdpa_gk [1, 2, 300, 64]", 0x6677948bb0f1d205),
    ("sdpa_gv [1, 2, 300, 64]", 0x737f62859c95ca80),
    ("sdpa_fwd [2, 2, 64, 16]", 0x0ee154d24d440900),
    ("sdpa_gq [2, 2, 64, 16]", 0x163a731421ff0006),
    ("sdpa_gk [2, 2, 64, 16]", 0x7ecb25193fa2bdd0),
    ("sdpa_gv [2, 2, 64, 16]", 0x9f24340773f5220c),
    ("rms_fwd", 0xabd4e1316a27eb03),
    ("rms_gx", 0xc9443895dbb5a0be),
    ("rms_gw", 0x4f69a30da2e929e3),
    ("rope_fwd", 0x67aef6f4d3e7c4bd),
    ("rope_bwd", 0x19893dad351dc381),
    ("ce_loss", 0x594e1e25e50639b4),
    ("ce_grad", 0x27da765c8a034a70),
    ("adamw_p", 0x44272ffc704755ad),
    ("adamw_m1", 0x23408c00ef235be5),
    ("adamw_m2", 0x14473793444e7cb3),
];

#[test]
#[ignore]
fn print_exact_digests() {
    for (name, digest) in digests(&backend(1)) {
        println!("    (\"{name}\", 0x{digest:016x}),");
    }
}

#[test]
fn exact_outputs_do_not_depend_on_thread_count() {
    let want = digests(&backend(1));
    assert!(!want.is_empty(), "no digests");
    for threads in THREADS {
        let got = digests(&backend(threads));
        assert_eq!(got.len(), want.len());
        for ((name, digest), (want_name, want)) in got.iter().zip(&want) {
            assert_eq!(name, want_name);
            assert_eq!(
                digest, want,
                "{name} threads {threads}: 0x{digest:016x} != one thread 0x{want:016x}"
            );
        }
    }
}

#[test]
fn exact_outputs_match_pre_pool_kernels_at_every_thread_count() {
    assert!(!GOLDEN.is_empty(), "golden digests missing");
    for threads in THREADS {
        let got = digests(&backend(threads));
        assert_eq!(got.len(), GOLDEN.len());
        for ((name, digest), (want_name, want)) in got.iter().zip(GOLDEN) {
            assert_eq!(name, want_name);
            if !RECORDED_HERE && LIBM_LN.contains(&name.as_str()) {
                if threads == THREADS[0] {
                    eprintln!("SKIP {name}: libm ln, golden recorded on Apple silicon");
                }
                continue;
            }
            assert_eq!(
                *digest, *want,
                "{name} threads {threads}: 0x{digest:016x} != golden 0x{want:016x}"
            );
        }
    }
}
