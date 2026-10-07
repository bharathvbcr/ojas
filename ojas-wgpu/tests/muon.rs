//! `WgpuBackend::muon_ns5_step` against `CpuBackend` (the f32 Newton-Schulz
//! reference in `ojas-cpu/src/optim.rs`): wide, tall and square matrices,
//! the nanolab shapes, momentum carried over steps, Nesterov on and off,
//! weight decay on and off, and the fail-closed paths (including
//! `Ns5Precision::Bf16`, which this backend refuses).
//!
//! Tolerance is the shared `TOL * max(1, max |cpu|)` over each tensor.

mod common;

use common::*;
use ojas_core::{Backend, DType, MuonNs5Config, Ns5Precision, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

fn bits(g: &WgpuBackend, t: &Tensor) -> Vec<u32> {
    g.download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn run(rows: usize, cols: usize, steps: u64, cfg: MuonNs5Config, seed: u64) {
    let c = cpu();
    let mut hp = host(seed, &[rows, cols]);
    let mut hm = Tensor::zeros(&[rows, cols], DType::F32, host_budget()).unwrap();
    let mut dp = up(&hp);
    let mut dm = up(&hm);
    let g = own();
    for step in 0..steps {
        let grad = host(seed + 100 + step, &[rows, cols]);
        let dg = up(&grad);
        let before = readbacks(&g);
        g.muon_ns5_step(&mut dp, &dg, &mut dm, cfg)
            .unwrap_or_else(|e| panic!("{rows}x{cols} step {step}: {e}"));
        assert_eq!(
            readbacks(&g),
            before,
            "muon {rows}x{cols} read a tensor back"
        );
        c.muon_ns5_step(&mut hp, &grad, &mut hm, cfg).unwrap();
    }
    let name = format!(
        "muon {rows}x{cols} nesterov={} wd={}",
        cfg.nesterov, cfg.weight_decay
    );
    // Positive control: a download through `g` moves its own counter.
    let before = readbacks(&g);
    g.download(&dp).unwrap();
    assert_eq!(
        readbacks(&g).0,
        before.0 + 1,
        "{name}: counter did not move"
    );
    close(&format!("{name} param"), &dp, &hp);
    close(&format!("{name} momentum"), &dm, &hm);
}

/// `|gpu - cpu| <= TOL * max |cpu|` with no floor at 1: the orthogonalized
/// update is small (entries near `1/sqrt(cols)`), so an absolute floor would
/// hide a wrong update.
fn relative(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(f64::from(v.abs())));
    assert!(scale > 0.0, "{name}: reference is all zero");
    let worst = ojas_kernels::max_abs(got, want).unwrap();
    assert!(
        worst <= TOL * scale,
        "{name}: |err| {worst:.3e} > {TOL:.1e} * {scale:.3e}"
    );
}

/// With a zero parameter, `lr` 1 and no decay, the new parameter is exactly
/// `-max(1, rows/cols)^0.5 * NS5(update)`: the Newton-Schulz output itself.
fn orthogonalized(rows: usize, cols: usize, seed: u64) {
    let c = cpu();
    let cfg = MuonNs5Config {
        lr: 1.0,
        momentum: 0.95,
        weight_decay: 0.0,
        nesterov: true,
        ns5: ojas_core::Ns5Precision::F32,
    };
    let mut hp = Tensor::zeros(&[rows, cols], DType::F32, host_budget()).unwrap();
    let mut hm = host(seed, &[rows, cols]);
    let mut dp = up(&hp);
    let mut dm = up(&hm);
    let grad = host(seed + 1, &[rows, cols]);
    gpu()
        .muon_ns5_step(&mut dp, &up(&grad), &mut dm, cfg)
        .unwrap();
    c.muon_ns5_step(&mut hp, &grad, &mut hm, cfg).unwrap();
    let name = format!("ns5 {rows}x{cols}");
    relative(&name, &down(&dp), &hp.to_f32_vec().unwrap());
    relative(
        &format!("{name} momentum"),
        &down(&dm),
        &hm.to_f32_vec().unwrap(),
    );
}

#[test]
fn newton_schulz_output_matches_cpu_relatively() {
    for &(r, c) in &[(8, 8), (17, 33), (33, 17), (64, 64), (127, 129)] {
        orthogonalized(r, c, 5700 + (r * 1000 + c) as u64);
    }
    orthogonalized(768, 768, 5750);
    orthogonalized(768, 2304, 5751);
    orthogonalized(2304, 768, 5752);
}

#[test]
fn small_wide_tall_and_square_match_cpu() {
    let cfg = MuonNs5Config::nanolab_default();
    for &(r, c) in &[
        (1, 1),
        (1, 7),
        (7, 1),
        (8, 8),
        (17, 33),
        (33, 17),
        (65, 63),
        (64, 64),
    ] {
        run(r, c, 3, cfg, 5000 + (r * 1000 + c) as u64);
    }
}

#[test]
fn nesterov_off_and_no_weight_decay_match_cpu() {
    let plain = MuonNs5Config {
        nesterov: false,
        ..MuonNs5Config::nanolab_default()
    };
    run(40, 24, 3, plain, 5100);
    run(24, 40, 3, plain, 5101);
    let no_decay = MuonNs5Config {
        weight_decay: 0.0,
        ..MuonNs5Config::nanolab_default()
    };
    run(40, 24, 3, no_decay, 5102);
    run(24, 40, 3, no_decay, 5103);
}

#[test]
fn nanolab_shapes_match_cpu() {
    let cfg = MuonNs5Config::nanolab_default();
    run(768, 768, 2, cfg, 5200);
    run(768, 2304, 1, cfg, 5201);
    run(2304, 768, 1, cfg, 5202);
}

#[test]
fn zero_gradient_leaves_the_parameter_bits_as_cpu_does() {
    let c = cpu();
    for wd in [0.0, 0.1] {
        let cfg = MuonNs5Config {
            weight_decay: wd,
            ..MuonNs5Config::nanolab_default()
        };
        let zeros = Tensor::zeros(&[12, 20], DType::F32, host_budget()).unwrap();
        let mut hp = host(5300, &[12, 20]);
        let mut hm = Tensor::zeros(&[12, 20], DType::F32, host_budget()).unwrap();
        let mut dp = up(&hp);
        let mut dm = up(&zeros);
        gpu()
            .muon_ns5_step(&mut dp, &up(&zeros), &mut dm, cfg)
            .unwrap();
        c.muon_ns5_step(&mut hp, &zeros, &mut hm, cfg).unwrap();
        let want: Vec<u32> = hp
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(bits(gpu(), &dp), want, "wd {wd}: parameter bits");
        assert!(bits(gpu(), &dm).iter().all(|b| *b == 0));
    }
}

#[test]
fn non_finite_gradient_leaves_state_untouched_and_is_named() {
    let g = &fresh();
    g.sync().unwrap();
    let p0 = host(5400, &[16, 24]);
    let m0 = host(5401, &[16, 24]);
    let mut p = g.upload(&p0).unwrap();
    let mut m = g.upload(&m0).unwrap();
    let mut bad = data(5402, 16 * 24);
    bad[77] = f32::NAN;
    let bad = g
        .upload(&Tensor::from_f32(&bad, &[16, 24], host_budget()).unwrap())
        .unwrap();
    g.muon_ns5_step(&mut p, &bad, &mut m, MuonNs5Config::nanolab_default())
        .expect("deferred: the step returns Ok and the device refuses it");
    match g.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "muon_ns5_step"),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    let want = |t: &Tensor| -> Vec<u32> {
        t.to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    assert_eq!(bits(g, &p), want(&p0), "param moved");
    assert_eq!(bits(g, &m), want(&m0), "momentum moved");

    // An overflowing momentum is refused the same way.
    let mut huge = g
        .upload(&Tensor::from_f32(&[f32::MAX; 16 * 24], &[16, 24], host_budget()).unwrap())
        .unwrap();
    let grad = g.upload(&host(5403, &[16, 24])).unwrap();
    g.muon_ns5_step(&mut p, &grad, &mut huge, MuonNs5Config::nanolab_default())
        .unwrap();
    assert!(matches!(g.sync(), Err(OjasError::NonFinite { .. })));
    assert_eq!(bits(g, &p), want(&p0), "param moved on overflow");
    g.sync().unwrap();
}

#[test]
fn an_unobserved_earlier_fault_does_not_block_a_clean_step() {
    // The device-side commit must decide on this call's values only.
    let g = &fresh();
    let c = cpu();
    g.sync().unwrap();
    let inf = g
        .upload(&Tensor::from_f32(&[f32::INFINITY, 1.0], &[2], host_budget()).unwrap())
        .unwrap();
    let _ = g.silu_forward(&inf).unwrap();
    let cfg = MuonNs5Config::nanolab_default();
    let mut hp = host(5500, &[9, 14]);
    let mut hm = Tensor::zeros(&[9, 14], DType::F32, host_budget()).unwrap();
    let grad = host(5501, &[9, 14]);
    let mut dp = g.upload(&hp).unwrap();
    let mut dm = g.upload(&hm).unwrap();
    g.muon_ns5_step(&mut dp, &g.upload(&grad).unwrap(), &mut dm, cfg)
        .unwrap();
    c.muon_ns5_step(&mut hp, &grad, &mut hm, cfg).unwrap();
    match g.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "silu_forward"),
        other => panic!("expected the earlier silu fault, got {other:?}"),
    }
    let got = g.download(&dp).unwrap().to_f32_vec().unwrap();
    close_vec("clean step after a fault", &got, &hp.to_f32_vec().unwrap());
}

#[test]
fn malformed_calls_are_refused_before_anything_is_written() {
    let g = &fresh();
    let cfg = MuonNs5Config::nanolab_default();
    let p0 = host(5600, &[4, 6]);
    let mut p = g.upload(&p0).unwrap();
    let mut m = g.upload(&host(5601, &[4, 6])).unwrap();
    let grad = g.upload(&host(5602, &[4, 6])).unwrap();
    let mut vec_p = g.upload(&host(5603, &[24])).unwrap();
    let mut vec_m = g.upload(&host(5604, &[24])).unwrap();
    let vec_g = g.upload(&host(5605, &[24])).unwrap();
    assert!(matches!(
        g.muon_ns5_step(&mut vec_p, &vec_g, &mut vec_m, cfg),
        Err(OjasError::Shape { .. })
    ));
    let wrong = g.upload(&host(5606, &[6, 4])).unwrap();
    assert!(matches!(
        g.muon_ns5_step(&mut p, &wrong, &mut m, cfg),
        Err(OjasError::Shape { .. })
    ));
    for (bad, nonfinite) in [
        (MuonNs5Config { lr: -1.0, ..cfg }, false),
        (
            MuonNs5Config {
                momentum: -0.5,
                ..cfg
            },
            false,
        ),
        (
            MuonNs5Config {
                weight_decay: -0.1,
                ..cfg
            },
            false,
        ),
        (
            MuonNs5Config {
                lr: f64::NAN,
                ..cfg
            },
            true,
        ),
        (
            MuonNs5Config {
                momentum: f64::INFINITY,
                ..cfg
            },
            true,
        ),
    ] {
        let r = g.muon_ns5_step(&mut p, &grad, &mut m, bad);
        match (nonfinite, &r) {
            (true, Err(OjasError::NonFinite { .. }))
            | (false, Err(OjasError::OutOfRange { .. })) => {}
            _ => panic!("{bad:?}: got {r:?}"),
        }
    }
    // bf16 Newton-Schulz is CPU and Metal only: refused before any write.
    let bf16 = MuonNs5Config {
        ns5: Ns5Precision::Bf16,
        ..cfg
    };
    assert!(matches!(
        g.muon_ns5_step(&mut p, &grad, &mut m, bf16),
        Err(OjasError::Unsupported { .. })
    ));
    // A parameter shared with another tensor is refused, not written.
    let alias = p.clone();
    assert!(g.muon_ns5_step(&mut p, &grad, &mut m, cfg).is_err());
    drop(alias);
    // A host tensor is a placement error.
    let mut hp = host(5607, &[4, 6]);
    assert!(matches!(
        g.muon_ns5_step(&mut hp, &grad, &mut m, cfg),
        Err(OjasError::Placement { .. })
    ));
    g.sync().unwrap();
    let want: Vec<u32> = p0
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(bits(g, &p), want);
}
