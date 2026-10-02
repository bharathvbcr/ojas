//! In-place optimizer steps are all-or-nothing (audit F2).
//!
//! `adamw_step`, `muon_ns5_step` and `clip_grad_norm` write several tensors.
//! A target that cannot be written (its storage is shared with a clone or a
//! narrowed view) must be found before the first write, so a refused step
//! leaves every tensor bit-identical: no moved parameter beside stale moments.

use ojas_core::{AdamWConfig, Backend, Budget, MuonNs5Config, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{bits, f32t};

fn wide() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 20))
}

fn snap(t: &Tensor) -> Vec<u32> {
    bits(&t.to_f32_vec().unwrap())
}

fn assert_shared_refusal<T: std::fmt::Debug>(what: &str, result: Result<T, OjasError>) {
    match result {
        Err(OjasError::Shape { detail, .. }) => {
            assert!(
                detail.contains("uniquely owned") || detail.contains("shared"),
                "{what}: {detail}"
            );
        }
        other => panic!("{what}: expected a Shape refusal of shared storage, got {other:?}"),
    }
}

#[test]
fn adamw_with_a_shared_second_moment_writes_nothing() {
    let cpu = wide();
    let mut p = f32t(&cpu, &[1.0, -2.0, 0.5], &[3]);
    let g = f32t(&cpu, &[0.5, 0.25, -1.0], &[3]);
    let mut m1 = f32t(&cpu, &[0.1, 0.2, 0.3], &[3]);
    let mut m2 = f32t(&cpu, &[0.3, 0.4, 0.5], &[3]);
    let alias = m2.clone();
    let before = (snap(&p), snap(&m1), snap(&m2));
    let cfg = AdamWConfig::nanolab(1e-2, 0.1);
    assert_shared_refusal(
        "adamw",
        cpu.adamw_step(&mut p, &g, &mut m1, &mut m2, 0, cfg),
    );
    assert_eq!(
        (snap(&p), snap(&m1), snap(&m2)),
        before,
        "adamw wrote before refusing"
    );
    // The same step on unshared state does move every tensor, so the
    // refusal above is not an update that would have been a no-op.
    drop(alias);
    cpu.adamw_step(&mut p, &g, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    assert_ne!(snap(&p), before.0);
    assert_ne!(snap(&m1), before.1);
    assert_ne!(snap(&m2), before.2);
}

#[test]
fn adamw_with_a_narrowed_second_moment_writes_nothing() {
    let cpu = wide();
    let mut p = f32t(&cpu, &[1.0, -2.0], &[2]);
    let g = f32t(&cpu, &[0.5, 0.25], &[2]);
    let mut m1 = f32t(&cpu, &[0.1, 0.2], &[2]);
    // m2 is a contiguous window into a larger allocation that is still alive.
    let parent = f32t(&cpu, &[9.0, 0.3, 0.4, 9.0], &[4]);
    let mut m2 = parent.narrow(4, &[2], &[1]).unwrap();
    let before = (snap(&p), snap(&m1), snap(&m2), snap(&parent));
    let cfg = AdamWConfig::nanolab(1e-2, 0.0);
    assert_shared_refusal(
        "adamw narrow",
        cpu.adamw_step(&mut p, &g, &mut m1, &mut m2, 3, cfg),
    );
    assert_eq!(
        (snap(&p), snap(&m1), snap(&m2), snap(&parent)),
        before,
        "adamw wrote before refusing"
    );
}

#[test]
fn muon_with_a_shared_momentum_writes_nothing() {
    let cpu = wide();
    let mut p = f32t(&cpu, &[1.0, -2.0, 0.5, 0.25, 0.75, -1.5], &[2, 3]);
    let g = f32t(&cpu, &[0.5, 0.25, -1.0, 0.1, -0.2, 0.3], &[2, 3]);
    let mut m = f32t(&cpu, &[0.1, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
    let alias = m.clone();
    let before = (snap(&p), snap(&m));
    let cfg = MuonNs5Config::nanolab_default();
    assert_shared_refusal("muon", cpu.muon_ns5_step(&mut p, &g, &mut m, cfg));
    assert_eq!((snap(&p), snap(&m)), before, "muon wrote before refusing");
    drop(alias);
    cpu.muon_ns5_step(&mut p, &g, &mut m, cfg).unwrap();
    assert_ne!(snap(&p), before.0);
    assert_ne!(snap(&m), before.1);
}

#[test]
fn clip_with_a_shared_later_gradient_scales_nothing() {
    let cpu = wide();
    let g0 = f32t(&cpu, &[3.0, 4.0], &[2]);
    let g1 = f32t(&cpu, &[12.0, 0.0], &[2]);
    let alias = g1.clone();
    let mut grads = vec![g0, g1];
    let before: Vec<Vec<u32>> = grads.iter().map(snap).collect();
    // Total norm 13 against max 1: the scale is below 1, so a write is due.
    assert_shared_refusal("clip", cpu.clip_grad_norm(&mut grads, 1.0));
    let after: Vec<Vec<u32>> = grads.iter().map(snap).collect();
    assert_eq!(
        after, before,
        "clip scaled grads[0] before refusing grads[1]"
    );
    drop(alias);
    let norm = cpu.clip_grad_norm(&mut grads, 1.0).unwrap();
    assert!((norm - 13.0).abs() < 1e-5, "{norm}");
    assert_ne!(snap(&grads[0]), before[0]);
    assert_ne!(snap(&grads[1]), before[1]);
}
