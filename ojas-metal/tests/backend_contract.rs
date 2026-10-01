//! Error and safety contract of `MetalBackend`: placement, non-finite
//! refusal without writes, head-dim and capacity refusals, causal masking,
//! and many threads on one backend.

#![cfg(target_os = "macos")]

mod common;

use std::sync::Arc;
use std::thread;

use common::*;
use ojas_core::{AdamWConfig, Backend, BackendId, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

fn is_placement(r: Result<Tensor, OjasError>) -> bool {
    matches!(
        r,
        Err(OjasError::Placement {
            expected: Some(BackendId::Metal),
            ..
        })
    )
}

#[test]
fn host_tensors_are_placement_errors_not_silent_uploads() {
    let m = metal();
    let x = rand(&[4, 3], 1, 1.0);
    let w = rand(&[5, 3], 2, 1.0);
    let dw = up(&m, &w);
    assert!(is_placement(m.linear_forward(&x, &dw)));
    assert!(is_placement(m.linear_forward(&up(&m, &x), &w)));
    assert!(is_placement(m.silu_forward(&x)));
    let mut hp = rand(&[3], 3, 1.0);
    let g = up(&m, &rand(&[3], 4, 1.0));
    let mut mm = up(&m, &host(&[0.0; 3], &[3]));
    let mut vv = up(&m, &host(&[0.0; 3], &[3]));
    let r = m.adamw_step(
        &mut hp,
        &g,
        &mut mm,
        &mut vv,
        0,
        AdamWConfig::nanolab(1e-3, 0.0),
    );
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
}

#[test]
fn a_second_metal_backend_does_not_accept_the_first_ones_buffers() {
    let (a, b) = (metal(), metal());
    let x = up(&a, &rand(&[2, 2], 1, 1.0));
    let r = b.silu_forward(&x);
    assert!(
        matches!(
            r,
            Err(OjasError::Backend {
                id: BackendId::Metal,
                ..
            })
        ),
        "{r:?}"
    );
    // Clones share the device, so they do accept it.
    let c = a.clone();
    assert!(c.silu_forward(&x).is_ok());
}

#[test]
fn upload_of_a_device_tensor_is_a_clone_and_of_a_foreign_device_is_placement() {
    let (a, b) = (metal(), metal());
    let x = up(&a, &rand(&[2, 2], 1, 1.0));
    let again = ok("re-upload", a.upload(&x));
    assert_eq!(again.device(), Some(BackendId::Metal));
    let r = b.upload(&x);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
}

#[test]
fn non_finite_inputs_are_refused_by_every_op_family() {
    let m = metal();
    let bad = up(&m, &host(&[1.0, f32::NAN, 2.0, 3.0], &[2, 2]));
    let inf = up(&m, &host(&[1.0, 2.0, f32::INFINITY, 3.0], &[2, 2]));
    let good = up(&m, &rand(&[2, 2], 1, 1.0));
    let w1 = up(&m, &rand(&[2], 2, 1.0));
    let nf = |r: Result<Tensor, OjasError>| matches!(r, Err(OjasError::NonFinite { .. }));
    assert!(nf(m.silu_forward(&bad)));
    assert!(nf(m.mul_forward(&good, &inf)));
    assert!(nf(m.residual_add_forward(&inf, &good)));
    assert!(nf(m.linear_forward(&bad, &good)));
    assert!(nf(m.linear_forward(&good, &inf)));
    assert!(nf(m.rms_norm_forward(&bad, &w1, 1e-6)));
    assert!(nf(m.rms_norm_forward(&good, &w1, f32::NAN)));
    let q = up(&m, &host(&[f32::NAN; 16], &[1, 1, 1, 16]));
    let k = up(&m, &rand(&[1, 1, 1, 16], 3, 1.0));
    assert!(nf(m.causal_sdpa_forward(&q, &k, &k)));
    let t = up(&m, &host_u32(&[0, 1], &[2]));
    assert!(nf(m.cross_entropy_mean_forward(&inf, &t, None)));
}

#[test]
fn non_finite_gradient_leaves_adamw_state_untouched() {
    let m = metal();
    let p0 = rand(&[67, 3], 5, 1.0);
    let mut p = up(&m, &p0);
    let mut m1 = up(&m, &host(&vec![0.25; 201], &[67, 3]));
    let mut m2 = up(&m, &host(&vec![0.5; 201], &[67, 3]));
    let mut g = values(201, 6, 1.0);
    g[150] = f32::NAN;
    let dg = up(&m, &host(&g, &[67, 3]));
    let r = m.adamw_step(
        &mut p,
        &dg,
        &mut m1,
        &mut m2,
        0,
        AdamWConfig::nanolab(1e-3, 0.1),
    );
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
    assert_eq!(down(&p), ok("p0", p0.to_f32_vec()));
    assert!(down(&m1).iter().all(|&v| v == 0.25));
    assert!(down(&m2).iter().all(|&v| v == 0.5));
}

/// A NaN or infinity in the very last element of any input, or an update
/// that overflows only there, is NonFinite and leaves the parameter and both
/// moments bit-identical: the refusal is decided before anything is written.
#[test]
fn adamw_refusal_at_the_last_element_leaves_every_tensor_bit_identical() {
    let m = metal();
    let shape = [3usize, 4099];
    let n = 3 * 4099;
    let bits = |t: &Tensor| -> Vec<u32> { down(t).iter().map(|v| v.to_bits()).collect() };
    let p0 = values(n, 21, 1.0);
    let m0 = values(n, 22, 0.1);
    let v0: Vec<f32> = values(n, 23, 0.01).iter().map(|v| v.abs()).collect();
    let g0 = values(n, 24, 1.0);
    let cfg = AdamWConfig::nanolab(1e-3, 0.1);
    for which in 0..4 {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut ins = [p0.clone(), g0.clone(), m0.clone(), v0.clone()];
            ins[which][n - 1] = bad;
            let mut p = up(&m, &host(&ins[0], &shape));
            let g = up(&m, &host(&ins[1], &shape));
            let mut m1 = up(&m, &host(&ins[2], &shape));
            let mut m2 = up(&m, &host(&ins[3], &shape));
            let before = (bits(&p), bits(&m1), bits(&m2));
            let r = m.adamw_step(&mut p, &g, &mut m1, &mut m2, 3, cfg);
            assert!(
                matches!(r, Err(OjasError::NonFinite { op: "adamw_step" })),
                "input {which} = {bad}: {r:?}"
            );
            assert_eq!(bits(&p), before.0, "input {which} = {bad}: param written");
            assert_eq!(bits(&m1), before.1, "input {which} = {bad}: moment1 written");
            assert_eq!(bits(&m2), before.2, "input {which} = {bad}: moment2 written");
        }
    }
    // Finite inputs whose update overflows only at the last element.
    let mut p_in = p0.clone();
    p_in[n - 1] = f32::MAX;
    let mut g_in = g0.clone();
    g_in[n - 1] = -1.0;
    let mut p = up(&m, &host(&p_in, &shape));
    let g = up(&m, &host(&g_in, &shape));
    let mut m1 = up(&m, &host(&vec![0.0; n], &shape));
    let mut m2 = up(&m, &host(&vec![0.0; n], &shape));
    let before = (bits(&p), bits(&m1), bits(&m2));
    let big = AdamWConfig::nanolab(1e32, 0.0);
    let r = m.adamw_step(&mut p, &g, &mut m1, &mut m2, 0, big);
    assert!(
        matches!(r, Err(OjasError::NonFinite { op: "adamw_step" })),
        "overflow: {r:?}"
    );
    assert_eq!(bits(&p), before.0, "overflow: param written");
    assert_eq!(bits(&m1), before.1, "overflow: moment1 written");
    assert_eq!(bits(&m2), before.2, "overflow: moment2 written");
    // The same tensors still step cleanly afterwards.
    ok("clean step", m.adamw_step(&mut p, &up(&m, &host(&g0, &shape)), &mut m1, &mut m2, 0, cfg));
}

#[test]
fn an_update_that_overflows_leaves_adamw_state_untouched() {
    let m = metal();
    let mut p = up(&m, &host(&[f32::MAX, 1.0], &[2]));
    let mut m1 = up(&m, &host(&[0.0, 0.0], &[2]));
    let mut m2 = up(&m, &host(&[0.0, 0.0], &[2]));
    let g = up(&m, &host(&[-1.0, 1.0], &[2]));
    let cfg = AdamWConfig::nanolab(f32::MAX as f64, 0.0);
    let r = m.adamw_step(&mut p, &g, &mut m1, &mut m2, 0, cfg);
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
    assert_eq!(down(&p), vec![f32::MAX, 1.0]);
    assert_eq!(down(&m1), vec![0.0, 0.0]);
}

#[test]
fn non_finite_gradient_leaves_muon_state_untouched() {
    let m = metal();
    let p0 = rand(&[8, 5], 7, 1.0);
    let mut p = up(&m, &p0);
    let mut mo = up(&m, &host(&[0.125; 40], &[8, 5]));
    let mut g = values(40, 8, 1.0);
    g[3] = f32::INFINITY;
    let dg = up(&m, &host(&g, &[8, 5]));
    let r = m.muon_ns5_step(
        &mut p,
        &dg,
        &mut mo,
        ojas_core::MuonNs5Config::nanolab_default(),
    );
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
    assert_eq!(down(&p), ok("p0", p0.to_f32_vec()));
    assert!(down(&mo).iter().all(|&v| v == 0.125));
}

#[test]
fn non_finite_gradient_is_not_clipped() {
    let m = metal();
    let mut grads = vec![
        up(&m, &host(&[3.0, 4.0], &[2])),
        up(&m, &host(&[f32::NAN, 1.0], &[2])),
    ];
    let r = m.clip_grad_norm(&mut grads, 0.1);
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
    assert_eq!(down(&grads[0]), vec![3.0, 4.0]);
}

#[test]
fn shared_parameter_storage_is_refused_before_any_in_place_write() {
    let m = metal();
    let mut p = up(&m, &rand(&[4], 1, 1.0));
    let alias = p.clone();
    let g = up(&m, &rand(&[4], 2, 1.0));
    let mut m1 = up(&m, &host(&[0.0; 4], &[4]));
    let mut m2 = up(&m, &host(&[0.0; 4], &[4]));
    let before = down(&alias);
    let r = m.adamw_step(
        &mut p,
        &g,
        &mut m1,
        &mut m2,
        0,
        AdamWConfig::nanolab(1e-2, 0.0),
    );
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    assert_eq!(down(&alias), before);
}

#[test]
fn out_of_range_token_ids_are_refused() {
    let m = metal();
    let table = up(&m, &rand(&[5, 3], 1, 1.0));
    let tok = up(&m, &host_u32(&[0, 5], &[2]));
    let r = m.embedding_forward(&table, &tok);
    assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}");
    let logits = up(&m, &rand(&[2, 5], 2, 1.0));
    let r = m.cross_entropy_mean_forward(&logits, &tok, None);
    assert!(matches!(r, Err(OjasError::OutOfRange { .. })), "{r:?}");
}

#[test]
fn an_all_ignored_batch_is_non_finite() {
    let m = metal();
    let logits = up(&m, &rand(&[2, 5], 2, 1.0));
    let tok = up(&m, &host_u32(&[9, 9], &[2]));
    let r = m.cross_entropy_mean_forward(&logits, &tok, Some(9));
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
    let r = m.cross_entropy_mean_backward(&logits, &tok, Some(9));
    assert!(matches!(r, Err(OjasError::NonFinite { .. })), "{r:?}");
}

#[test]
fn head_dim_above_64_is_refused_not_truncated() {
    let m = metal();
    // Derived from the core limit, so raising it keeps this a refusal test.
    let over = ojas_core::METAL_MAX_HEAD_DIM + 16;
    let q = up(&m, &rand(&[1, 1, 4, over as usize], 1, 1.0));
    let r = m.causal_sdpa_forward(&q, &q, &q);
    assert!(
        matches!(
            r,
            Err(OjasError::UnsupportedHeadDim { head_dim, limit })
                if head_dim == over && limit == ojas_core::METAL_MAX_HEAD_DIM
        ),
        "{r:?}"
    );
    let r = m.causal_sdpa_backward(&q, &q, &q, &q);
    assert!(
        matches!(r, Err(OjasError::UnsupportedHeadDim { .. })),
        "{r:?}"
    );
}

#[test]
fn shape_errors_match_the_cpu_classification() {
    let (m, c) = (metal(), cpu());
    let x = rand(&[3, 4], 1, 1.0);
    let w = rand(&[5, 6], 2, 1.0);
    let odd = rand(&[3, 5], 3, 1.0);
    type Case<'a> = (
        &'a str,
        Result<Tensor, OjasError>,
        Result<Tensor, OjasError>,
    );
    let cases: [Case; 3] = [
        (
            "linear in-features",
            c.linear_forward(&x, &w),
            m.linear_forward(&up(&m, &x), &up(&m, &w)),
        ),
        (
            "rope odd dim",
            c.rope_half_split_forward(&odd, &odd, &odd),
            m.rope_half_split_forward(&up(&m, &odd), &up(&m, &odd), &up(&m, &odd)),
        ),
        (
            "mul mismatch",
            c.mul_forward(&x, &odd),
            m.mul_forward(&up(&m, &x), &up(&m, &odd)),
        ),
    ];
    for (what, want, got) in cases {
        let (Err(want), Err(got)) = (want, got) else {
            panic!("{what}: both backends must refuse");
        };
        assert_eq!(
            std::mem::discriminant(&want),
            std::mem::discriminant(&got),
            "{what}: cpu {want:?} metal {got:?}"
        );
    }
}

#[test]
fn causal_attention_does_not_read_future_keys() {
    let m = metal();
    let (t, d) = (33usize, 64usize);
    let q = rand(&[1, 2, t, d], 1, 1.0);
    let k = rand(&[1, 2, t, d], 2, 1.0);
    let v = rand(&[1, 2, t, d], 3, 1.0);
    let base = down(&ok(
        "base",
        m.causal_sdpa_forward(&up(&m, &q), &up(&m, &k), &up(&m, &v)),
    ));
    // Replace every key and value at positions > cut with huge values.
    let cut = 20;
    let mut k2 = ok("k", k.to_f32_vec());
    let mut v2 = ok("v", v.to_f32_vec());
    for h in 0..2 {
        for pos in cut + 1..t {
            for j in 0..d {
                let i = (h * t + pos) * d + j;
                k2[i] = 50.0;
                v2[i] = 1e6;
            }
        }
    }
    let (k2, v2) = (host(&k2, &[1, 2, t, d]), host(&v2, &[1, 2, t, d]));
    let out = down(&ok(
        "leak",
        m.causal_sdpa_forward(&up(&m, &q), &up(&m, &k2), &up(&m, &v2)),
    ));
    for h in 0..2 {
        for pos in 0..=cut {
            for j in 0..d {
                let i = (h * t + pos) * d + j;
                assert_eq!(out[i], base[i], "head {h} pos {pos} saw a future key");
            }
        }
    }
    // And the poisoned future rows did change, so the probe reaches them.
    let i = (cut + 1) * d;
    assert_ne!(out[i], base[i]);

    // Backward: dk and dv at future positions get no gradient from earlier
    // queries' outputs.
    let mut g = vec![0.0f32; 2 * t * d];
    for h in 0..2 {
        for pos in 0..=cut {
            for j in 0..d {
                g[(h * t + pos) * d + j] = 1.0;
            }
        }
    }
    let g = up(&m, &host(&g, &[1, 2, t, d]));
    let (_, dk, dv) = ok(
        "bwd",
        m.causal_sdpa_backward(&up(&m, &q), &up(&m, &k), &up(&m, &v), &g),
    );
    let (dk, dv) = (down(&dk), down(&dv));
    for h in 0..2 {
        for pos in cut + 1..t {
            for j in 0..d {
                let i = (h * t + pos) * d + j;
                assert_eq!(dk[i], 0.0, "dk leak at head {h} pos {pos}");
                assert_eq!(dv[i], 0.0, "dv leak at head {h} pos {pos}");
            }
        }
    }
}

#[test]
fn over_budget_outputs_are_capacity_exceeded_before_device_allocation() {
    // 20 KiB: one 16 KiB input fits, a second 16 KiB output does not.
    let small = ok("new", MetalBackend::new(Budget::new(20 * 1024)));
    let xs = up(&small, &rand(&[64, 64], 1, 1.0));
    assert_eq!(ok("live", small.budget().live_bytes()), 64 * 64 * 4);
    let r = small.residual_add_forward(&xs, &xs);
    assert!(
        matches!(
            r,
            Err(OjasError::CapacityExceeded {
                requested: 16384,
                cap: 20480,
                live: 16384
            })
        ),
        "{r:?}"
    );
    let r = small.linear_backward(&xs, &xs, &xs);
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
    // The refusals did not leak budget, and a buffer's drop returns it.
    assert_eq!(ok("live", small.budget().live_bytes()), 64 * 64 * 4);
    drop(xs);
    assert_eq!(ok("live", small.budget().live_bytes()), 0);
    // An upload larger than the cap is refused before the device sees it.
    let r = small.upload(&rand(&[1, 1 << 13], 3, 1.0));
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
    // Optimizer scratch counts too: Muon's p, momentum and g fit (12 KiB) but
    // its Newton-Schulz scratch does not, so nothing moves.
    let mut p = up(&small, &rand(&[32, 32], 4, 1.0));
    let mut mo = up(&small, &host(&[0.0; 1024], &[32, 32]));
    let g = up(&small, &rand(&[32, 32], 5, 1.0));
    let before = down(&p);
    let r = small.muon_ns5_step(
        &mut p,
        &g,
        &mut mo,
        ojas_core::MuonNs5Config {
            lr: 0.02,
            momentum: 0.95,
            nesterov: true,
            weight_decay: 0.0,
        },
    );
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
    assert_eq!(down(&p), before);
    // AdamW updates in place with no scratch, so the same tensors step
    // inside the budget and charge nothing beyond them.
    let mut m2 = up(&small, &host(&[0.0; 1024], &[32, 32]));
    let live = ok("live", small.budget().live_bytes());
    ok(
        "adamw in budget",
        small.adamw_step(&mut p, &g, &mut mo, &mut m2, 0, AdamWConfig::nanolab(1e-3, 0.0)),
    );
    assert_eq!(ok("live", small.budget().live_bytes()), live);
    assert_ne!(down(&p), before);
}

/// `Backend::download` charges the backend's own budget, so a count read
/// there moves exactly when this backend reads back, whatever other threads
/// and backends do at the same time; a copy into another budget does not
/// move it.
#[test]
fn downloads_are_counted_on_the_backends_own_budget() {
    let m = metal();
    let x = up(&m, &rand(&[3, 5], 1, 1.0));
    assert_eq!(m.budget().device_readbacks(), (0, 0));
    let noise = thread::spawn(|| {
        let other = metal();
        let y = up(&other, &rand(&[64], 2, 1.0));
        for _ in 0..50 {
            ok("other download", other.download(&y));
        }
        other.budget().device_readbacks()
    });
    ok("download", m.download(&x));
    assert_eq!(m.budget().device_readbacks(), (1, 60));
    ok("elsewhere", x.to_host(&Budget::new(1 << 20)));
    let y = ok("op", m.silu_forward(&x));
    assert_eq!(
        m.budget().device_readbacks(),
        (1, 60),
        "an op or a copy into another budget is not this backend's readback"
    );
    ok("download", m.download(&y));
    assert_eq!(m.budget().device_readbacks(), (2, 120));
    let theirs = noise.join().expect("noise thread");
    assert_eq!(theirs, (50, 50 * 256));
    assert_eq!(m.budget().device_readbacks(), (2, 120));
}

#[test]
fn many_threads_on_one_backend_get_their_own_answers() {
    let m = Arc::new(metal());
    let c = cpu();
    let w = rand(&[13, 11], 9, 1.0);
    let want: Vec<Vec<f32>> = (0..16)
        .map(|i| {
            let x = rand(&[7, 11], 100 + i, 1.0);
            ok("cpu", ok("cpu", c.linear_forward(&x, &w)).to_f32_vec())
        })
        .collect();
    let dw = Arc::new(up(&m, &w));
    let handles: Vec<_> = (0..16u64)
        .map(|i| {
            let m = Arc::clone(&m);
            let dw = Arc::clone(&dw);
            thread::spawn(move || {
                let mut last = Vec::new();
                for _ in 0..25 {
                    let x = up(&m, &rand(&[7, 11], 100 + i, 1.0));
                    let y = ok("metal", m.linear_forward(&x, &dw));
                    let y = ok("silu", m.silu_forward(&y));
                    let y = ok("bwd", m.silu_backward(&y, &y));
                    let _ = y;
                    last = down(&ok("metal", m.linear_forward(&x, &dw)));
                }
                last
            })
        })
        .collect();
    for (i, h) in handles.into_iter().enumerate() {
        let got = match h.join() {
            Ok(v) => v,
            Err(_) => panic!("thread {i} panicked"),
        };
        close(&format!("thread {i}"), &got, &want[i], 2e-4, 2e-4);
    }
}
