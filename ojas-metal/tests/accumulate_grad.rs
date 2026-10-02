//! `Backend::accumulate_grad` on Metal (T2, gate G2): `acc += grad` in place
//! when `acc` solely owns its device buffer, bit-identical to the CPU
//! reference, and a non-finite sum leaves `acc` bit-identical.
#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::Arc;

use common::*;
use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

const OP: &str = "accumulate_grad";

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|x| x.to_bits()).collect()
}

fn buffer_ptr(t: &Tensor) -> *const () {
    match t.device_buffer() {
        Some(b) => Arc::as_ptr(b) as *const (),
        None => panic!("not a device tensor"),
    }
}

#[test]
fn matches_cpu_bit_for_bit_in_the_same_buffer() {
    let m = metal();
    let c = cpu();
    for (i, shape) in [vec![1], vec![3, 4099], vec![768, 768], vec![4, 12, 64]]
        .into_iter()
        .enumerate()
    {
        let seed = 10 * i as u64;
        let (a, g) = (rand(&shape, seed, 3.0), rand(&shape, seed + 1, 2.0));
        let mut want = a.clone();
        ok("cpu", c.accumulate_grad(&mut want, &g));
        let mut acc = up(&m, &a);
        let before = buffer_ptr(&acc);
        ok("metal", m.accumulate_grad(&mut acc, &up(&m, &g)));
        assert_eq!(buffer_ptr(&acc), before, "{shape:?}: not written in place");
        let want_bits: Vec<u32> = ok("cpu out", want.to_f32_vec())
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(bits(&acc), want_bits, "{shape:?}: differs from the CPU");
    }
}

#[test]
fn runs_in_place_under_a_budget_with_no_room_for_a_new_sum() {
    let n = 3 * 4099;
    // Exactly acc and grad: a composition that allocates the sum is refused.
    let m = ok("metal", MetalBackend::new(Budget::new(2 * 4 * n as u64)));
    let (a, g) = (values(n, 1, 1.0), values(n, 2, 1.0));
    let mut acc = up(&m, &host(&a, &[3, 4099]));
    let grad = up(&m, &host(&g, &[3, 4099]));
    ok("accumulate", m.accumulate_grad(&mut acc, &grad));
    let want: Vec<f32> = a.iter().zip(&g).map(|(x, y)| x + y).collect();
    assert_eq!(down(&acc), want);
    assert_eq!(ok("live", m.budget().live_bytes()), 2 * 4 * n as u64);
}

#[test]
fn a_shared_accumulator_gets_a_new_buffer_and_the_other_handle_is_untouched() {
    let m = metal();
    let shape = [5, 33];
    let (a, g) = (rand(&shape, 3, 1.0), rand(&shape, 4, 1.0));
    let mut acc = up(&m, &a);
    let other = acc.clone();
    let before = bits(&other);
    ok("accumulate", m.accumulate_grad(&mut acc, &up(&m, &g)));
    assert_eq!(bits(&other), before, "the shared handle was written");
    let want: Vec<f32> = ok("a", a.to_f32_vec())
        .iter()
        .zip(ok("g", g.to_f32_vec()))
        .map(|(x, y)| x + y)
        .collect();
    assert_eq!(down(&acc), want);
    assert!(acc.device_buffer_mut().is_ok(), "acc must end uniquely owned");
}

#[test]
fn a_non_finite_sum_is_refused_and_leaves_acc_bit_identical() {
    let m = metal();
    let shape = [3, 4099];
    let n = 3 * 4099;
    let poison = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    // (acc value, grad value) at the last element.
    let mut cases: Vec<(f32, f32)> = Vec::new();
    for p in poison {
        cases.push((p, 1.0));
        cases.push((1.0, p));
    }
    cases.push((f32::MAX, f32::MAX));
    cases.push((-f32::MAX, -f32::MAX));
    for shared in [false, true] {
        for &(av, gv) in &cases {
            let mut a = values(n, 5, 1.0);
            let mut g = values(n, 6, 1.0);
            a[n - 1] = av;
            g[n - 1] = gv;
            let mut acc = up(&m, &host(&a, &shape));
            let keep = shared.then(|| acc.clone());
            let before = bits(&acc);
            let ptr = buffer_ptr(&acc);
            let r = m.accumulate_grad(&mut acc, &up(&m, &host(&g, &shape)));
            deferred(&m, &format!("shared {shared}, acc {av}, grad {gv}"), r, OP);
            assert_eq!(bits(&acc), before, "acc {av}, grad {gv}: acc changed");
            if let Some(other) = &keep {
                // A shared acc takes a new buffer when the call returns Ok;
                // on a fault it holds the old values, and the other handle
                // is untouched.
                assert_ne!(buffer_ptr(&acc), ptr, "a shared acc gets a new buffer");
                assert_eq!(bits(other), before, "acc {av}, grad {gv}: other handle changed");
            } else {
                assert_eq!(buffer_ptr(&acc), ptr, "a unique acc keeps its buffer");
            }
            drop(keep);
        }
    }
    // A clean call still works afterwards.
    let mut acc = up(&m, &rand(&shape, 7, 1.0));
    ok("clean", m.accumulate_grad(&mut acc, &up(&m, &rand(&shape, 8, 1.0))));
    ok("nothing pending", m.sync());
}

#[test]
fn shape_and_placement_errors_leave_acc_untouched() {
    let m = metal();
    let mut acc = up(&m, &rand(&[4, 8], 9, 1.0));
    let before = bits(&acc);
    let r = m.accumulate_grad(&mut acc, &up(&m, &rand(&[8, 4], 10, 1.0)));
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    let r = m.accumulate_grad(&mut acc, &rand(&[4, 8], 11, 1.0));
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    assert_eq!(bits(&acc), before);
    let mut host_acc = rand(&[4, 8], 12, 1.0);
    let r = m.accumulate_grad(&mut host_acc, &up(&m, &rand(&[4, 8], 13, 1.0)));
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
}
