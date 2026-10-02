//! Host copies of op inputs are charged to the backend's budget (audit F7).
//!
//! An op that decodes its tensor inputs into host vectors before computing
//! holds copies as large as the inputs, so a budget that has room for the
//! output alone must refuse rather than let the process hold inputs, copies
//! and output at once with only the output counted.
//!
//! Cross-entropy reads its logits and targets in place and makes no copy, so
//! its charge is exactly its output: that much room succeeds and one f32
//! less refuses.
//!
//! The inputs are built under their own budget, so the backend's budget
//! sees only what the op itself charges.

use ojas_core::{Backend, Budget, OjasError, Tensor, RMS_NORM_EPS};
use ojas_cpu::CpuBackend;

const F32: u64 = 4;

type Op = Box<dyn Fn(&CpuBackend) -> Result<Tensor, OjasError>>;

struct Case {
    name: &'static str,
    /// Bytes of every tensor input the op reads.
    input_bytes: u64,
    /// Bytes of the tensor the op returns.
    output_bytes: u64,
    /// The op reads its inputs in place and charges only its output.
    in_place: bool,
    run: Op,
}

fn f32t(budget: &Budget, n: usize, shape: &[usize]) -> Tensor {
    let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect();
    Tensor::from_f32(&data, shape, budget).unwrap()
}

fn cases(inputs: &Budget) -> Vec<Case> {
    let x = f32t(inputs, 64, &[4, 16]);
    let y = f32t(inputs, 64, &[4, 16]);
    let w16 = f32t(inputs, 16, &[16]);
    let logits = f32t(inputs, 32, &[4, 8]);
    let targets = Tensor::from_u32(&[1, 7, 0, 3], &[4], inputs).unwrap();
    let qkv = f32t(inputs, 32, &[1, 2, 4, 4]);
    let lin_x = f32t(inputs, 32, &[4, 8]);
    let lin_w = f32t(inputs, 24, &[3, 8]);
    vec![
        Case {
            name: "silu_forward",
            input_bytes: 64 * F32,
            output_bytes: 64 * F32,
            in_place: false,
            run: {
                let x = x.clone();
                Box::new(move |cpu| cpu.silu_forward(&x))
            },
        },
        Case {
            name: "mul_forward",
            input_bytes: 128 * F32,
            output_bytes: 64 * F32,
            in_place: false,
            run: {
                let (x, y) = (x.clone(), y.clone());
                Box::new(move |cpu| cpu.mul_forward(&x, &y))
            },
        },
        Case {
            name: "residual_add_forward",
            input_bytes: 128 * F32,
            output_bytes: 64 * F32,
            in_place: false,
            run: {
                let (x, y) = (x.clone(), y.clone());
                Box::new(move |cpu| cpu.residual_add_forward(&x, &y))
            },
        },
        Case {
            name: "rms_norm_forward",
            input_bytes: 80 * F32,
            output_bytes: 64 * F32,
            in_place: false,
            run: {
                let (x, w) = (x.clone(), w16.clone());
                Box::new(move |cpu| cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS))
            },
        },
        Case {
            name: "cross_entropy_mean_forward",
            input_bytes: 32 * F32 + 4 * 4,
            output_bytes: F32,
            in_place: true,
            run: {
                let (l, t) = (logits.clone(), targets.clone());
                Box::new(move |cpu| cpu.cross_entropy_mean_forward(&l, &t, None))
            },
        },
        Case {
            name: "cross_entropy_mean_backward",
            input_bytes: 32 * F32 + 4 * 4,
            output_bytes: 32 * F32,
            in_place: true,
            run: {
                let (l, t) = (logits.clone(), targets.clone());
                Box::new(move |cpu| cpu.cross_entropy_mean_backward(&l, &t, None))
            },
        },
        Case {
            name: "causal_sdpa_forward",
            input_bytes: 96 * F32,
            output_bytes: 32 * F32,
            in_place: false,
            run: {
                let q = qkv.clone();
                Box::new(move |cpu| cpu.causal_sdpa_forward(&q, &q, &q))
            },
        },
        Case {
            name: "linear_forward",
            input_bytes: 56 * F32,
            output_bytes: 12 * F32,
            in_place: false,
            run: Box::new(move |cpu| cpu.linear_forward(&lin_x, &lin_w)),
        },
    ]
}

#[test]
fn a_budget_with_room_for_the_output_but_not_the_input_copies_refuses() {
    let inputs = Budget::new(1 << 20);
    let mut undercharged = Vec::new();
    for case in cases(&inputs).into_iter().filter(|c| !c.in_place) {
        // Everything but the last f32 of inputs plus output.
        let cap = case.input_bytes + case.output_bytes - F32;
        assert!(cap >= case.output_bytes);
        let tight = CpuBackend::new(Budget::new(cap));
        match (case.run)(&tight) {
            Err(OjasError::CapacityExceeded { .. }) => {}
            Ok(_) => undercharged.push(format!(
                "{}: succeeded under a {cap}-byte cap; inputs {} + output {}",
                case.name, case.input_bytes, case.output_bytes
            )),
            Err(other) => panic!("{}: expected CapacityExceeded, got {other:?}", case.name),
        }
        assert_eq!(
            tight.budget().live_bytes().unwrap(),
            0,
            "{}: a refused op left a charge behind",
            case.name
        );
    }
    assert!(
        undercharged.is_empty(),
        "input copies not charged:\n{}",
        undercharged.join("\n")
    );
}

/// An op that reads in place succeeds with room for exactly its output and
/// refuses with one f32 less, leaving nothing charged.
#[test]
fn in_place_readers_charge_exactly_their_output() {
    let inputs = Budget::new(1 << 20);
    for case in cases(&inputs).into_iter().filter(|c| c.in_place) {
        let fits = CpuBackend::new(Budget::new(case.output_bytes));
        let out = (case.run)(&fits)
            .unwrap_or_else(|e| panic!("{}: refused with room for its output: {e:?}", case.name));
        assert_eq!(fits.budget().live_bytes().unwrap(), case.output_bytes, "{}", case.name);
        drop(out);
        let tight = CpuBackend::new(Budget::new(case.output_bytes - F32));
        match (case.run)(&tight) {
            Err(OjasError::CapacityExceeded { .. }) => {}
            other => panic!(
                "{}: expected CapacityExceeded one f32 short of its output, got {:?}",
                case.name,
                other.map(|_| ())
            ),
        }
        assert_eq!(tight.budget().live_bytes().unwrap(), 0, "{}", case.name);
    }
}

#[test]
fn input_charges_are_released_when_the_op_returns() {
    let inputs = Budget::new(1 << 20);
    for case in cases(&inputs) {
        let cpu = CpuBackend::new(Budget::new(1 << 20));
        let out = (case.run)(&cpu).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert_eq!(
            cpu.budget().live_bytes().unwrap(),
            case.output_bytes,
            "{}: only the returned tensor stays charged",
            case.name
        );
        drop(out);
        assert_eq!(cpu.budget().live_bytes().unwrap(), 0, "{}", case.name);
    }
}

/// Every operand is validated before the first copy is charged, so a NaN in
/// a later operand is `NonFinite` under a budget with no room at all, not a
/// `CapacityExceeded` for an earlier operand's copy.
#[test]
fn an_invalid_later_operand_is_reported_before_any_charge() {
    let inputs = Budget::new(1 << 20);
    let good = f32t(&inputs, 8, &[2, 4]);
    let bad = Tensor::from_f32(
        &[1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &inputs,
    )
    .unwrap();
    // `linear_backward` of `[2, 4]` by `[2, 4]` takes a `[2, 2]` gradient. A
    // `[2, 4]` NaN gradient is malformed, and a shape error is reported
    // before any NaN scan (docs/shape-contract.md D3).
    let bad_grad = Tensor::from_f32(&[1.0, f32::NAN, 3.0, 4.0], &[2, 2], &inputs).unwrap();
    let empty = CpuBackend::new(Budget::new(0));
    for (name, result) in [
        ("mul_forward", empty.mul_forward(&good, &bad).map(|_| ())),
        (
            "linear_backward",
            empty.linear_backward(&good, &good, &bad_grad).map(|_| ()),
        ),
        (
            "clip_grad_norm",
            empty
                .clip_grad_norm(&mut [good.clone(), bad.clone()], 1.0)
                .map(|_| ()),
        ),
    ] {
        match result {
            Err(OjasError::NonFinite { .. }) => {}
            other => panic!("{name}: expected NonFinite, got {other:?}"),
        }
    }
    let ids = Tensor::from_f32(&[0.0, 1.0], &[2], &inputs).unwrap();
    match empty.cross_entropy_mean_forward(&good, &ids, None) {
        Err(OjasError::Dtype { .. }) => {}
        other => panic!("cross_entropy targets: expected Dtype, got {other:?}"),
    }
}
