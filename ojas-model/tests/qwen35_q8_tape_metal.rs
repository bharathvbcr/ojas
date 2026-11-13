//! The resident q8 base on the Tape on Metal (card ft-5f743c19, qv2 W3): the packed weights are
//! uploaded once at load and the host copies are dropped, a linear over one dequantises it on the
//! device, and the loss and every adapter gradient equal the fake-quant dense base's bit for bit
//! (the same `dequant_affine_bf16` output and the same `linear_forward` / `linear_backward_input`
//! over it). Run with `--test-threads=1`. A missing device fails unless `OJAS_ALLOW_NO_GPU=1`.

#![cfg(target_os = "macos")]

mod common;
mod qwen35_common;
mod qwen35_q8_common;

use ojas_autograd::Tape;
use ojas_core::{Autocast, BackendId, Budget, DType, OjasError};
use ojas_metal::MetalBackend;
use ojas_model::qwen35::{bind_frozen_base, load_hf_q8_pinned, FrozenWeight};
use ojas_model::ActivationCheckpoint;
use qwen35_common::lora::*;
use qwen35_common::*;
use qwen35_q8_common::*;

fn metal() -> Option<MetalBackend> {
    match MetalBackend::new(Budget::new(8 << 30)) {
        Ok(m) => Some(m),
        Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Metal device: {e:?}");
            None
        }
        Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn fixed(run: &LoraRun) -> Vec<Vec<u32>> {
    run.grads
        .iter()
        .flat_map(|(_, a, b)| [bits(a), bits(b)])
        .collect()
}

/// Metal over the fq8 dense base against Metal over the packed q8 base, both on the one device,
/// non-zero `B` so every layer's dX reaches an adapter.
fn run_pair(m: &MetalBackend, activations: ActivationCheckpoint) -> (LoraRun, LoraRun) {
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let opts = || Opts {
        activations,
        ..fx.opts()
    };
    let dense = lora_step(m.clone(), &fq8_host(), &sites, &d, &opts());
    let base = load_q8(m);
    let packed = lora_step_base(m.clone(), &base, &sites, &d, &opts());
    (dense, packed)
}

fn assert_same(packed: &LoraRun, dense: &LoraRun) {
    assert_eq!(
        packed.loss.to_bits(),
        dense.loss.to_bits(),
        "loss {} against {}",
        packed.loss,
        dense.loss
    );
    assert_eq!(fixed(packed), fixed(dense), "adapter gradients differ");
    assert!(packed
        .grads
        .iter()
        .any(|(_, a, b)| a.iter().chain(b).any(|&x| x != 0.0)));
}

#[test]
fn q8_base_loss_and_adapter_grads_equal_fq8_bitwise_off_metal() {
    let Some(m) = metal() else { return };
    let (dense, packed) = run_pair(&m, ActivationCheckpoint::Off);
    assert_same(&packed, &dense);
}

#[test]
fn q8_base_loss_and_adapter_grads_equal_fq8_bitwise_blocks_metal() {
    let Some(m) = metal() else { return };
    let (dense, packed) = run_pair(&m, ActivationCheckpoint::Blocks);
    assert_same(&packed, &dense);
}

/// Inside a bf16 autocast region the dequantisation passes through, on Metal too.
#[test]
fn q8_base_equals_fq8_bitwise_under_bf16_autocast_metal() {
    let Some(m) = metal() else { return };
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let opts = || Opts {
        bf16: true,
        ..fx.opts()
    };
    let dense = lora_step(Autocast::new(m.clone()), &fq8_host(), &sites, &d, &opts());
    let base = load_q8(&m);
    let packed = lora_step_base(Autocast::new(m.clone()), &base, &sites, &d, &opts());
    assert_same(&packed, &dense);
}

/// All 16 parts, per field, through the device's own dequantisation.
#[test]
fn every_frozen_weight_dequantises_to_the_fq8_field_metal() {
    let Some(m) = metal() else { return };
    let (q8, fq8) = (load_q8(&m), fq8_host());
    let (packed, dense) = (wrapped_fields(&q8), wrapped_fields(&fq8));
    assert_eq!((packed.len(), dense.len()), (WRAPPED_PARTS, WRAPPED_PARTS));
    for ((name, w), (other, want)) in packed.into_iter().zip(dense) {
        assert_eq!(name, other);
        let FrozenWeight::Quant(q) = w else {
            panic!("{name} is not packed");
        };
        assert_eq!(q.backend(), BackendId::Metal, "{name}");
        assert!(
            q.packed().device().is_some(),
            "{name}: parts are on the device"
        );
        assert_eq!([q.rows(), q.cols()], want.shape(), "{name}");
        assert_eq!(
            host_bits(&m, &dequant(&m, q)),
            want.bf16_slice().unwrap(),
            "{name}"
        );
    }
}

/// The counts of the CPU tape, on the device: nothing at the bind, 16 per forward, 16 more per
/// backward, and 32 more under Blocks (the replay and the dX).
#[test]
fn a_step_dequantises_per_linear_use_on_metal() {
    let Some(m) = metal() else { return };
    let base = load_q8(&m);
    let off = counted_step(m.clone(), &base, ActivationCheckpoint::Off, false);
    assert_eq!(
        (off.after_bind, off.after_forward, off.after_backward),
        (0, WRAPPED_PARTS, 2 * WRAPPED_PARTS)
    );
    let blocks = counted_step(m.clone(), &base, ActivationCheckpoint::Blocks, false);
    assert_eq!(
        (
            blocks.after_bind,
            blocks.after_forward,
            blocks.after_backward
        ),
        (0, WRAPPED_PARTS, 3 * WRAPPED_PARTS)
    );
    // The control: dequantising at bind counts at the bind and never again.
    let o1 = counted_step(m, &base, ActivationCheckpoint::Off, true);
    assert_eq!(
        (o1.after_bind, o1.after_forward, o1.after_backward),
        (WRAPPED_PARTS, WRAPPED_PARTS, WRAPPED_PARTS)
    );
}

/// Uploaded once, host copies dropped: after the load the host budget holds only the dense
/// tensors, and every part of every packed weight is a device tensor.
#[test]
fn the_load_leaves_no_host_copy_of_a_packed_weight() {
    let Some(m) = metal() else { return };
    let host = budget();
    let base = load_hf_q8_pinned(
        &spec(),
        &q8_path(),
        &q8_identity(),
        &q8_config(),
        PREFIX,
        DType::Bf16,
        DType::F32,
        &host,
        &m,
    )
    .unwrap();
    assert_eq!(host.live_bytes().unwrap(), dense_host_bytes(&base));
    for (name, w) in wrapped_fields(&base) {
        let FrozenWeight::Quant(q) = w else {
            panic!("{name} is not packed");
        };
        for part in [q.packed(), q.scales(), q.biases()] {
            assert_eq!(part.device(), Some(BackendId::Metal), "{name}");
        }
    }
}

/// W3-D9: host parts on a device tape are `Placement`, at the bind and at the linear.
#[test]
fn host_placed_parts_on_a_metal_tape_are_a_placement_refusal() {
    let Some(m) = metal() else { return };
    let host_base = load_q8(&cpu());
    let mut tape = Tape::new(m.clone());
    match bind_frozen_base(&mut tape, &host_base) {
        Err(OjasError::Placement {
            op,
            expected,
            found,
        }) => {
            assert_eq!(op, "qwen35::bind_frozen_base");
            assert_eq!((expected, found), (Some(BackendId::Metal), None));
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    let FrozenWeight::Quant(q) = wrapped_fields(&host_base)[0].1 else {
        panic!("not packed");
    };
    let x = tape
        .leaf(ojas_core::Tensor::from_f32(&[0.5f32; 64], &[1, 64], &budget()).unwrap())
        .unwrap();
    match tape.linear_q(x, q) {
        Err(OjasError::Placement { op, .. }) => assert_eq!(op, "Tape::linear_q"),
        other => panic!("{other:?}"),
    }
}

/// Parts uploaded by one Metal device are refused by another's tape, by that backend's own
/// operand check at the first dequantisation, and the refusal leaves nothing recorded.
#[test]
fn parts_from_another_metal_device_are_refused() {
    let Some(first) = metal() else { return };
    let Some(second) = metal() else { return };
    let base = load_q8(&first);
    let FrozenWeight::Quant(q) = wrapped_fields(&base)[0].1 else {
        panic!("not packed");
    };
    let mut tape = Tape::new(second);
    let x = tape
        .leaf(ojas_core::Tensor::from_f32(&vec![0.5; q.cols()], &[1, q.cols()], &budget()).unwrap())
        .unwrap();
    match tape.linear_q(x, q) {
        Err(OjasError::Backend { id, detail }) => {
            assert_eq!(id, BackendId::Metal);
            assert!(detail.contains("different MetalBackend device"), "{detail}");
        }
        other => panic!("{other:?}"),
    }
    assert!(tape.value(ojas_autograd::Var(1)).is_err());
}

/// `place` checks the room for all three parts before the first upload: a budget that cannot hold
/// them refuses with nothing charged and nothing uploaded.
#[test]
fn place_refuses_before_any_upload_when_the_budget_is_short() {
    let Some(m) = metal() else { return };
    let host_base = load_q8(&cpu());
    let FrozenWeight::Quant(q) = wrapped_fields(&host_base)[0].1 else {
        panic!("not packed");
    };
    let want = q.nbytes().unwrap() as u64;
    // The same device and link, a budget one byte short of the three parts.
    let short_budget = Budget::new(want - 1);
    let short = m.with_budget(short_budget.clone());
    let parts = ojas_core::AffineQuantized::from_parts(
        q.spec(),
        q.rows(),
        q.cols(),
        q.packed().clone(),
        q.scales().clone(),
        q.biases().clone(),
    )
    .unwrap();
    match parts.place(&short) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("{:?}", other.map(|_| ())),
    }
    assert_eq!(short_budget.live_bytes().unwrap(), 0);
    assert_eq!(short_budget.peak_bytes(), 0);
}
