//! A1 `Tape::backward_seeded` (gate G6) and A2 `Tape::take_grad`.
//!
//! G6: a seed of 0.5 gives exactly 0.5x the seed-1 gradients. Scaling by a
//! power of two commutes with every rounding in the backward chain, so the
//! comparison is bit for bit on each device the tests reach here: the CPU
//! tape (`Exact` and the default `Fast`), and the `Resident` doubles that
//! drive the tape's host and device paths with CPU kernels. `wgpu_tape.rs`
//! runs the same gate on a real wgpu adapter. All data stay in a
//! normal range so no product underflows.

mod common;

use common::{bits, data, Resident};
use ojas_autograd::{Tape, Var};
use ojas_core::{Backend, BackendId, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const VOCAB: usize = 7;
const DIM: usize = 5;
const ROWS: usize = 3;

struct Graph {
    params: [Var; 3],
    root: Var,
}

/// Embedding, linear, silu residual, two reshapes, output linear,
/// cross-entropy. With `doubled` the root is `loss + loss`, so the
/// cross-entropy is not the root; otherwise it is.
fn record<B: Backend>(tape: &mut Tape<B>, doubled: bool) -> Result<Graph, OjasError> {
    let budget = Budget::new(1 << 20);
    let table = tape.leaf(Tensor::from_f32(
        &data(1, VOCAB * DIM),
        &[VOCAB, DIM],
        &budget,
    )?)?;
    let w1 = tape.leaf(Tensor::from_f32(&data(2, DIM * DIM), &[DIM, DIM], &budget)?)?;
    let w2 = tape.leaf(Tensor::from_f32(
        &data(3, VOCAB * DIM),
        &[VOCAB, DIM],
        &budget,
    )?)?;
    let ids = Tensor::from_u32(&[1, 4, 6], &[ROWS], &budget)?;
    let targets = Tensor::from_u32(&[2, 0, 5], &[ROWS], &budget)?;
    let e = tape.embedding(table, ids)?;
    let h = tape.linear(e, w1)?;
    let s = tape.silu(h)?;
    let r = tape.add(h, s)?;
    let r3 = tape.reshape(r, &[1, ROWS, DIM])?;
    let r2 = tape.reshape(r3, &[ROWS, DIM])?;
    let logits = tape.linear(r2, w2)?;
    let loss = tape.cross_entropy(logits, targets, None)?;
    let root = if doubled { tape.add(loss, loss)? } else { loss };
    Ok(Graph {
        params: [table, w1, w2],
        root,
    })
}

fn host_bits<B: Backend>(tape: &Tape<B>, var: Var) -> Vec<u32> {
    let grad = tape.grad(var).expect("leaf gradient");
    bits(&tape.backend().download(grad).unwrap())
}

/// Leaf gradient bits of `record` seeded with `seed`.
fn grads<B: Backend>(backend: B, doubled: bool, seed: f32) -> [Vec<u32>; 3] {
    let mut tape = Tape::new(backend);
    let g = record(&mut tape, doubled).unwrap();
    tape.backward_seeded(g.root, seed).unwrap();
    g.params.map(|p| host_bits(&tape, p))
}

fn scaled(want: &[u32], seed: f32) -> Vec<u32> {
    want.iter()
        .map(|b| (f32::from_bits(*b) * seed).to_bits())
        .collect()
}

/// Seed `s` gives exactly `s` times the seed-1 gradients, for power-of-two
/// `s`, with the cross-entropy at the root and below it.
fn gate_g6<B: Backend>(make: impl Fn() -> B, name: &str) {
    for doubled in [false, true] {
        let one = grads(make(), doubled, 1.0);
        for seed in [0.5f32, 0.25, 0.125] {
            let got = grads(make(), doubled, seed);
            for (i, (g, w)) in got.iter().zip(&one).enumerate() {
                assert_eq!(
                    g,
                    &scaled(w, seed),
                    "{name}: doubled {doubled} seed {seed} param {i} is not seed x the seed-1 gradient"
                );
            }
        }
    }
}

fn cpu_exact() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact)
}

#[test]
fn g6_half_seed_halves_gradients_on_cpu_exact() {
    gate_g6(cpu_exact, "cpu exact");
}

#[test]
fn g6_half_seed_halves_gradients_on_cpu_default_numerics() {
    gate_g6(|| CpuBackend::new(Budget::new(1 << 26)), "cpu default");
}

#[test]
fn g6_half_seed_halves_gradients_on_the_host_double() {
    gate_g6(Resident::host, "host double");
}

/// The root-CE shortcut in `tape.rs`: on a device backend a cross-entropy at
/// the root used to pass its unscaled gradient through, which is right only
/// for a seed of 1. This is the case that fails if the shortcut ignores the
/// seed.
#[test]
fn g6_half_seed_halves_gradients_on_the_device_double() {
    gate_g6(Resident::new, "device double");
}

/// Same graph across the reachable devices: each device double equals the
/// CPU `Exact` tape bit for bit at seed 0.5, so the seed means the same
/// thing on every one of them.
#[test]
fn g6_half_seed_agrees_across_devices() {
    for doubled in [false, true] {
        let want = grads(cpu_exact(), doubled, 0.5);
        assert_eq!(grads(Resident::host(), doubled, 0.5), want, "host double");
        assert_eq!(grads(Resident::new(), doubled, 0.5), want, "device double");
    }
}

/// A seed that is not a power of two still scales: 1/3 (K = 3), within
/// f32 rounding of the scaled seed-1 gradient.
#[test]
fn a_non_power_of_two_seed_scales_within_rounding() {
    let seed = 1.0f32 / 3.0;
    for doubled in [false, true] {
        let one = grads(cpu_exact(), doubled, 1.0);
        let got = grads(cpu_exact(), doubled, seed);
        for (i, (g, w)) in got.iter().zip(&one).enumerate() {
            let want: Vec<f64> = w
                .iter()
                .map(|b| f64::from(f32::from_bits(*b)) * f64::from(seed))
                .collect();
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert!(scale > 0.0, "param {i} has an all-zero gradient");
            for (gb, w) in g.iter().zip(&want) {
                let g = f64::from(f32::from_bits(*gb));
                assert!(
                    (g - w).abs() <= 1e-5 * scale,
                    "param {i}: {g} vs {w} (scale {scale})"
                );
            }
        }
    }
    for doubled in [false, true] {
        let host = grads(Resident::host(), doubled, seed);
        let dev = grads(Resident::new(), doubled, seed);
        assert_eq!(host, dev, "host and device paths disagree at seed 1/3");
    }
}

#[test]
fn backward_is_backward_seeded_at_one() {
    for doubled in [false, true] {
        let mut tape = Tape::new(cpu_exact());
        let g = record(&mut tape, doubled).unwrap();
        tape.backward(g.root).unwrap();
        let plain = g.params.map(|p| host_bits(&tape, p));
        assert_eq!(plain, grads(cpu_exact(), doubled, 1.0));
    }
}

/// The seed reaches every kind of root, not only a loss: a non-scalar root
/// seeded with `s` gives `s` times the gradient of its sum.
#[test]
fn a_non_scalar_root_is_seeded_in_every_element() {
    let cpu = cpu_exact();
    let x0 = [0.3f32, -1.2, 2.5, 0.75];
    for seed in [1.0f32, 0.5, 3.0] {
        let mut tape = Tape::new(cpu.clone());
        let x = tape
            .leaf(Tensor::from_f32(&x0, &[2, 2], cpu.budget()).unwrap())
            .unwrap();
        let y = tape.mul(x, x).unwrap();
        tape.backward_seeded(y, seed).unwrap();
        let got = tape.grad(x).unwrap().to_f32_vec().unwrap();
        let want: Vec<f32> = x0.iter().map(|a| 2.0 * a * seed).collect();
        assert_eq!(got, want, "seed {seed}");
    }
}

#[test]
fn backward_seeded_refuses_seeds_a_training_loop_never_passes() {
    for make in [Resident::host, Resident::new] {
        let mut tape = Tape::new(make());
        let g = record(&mut tape, false).unwrap();
        tape.backward(g.root).unwrap();
        assert!(tape.grad(g.params[0]).is_some());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            match tape.backward_seeded(g.root, bad) {
                Err(OjasError::NonFinite { .. }) => {}
                other => panic!("seed {bad}: expected NonFinite, got {other:?}"),
            }
        }
        for bad in [
            0.0f32,
            -0.0,
            -0.5,
            -1.0,
            f32::MIN_POSITIVE / 2.0,
            -f32::MIN_POSITIVE,
        ] {
            match tape.backward_seeded(g.root, bad) {
                Err(OjasError::OutOfRange { .. }) => {}
                other => panic!("seed {bad:e}: expected OutOfRange, got {other:?}"),
            }
            for p in g.params {
                assert!(
                    tape.grad(p).is_none(),
                    "a refused seed {bad:e} left a stale gradient"
                );
            }
        }
        // The smallest normal and a large seed are accepted.
        tape.backward_seeded(g.root, f32::MIN_POSITIVE).unwrap();
        tape.backward_seeded(g.root, 1024.0).unwrap();
        assert!(tape.grad(g.params[0]).is_some());
    }
}

#[test]
fn a_variable_off_the_tape_clears_gradients_too() {
    let mut tape = Tape::new(cpu_exact());
    let g = record(&mut tape, false).unwrap();
    tape.backward(g.root).unwrap();
    assert!(matches!(
        tape.backward_seeded(Var(10_000), 0.5),
        Err(OjasError::OutOfRange { .. })
    ));
    for p in g.params {
        assert!(tape.grad(p).is_none());
    }
}

/// A2: the moved gradient is uniquely owned, so an in-place write succeeds,
/// where a clone of the tape's handle is refused; a second take is `None`.
#[test]
fn take_grad_moves_a_uniquely_owned_gradient_out() {
    let mut tape = Tape::new(cpu_exact());
    let g = record(&mut tape, true).unwrap();
    tape.backward_seeded(g.root, 0.5).unwrap();
    for p in g.params {
        let want = bits(tape.grad(p).unwrap());
        let mut shared = tape.grad(p).cloned().unwrap();
        let n = shared.num_elements().unwrap();
        assert!(
            shared.ensure_writable_f32(n).is_err(),
            "negative control: a clone of the tape's handle must be shared"
        );
        drop(shared);

        let mut taken = tape.take_grad(p).expect("gradient");
        assert_eq!(bits(&taken), want, "take_grad changed the values");
        assert!(tape.grad(p).is_none(), "the tape still holds the gradient");
        assert!(
            tape.take_grad(p).is_none(),
            "a second take returned a value"
        );
        taken
            .ensure_writable_f32(n)
            .expect("the taken gradient is shared");
        let halved: Vec<f32> = taken
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v * 0.5)
            .collect();
        taken.write_f32(&halved).unwrap();
        // An in-place backend op on the moved tensor.
        let other =
            Tensor::from_f32(&vec![0.25; n], taken.shape(), tape.backend().budget()).unwrap();
        tape.backend()
            .clip_grad_norm(std::slice::from_mut(&mut taken), 1e-3)
            .expect("in-place clip on the taken gradient");
        tape.backend().accumulate_grad(&mut taken, &other).unwrap();
    }
    tape.clear();
    assert!(tape.take_grad(g.params[0]).is_none());
}

#[test]
fn take_grad_on_the_device_double_moves_a_sole_device_buffer() {
    let mut tape = Tape::new(Resident::new());
    let g = record(&mut tape, false).unwrap();
    tape.backward_seeded(g.root, 0.5).unwrap();
    for p in g.params {
        let mut shared = tape.grad(p).cloned().unwrap();
        assert!(shared.device_buffer_mut().is_err(), "negative control");
        drop(shared);
        let mut taken = tape.take_grad(p).unwrap();
        assert_eq!(taken.device(), Some(BackendId::Wgpu));
        taken
            .device_buffer_mut()
            .expect("the taken device gradient is shared");
        assert!(tape.take_grad(p).is_none());
    }
}

/// A leaf that is itself the root, and a leaf reached only through a
/// reshape view, both hand out a sole owner.
#[test]
fn take_grad_is_unique_through_views_and_at_the_root() {
    let cpu = cpu_exact();
    let mut tape = Tape::new(cpu.clone());
    let x = tape
        .leaf(Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[4], cpu.budget()).unwrap())
        .unwrap();
    let v = tape.reshape(x, &[2, 2]).unwrap();
    tape.backward_seeded(v, 0.5).unwrap();
    let mut taken = tape.take_grad(x).unwrap();
    assert_eq!(taken.to_f32_vec().unwrap(), vec![0.5; 4]);
    taken
        .ensure_writable_f32(4)
        .expect("reshape-view gradient is shared");

    tape.backward_seeded(x, 0.25).unwrap();
    let mut taken = tape.take_grad(x).unwrap();
    assert_eq!(taken.to_f32_vec().unwrap(), vec![0.25; 4]);
    taken
        .ensure_writable_f32(4)
        .expect("root gradient is shared");
}

#[test]
fn take_grad_off_the_tape_is_none() {
    let mut tape = Tape::new(cpu_exact());
    assert!(tape.take_grad(Var(0)).is_none());
    let g = record(&mut tape, false).unwrap();
    assert!(tape.take_grad(g.params[0]).is_none(), "no backward yet");
    assert!(tape.take_grad(Var(10_000)).is_none());
}
