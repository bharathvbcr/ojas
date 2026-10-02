//! A3 `Tape::linear_cross_entropy`: the fused head through
//! `Backend::linear_cross_entropy_mean`.
//!
//! The tape-mechanics tests run on the `Resident` doubles (`tests/common`),
//! whose fused op composes the unfused CPU kernels, so the fused tape and the
//! unfused tape path (`linear` → `cross_entropy`) agree bit for bit there.
//! `cpu_tape_fused_ce_matches_the_unfused_path` runs the real `CpuBackend`
//! fused kernel against the unfused path at G3's 1e-6 relative tolerance.

mod common;

use common::{bits, ce, data, lin, Fused, Resident, Rng};
use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Backend, BackendId, Budget, CeChunk, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const CHUNK: CeChunk = CeChunk { rows: 2, cols: 3 };

/// Shapes of a small head: `x` is `[N, D]` (a non-leaf, from an embedding
/// and a linear), `w` is `[V, D]` and tied: it is both the embedding table
/// and the head weight, so its gradient sums two contributions.
const N: usize = 5;
const D: usize = 4;
const V: usize = 7;

struct Head {
    w: Var,
    w1: Var,
    loss: Var,
    root: Var,
}

#[derive(Clone, Copy)]
enum Path {
    Unfused,
    Fused,
}

fn targets(ignore: Option<u32>) -> Vec<u32> {
    let mut t: Vec<u32> = (0..N).map(|i| ((i * 3 + 1) % V) as u32).collect();
    if let Some(sentinel) = ignore {
        t[1] = sentinel;
        t[3] = sentinel;
    }
    t
}

fn record<B: Backend>(
    tape: &mut Tape<B>,
    path: Path,
    ignore: Option<u32>,
    doubled: bool,
) -> Result<Head, OjasError> {
    let budget = Budget::new(1 << 20);
    let w = tape.leaf(Tensor::from_f32(&data(5, V * D), &[V, D], &budget)?)?;
    let w1 = tape.leaf(Tensor::from_f32(&data(6, D * D), &[D, D], &budget)?)?;
    let ids: Vec<u32> = (0..N).map(|i| ((i * 2 + 3) % V) as u32).collect();
    let e = tape.embedding(w, Tensor::from_u32(&ids, &[N], &budget)?)?;
    let x = tape.linear(e, w1)?;
    let t = Tensor::from_u32(&targets(ignore), &[N], &budget)?;
    let loss = match path {
        Path::Unfused => {
            let logits = tape.linear(x, w)?;
            tape.cross_entropy(logits, t, ignore)?
        }
        Path::Fused => tape.linear_cross_entropy(x, w, t, ignore, CHUNK)?,
    };
    let root = if doubled { tape.add(loss, loss)? } else { loss };
    Ok(Head { w, w1, loss, root })
}

struct Run {
    loss: Vec<u32>,
    grads: [Vec<u32>; 2],
}

fn host_bits<B: Backend>(tape: &Tape<B>, t: &Tensor) -> Vec<u32> {
    bits(&tape.backend().download(t).unwrap())
}

fn run<B: Backend>(backend: B, path: Path, ignore: Option<u32>, doubled: bool, seed: f32) -> Run {
    let mut tape = Tape::new(backend);
    let h = record(&mut tape, path, ignore, doubled).unwrap();
    tape.backward_seeded(h.root, seed).unwrap();
    let grads = [h.w, h.w1].map(|v| host_bits(&tape, tape.grad(v).expect("leaf gradient")));
    Run {
        loss: host_bits(&tape, tape.value(h.loss).unwrap()),
        grads,
    }
}

/// (a) The fused tape equals the unfused tape path: loss and both leaf
/// gradients, at seeds 1 and 0.5, with the loss at the root and below it,
/// with and without an ignore index, on the host and device paths.
#[test]
fn fused_equals_unfused_tape_path() {
    for make in [Resident::host, Resident::new] {
        for ignore in [None, Some(99)] {
            for doubled in [false, true] {
                for seed in [1.0f32, 0.5, 0.25] {
                    let want = run(make(), Path::Unfused, ignore, doubled, seed);
                    let got = run(make(), Path::Fused, ignore, doubled, seed);
                    let what = format!("ignore {ignore:?} doubled {doubled} seed {seed}");
                    assert_eq!(got.loss, want.loss, "{what}: loss");
                    assert_eq!(got.grads[0], want.grads[0], "{what}: tied w grad");
                    assert_eq!(got.grads[1], want.grads[1], "{what}: w1 grad");
                }
            }
        }
    }
}

/// G6 through the fused op: a power-of-two seed scales its gradients
/// exactly, on both paths.
#[test]
fn fused_seed_scales_exactly() {
    for make in [Resident::host, Resident::new] {
        for doubled in [false, true] {
            let one = run(make(), Path::Fused, Some(99), doubled, 1.0);
            let half = run(make(), Path::Fused, Some(99), doubled, 0.5);
            for (h, o) in half.grads.iter().zip(&one.grads) {
                let want: Vec<u32> = o
                    .iter()
                    .map(|b| (f32::from_bits(*b) * 0.5).to_bits())
                    .collect();
                assert_eq!(h, &want, "doubled {doubled}");
            }
        }
    }
}

/// Forward calls the backend once with gradients; backward only scales
/// them. A second backward over the same node recomputes them through the
/// backend, because the first one moved them out, and gets the same bits.
#[test]
fn backward_scales_the_stored_gradients_and_a_repeat_recomputes() {
    for make in [Resident::host, Resident::new] {
        let mut tape = Tape::new(make());
        let h = record(&mut tape, Path::Fused, None, false).unwrap();
        assert_eq!(tape.backend().fused_calls(), 1);
        tape.backward_seeded(h.root, 0.5).unwrap();
        assert_eq!(tape.backend().fused_calls(), 1, "backward called the op");
        let first = [h.w, h.w1].map(|v| host_bits(&tape, tape.grad(v).unwrap()));
        tape.backward_seeded(h.root, 0.5).unwrap();
        assert_eq!(tape.backend().fused_calls(), 2, "repeat did not recompute");
        let second = [h.w, h.w1].map(|v| host_bits(&tape, tape.grad(v).unwrap()));
        assert_eq!(first, second, "repeat backward changed the gradients");
        tape.backward(h.root).unwrap();
        let unit = [h.w, h.w1].map(|v| host_bits(&tape, tape.grad(v).unwrap()));
        for (s, u) in second.iter().zip(&unit) {
            let want: Vec<u32> = u
                .iter()
                .map(|b| (f32::from_bits(*b) * 0.5).to_bits())
                .collect();
            assert_eq!(s, &want);
        }
    }
}

/// The device path reads nothing back during forward or backward, at any
/// seed, and keeps every gradient on the device.
#[test]
fn fused_on_the_device_reads_nothing_back() {
    for seed in [1.0f32, 0.5] {
        let mut tape = Tape::new(Resident::new());
        let before = tape.backend().budget().device_readbacks();
        let h = record(&mut tape, Path::Fused, Some(99), false).unwrap();
        tape.backward_seeded(h.root, seed).unwrap();
        assert_eq!(tape.backend().budget().device_readbacks(), before);
        for v in [h.w, h.w1] {
            assert_eq!(tape.grad(v).unwrap().device(), Some(BackendId::Wgpu));
        }
    }
}

/// A2 with A3: the head weight's gradient is moved out of the fused record,
/// not shared with it, so `take_grad` hands out a sole owner, at seed 1
/// (where the gradient is passed through unscaled) and at seed 0.5.
#[test]
fn a_fused_head_weight_gradient_is_uniquely_owned() {
    let budget = Budget::new(1 << 20);
    for seed in [1.0f32, 0.5] {
        // Host path: an untied head weight whose only use is the fused op.
        let mut tape = Tape::new(Resident::host());
        let x = tape
            .leaf(Tensor::from_f32(&data(1, N * D), &[N, D], &budget).unwrap())
            .unwrap();
        let w = tape
            .leaf(Tensor::from_f32(&data(2, V * D), &[V, D], &budget).unwrap())
            .unwrap();
        let t = Tensor::from_u32(&targets(None), &[N], &budget).unwrap();
        let loss = tape.linear_cross_entropy(x, w, t, None, CHUNK).unwrap();
        tape.backward_seeded(loss, seed).unwrap();
        for (v, n) in [(x, N * D), (w, V * D)] {
            let mut g = tape.take_grad(v).unwrap();
            g.ensure_writable_f32(n)
                .unwrap_or_else(|e| panic!("seed {seed}: host gradient is shared: {e}"));
        }

        // Device path.
        let mut tape = Tape::new(Resident::new());
        let x = tape
            .leaf(Tensor::from_f32(&data(1, N * D), &[N, D], &budget).unwrap())
            .unwrap();
        let w = tape
            .leaf(Tensor::from_f32(&data(2, V * D), &[V, D], &budget).unwrap())
            .unwrap();
        let t = Tensor::from_u32(&targets(None), &[N], &budget).unwrap();
        let loss = tape.linear_cross_entropy(x, w, t, None, CHUNK).unwrap();
        tape.backward_seeded(loss, seed).unwrap();
        for v in [x, w] {
            let mut g = tape.take_grad(v).unwrap();
            g.device_buffer_mut()
                .unwrap_or_else(|e| panic!("seed {seed}: device gradient is shared: {e}"));
        }
    }
}

/// (b) f64 gradcheck of the fused op on the tape, over random shapes, with
/// an ignore index and a random seed-like scale on the loss.
#[test]
fn fused_gradcheck_random_shapes() {
    const H: f64 = 1e-4;
    const ATOL: f64 = 1e-4;
    const RTOL: f64 = 1e-3;
    let mut rng = Rng(17);
    for case in 0..12 {
        let n = 1 + rng.below(6);
        let d = [1, 2, 3, 5, 9][rng.below(5)];
        let v = 2 + rng.below(9);
        let ignore = if case % 2 == 0 {
            None
        } else {
            Some(v as u32 + 2)
        };
        let mut tg: Vec<u32> = (0..n).map(|_| rng.below(v) as u32).collect();
        if let (Some(sentinel), true) = (ignore, n > 1) {
            tg[rng.below(n)] = sentinel;
        }
        let x0 = rng.vec(n * d, 1.0);
        let w0 = rng.vec(v * d, 1.0);
        let chunk = CeChunk {
            rows: 1 + rng.below(n),
            cols: 1 + rng.below(v),
        };
        let seed = [1.0f32, 0.5, 0.25][rng.below(3)];

        let double = Resident::host();
        let f = |a: &[f64]| -> Vec<f32> { a.iter().map(|&x| x as f32).collect() };
        let mut tape = Tape::new(double);
        let budget = tape.backend().budget().clone();
        let x = tape
            .leaf(Tensor::from_f32(&f(&x0), &[n, d], &budget).unwrap())
            .unwrap();
        let w = tape
            .leaf(Tensor::from_f32(&f(&w0), &[v, d], &budget).unwrap())
            .unwrap();
        let t = Tensor::from_u32(&tg, &[n], &budget).unwrap();
        let loss = tape.linear_cross_entropy(x, w, t, ignore, chunk).unwrap();
        let got = f64::from(tape.value(loss).unwrap().to_f32_vec().unwrap()[0]);
        let want = ce(&lin(&x0, &w0, d, v), &tg, v, ignore);
        assert!(
            (got - want).abs() < 1e-5 * (1.0 + want.abs()),
            "case {case}: loss {got} vs {want}"
        );
        tape.backward_seeded(loss, seed).unwrap();
        let s = f64::from(seed);
        let nx = central_diff(&x0, H, |p| Ok(s * ce(&lin(p, &w0, d, v), &tg, v, ignore))).unwrap();
        let nw = central_diff(&w0, H, |p| Ok(s * ce(&lin(&x0, p, d, v), &tg, v, ignore))).unwrap();
        for (name, var, numeric) in [("x", x, nx), ("w", w, nw)] {
            let analytic = tape.grad(var).unwrap().to_f32_vec().unwrap();
            gradients_match(&analytic, &numeric, ATOL, RTOL)
                .unwrap_or_else(|e| panic!("case {case} n{n} d{d} v{v} {name}: {e}"));
        }
    }
}

/// (c) A row whose target is the ignore index contributes nothing: its
/// input-gradient row is exactly zero, and the loss and weight gradient
/// equal a run over the valid rows alone.
#[test]
fn ignored_rows_contribute_nothing() {
    let budget = Budget::new(1 << 20);
    let all = targets(Some(99));
    let keep: Vec<usize> = (0..N).filter(|&i| all[i] != 99).collect();
    let x_all = data(1, N * D);
    let x_kept: Vec<f32> = keep
        .iter()
        .flat_map(|&i| x_all[i * D..(i + 1) * D].to_vec())
        .collect();
    let t_kept: Vec<u32> = keep.iter().map(|&i| all[i]).collect();
    let w0 = data(2, V * D);
    for make in [Resident::host, Resident::new] {
        let mut full = Tape::new(make());
        let x = full
            .leaf(Tensor::from_f32(&x_all, &[N, D], &budget).unwrap())
            .unwrap();
        let w = full
            .leaf(Tensor::from_f32(&w0, &[V, D], &budget).unwrap())
            .unwrap();
        let t = Tensor::from_u32(&all, &[N], &budget).unwrap();
        let loss = full.linear_cross_entropy(x, w, t, Some(99), CHUNK).unwrap();
        full.backward(loss).unwrap();

        let mut kept = Tape::new(make());
        let n = keep.len();
        let kx = kept
            .leaf(Tensor::from_f32(&x_kept, &[n, D], &budget).unwrap())
            .unwrap();
        let kw = kept
            .leaf(Tensor::from_f32(&w0, &[V, D], &budget).unwrap())
            .unwrap();
        let kt = Tensor::from_u32(&t_kept, &[n], &budget).unwrap();
        let kloss = kept.linear_cross_entropy(kx, kw, kt, None, CHUNK).unwrap();
        kept.backward(kloss).unwrap();

        let down = |tape: &Tape<Resident>, t: &Tensor| {
            tape.backend().download(t).unwrap().to_f32_vec().unwrap()
        };
        let close = |a: &[f32], b: &[f32], what: &str| {
            assert_eq!(a.len(), b.len());
            for (p, q) in a.iter().zip(b) {
                assert!(
                    (p - q).abs() <= 1e-6 * q.abs().max(1e-6),
                    "{what}: {p} vs {q}"
                );
            }
        };
        close(
            &down(&full, full.value(loss).unwrap()),
            &down(&kept, kept.value(kloss).unwrap()),
            "loss",
        );
        close(
            &down(&full, full.grad(w).unwrap()),
            &down(&kept, kept.grad(kw).unwrap()),
            "weight gradient",
        );
        let gx = down(&full, full.grad(x).unwrap());
        let gk = down(&kept, kept.grad(kx).unwrap());
        for row in 0..N {
            let r = &gx[row * D..(row + 1) * D];
            if all[row] == 99 {
                assert!(
                    r.iter().all(|v| v.to_bits() == 0),
                    "ignored row {row} has gradient {r:?}"
                );
            } else {
                let k = keep.iter().position(|&i| i == row).unwrap();
                close(r, &gk[k * D..(k + 1) * D], "kept row");
            }
        }
    }
}

/// Every row ignored: the backend's `NonFinite` comes back and nothing is
/// recorded.
#[test]
fn an_all_ignored_batch_is_nonfinite_and_records_nothing() {
    let budget = Budget::new(1 << 20);
    for make in [Resident::host, Resident::new] {
        let mut tape = Tape::new(make());
        let x = tape
            .leaf(Tensor::from_f32(&data(1, N * D), &[N, D], &budget).unwrap())
            .unwrap();
        let w = tape
            .leaf(Tensor::from_f32(&data(2, V * D), &[V, D], &budget).unwrap())
            .unwrap();
        let t = Tensor::from_u32(&[7; N], &[N], &budget).unwrap();
        match tape.linear_cross_entropy(x, w, t, Some(7), CHUNK) {
            Err(OjasError::NonFinite { .. }) => {}
            other => panic!("expected NonFinite, got {other:?}"),
        }
        let next = tape
            .leaf(Tensor::from_f32(&[1.0], &[1], &budget).unwrap())
            .unwrap();
        assert_eq!(next, Var(2), "a failed fused call was recorded");
    }
}

/// (d) A backend without the fused op returns `Unsupported`; the tape
/// returns it and records nothing. There is no fallback to the unfused
/// path, which would have returned a loss.
#[test]
fn unsupported_propagates_with_no_fallback() {
    let budget = Budget::new(1 << 20);
    for make in [Resident::host, Resident::new] {
        let mut tape = Tape::new(make().without_fused());
        let x = tape
            .leaf(Tensor::from_f32(&data(1, N * D), &[N, D], &budget).unwrap())
            .unwrap();
        let w = tape
            .leaf(Tensor::from_f32(&data(2, V * D), &[V, D], &budget).unwrap())
            .unwrap();
        let t = Tensor::from_u32(&targets(None), &[N], &budget).unwrap();
        match tape.linear_cross_entropy(x, w, t, None, CHUNK) {
            Err(OjasError::Unsupported { op, .. }) => {
                assert_eq!(op, "linear_cross_entropy_mean");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        assert_eq!(tape.backend().fused_calls(), 1);
        let next = tape
            .leaf(Tensor::from_f32(&[1.0], &[1], &budget).unwrap())
            .unwrap();
        assert_eq!(next, Var(2), "the refused call recorded a node");
    }
}

/// A backend that breaks the `LinearCe` contract (no gradients, or a
/// gradient of the wrong shape) is refused with `Backend`, not unwrapped
/// or passed on to fail later in backward.
#[test]
fn a_backend_breaking_the_fused_contract_is_refused() {
    let budget = Budget::new(1 << 20);
    for mode in [Fused::NoGrads, Fused::WrongShape] {
        for make in [Resident::host, Resident::new] {
            let mut tape = Tape::new(make().with_fused(mode));
            let x = tape
                .leaf(Tensor::from_f32(&data(1, N * D), &[N, D], &budget).unwrap())
                .unwrap();
            let w = tape
                .leaf(Tensor::from_f32(&data(2, V * D), &[V, D], &budget).unwrap())
                .unwrap();
            let t = Tensor::from_u32(&targets(None), &[N], &budget).unwrap();
            match tape.linear_cross_entropy(x, w, t, None, CHUNK) {
                Err(OjasError::Backend { detail, .. }) => {
                    assert!(detail.contains("linear_cross_entropy_mean"), "{detail}");
                }
                other => panic!("{mode:?}: expected Backend, got {other:?}"),
            }
            let next = tape
                .leaf(Tensor::from_f32(&[1.0], &[1], &budget).unwrap())
                .unwrap();
            assert_eq!(next, Var(2), "{mode:?}: a refused call recorded a node");
        }
    }
}

#[test]
fn off_tape_operands_are_refused_before_the_backend_is_called() {
    let budget = Budget::new(1 << 20);
    let mut tape = Tape::new(Resident::host());
    let x = tape
        .leaf(Tensor::from_f32(&data(1, N * D), &[N, D], &budget).unwrap())
        .unwrap();
    let t = Tensor::from_u32(&targets(None), &[N], &budget).unwrap();
    assert!(matches!(
        tape.linear_cross_entropy(x, Var(9), t, None, CHUNK),
        Err(OjasError::OutOfRange { .. })
    ));
    assert_eq!(tape.backend().fused_calls(), 0);
}

/// The real `CpuBackend`. Until CPU implements the fused op it must refuse
/// with `Unsupported`, and the tape must pass that on; once it does, the
/// fused tape must match the unfused one at 1e-6 relative (G3 (a)). The
/// test prints which branch it ran.
#[test]
fn cpu_tape_fused_ce_matches_the_unfused_path() {
    // CpuBackend implements the fused op natively (ojas-cpu fused_ce.rs), so
    // an Unsupported here is a failure, not a branch.
    let cpu = || CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact);
    for ignore in [None, Some(99)] {
        for doubled in [false, true] {
            for seed in [1.0f32, 0.5, 0.25] {
                let want = run(cpu(), Path::Unfused, ignore, doubled, seed);
                let got = run(cpu(), Path::Fused, ignore, doubled, seed);
                let what = format!("ignore {ignore:?} doubled {doubled} seed {seed}");
                let rel = |a: &[u32], b: &[u32], part: &str| {
                    assert_eq!(a.len(), b.len(), "{what} {part}: length");
                    for (p, q) in a.iter().zip(b) {
                        let (p, q) = (f32::from_bits(*p), f32::from_bits(*q));
                        assert!(
                            (p - q).abs() <= 1e-6 * q.abs().max(1e-6),
                            "{what} {part}: fused {p} unfused {q}"
                        );
                    }
                };
                rel(&got.loss, &want.loss, "loss");
                rel(&got.grads[0], &want.grads[0], "tied w grad");
                rel(&got.grads[1], &want.grads[1], "w1 grad");
            }
        }
    }
}
