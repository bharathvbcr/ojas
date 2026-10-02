//! Randomized tape stress: many iterations of a seeded random op sequence,
//! then `backward_seeded`, `take_grad` into trainer-owned accumulators
//! (`Backend::accumulate_grad`), and `clear`, the way §3's training step
//! drives the tape.
//!
//! Checks, per run:
//! - after every `clear`, `Budget::live_bytes` is back to the baseline plus
//!   the accumulators, and after the run it is back to the baseline, so
//!   neither the tape nor a taken gradient leaks a charge;
//! - every taken gradient is uniquely owned, and a second take is `None`;
//! - the same seed gives the same bits: run to run, between the host and
//!   device paths, and (on CPU `Exact`) between one thread and four.
//!
//! The tape borrows its backend (`Tape<&B>`, through ojas-core's blanket
//! `impl Backend for &B`), so the budget can be read after the tape drops.

mod common;

use common::{Resident, Rng};
use ojas_autograd::{Tape, Var};
use ojas_core::{
    Backend, BackendId, Budget, CeChunk, DType, Numerics, OjasError, Tensor, RMS_NORM_EPS,
};
use ojas_cpu::CpuBackend;

const D: usize = 4;
const V: usize = 5;
const ITERS: usize = 150;
/// Fixed-shape parameters a trainer accumulates: `wa`, `wb`, `norm`, `head`.
const PARAM_SHAPES: [&[usize]; 4] = [&[D, D], &[D, D], &[D], &[V, D]];
const SEEDS: [f32; 5] = [1.0, 0.5, 0.25, 1.0 / 3.0, 2.0];

struct Outcome {
    /// Bits of every taken gradient, in order, per iteration.
    digest: Vec<Vec<u32>>,
    /// Final accumulator bits.
    accs: Vec<Vec<u32>>,
}

fn host_f32(rng: &mut Rng, n: usize, budget: &Budget, shape: &[usize]) -> Tensor {
    let v: Vec<f32> = rng.vec(n, 1.0).iter().map(|&x| x as f32).collect();
    Tensor::from_f32(&v, shape, budget).unwrap()
}

fn assert_sole_owner(t: &mut Tensor) {
    let n = t.num_elements().unwrap();
    match t.device() {
        None => t
            .ensure_writable_f32(n)
            .expect("taken host gradient is shared"),
        Some(_) => {
            t.device_buffer_mut()
                .expect("taken device gradient is shared");
        }
    }
}

/// One random graph on `tape`. Returns the root, the leaves in the order
/// `[x, wa, wb, norm, head]`, and the root kind: 0 cross-entropy, 1
/// `loss + loss`, 2 a non-scalar activation, 3 the fused head.
fn record<B: Backend>(
    tape: &mut Tape<B>,
    rng: &mut Rng,
    fused: bool,
) -> Result<(Var, [Var; 5], usize), OjasError> {
    let budget = tape.backend().budget().clone();
    let rows = 1 + rng.below(4);
    let x = tape.leaf(host_f32(rng, rows * D, &budget, &[rows, D]))?;
    let wa = tape.leaf(host_f32(rng, D * D, &budget, &[D, D]))?;
    let wb = tape.leaf(host_f32(rng, D * D, &budget, &[D, D]))?;
    let norm = tape.leaf(host_f32(rng, D, &budget, &[D]))?;
    let head = tape.leaf(host_f32(rng, V * D, &budget, &[V, D]))?;
    let mut live = vec![x];
    let mut h = x;
    for _ in 0..1 + rng.below(6) {
        let other = live[rng.below(live.len())];
        h = match rng.below(6) {
            0 => tape.linear(h, wa)?,
            1 => tape.linear(h, wb)?,
            2 => tape.silu(h)?,
            3 => tape.rms_norm(h, norm, RMS_NORM_EPS)?,
            4 => tape.add(h, other)?,
            _ => {
                let flat = tape.reshape(h, &[rows * D])?;
                let back = tape.reshape(flat, &[rows, D])?;
                tape.mul(back, other)?
            }
        };
        live.push(h);
    }
    let mut targets: Vec<u32> = (0..rows).map(|_| rng.below(V) as u32).collect();
    let ignore = if rng.below(2) == 0 {
        None
    } else {
        Some(V as u32 + 1)
    };
    if let (Some(sentinel), true) = (ignore, rows > 1) {
        targets[1 + rng.below(rows - 1)] = sentinel;
    }
    let t = Tensor::from_u32(&targets, &[rows], &budget)?;
    let kinds = if fused { 4 } else { 3 };
    let kind = rng.below(kinds);
    let root = match kind {
        0 => {
            let logits = tape.linear(h, head)?;
            tape.cross_entropy(logits, t, ignore)?
        }
        1 => {
            let logits = tape.linear(h, head)?;
            let loss = tape.cross_entropy(logits, t, ignore)?;
            tape.add(loss, loss)?
        }
        2 => h,
        _ => {
            let chunk = CeChunk {
                rows: 1 + rng.below(rows),
                cols: 1 + rng.below(V),
            };
            tape.linear_cross_entropy(h, head, t, ignore, chunk)?
        }
    };
    Ok((root, [x, wa, wb, norm, head], kind))
}

fn stress<B: Backend>(backend: &B, seed: u64, fused: bool, check_bytes: bool) -> Outcome {
    let budget = backend.budget();
    let baseline = budget.live_bytes().unwrap();
    let mut rng = Rng(seed);
    let mut accs: Vec<Tensor> = PARAM_SHAPES
        .iter()
        .map(|s| {
            let zeros = Tensor::zeros(s, DType::F32, budget).unwrap();
            backend.upload(&zeros).unwrap()
        })
        .collect();
    let with_accs = budget.live_bytes().unwrap();
    let mut digest = Vec::with_capacity(ITERS);
    let mut tape = Tape::new(backend);
    let mut kinds = [0usize; 4];
    for iter in 0..ITERS {
        tape.clear();
        let (root, leaves, kind) = record(&mut tape, &mut rng, fused).unwrap();
        kinds[kind] += 1;
        let seed = SEEDS[rng.below(SEEDS.len())];
        tape.backward_seeded(root, seed)
            .unwrap_or_else(|e| panic!("iter {iter}: backward: {e}"));
        let mut row = Vec::new();
        for (i, leaf) in leaves.iter().enumerate() {
            let Some(mut g) = tape.take_grad(*leaf) else {
                row.push(u32::MAX);
                continue;
            };
            assert!(tape.take_grad(*leaf).is_none(), "iter {iter}: second take");
            assert_sole_owner(&mut g);
            let host = backend.download(&g).unwrap();
            row.extend(host.to_f32_vec().unwrap().iter().map(|v| v.to_bits()));
            if i > 0 {
                backend
                    .accumulate_grad(&mut accs[i - 1], &g)
                    .unwrap_or_else(|e| panic!("iter {iter}: accumulate: {e}"));
            }
        }
        digest.push(row);
        tape.clear();
        if check_bytes {
            assert_eq!(
                budget.live_bytes().unwrap(),
                with_accs,
                "iter {iter}: a cleared tape left bytes charged"
            );
        }
    }
    drop(tape);
    let reached = if fused { 4 } else { 3 };
    assert!(
        kinds[..reached].iter().all(|&k| k > 0),
        "a root kind was never drawn: {kinds:?}"
    );
    let accs_bits = accs
        .iter()
        .map(|a| {
            let host = backend.download(a).unwrap();
            host.to_f32_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect();
    drop(accs);
    if check_bytes {
        assert_eq!(
            budget.live_bytes().unwrap(),
            baseline,
            "the run left bytes charged"
        );
    }
    Outcome {
        digest,
        accs: accs_bits,
    }
}

fn exact(threads: Option<usize>) -> CpuBackend {
    let budget = Budget::new(1 << 26);
    let cpu = match threads {
        None => CpuBackend::new(budget),
        Some(n) => CpuBackend::with_threads(budget, n).unwrap(),
    };
    cpu.with_numerics(Numerics::Exact)
}

fn same(a: &Outcome, b: &Outcome, what: &str) {
    assert_eq!(a.digest.len(), b.digest.len());
    for (i, (x, y)) in a.digest.iter().zip(&b.digest).enumerate() {
        assert_eq!(x, y, "{what}: iteration {i} gradients differ");
    }
    assert_eq!(a.accs, b.accs, "{what}: accumulators differ");
}

#[test]
fn cpu_tape_stress_has_no_leaks_and_is_deterministic() {
    for seed in [1u64, 2, 3] {
        let first = stress(&exact(None), seed, false, true);
        let again = stress(&exact(None), seed, false, true);
        same(&first, &again, "cpu run to run");
        let default = CpuBackend::new(Budget::new(1 << 26));
        let fast = stress(&default, seed, false, true);
        let fast_again = stress(&default, seed, false, true);
        same(&fast, &fast_again, "cpu default numerics run to run");
    }
}

#[test]
fn cpu_exact_stress_bits_do_not_depend_on_threads() {
    for seed in [4u64, 5] {
        let one = stress(&exact(None), seed, false, true);
        let four = stress(&exact(Some(4)), seed, false, false);
        same(&one, &four, "1 vs 4 threads");
    }
}

#[test]
fn double_stress_with_the_fused_head_has_no_leaks_and_paths_agree() {
    for seed in [6u64, 7, 8] {
        let host = stress(&Resident::host(), seed, true, true);
        let host_again = stress(&Resident::host(), seed, true, true);
        same(&host, &host_again, "host double run to run");
        let device = stress(&Resident::new(), seed, true, true);
        same(&host, &device, "host vs device double");
        // The unfused CPU Exact tape on the same sequence, without the fused
        // root kind, equals the doubles on that sequence too.
        let cpu = stress(&exact(None), seed, false, true);
        let host_unfused = stress(&Resident::host(), seed, false, true);
        same(&cpu, &host_unfused, "cpu vs host double, unfused");
    }
}

#[test]
fn a_device_stress_keeps_gradients_on_the_device() {
    let double = Resident::new();
    let mut tape = Tape::new(&double);
    let mut rng = Rng(9);
    for _ in 0..20 {
        tape.clear();
        let (root, leaves, _) = record(&mut tape, &mut rng, true).unwrap();
        tape.backward_seeded(root, 0.5).unwrap();
        for leaf in leaves {
            if let Some(g) = tape.take_grad(leaf) {
                assert_eq!(g.device(), Some(BackendId::Wgpu));
            }
        }
    }
}
