//! `Tape<WgpuBackend>` against `Tape<CpuBackend>` on the same graph. A missing
//! adapter fails unless `OJAS_ALLOW_NO_GPU=1`. Counts readbacks on the wgpu
//! backend's own budget (`Budget::device_readbacks`), so parallel tests cannot
//! disturb it. The CPU reference runs `Numerics::Exact`.

use std::sync::{Mutex, MutexGuard};

use ojas_autograd::{Tape, Var};
use ojas_core::{Backend, BackendId, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_wgpu::WgpuBackend;

const TOL: f64 = 1e-4;
const VOCAB: usize = 11;
const DIM: usize = 8;
const ROWS: usize = 6;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn data(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed * 97)
                % 1000;
            v as f32 / 500.0 - 1.0
        })
        .collect()
}

struct Graph {
    params: [Var; 3],
    loss: Var,
    root: Var,
}

/// Embedding, linear, silu residual, two reshapes, output linear,
/// cross-entropy. With `doubled` the root is `loss + loss`.
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
    let ids: Vec<u32> = (0..ROWS).map(|i| ((i * 5 + 1) % VOCAB) as u32).collect();
    let tgt: Vec<u32> = (0..ROWS).map(|i| ((i * 3) % VOCAB) as u32).collect();
    let e = tape.embedding(table, Tensor::from_u32(&ids, &[ROWS], &budget)?)?;
    let h = tape.linear(e, w1)?;
    let s = tape.silu(h)?;
    let r = tape.add(h, s)?;
    let r3 = tape.reshape(r, &[2, ROWS / 2, DIM])?;
    let r2 = tape.reshape(r3, &[ROWS, DIM])?;
    let logits = tape.linear(r2, w2)?;
    let loss = tape.cross_entropy(logits, Tensor::from_u32(&tgt, &[ROWS], &budget)?, None)?;
    let root = if doubled { tape.add(loss, loss)? } else { loss };
    Ok(Graph {
        params: [table, w1, w2],
        loss,
        root,
    })
}

fn close(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(1.0f64, |m, v| m.max(f64::from(v.abs())));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(*g) - f64::from(*w)).abs();
        assert!(err <= TOL * scale, "{name}[{i}]: wgpu {g} cpu {w}");
    }
}

fn run(gpu: WgpuBackend, doubled: bool) {
    let mut cpu_tape =
        Tape::new(CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact));
    let cg = record(&mut cpu_tape, doubled).unwrap();
    cpu_tape.backward(cg.root).unwrap();

    let mut tape = Tape::new(gpu);
    let before = tape.backend().budget().device_readbacks();
    let g = record(&mut tape, doubled).expect("forward on wgpu");
    tape.backward(g.root).expect("backward on wgpu");
    assert_eq!(
        tape.backend().budget().device_readbacks(),
        before,
        "the tape read a tensor back"
    );

    let loss = tape
        .backend()
        .download(tape.value(g.loss).unwrap())
        .unwrap();
    assert_eq!(
        tape.backend().budget().device_readbacks().0,
        before.0 + 1,
        "the loss is the one readback"
    );
    let got = loss.to_f32_vec().unwrap()[0];
    let want = cpu_tape.value(cg.loss).unwrap().to_f32_vec().unwrap()[0];
    let rel =
        (f64::from(got) - f64::from(want)).abs() / f64::from(want.abs()).max(f64::MIN_POSITIVE);
    assert!(rel <= TOL, "loss: wgpu {got} cpu {want} rel {rel:.3e}");

    for (i, (var, cvar)) in g.params.iter().zip(&cg.params).enumerate() {
        let grad = tape.grad(*var).expect("leaf gradient");
        assert_eq!(
            grad.device(),
            Some(BackendId::Wgpu),
            "param {i} grad left the device"
        );
        let got = tape.backend().download(grad).unwrap().to_f32_vec().unwrap();
        let want = cpu_tape.grad(*cvar).unwrap().to_f32_vec().unwrap();
        close(&format!("param {i} grad"), &got, &want);
    }
}

/// Opt-in that lets a machine with no wgpu adapter skip these tests. Unset,
/// a missing adapter fails the test, so it is never counted as a pass.
const ALLOW_NO_GPU: &str = "OJAS_ALLOW_NO_GPU";

/// What to do when the adapter does not open: `Ok` to skip only when
/// `allow` is exactly `"1"`, otherwise the failure text.
fn no_adapter(allow: Option<&str>, err: &dyn std::fmt::Display) -> Result<(), String> {
    if allow == Some("1") {
        Ok(())
    } else {
        Err(format!(
            "WgpuBackend::open failed: {err}. This test needs a wgpu adapter; set \
             {ALLOW_NO_GPU}=1 to skip it explicitly on a machine without one"
        ))
    }
}

fn open() -> Option<WgpuBackend> {
    match WgpuBackend::open(Budget::new(1 << 28)) {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            let allow = std::env::var(ALLOW_NO_GPU).ok();
            match no_adapter(allow.as_deref(), &err) {
                Ok(()) => {
                    eprintln!("SKIP ({ALLOW_NO_GPU}=1): no wgpu adapter: {err}");
                    None
                }
                Err(text) => panic!("{text}"),
            }
        }
    }
}

#[test]
fn a_missing_adapter_fails_unless_explicitly_allowed() {
    let err = "no adapter";
    for refused in [None, Some(""), Some("0"), Some("true"), Some("yes")] {
        let text = no_adapter(refused, &err).unwrap_err();
        assert!(text.contains(ALLOW_NO_GPU), "{text}");
    }
    assert_eq!(no_adapter(Some("1"), &err), Ok(()));
}

#[test]
fn wgpu_tape_matches_cpu_and_reads_back_only_the_loss() {
    let _serial = serial();
    if let Some(gpu) = open() {
        run(gpu, false);
    }
}

#[test]
fn wgpu_tape_scales_a_non_root_cross_entropy_on_the_device() {
    let _serial = serial();
    if let Some(gpu) = open() {
        run(gpu, true);
    }
}
