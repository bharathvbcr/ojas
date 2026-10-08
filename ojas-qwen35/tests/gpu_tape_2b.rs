//! The real Qwen3.5-2B forward and backward through ojas's Tape on Metal
//! (`ojas_model::qwen35`), against this crate's provider (tessl's
//! `Qwen35Model::train_step`, which `gpu_parity.rs` holds to transformers)
//! on the same tokens. `#[ignore]` and named `gpu_*`, as in
//! `gpu_parity.rs`: run it alone, in the agreed GPU window.
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test gpu_tape_2b \
//!     -- --ignored --test-threads=1 --nocapture gpu_real_2b
//! ```
//!
//! The snapshot is `QWEN35_2B_SNAPSHOT`, or the one snapshot of
//! `Qwen/Qwen3.5-2B` in the Hugging Face cache. `QWEN35_TAPE_TOKENS` sets the
//! sequence length (default 128).
//!
//! The two never hold the model at once: the provider (about 40 GB with a
//! staged gradient table) steps, its loss and a subset of its gradients are
//! copied to the host, and it is dropped before the tape loads the weights.
//! The subset is tessl's for its real-2B reference: every 1-D parameter and
//! every matrix of layer 0 (gated delta net) and layer 3 (attention). The
//! bounds are tessl's for the real 2B against transformers: loss within 1e-4
//! relative, each gradient within 1e-2 of its own peak. Each figure is
//! printed.

#![cfg(target_os = "macos")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, Tensor};
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use ojas_model::qwen35::{bind, forward_loss, fuse_grads, load_hf, Qwen35Tables};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_qwen35::{Numerics, Qwen35Step, Sequence, Snapshot, Which};

const LOSS_BOUND: f64 = 1e-4;
const GRAD_BOUND: f64 = 1e-2;
const GIB: u64 = 1 << 30;

fn snapshot_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("QWEN35_2B_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let root = PathBuf::from(std::env::var_os("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots");
    let dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    assert_eq!(
        dirs.len(),
        1,
        "expected one snapshot under {}",
        root.display()
    );
    dirs[0].clone()
}

fn tokens() -> usize {
    std::env::var("QWEN35_TAPE_TOKENS")
        .map(|s| s.parse().expect("QWEN35_TAPE_TOKENS"))
        .unwrap_or(128)
}

/// `n` deterministic ids below 150000 (ordinary text tokens), skipping the
/// vision special tokens the config declares.
fn ids(n: usize, reserved: &[u32]) -> Vec<u32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let id = (s % 150_000) as u32;
        if !reserved.contains(&id) {
            out.push(id);
        }
    }
    out
}

/// tessl's real-2B subset: every 1-D tensor, and layers 0 and 3 whole.
fn compared(name: &str, shape: &[usize]) -> bool {
    shape.len() == 1 || name.starts_with("layers.0.") || name.starts_with("layers.3.")
}

fn gib(b: u64) -> f64 {
    b as f64 / GIB as f64
}

#[test]
#[ignore]
fn gpu_real_2b_tape_forward_backward_matches_the_provider() {
    let snap = Snapshot::from_dir(&snapshot_dir()).unwrap();
    let cfg = snap.config.clone();
    let reserved: Vec<u32> = cfg.reserved_token_ids.iter().map(|(_, id)| *id).collect();
    let ids = ids(tokens(), &reserved);
    let n = ids.len() - 1;

    // The provider: one step, the loss and the subset to the host, dropped.
    let (want_loss, want) = {
        let mut step = Qwen35Step::open(&snap, Numerics::ExactF32, Budget::new(16 * GIB)).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        let pending = step
            .forward(&Sequence {
                ids: &ids,
                letter_rows: &rows,
                letter_targets: &ids[1..],
                letter_scale: 1.0 / n as f32,
                span_positions: &[],
            })
            .unwrap();
        let loss = pending.letter_ce_sum() / n as f64;
        step.backward(pending, None).unwrap();
        let names: Vec<String> = step
            .parameter_table()
            .iter()
            .filter(|t| compared(&t.name, &t.shape))
            .map(|t| t.name.clone())
            .collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let grads: BTreeMap<String, Vec<f32>> = step
            .read_entries(Which::Gradients, &refs)
            .unwrap()
            .into_iter()
            .map(|g| (g.name, g.tensor.to_f32_vec().unwrap()))
            .collect();
        (loss, grads)
    };
    eprintln!(
        "provider: loss {want_loss:.8} over {n} tokens, {} gradients kept",
        want.len()
    );

    // The tape, on Metal, each layer a checkpointed segment.
    let spec = cfg.tape_spec();
    let m = MetalBackend::new(Budget::new(40 * GIB)).unwrap();
    let mem = |what: &str| {
        let r = m.memory().unwrap();
        eprintln!(
            "{what}: {:.2} GiB allocated of a {:.2} GiB working set",
            gib(r.allocated),
            gib(r.recommended_working_set)
        );
        r.allocated
    };
    mem("tape: device open");
    let host_budget = Budget::new(16 * GIB);
    let device = {
        let file = SafeTensors::open(&snap.weights_path).unwrap();
        let host = load_hf(&spec, &file, &cfg.tower_prefix, &host_budget).unwrap();
        host.try_map(&mut |t| m.upload(t)).unwrap()
    };
    m.sync().unwrap();
    mem("tape: weights resident");
    let tables = Qwen35Tables::new(&spec, n, &host_budget)
        .unwrap()
        .upload(&m)
        .unwrap();
    let ids_t = Tensor::from_u32(&ids[..n], &[1, n], &host_budget).unwrap();
    let targets = Tensor::from_u32(&ids[1..], &[1, n], &host_budget).unwrap();
    let mut tape = Tape::new(m.clone());
    let vars = bind(&mut tape, &device).unwrap();
    drop(device);
    let loss = forward_loss(
        &mut tape,
        &spec,
        &vars,
        &ids_t,
        &targets,
        &tables,
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Blocks,
    )
    .unwrap();
    m.sync().unwrap();
    let after_forward = mem("tape: after forward");
    tape.backward(loss).unwrap();
    m.sync().unwrap();
    let after_backward = mem("tape: after backward");
    let got_loss = f64::from(
        tape.value(loss)
            .unwrap()
            .to_host(&host_budget)
            .unwrap()
            .to_f32_vec()
            .unwrap()[0],
    );
    let grads = vars
        .try_map(&mut |v| tape.take_grad(*v).ok_or("no gradient"))
        .unwrap();
    drop(tape);
    eprintln!(
        "tape peak sampled: {:.2} GiB",
        gib(after_forward.max(after_backward))
    );

    let rel = (got_loss - want_loss).abs() / want_loss.abs();
    eprintln!("tape loss {got_loss:.8} vs provider {want_loss:.8} (rel {rel:.2e})");
    assert!(got_loss.is_finite());
    assert!(rel <= LOSS_BOUND, "loss {got_loss} vs {want_loss}");

    let fused = fuse_grads(&spec, &grads, "", &host_budget).unwrap();
    let mut seen = 0;
    let mut worst = (0.0f64, String::new());
    for g in fused.iter().filter(|g| compared(&g.name, &g.shape)) {
        let w = want
            .get(&g.name)
            .unwrap_or_else(|| panic!("{}: not read from the provider", g.name));
        assert_eq!(w.len(), g.values.len(), "{}", g.name);
        let peak = w.iter().fold(0.0f64, |p, x| p.max(f64::from(x.abs())));
        assert!(
            peak > 0.0,
            "{}: an all-zero reference checks nothing",
            g.name
        );
        let err = g
            .values
            .iter()
            .zip(w)
            .map(|(a, b)| {
                assert!(a.is_finite(), "{}: non-finite", g.name);
                f64::from((a - b).abs())
            })
            .fold(0.0, f64::max);
        let r = err / peak;
        eprintln!("{}: {r:.3e} of peak {peak:.3e}", g.name);
        assert!(r <= GRAD_BOUND, "{}: {r:.3e} of its peak", g.name);
        if r > worst.0 {
            worst = (r, g.name.clone());
        }
        seen += 1;
    }
    assert_eq!(seen, want.len(), "every kept provider gradient is compared");
    eprintln!("worst gradient {:.3e} ({}) over {seen}", worst.0, worst.1);
}
