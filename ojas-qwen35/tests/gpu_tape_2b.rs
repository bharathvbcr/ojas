//! The real Qwen3.5-2B-Base forward and backward through ojas's Tape on
//! Metal (`ojas_model::qwen35`), against transformers' own float32 step on
//! the same tokens: tessl's real-2B training reference
//! (`tools/qwen35_ref/make_train_fixture.py 2b --tokens 128`, a CPU process).
//! `#[ignore]` and named `gpu_*`, as in `gpu_parity.rs`: run it alone, under
//! the machine's heavy-job lock.
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test gpu_tape_2b \
//!     -- --ignored --test-threads=1 --nocapture gpu_real_2b
//! ```
//!
//! The reference is `QWEN35_TRAIN_REF_DIR`, or `tessl/target/qwen35_train_ref`;
//! the snapshot is `QWEN35_2B_SNAPSHOT`, or the one snapshot of
//! `Qwen/Qwen3.5-2B-Base` in the Hugging Face cache, and must hold the
//! weights file the reference was made from (`meta.txt`).
//!
//! The provider is not in this process (gp-qwen35-2b-metal-tape-run, Fable,
//! 2026-10-15): it holds 30.13 GB at open and stages a whole f32 table to
//! read any gradient back, over this Mac's 32 GiB memguard. Its leg on the
//! same reference is tessl's `real_2b_step_matches_transformers`, run in its
//! own process. The bounds are that test's for the real 2B against
//! transformers: loss within 1e-4 relative, each gradient within 1e-2 of its
//! own peak, every gradient the reference kept (every 1-D parameter and conv
//! weight, layers 0 and 3 whole, the embedding's rows for the prompt's ids
//! and 64 others). The device budget is below the guard, so a step that does
//! not fit is a typed refusal here, not a SIGTERM.
//!
//! One `ROW ` line (JSON) for `bench/results/`: the reference's source, the
//! resident and exact peak device bytes, the loss and the worst gradient. A
//! refusal at any phase is a row too, naming the phase, and the test then
//! fails; the row is printed before the bounds are asserted.

#![cfg(target_os = "macos")]

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, DType, Tensor};
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use ojas_model::qwen35::{bind, forward_loss, fuse_grads, load_hf, Qwen35Tables};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_qwen35::Snapshot;

use common::{ids, npy, reference_files, refused, rel, snapshot_dir, train_ref_dir, Row};

const LOSS_BOUND: f64 = 1e-4;
const GRAD_BOUND: f64 = 1e-2;
const GIB: u64 = 1 << 30;
/// Below the 32 GiB memguard (the Tape's full-gradient step at T = 2,048 f32
/// peaked under 23 GB, bench/results/2026-10-11-bf16-frozen-base).
const DEVICE_BUDGET: u64 = 28 * GIB;

/// The weights file under `dir`: its final path (every link resolved, to
/// tell two snapshots of one file apart from two files), and the name of
/// the Hugging Face cache blob the snapshot links to, which for an LFS file
/// is its sha256 (`None` when the file is not such a link).
fn weights_file(dir: &Path) -> (PathBuf, Option<String>) {
    let path = Snapshot::from_dir(dir).unwrap().weights_path;
    let real = std::fs::canonicalize(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let blob = std::fs::read_link(&path)
        .ok()
        .and_then(|l| l.file_name().map(|n| n.to_string_lossy().into_owned()));
    (real, blob)
}

/// The snapshot directory the reference was made from (`meta.txt`'s
/// `model` line).
fn reference_snapshot(ref_dir: &Path) -> PathBuf {
    let meta = std::fs::read_to_string(ref_dir.join("meta.txt"))
        .unwrap_or_else(|e| panic!("{}: {e}", ref_dir.join("meta.txt").display()));
    let line = meta
        .lines()
        .find_map(|l| l.strip_prefix("model "))
        .expect("meta.txt names its model");
    PathBuf::from(line.trim())
}

#[test]
#[ignore = "needs the real 2B snapshot, the transformers reference (QWEN35_TRAIN_REF_DIR) and a Metal runtime; run alone under the heavy-job lock"]
fn gpu_real_2b_tape_forward_backward_matches_transformers() {
    let ref_dir = train_ref_dir();
    let snap_dir = snapshot_dir("QWEN35_2B_SNAPSHOT");
    let (file, sha256) = weights_file(&snap_dir);
    let (ref_file, _) = weights_file(&reference_snapshot(&ref_dir));
    assert_eq!(
        file, ref_file,
        "the snapshot's weights are not the file the reference was made from"
    );
    let snap = Snapshot::from_dir(&snap_dir).unwrap();
    let cfg = snap.config.clone();
    let spec = cfg.tape_spec();
    let ids = ids(&ref_dir);
    let n = ids.len() - 1;
    let want_loss = npy(&ref_dir.join("loss.npy"))[0];

    let mut r = Row::new();
    r.text("arm", "tape-f32-vs-transformers");
    r.text("snapshot", &snap_dir.display().to_string());
    r.text(
        "weights_sha256",
        sha256.as_deref().unwrap_or("not a Hugging Face cache link"),
    );
    r.text("reference", &ref_dir.display().to_string());
    r.num("tokens", ids.len() as u64);
    r.num("device_budget_bytes", DEVICE_BUDGET);

    let m = match MetalBackend::new(Budget::new(DEVICE_BUDGET)) {
        Ok(m) => m,
        Err(e) => refused(r, "MetalBackend::new", e),
    };
    let host_budget = Budget::new(24 * GIB);
    let device = {
        let file = SafeTensors::open(&snap.weights_path).unwrap();
        let host = match load_hf(
            &spec,
            &file,
            &cfg.tower_prefix,
            DType::F32,
            DType::F32,
            &host_budget,
        ) {
            Ok(h) => h,
            Err(e) => refused(r, "load_hf", e),
        };
        match host.try_map(&mut |t| m.upload(t)) {
            Ok(d) => d,
            Err(e) => refused(r, "upload", e),
        }
    };
    if let Err(e) = m.sync() {
        refused(r, "sync weights", e);
    }
    let resident = m.memory_and_reset_peak().unwrap();
    r.num("resident_bytes", resident.allocated);
    r.num("working_set_bytes", resident.recommended_working_set);
    r.phase("weights resident");
    let tables = Qwen35Tables::new(&spec, n, &host_budget)
        .unwrap()
        .upload(&m)
        .unwrap();
    let ids_t = Tensor::from_u32(&ids[..n], &[1, n], &host_budget).unwrap();
    let targets = Tensor::from_u32(&ids[1..], &[1, n], &host_budget).unwrap();
    let mut tape = Tape::new(m.clone());
    let vars = bind(&mut tape, &device).unwrap();
    drop(device);
    let t0 = Instant::now();
    let loss = match forward_loss(
        &mut tape,
        &spec,
        &vars,
        &ids_t,
        &targets,
        &tables,
        None,
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Blocks,
    ) {
        Ok(l) => l,
        Err(e) => refused(r, "forward", e),
    };
    if let Err(e) = tape.backward(loss) {
        refused(r, "backward", e);
    }
    if let Err(e) = m.sync() {
        refused(r, "sync step", e);
    }
    r.num("step_ms", t0.elapsed().as_millis() as u64);
    r.num("step_peak_bytes", m.memory().unwrap().peak_allocated);
    r.phase("step");
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

    // Every reference gradient, under transformers' names; the embedding by
    // the rows the reference kept.
    let fused = fuse_grads(&spec, &grads, "model.", &host_budget).unwrap();
    let mut results: Vec<(String, f64)> = Vec::new();
    for g in &fused {
        let got: Vec<f32> = if g.name == "model.embed_tokens.weight" {
            let h = g.shape[1];
            npy(&ref_dir.join("embed_rows.npy"))
                .iter()
                .flat_map(|&row| g.values[row as usize * h..][..h].to_vec())
                .collect()
        } else {
            g.values.clone()
        };
        let file = if g.name == "model.embed_tokens.weight" {
            ref_dir.join("grad.model.embed_tokens.weight.rows.npy")
        } else {
            ref_dir.join(format!("grad.{}.npy", g.name))
        };
        if !file.exists() {
            continue;
        }
        assert!(got.iter().all(|x| x.is_finite()), "{}: non-finite", g.name);
        let x = rel(&got, &npy(&file));
        eprintln!("{}: {x:.3e} of its peak", g.name);
        results.push((g.name.clone(), x));
    }
    let loss_rel = (got_loss - want_loss).abs() / want_loss.abs();
    let worst =
        results.iter().cloned().fold(
            (String::new(), 0.0f64),
            |w, x| if x.1 > w.1 { x } else { w },
        );
    eprintln!("tape loss {got_loss:.8} vs transformers {want_loss:.8} (rel {loss_rel:.2e})");
    eprintln!(
        "worst gradient {:.3e} ({}) over {}",
        worst.1,
        worst.0,
        results.len()
    );
    r.text("loss", &format!("{got_loss:.8}"));
    r.text("reference_loss", &format!("{want_loss:.8}"));
    r.text("loss_rel", &format!("{loss_rel:.3e}"));
    r.num("gradients_compared", results.len() as u64);
    r.text("worst_grad", &worst.0);
    r.text("worst_grad_rel", &format!("{:.3e}", worst.1));
    let within = got_loss.is_finite()
        && loss_rel <= LOSS_BOUND
        && results.iter().all(|(_, x)| *x <= GRAD_BOUND);
    r.text("status", if within { "ok" } else { "out of bounds" });
    r.print();

    assert_eq!(
        results.len(),
        reference_files(&ref_dir),
        "every reference gradient is compared"
    );
    assert!(
        loss_rel <= LOSS_BOUND,
        "loss {got_loss} vs transformers {want_loss}"
    );
    for (name, x) in &results {
        assert!(*x <= GRAD_BOUND, "{name}: {x:.3e} of its peak");
    }
}
