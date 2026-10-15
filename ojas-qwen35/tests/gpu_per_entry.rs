//! State reads, `save_state` and `load_state` go entry by entry through
//! tessl's host-mapped per-entry API, so none of them allocates device
//! memory. Opens a Metal runtime: `#[ignore]`, run in the agreed GPU window.
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test gpu_per_entry -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! The tiny tests use tessl's committed fixture. `gpu_real_2b_read_peaks`
//! needs the real 2B (`QWEN35_2B_SNAPSHOT`, or the one snapshot of
//! `Qwen/Qwen3.5-2B` in the Hugging Face cache) and prints the device peak of
//! each read; it never writes a state directory (the 2B's is about 22 GB).

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_core::Budget;
use ojas_qwen35::{
    AdamWHyper, GroupSpec, LrRule, Numerics, OptimizerPlan, Precision, Qwen35Step, Select,
    Sequence, Snapshot, WdRule, Which,
};

const ALL: [Which; 4] = [
    Which::Parameters,
    Which::Gradients,
    Which::ExpAvg,
    Which::ExpAvgSq,
];

fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl/tests/fixtures/qwen35_train")
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "ojas-qwen35-per-entry-{name}-{}",
        std::process::id()
    ));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    d
}

fn open_tiny() -> Qwen35Step {
    let snap = Snapshot::from_dir(&tiny_dir()).unwrap();
    Qwen35Step::open(
        &snap,
        Numerics::ExactF32,
        Precision::F32,
        Budget::new(1 << 30),
    )
    .unwrap()
}

fn backward_one(s: &mut Qwen35Step, salt: u32) {
    let ids: Vec<u32> = (0..12u32).map(|i| (i * 7 + 3 + salt) % 64).collect();
    let rows: Vec<u32> = (0..11u32).collect();
    let pending = s
        .forward(&Sequence {
            ids: &ids,
            letter_rows: &rows,
            letter_targets: &ids[1..],
            letter_scale: 1.0 / 11.0,
            span_positions: &[],
        })
        .unwrap();
    s.backward(pending, None).unwrap();
}

fn take_step(s: &mut Qwen35Step) {
    let plan = OptimizerPlan::build(
        s.parameter_table(),
        &GroupSpec {
            lr: vec![LrRule {
                label: "all".into(),
                select: Select::All,
                lr_scale: 1.0,
            }],
            weight_decay: vec![WdRule {
                label: "all".into(),
                select: Select::All,
                weight_decay: 0.01,
            }],
        },
    )
    .unwrap();
    let hyper = AdamWHyper {
        lr: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    s.adamw_step(&hyper, &plan).unwrap();
}

fn table_bits(s: &Qwen35Step, which: Which) -> Vec<(String, Vec<u32>)> {
    s.read_table(which)
        .unwrap()
        .into_iter()
        .map(|t| {
            let b = t
                .tensor
                .to_f32_vec()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            (t.name, b)
        })
        .collect()
}

/// A provider that has taken one AdamW step and holds a fresh gradient.
fn stepped() -> Qwen35Step {
    let mut s = open_tiny();
    backward_one(&mut s, 0);
    take_step(&mut s);
    backward_one(&mut s, 5);
    s
}

#[test]
#[ignore]
fn gpu_read_entries_are_the_table_rows() {
    let s = stepped();
    // A matrix tessl stores transposed, a 1-D tensor and the embedding.
    let names: Vec<String> = s
        .parameter_table()
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| {
            n.ends_with("mlp.gate_proj.weight")
                || n.ends_with("norm.weight")
                || n == "embed_tokens.weight"
        })
        .collect();
    assert!(names.len() >= 3);
    let picks: Vec<&str> = names.iter().rev().map(String::as_str).collect();
    for which in ALL {
        let whole = table_bits(&s, which);
        let some = s.read_entries(which, &picks).unwrap();
        assert_eq!(some.len(), picks.len());
        for t in some {
            let bits: Vec<u32> = t
                .tensor
                .to_f32_vec()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let want = whole.iter().find(|(n, _)| *n == t.name).unwrap();
            assert_eq!(bits, want.1, "{which:?} {}", t.name);
        }
    }
    assert!(
        table_bits(&s, Which::Gradients)
            .iter()
            .any(|(_, b)| b.iter().any(|&x| x != 0)),
        "the gradient table read back as all zero"
    );
}

#[test]
#[ignore]
fn gpu_state_round_trip_is_bit_identical() {
    let mut s = stepped();
    s.discard_gradients();
    let saved: Vec<_> = [Which::Parameters, Which::ExpAvg, Which::ExpAvgSq]
        .into_iter()
        .map(|w| (w, table_bits(&s, w)))
        .collect();
    let steps = s.step_count();
    let dir = scratch("round-trip");
    s.save_state(&dir).unwrap();

    // Move every table, then restore.
    backward_one(&mut s, 11);
    take_step(&mut s);
    assert_ne!(table_bits(&s, Which::Parameters), saved[0].1);
    s.load_state(&dir).unwrap();
    assert_eq!(s.step_count(), steps);
    for (w, want) in &saved {
        assert_eq!(&table_bits(&s, *w), want, "{w:?} after load_state");
    }

    // A fresh provider loads the same state to the same bits.
    let mut fresh = open_tiny();
    fresh.load_state(&dir).unwrap();
    for (w, want) in &saved {
        assert_eq!(&table_bits(&fresh, *w), want, "{w:?} in a fresh provider");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Reads, `save_state` and `load_state` allocate nothing on the device: the
/// pool's peak since the reset is what was already allocated. A whole-table
/// staging (the old path) raised it by the size of a table.
#[test]
#[ignore]
fn gpu_state_io_allocates_no_device_memory() {
    let mut s = stepped();
    let before = s.device_bytes().allocated;
    let check = |s: &Qwen35Step, what: &str| {
        let peak = s.device_bytes().peak;
        eprintln!("{what}: peak {peak} B, allocated before {before} B");
        assert!(
            peak <= before,
            "{what} raised the device peak to {peak} B from {before} B"
        );
    };
    s.reset_peak_device_bytes();
    for which in ALL {
        s.read_table(which).unwrap();
    }
    check(&s, "read_table x4");

    s.discard_gradients();
    let dir = scratch("alloc");
    s.reset_peak_device_bytes();
    s.save_state(&dir).unwrap();
    check(&s, "save_state");

    s.reset_peak_device_bytes();
    s.load_state(&dir).unwrap();
    check(&s, "load_state");
    std::fs::remove_dir_all(&dir).unwrap();
}

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

/// The device peak of a read of each table on the real 2B after one 128-token
/// ExactF32 backward, against the recommended working set. Three entries are
/// read (the 2 GB embedding, one MLP matrix and the final norm): the new path
/// moves just those, the old one staged the whole table first.
#[test]
#[ignore]
fn gpu_real_2b_read_peaks() {
    let snap = Snapshot::from_dir(&snapshot_dir()).unwrap();
    let mut s = Qwen35Step::open(
        &snap,
        Numerics::ExactF32,
        Precision::F32,
        Budget::new(6 << 30),
    )
    .unwrap();
    let ids: Vec<u32> = (0..128u32).map(|i| (i * 7919 + 13) % 100_000).collect();
    let rows: Vec<u32> = (0..127u32).collect();
    let pending = s
        .forward(&Sequence {
            ids: &ids,
            letter_rows: &rows,
            letter_targets: &ids[1..],
            letter_scale: 1.0 / 127.0,
            span_positions: &[],
        })
        .unwrap();
    s.backward(pending, None).unwrap();
    let recommended = tessl::GpuRuntime::new()
        .unwrap()
        .memory_info()
        .recommended_working_set;
    let picks = [
        "embed_tokens.weight",
        "layers.0.mlp.gate_proj.weight",
        "norm.weight",
    ];
    for which in ALL {
        let before = s.device_bytes().allocated;
        s.reset_peak_device_bytes();
        let got = s.read_entries(which, &picks).unwrap();
        drop(got);
        let peak = s.device_bytes().peak;
        eprintln!(
            "2B read {which:?}: allocated before {before} B, peak {peak} B (+{} B), recommended working set {recommended} B",
            peak.saturating_sub(before)
        );
        assert!(peak <= recommended);
    }
}
