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

mod support;

use std::path::PathBuf;

use ojas_core::Budget;
use ojas_qwen35::{Numerics, Precision, Qwen35Step, Sequence, Snapshot, Which};
use support::{backward_one, open_tiny, scratch, stepped, table_bits, take_step};

const ALL: [Which; 4] = [
    Which::Parameters,
    Which::Gradients,
    Which::ExpAvg,
    Which::ExpAvgSq,
];

#[test]
#[ignore = "opens a Metal runtime; run in the GPU window with --release -- --ignored --test-threads=1"]
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
#[ignore = "opens a Metal runtime; run in the GPU window with --release -- --ignored --test-threads=1"]
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
#[ignore = "opens a Metal runtime; run in the GPU window with --release -- --ignored --test-threads=1"]
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
/// backward, against the recommended working set and against the bytes read.
///
/// Opened as `Precision::Bf16` + `Bf16Operands`, not f32. `Qwen35Step::open`
/// allocates the whole 16 B/param state (weights, gradient bank, both AdamW
/// moments: 30.1 GB on the 2B) at either precision, and there is no open
/// without the optimizer. The f32 model also packs a 2 GB `[hidden, vocab]`
/// head copy at the first forward, which put the process at 34.44 GB peak
/// footprint (`/usr/bin/time -l`) after the backward alone, over memguard's
/// 32 GiB (34.36 GB), so the run was SIGTERMed. The bf16 tower has no head
/// copy. The path under test is the same: `read_entries` goes through
/// `read_parameter_host`, `read_gradient_host` and `read_adamw_moment_host`
/// per entry at both precisions.
///
/// The picks are an MLP matrix, a second MLP matrix and the final norm, not
/// the 2 GB embedding: reading it costs 4 GB of host memory (the entry, then
/// its tensor), which on top of the 30.1 GB resident state exceeds 32 GiB at
/// either precision. The read path does not depend on entry size, and the
/// whole-table staging of the old path (several GB per table) is far above
/// what these picks weigh.
///
/// Each read's device peak must stay within the picked entries' f32 bytes of
/// what was allocated before it (the new path measures about zero; the slack
/// admits one staged copy of the picks, not of a table).
#[test]
#[ignore = "needs the real 2B snapshot (QWEN35_2B_SNAPSHOT or the Hugging Face cache) and a Metal runtime; run alone under the heavy-job lock"]
fn gpu_real_2b_read_peaks() {
    let snap = Snapshot::from_dir(&snapshot_dir()).unwrap();
    let mut s = Qwen35Step::open(
        &snap,
        Numerics::Bf16Operands,
        Precision::Bf16,
        Budget::new(6 << 30),
    )
    .unwrap();
    eprintln!("2B opened: {:?}", s.device_bytes());
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
    eprintln!("2B backward done: {:?}", s.device_bytes());
    let recommended = tessl::GpuRuntime::new()
        .unwrap()
        .memory_info()
        .recommended_working_set;
    let picks = [
        "layers.0.mlp.gate_proj.weight",
        "layers.3.mlp.down_proj.weight",
        "norm.weight",
    ];
    let table = s.parameter_table();
    let numel = |name: &str| -> u64 {
        table
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is not in the parameter table"))
            .shape
            .iter()
            .map(|&d| d as u64)
            .product()
    };
    let picked_bytes: u64 = picks.iter().map(|n| 4 * numel(n)).sum();
    let table_bytes: u64 = table
        .iter()
        .map(|t| 4 * t.shape.iter().map(|&d| d as u64).product::<u64>())
        .sum();
    assert!(
        picked_bytes * 8 < table_bytes,
        "the picks ({picked_bytes} B) must be far below a table ({table_bytes} B)"
    );
    for which in ALL {
        let before = s.device_bytes().allocated;
        s.reset_peak_device_bytes();
        let got = s.read_entries(which, &picks).unwrap();
        drop(got);
        let peak = s.device_bytes().peak;
        let rise = peak.saturating_sub(before);
        eprintln!(
            "2B read {which:?}: allocated before {before} B, peak {peak} B (+{rise} B), picked {picked_bytes} B, table {table_bytes} B, recommended working set {recommended} B"
        );
        assert!(peak <= recommended);
        assert!(
            rise <= picked_bytes,
            "{which:?}: the read raised the device peak by {rise} B for {picked_bytes} B of entries (a {table_bytes} B table)"
        );
    }
}
