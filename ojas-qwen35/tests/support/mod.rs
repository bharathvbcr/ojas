//! Helpers shared by the tiny-fixture GPU tests.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use ojas_core::Budget;
use ojas_qwen35::{
    AdamWHyper, GroupSpec, LrRule, Numerics, OptimizerPlan, Precision, Qwen35Step, Select,
    Sequence, Snapshot, WdRule, Which,
};

pub fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl/tests/fixtures/qwen35_train")
}

pub fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "ojas-qwen35-per-entry-{name}-{}",
        std::process::id()
    ));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    d
}

pub fn open_tiny() -> Qwen35Step {
    let snap = Snapshot::from_dir(&tiny_dir()).unwrap();
    Qwen35Step::open(
        &snap,
        Numerics::ExactF32,
        Precision::F32,
        Budget::new(1 << 30),
    )
    .unwrap()
}

pub fn backward_one(s: &mut Qwen35Step, salt: u32) {
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

pub fn take_step(s: &mut Qwen35Step) {
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

pub fn table_bits(s: &Qwen35Step, which: Which) -> Vec<(String, Vec<u32>)> {
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
pub fn stepped() -> Qwen35Step {
    let mut s = open_tiny();
    backward_one(&mut s, 0);
    take_step(&mut s);
    backward_one(&mut s, 5);
    s
}

pub fn cut_short(dir: &Path, prefix: &str) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with(prefix))
        .collect();
    files.sort();
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(files.last().unwrap())
        .unwrap();
    f.set_len(f.metadata().unwrap().len() - 8).unwrap();
}
