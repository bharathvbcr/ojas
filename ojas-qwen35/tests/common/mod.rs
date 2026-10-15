//! Helpers shared by the GPU tests (`gpu_parity.rs`, `gpu_tape_2b.rs`,
//! `gpu_memory_rows.rs`): the real-2B snapshot, tessl's `.npy` references,
//! and the `ROW ` line a measurement prints for `bench/results/`.
//!
//! Each test binary compiles this module and uses part of it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use tessl::npy::read_npy;

/// The canonical tessl checkout beside ojas, through which its fixtures and
/// `target/qwen35_train_ref` are read.
pub fn tessl_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl")
}

/// The real Qwen3.5-2B-Base snapshot: `$env`, or the one snapshot of
/// `Qwen/Qwen3.5-2B-Base` in the Hugging Face cache.
pub fn snapshot_dir(env: &str) -> PathBuf {
    if let Some(d) = std::env::var_os(env) {
        return PathBuf::from(d);
    }
    let root = PathBuf::from(std::env::var_os("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots");
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

/// tessl's real-2B training reference (`tools/qwen35_ref/make_train_fixture.py
/// 2b`): `QWEN35_TRAIN_REF_DIR`, or `tessl/target/qwen35_train_ref`.
pub fn train_ref_dir() -> PathBuf {
    std::env::var_os("QWEN35_TRAIN_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| tessl_root().join("target/qwen35_train_ref"))
}

/// An `.npy` file's values as f64, whatever its float or int dtype.
pub fn npy(path: &Path) -> Vec<f64> {
    let a = read_npy(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if let Ok(s) = a.f32_slice() {
        s.iter().map(|&x| f64::from(x)).collect()
    } else if let Ok(s) = a.f64_slice() {
        s.to_vec()
    } else {
        a.i64_slice().unwrap().iter().map(|&x| x as f64).collect()
    }
}

/// The token ids of a reference directory (`ids.npy`).
pub fn ids(dir: &Path) -> Vec<u32> {
    npy(&dir.join("ids.npy"))
        .iter()
        .map(|&x| x as u32)
        .collect()
}

/// How many reference gradients (`grad.*.npy`) `dir` holds.
pub fn reference_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("grad.")
        })
        .count()
}

/// Worst `|got - want| / max|want|`.
pub fn rel(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(&g, w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max)
        / peak
}

/// One measurement's fields, printed as a JSON object on a line starting
/// `ROW `.
pub struct Row {
    fields: Vec<(&'static str, String)>,
}

impl Row {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    pub fn num(&mut self, k: &'static str, v: u64) {
        self.fields.push((k, v.to_string()));
    }

    pub fn text(&mut self, k: &'static str, v: &str) {
        let escaped: String = v
            .chars()
            .flat_map(|c| match c {
                '"' => vec!['\\', '"'],
                '\\' => vec!['\\', '\\'],
                '\n' => vec!['\\', 'n'],
                c => vec![c],
            })
            .collect();
        self.fields.push((k, format!("\"{escaped}\"")));
    }

    /// Print the row so far with `status` set to the phase reached, then
    /// drop the status again: a process stopped later (the machine's memory
    /// guard) still leaves what it held here.
    pub fn phase(&mut self, status: &str) {
        self.text("status", status);
        self.print();
        self.fields.pop();
    }

    pub fn print(&self) {
        let body: Vec<String> = self
            .fields
            .iter()
            .map(|(k, v)| format!("\"{k}\":{v}"))
            .collect();
        println!("ROW {{{}}}", body.join(","));
    }
}

/// Print the row with the refusal and fail.
pub fn refused(mut row: Row, phase: &str, err: impl std::fmt::Debug) -> ! {
    row.text("status", "refused");
    row.text("phase", phase);
    row.text("error", &format!("{err:?}"));
    row.print();
    panic!("{phase}: {err:?}");
}
