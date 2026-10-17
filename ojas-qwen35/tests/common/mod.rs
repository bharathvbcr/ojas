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
#[derive(Clone)]
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

    /// A real number, printed with six decimals.
    pub fn real(&mut self, k: &'static str, v: f64) {
        self.fields.push((k, format!("{v:.6}")));
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

/// This process's `phys_footprint` bytes: the figure memguard and
/// `tools/mac_heavy.sh` act on (Metal allocations are in it, `ps` RSS does
/// not show them). `now` is the current value, `lifetime_max` the process's
/// high-water mark since it started (`ri_lifetime_max_phys_footprint`).
#[derive(Clone, Copy, Debug)]
pub struct Footprint {
    pub now: u64,
    pub lifetime_max: u64,
}

/// `rusage_info_v4` from `<sys/resource.h>`: a 16-byte uuid, then 35 `u64`
/// fields. The two read here sit at 0-based positions 7 and 28 of those 35,
/// the order `Lappi-decision/tools/mac_footprint.py` lists.
#[repr(C)]
struct RusageInfoV4 {
    uuid: [u8; 16],
    fields: [u64; 35],
}

const RUSAGE_INFO_V4: i32 = 4;
const PHYS_FOOTPRINT: usize = 7;
const LIFETIME_MAX_PHYS_FOOTPRINT: usize = 28;

extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
}

/// This process's [`Footprint`], or the `proc_pid_rusage` status that
/// refused it: a footprint that could not be read is never reported as 0.
pub fn footprint() -> Result<Footprint, i32> {
    let mut info = RusageInfoV4 {
        uuid: [0; 16],
        fields: [0; 35],
    };
    // SAFETY: `info` is a live, writable `rusage_info_v4`-sized buffer, and
    // flavor 4 is the layout `RusageInfoV4` mirrors.
    let status = unsafe {
        proc_pid_rusage(
            std::process::id() as i32,
            RUSAGE_INFO_V4,
            std::ptr::addr_of_mut!(info),
        )
    };
    if status != 0 {
        return Err(status);
    }
    Ok(Footprint {
        now: info.fields[PHYS_FOOTPRINT],
        lifetime_max: info.fields[LIFETIME_MAX_PHYS_FOOTPRINT],
    })
}

/// [`footprint`], or a refusal row from `row` if it cannot be read.
pub fn footprint_or_refuse(row: &Row) -> Footprint {
    footprint().unwrap_or_else(|code| {
        refused(
            row.clone(),
            "footprint",
            format!("proc_pid_rusage status {code}"),
        )
    })
}

/// The one `.safetensors` file of a snapshot whose `weights_path` is an
/// index (`model.safetensors.index.json`): `SafeTensors::open` reads a
/// single file, not an index. A snapshot with another number of shards is
/// refused by name, not read partially.
pub fn single_weights_file(weights_path: &Path) -> PathBuf {
    if weights_path.extension().is_none_or(|e| e != "json") {
        return weights_path.to_path_buf();
    }
    let dir = weights_path.parent().expect("index has a directory");
    let shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    assert_eq!(
        shards.len(),
        1,
        "{} names {} shard files; this arm reads a single-file snapshot",
        weights_path.display(),
        shards.len()
    );
    shards[0].clone()
}
