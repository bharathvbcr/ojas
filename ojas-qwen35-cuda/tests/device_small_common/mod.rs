//! What lane L-cuda-small's kernel tests (`tests/device_k{3,4,6,7,9,10}_*.rs`)
//! share: the embedded goldens, the pre-registered bounds, and the checks.
//!
//! **Goldens travel in the binary.** Each test file embeds the float64 torch
//! goldens it reads with [`golden!`] (`include_bytes!`), and
//! [`Goldens::verify`] checks every embedded file's sha256 against the
//! embedded `manifest.json` (L-cuda-oracle's pins), so a device test binary
//! copied to the box needs no checkout and cannot run on drifted bytes.
//!
//! **The comparisons.** Kernels take f32, the goldens are f64. Every test
//! rounds the golden inputs to f32 once and evaluates L-cuda-oracle's float64
//! reference (`tests/reference/`) on exactly those rounded values; a kernel
//! (on the device) or its host mirror is held to that reference within the
//! tessl bound named below. The device is also held **bit for bit** to the
//! mirror where the mirror is bitwise, and to a repeat run.
//!
//! **Bounds, written before any run**, each tessl's own bound for the same
//! kernel on Metal against float64:
//!
//! | output | bound | source |
//! | --- | --- | --- |
//! | `rms_norm`, gated norm, conv forward | `abs 1e-6 + rel 1e-5` | `tessl/tests/qwen35_kernels.rs:1071,1118,1174` |
//! | output gate forward | `abs 1e-7 + rel 1e-5` | `tessl/tests/qwen35_kernels.rs:1617` |
//! | q/k norm + RoPE forward | `abs 2e-5` | `tessl/tests/qwen35_kernels.rs:1556,1562` |
//! | every backward, the K3 gates forward | `1e-4 * max|ref|` | `tessl/tests/qwen35_bwd.rs:5,43-66` |
//! | K9 gather | bit-exact | the f32 table's bits |
//! | K10 per-row loss and loss | `1e-5 + 1e-5|ref|` | `tessl/tests/cross_entropy.rs:12,185` |
//! | K10 gradients, f32 / bf16 operands | `1e-4` / `2^-7` of `max|ref|` | `tessl/tests/cross_entropy.rs:12-14,252-258` |
#![allow(dead_code, unused_imports)]

use ojas_qwen35_cuda::check::{Check, Status};

use crate::reference::{goldens, npy, sha256};

/// One embedded golden: its name (the file stem) and its bytes.
pub type Embedded = (&'static str, &'static [u8]);

/// Embeds `tests/fixtures/goldens/<name>.npy` as an [`Embedded`].
macro_rules! golden {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/goldens/",
                $name,
                ".npy"
            )) as &'static [u8],
        )
    };
}
pub(crate) use golden;

/// L-cuda-oracle's manifest, embedded with the goldens it pins.
pub const MANIFEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/goldens/manifest.json"
));

/// The norms' `rms_norm_eps` (the goldens' and the 2B's).
pub const EPS: f32 = 1e-6;

pub use ojas_qwen35_cuda::small_common::{
    BWD_PEAK_REL, BWD_PEAK_SOURCE, CE_BF16_GRAD_PEAK_REL, CE_EXACT_GRAD_PEAK_REL, CE_LOSS_TOL,
    CE_SOURCE, GATE_FWD_SOURCE, GATE_FWD_TOL, NORM_FWD_SOURCE, NORM_FWD_TOL, QK_FWD_ABS,
    QK_FWD_SOURCE,
};

/// The goldens one test file embeds.
pub struct Goldens(pub &'static [Embedded]);

impl Goldens {
    /// Every embedded file hashes to the manifest's pin, and its name is in
    /// the manifest. Returns how many were checked.
    pub fn verify(&self) -> usize {
        let pins = goldens::manifest_files(MANIFEST);
        for (name, bytes) in self.0 {
            let file = format!("{name}.npy");
            let want = pins
                .iter()
                .find(|(f, _)| *f == file)
                .unwrap_or_else(|| panic!("{file} is not pinned by manifest.json"));
            assert_eq!(
                sha256::hex(bytes),
                want.1,
                "{file}: embedded bytes differ from the manifest's sha256"
            );
        }
        assert!(!self.0.is_empty(), "no goldens embedded: a vacuous suite");
        self.0.len()
    }

    fn npy(&self, name: &str) -> npy::Npy {
        let bytes = self
            .0
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("golden {name} is not embedded in this test"))
            .1;
        npy::parse(bytes).unwrap_or_else(|e| panic!("golden {name}: {e}"))
    }

    /// A `<f8` golden.
    pub fn f64s(&self, name: &str) -> Vec<f64> {
        self.shaped(name).1
    }

    /// `(shape, data)` of a `<f8` golden.
    pub fn shaped(&self, name: &str) -> (Vec<usize>, Vec<f64>) {
        let a = self.npy(name);
        let d = a
            .f64s()
            .unwrap_or_else(|e| panic!("golden {name}: {e}"))
            .to_vec();
        (a.shape, d)
    }

    /// An `<i8` golden of non-negative indices.
    pub fn indices(&self, name: &str) -> Vec<usize> {
        self.npy(name)
            .i64s()
            .unwrap_or_else(|e| panic!("golden {name}: {e}"))
            .iter()
            .map(|&x| usize::try_from(x).unwrap_or_else(|_| panic!("golden {name}: index {x}")))
            .collect()
    }
}

/// `x` rounded to f32 (nearest-even), the kernels' inputs.
pub fn f32s(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}

/// `x` widened to f64 (exact): the reference's view of the rounded inputs.
pub fn wide(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// [`EPS`] as the reference sees it.
pub fn eps64() -> f64 {
    f64::from(EPS)
}

/// `n` as u32 (test sizes are small).
pub fn u32_of(n: usize) -> u32 {
    u32::try_from(n).unwrap_or_else(|_| panic!("{n} does not fit u32"))
}

/// Panics with every non-passing check's detail.
pub fn assert_pass(checks: &[Check]) {
    assert!(!checks.is_empty(), "no checks ran");
    for c in checks {
        eprintln!("{} {}: {}", c.status.name(), c.name, c.detail);
    }
    let bad: Vec<String> = checks
        .iter()
        .filter(|c| c.status != Status::Pass)
        .map(|c| format!("{} {}: {}", c.status.name(), c.name, c.detail))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Bitwise equality of two f32 slices, naming the first difference.
pub fn assert_bits(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        let n = (0..got.len())
            .filter(|&i| got[i].to_bits() != want[i].to_bits())
            .count();
        panic!(
            "{label}: {n} of {} elements differ; first at {i}: got {:e} ({:#x}), want {:e} ({:#x})",
            got.len(),
            got[i],
            got[i].to_bits(),
            want[i],
            want[i].to_bits()
        );
    }
}

/// A device runtime with the default budget.
#[cfg(feature = "cuda")]
pub fn runtime() -> ojas_qwen35_cuda::runtime::CudaRuntime {
    ojas_qwen35_cuda::runtime::CudaRuntime::open(ojas_qwen35_cuda::runtime::RuntimeConfig::default())
        .unwrap_or_else(|e| panic!("CudaRuntime::open: {e}"))
}
