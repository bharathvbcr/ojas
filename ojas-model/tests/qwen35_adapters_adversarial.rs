//! Adversarial regressions for the adapter checkpoint, the gradient bank and
//! the all-or-nothing AdamW. Each test under "defects" fails against the
//! code as first written; "already handled" pins attacks the code resisted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ojas_core::{AdamWConfig, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::{write_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::lora::LoraConfig;
use ojas_model::qwen35::adapters::{lora_a_name, lora_b_name};
use ojas_model::qwen35::adapters::{
    read_adapter_checkpoint, write_adapter_checkpoint, AdapterCheckpoint, AdapterEntry,
    AdapterError, BaseIdentity, DataState, Expected, ExpectedEntry, ADAPTERS_FILE, KEY_DIGEST,
    KEY_LAYOUT, KEY_RANK, KEY_RNG, KEY_STEP, MOMENTS_FILE,
};
use ojas_model::qwen35::bank::{AdapterBank, BankEntry, BankError, BankMode, Residence};
use ojas_model::qwen35::optim::{adamw_all_or_nothing, AdamWEntry, OptimError};

const BASE: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

static DIRS: AtomicU64 = AtomicU64::new(0);

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ojas adv {}-{n} ünï", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }

    fn ckpt(&self) -> PathBuf {
        self.0.join("run 日本")
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn entry(name: &str, shape: &[usize], salt: usize) -> AdapterEntry {
    let n: usize = shape.iter().product();
    let v = |s: usize| -> Vec<f32> { (0..n).map(|i| ((i + s) % 7) as f32 * 0.25 - 0.5).collect() };
    AdapterEntry {
        name: name.to_string(),
        shape: shape.to_vec(),
        values: v(salt),
        moment1: v(salt + 1),
        moment2: v(salt + 2).into_iter().map(f32::abs).collect(),
    }
}

fn checkpoint(step: u64) -> AdapterCheckpoint {
    AdapterCheckpoint {
        lora: LoraConfig::new(4, 8.0, 0.0).unwrap(),
        step,
        data: DataState {
            rng_state: vec![1, 2, 3, 4],
            consumed_digest: "00ff".repeat(4),
        },
        base: BaseIdentity {
            file_name: "model.safetensors".to_string(),
            sha256: BASE.to_string(),
            quantizer: None,
        },
        entries: vec![
            entry("m.lora_A.weight", &[4, 3], 0),
            entry("m.lora_B.weight", &[2, 4], 1),
        ],
    }
}

fn expected(c: &AdapterCheckpoint) -> Expected {
    Expected {
        base: c.base.clone(),
        lora: c.lora,
        entries: c
            .entries
            .iter()
            .map(|e| ExpectedEntry {
                name: e.name.clone(),
                shape: e.shape.clone(),
            })
            .collect(),
    }
}

fn same(a: &AdapterCheckpoint, b: &AdapterCheckpoint) {
    assert_eq!(a.step, b.step);
    assert_eq!(a.data, b.data);
    assert_eq!(a.entries.len(), b.entries.len());
    for (x, y) in a.entries.iter().zip(&b.entries) {
        assert_eq!(x.name, y.name);
        assert_eq!(x.shape, y.shape);
        assert_eq!(bits(&x.values), bits(&y.values));
        assert_eq!(bits(&x.moment1), bits(&y.moment1));
        assert_eq!(bits(&x.moment2), bits(&y.moment2));
    }
}

struct Raw {
    name: String,
    dtype: StDtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn tamper(dir: &Path, file: &str, edit: impl FnOnce(&mut Vec<Raw>, &mut BTreeMap<String, String>)) {
    let path = dir.join(file);
    let (mut raws, mut meta) = {
        let st = SafeTensors::open(&path).unwrap();
        let names: Vec<String> = st.names().map(str::to_string).collect();
        let raws: Vec<Raw> = names
            .iter()
            .map(|n| {
                let info = st.info(n).unwrap();
                Raw {
                    name: n.clone(),
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data: st.read_bytes(n).unwrap(),
                }
            })
            .collect();
        (raws, st.metadata().clone())
    };
    edit(&mut raws, &mut meta);
    let items: Vec<TensorOut<'_>> = raws
        .iter()
        .map(|r| TensorOut {
            name: &r.name,
            dtype: r.dtype,
            shape: &r.shape,
            data: &r.data,
        })
        .collect();
    let md: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    write_safetensors(&path, &items, &md).unwrap();
}

fn saved(t: &Tmp, step: u64) -> (PathBuf, AdapterCheckpoint) {
    let c = checkpoint(step);
    let dir = t.ckpt();
    write_adapter_checkpoint(&dir, &c).unwrap();
    (dir, c)
}

// ---------------------------------------------------------------- defects

#[test]
fn a_non_canonical_step_or_rank_is_corruption() {
    for (key, text) in [
        (KEY_STEP, "+37"),
        (KEY_STEP, "037"),
        (KEY_STEP, " 37"),
        (KEY_STEP, "37 "),
        (KEY_RANK, "+4"),
        (KEY_RANK, "04"),
    ] {
        let t = Tmp::new();
        let (dir, c) = saved(&t, 37);
        tamper(&dir, ADAPTERS_FILE, |_, meta| {
            meta.insert(key.to_string(), text.to_string());
        });
        let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
        assert!(
            matches!(err, AdapterError::Corrupt { .. }),
            "{key}={text:?}: {err}"
        );
    }
}

#[test]
fn the_version_of_both_files_is_judged_before_other_metadata() {
    // The adapters file is a supported version but its RNG state is junk;
    // the moments file is a version this reader does not know. The document
    // promises the version first, in both files.
    let t = Tmp::new();
    let (dir, c) = saved(&t, 5);
    tamper(&dir, ADAPTERS_FILE, |_, meta| {
        meta.insert(KEY_RNG.to_string(), "zz".to_string());
    });
    tamper(&dir, MOMENTS_FILE, |_, meta| {
        meta.insert(KEY_LAYOUT.to_string(), "2".to_string());
    });
    let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
    assert!(matches!(err, AdapterError::UnknownVersion { .. }), "{err}");
}

#[test]
fn an_oversized_digest_is_refused_on_read_as_on_write() {
    let t = Tmp::new();
    let (dir, c) = saved(&t, 5);
    tamper(&dir, ADAPTERS_FILE, |_, meta| {
        meta.insert(KEY_DIGEST.to_string(), "ab".repeat(2049));
    });
    let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
    assert!(matches!(err, AdapterError::Corrupt { .. }), "{err}");
}

#[test]
fn a_resume_that_expects_no_entries_is_refused() {
    let t = Tmp::new();
    let (dir, c) = saved(&t, 5);
    let mut want = expected(&c);
    want.entries.clear();
    let err = read_adapter_checkpoint(&dir, &want).unwrap_err();
    assert!(matches!(err, AdapterError::InvalidField { .. }), "{err}");
}

#[test]
fn a_refused_overwrite_empties_the_slot_so_later_adds_cannot_extend_the_last_step() {
    let (backend, b) = (
        CpuBackend::new(Budget::new(1 << 24)).with_numerics(Numerics::Exact),
        Budget::new(1 << 24),
    );
    let mut bank = AdapterBank::new(vec![
        BankEntry {
            name: "a".to_string(),
            shape: vec![2, 2],
            residence: Residence::Backend,
        },
        BankEntry {
            name: "h".to_string(),
            shape: vec![3],
            residence: Residence::Host,
        },
    ])
    .unwrap();
    let good = Tensor::from_f32(&[1.0; 4], &[2, 2], &b).unwrap();
    bank.accumulate(&backend, 0, BankMode::Overwrite, good)
        .unwrap();
    bank.accumulate_host(1, BankMode::Overwrite, &[1.0; 3])
        .unwrap();

    let wrong = Tensor::from_f32(&[1.0; 4], &[4], &b).unwrap();
    assert!(matches!(
        bank.accumulate(&backend, 0, BankMode::Overwrite, wrong),
        Err(BankError::Shape { .. })
    ));
    assert!(matches!(
        bank.accumulate_host(1, BankMode::Overwrite, &[1.0; 2]),
        Err(BankError::Shape { .. })
    ));
    // The previous step's sums are gone: the next Add has nothing to extend.
    let next = Tensor::from_f32(&[1.0; 4], &[2, 2], &b).unwrap();
    assert!(matches!(
        bank.accumulate(&backend, 0, BankMode::Add, next),
        Err(BankError::AddWithoutBase { .. })
    ));
    assert!(matches!(
        bank.accumulate_host(1, BankMode::Add, &[1.0; 3]),
        Err(BankError::AddWithoutBase { .. })
    ));
    assert!(!bank.is_complete());
    assert!(bank.host_values(1).is_none());
}

#[test]
fn a_shared_parameter_or_moment_is_refused_before_any_tensor_moves() {
    let (backend, b) = (
        CpuBackend::new(Budget::new(1 << 24)).with_numerics(Numerics::Exact),
        Budget::new(1 << 24),
    );
    let mut s = states(&b);
    let g = grads(&b);
    let before = snapshot(&s);
    // A clone the caller kept alive: the backend cannot write in place at
    // this entry, and entries 0 and 1 would already have moved.
    let keep_param = s[2].p.clone();
    let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 0, &b).unwrap_err();
    assert!(matches!(err, OptimError::Refused { entry: 2, .. }), "{err}");
    assert_eq!(snapshot(&s), before);
    drop(keep_param);

    let keep_moment = s[1].m2.clone();
    let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 0, &b).unwrap_err();
    assert!(matches!(err, OptimError::Refused { entry: 1, .. }), "{err}");
    assert_eq!(snapshot(&s), before);
    drop(keep_moment);

    adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 0, &b).unwrap();
    assert_ne!(snapshot(&s), before);
}

#[test]
fn a_non_finite_host_parameter_or_moment_is_refused_before_any_tensor_moves() {
    let (backend, b) = (
        CpuBackend::new(Budget::new(1 << 24)).with_numerics(Numerics::Exact),
        Budget::new(1 << 24),
    );
    let g = grads(&b);
    for (which, bad) in [
        (0usize, f32::NAN),
        (1, f32::INFINITY),
        (2, f32::NEG_INFINITY),
    ] {
        let mut s = states(&b);
        let poisoned = vec![bad; 2];
        let t = Tensor::from_f32(&poisoned, &[2], &b).unwrap();
        match which {
            0 => s[2].p = t,
            1 => s[2].m1 = t,
            _ => s[2].m2 = t,
        }
        let before = snapshot(&s);
        let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 3, &b).unwrap_err();
        assert!(
            matches!(err, OptimError::Refused { entry: 2, .. }),
            "{which}: {err}"
        );
        assert_eq!(snapshot(&s), before, "{which}");
    }
}

// -------------------------------------------------------- already handled

#[test]
fn step_zero_and_u64_max_round_trip() {
    for step in [0u64, u64::MAX] {
        let t = Tmp::new();
        let (dir, c) = saved(&t, step);
        same(&c, &read_adapter_checkpoint(&dir, &expected(&c)).unwrap());
    }
}

#[test]
fn awkward_entry_names_round_trip() {
    let t = Tmp::new();
    let mut c = checkpoint(1);
    let awkward = "日本.\"q\".back\\slash x";
    c.entries.push(entry(&lora_a_name(awkward), &[4, 3], 3));
    c.entries.push(entry(&lora_b_name(awkward), &[2, 4], 4));
    c.entries.push(entry(&lora_a_name("a.b.c.d.e"), &[4, 1], 5));
    c.entries.push(entry(&lora_b_name("a.b.c.d.e"), &[1, 4], 6));
    let dir = t.ckpt();
    write_adapter_checkpoint(&dir, &c).unwrap();
    let back = read_adapter_checkpoint(&dir, &expected(&c)).unwrap();
    same(&c, &back);
}

#[test]
fn a_bare_awkward_entry_name_is_refused() {
    let t = Tmp::new();
    let mut c = checkpoint(1);
    c.entries
        .push(entry("日本.\"q\".back\\slash x.weight", &[2], 3));
    let err = write_adapter_checkpoint(&t.ckpt(), &c).unwrap_err();
    assert!(
        matches!(&err, AdapterError::UnknownEntryName(n) if n == "日本.\"q\".back\\slash x.weight"),
        "{err}"
    );
}

#[test]
fn a_trailing_slash_path_works_and_a_stale_stage_is_swept() {
    let t = Tmp::new();
    let c = checkpoint(2);
    let plain = t.ckpt();
    let slashed = PathBuf::from(format!("{}/", plain.display()));
    write_adapter_checkpoint(&slashed, &c).unwrap();
    same(
        &c,
        &read_adapter_checkpoint(&slashed, &expected(&c)).unwrap(),
    );

    let name = plain.file_name().unwrap().to_string_lossy().into_owned();
    let stale =
        t.0.join(format!(".{name}.{}.987654.stage", std::process::id()));
    std::fs::create_dir(&stale).unwrap();
    std::fs::write(stale.join("junk"), b"x").unwrap();
    write_adapter_checkpoint(&plain, &checkpoint(3)).unwrap();
    assert!(!stale.exists());
    same(
        &checkpoint(3),
        &read_adapter_checkpoint(&plain, &expected(&c)).unwrap(),
    );
}

#[cfg(unix)]
#[test]
fn a_symlinked_target_or_a_file_target_is_refused_and_left_alone() {
    let t = Tmp::new();
    let real = t.0.join("real");
    write_adapter_checkpoint(&real, &checkpoint(7)).unwrap();
    let link = t.0.join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    assert!(write_adapter_checkpoint(&link, &checkpoint(8)).is_err());
    same(
        &checkpoint(7),
        &read_adapter_checkpoint(&real, &expected(&checkpoint(7))).unwrap(),
    );

    let file = t.0.join("a file");
    std::fs::write(&file, b"keep").unwrap();
    assert!(write_adapter_checkpoint(&file, &checkpoint(8)).is_err());
    assert_eq!(std::fs::read(&file).unwrap(), b"keep");
}

// ------------------------------------------------------- optimiser helpers

const SHAPES: [&[usize]; 3] = [&[3], &[4], &[2]];
const NAMES: [&str; 3] = ["a", "b", "c"];

struct State {
    p: Tensor,
    m1: Tensor,
    m2: Tensor,
}

fn states(b: &Budget) -> Vec<State> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n: usize = s.iter().product();
            let v = |k: usize| -> Vec<f32> {
                (0..n)
                    .map(|j| ((j + i + k) % 5) as f32 * 0.1 + 0.05)
                    .collect()
            };
            State {
                p: Tensor::from_f32(&v(0), s, b).unwrap(),
                m1: Tensor::from_f32(&v(1), s, b).unwrap(),
                m2: Tensor::from_f32(&v(2), s, b).unwrap(),
            }
        })
        .collect()
}

fn grads(b: &Budget) -> Vec<Tensor> {
    SHAPES
        .iter()
        .map(|s| {
            let n: usize = s.iter().product();
            Tensor::from_f32(&vec![0.5; n], s, b).unwrap()
        })
        .collect()
}

fn snapshot(s: &[State]) -> Vec<Vec<u32>> {
    s.iter()
        .flat_map(|s| [&s.p, &s.m1, &s.m2].map(|t| bits(&t.to_f32_vec().unwrap())))
        .collect()
}

fn entries<'a>(s: &'a mut [State], g: &'a [Tensor]) -> Vec<AdamWEntry<'a>> {
    s.iter_mut()
        .zip(g)
        .enumerate()
        .map(|(i, (s, g))| AdamWEntry {
            name: NAMES[i],
            param: &mut s.p,
            grad: g,
            moment1: &mut s.m1,
            moment2: &mut s.m2,
            config: AdamWConfig {
                lr: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        })
        .collect()
}
