//! The adapter checkpoint (`docs/adapter-checkpoint.md`): a bit-exact round
//! trip, the file and metadata contract Lappi's converter reads, every
//! refusal on resume, truncated and tampered files, replacement of an
//! earlier checkpoint, and a directory path with spaces and non-ASCII text.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ojas_core::Budget;
use ojas_io::{write_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::lora::LoraConfig;
use ojas_model::qwen35::adapters::{
    lora_a_name, lora_b_name, read_adapter_checkpoint, write_adapter_checkpoint, AdapterCheckpoint,
    AdapterEntry, AdapterError, BaseIdentity, DataState, Expected, ExpectedEntry, QuantizerRecord,
    ADAPTERS_FILE, ADAPTER_LAYOUT_VERSION, HOST_ENTRY_NAMES, KEY_BASE, KEY_LAYOUT, KEY_PAYLOAD,
    KEY_STEP, MOMENTS_FILE,
};

const MODULE: &str = "model.language_model.layers.3.self_attn.q_proj";
const BASE: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

static DIRS: AtomicU64 = AtomicU64::new(0);

/// A fresh directory whose own name has a space and non-ASCII text.
struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        let n = DIRS.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("ojas adapters {}-{n} ünï 日本", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }

    fn ckpt(&self) -> PathBuf {
        self.0.join("run 1 (résumé)")
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

/// Values with the awkward bit patterns a text or lossy format would lose:
/// negative zero, subnormals, the largest finite, ordinary fractions.
fn pattern(n: usize, salt: usize) -> Vec<f32> {
    let odd = [
        -0.0f32,
        0.0,
        f32::MIN_POSITIVE / 4.0,
        -f32::MIN_POSITIVE / 8.0,
        f32::MAX,
        f32::MIN,
        1.0 / 3.0,
        -1.0e-20,
    ];
    (0..n)
        .map(|i| {
            if i < odd.len() {
                odd[(i + salt) % odd.len()]
            } else {
                ((i * 31 + salt * 17) % 101) as f32 * 0.0137 - 0.6
            }
        })
        .collect()
}

fn nonneg(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| x.abs()).collect()
}

fn entry(name: String, shape: &[usize], salt: usize) -> AdapterEntry {
    let n: usize = shape.iter().product();
    AdapterEntry {
        name,
        shape: shape.to_vec(),
        values: pattern(n, salt),
        moment1: pattern(n, salt + 3),
        moment2: nonneg(pattern(n, salt + 5)),
    }
}

fn lora() -> LoraConfig {
    LoraConfig::new(16, 16.0, 0.0).unwrap()
}

fn quant() -> QuantizerRecord {
    QuantizerRecord {
        format: "mlx-affine".to_string(),
        group_size: "64".to_string(),
        mlx_version: "0.31.0".to_string(),
    }
}

fn base(quantizer: Option<QuantizerRecord>) -> BaseIdentity {
    BaseIdentity {
        file_name: "Qwen3.5-2B-Base-fakequant-bf16.safetensors".to_string(),
        sha256: BASE.to_string(),
        quantizer,
    }
}

fn checkpoint(step: u64) -> AdapterCheckpoint {
    AdapterCheckpoint {
        lora: lora(),
        step,
        data: DataState {
            rng_state: vec![0, 1, 2, 0xfe, 0xff, 0x80],
            consumed_digest: "00ff10aa".repeat(8),
        },
        base: base(Some(quant())),
        entries: vec![
            entry(lora_a_name(MODULE), &[16, 5], 0),
            entry(lora_b_name(MODULE), &[7, 16], 1),
            entry("d17.delta".to_string(), &[3], 2),
            entry("span_head.start_proj.weight".to_string(), &[2, 3], 3),
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

fn assert_same_bits(a: &AdapterCheckpoint, b: &AdapterCheckpoint) {
    assert_eq!(a.step, b.step);
    assert_eq!(a.data, b.data);
    assert_eq!(a.base, b.base);
    assert_eq!(a.lora.rank(), b.lora.rank());
    assert_eq!(a.lora.alpha().to_bits(), b.lora.alpha().to_bits());
    assert_eq!(a.lora.dropout(), 0.0);
    assert_eq!(a.entries.len(), b.entries.len());
    for (x, y) in a.entries.iter().zip(&b.entries) {
        assert_eq!(x.name, y.name);
        assert_eq!(x.shape, y.shape);
        assert_eq!(bits(&x.values), bits(&y.values), "{} values", x.name);
        assert_eq!(bits(&x.moment1), bits(&y.moment1), "{} m1", x.name);
        assert_eq!(bits(&x.moment2), bits(&y.moment2), "{} m2", x.name);
    }
}

struct Raw {
    name: String,
    dtype: StDtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn load_raw(path: &Path) -> (Vec<Raw>, BTreeMap<String, String>) {
    let st = SafeTensors::open(path).unwrap();
    let names: Vec<String> = st.names().map(str::to_string).collect();
    let raws = names
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
}

fn store_raw(path: &Path, raws: &[Raw], meta: &BTreeMap<String, String>) {
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
    write_safetensors(path, &items, &md).unwrap();
}

/// Rewrite one file of the checkpoint after `edit` changed its tensors and
/// metadata.
fn tamper(dir: &Path, file: &str, edit: impl FnOnce(&mut Vec<Raw>, &mut BTreeMap<String, String>)) {
    let path = dir.join(file);
    let (mut raws, mut meta) = load_raw(&path);
    edit(&mut raws, &mut meta);
    store_raw(&path, &raws, &meta);
}

/// The integrity hash the format records (`docs/adapter-checkpoint.md`):
/// FNV-1a 64 over the little-endian tensor bytes in ascending name order,
/// then over every metadata pair but the hash itself, sorted by key, as
/// `len(key)` u64 LE, key, `len(value)` u64 LE, value. Written out here so
/// the test pins the documented algorithm and not the writer's own helper.
fn rehash(raws: &[Raw], meta: &mut BTreeMap<String, String>) {
    let mut sorted: Vec<&Raw> = raws.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            h = (h ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for r in sorted {
        feed(&r.data);
    }
    for (k, v) in meta.iter().filter(|(k, _)| k.as_str() != KEY_PAYLOAD) {
        for s in [k, v] {
            feed(&(s.len() as u64).to_le_bytes());
            feed(s.as_bytes());
        }
    }
    meta.insert(KEY_PAYLOAD.to_string(), format!("{h:016x}"));
}

fn saved(t: &Tmp) -> (PathBuf, AdapterCheckpoint) {
    let c = checkpoint(37);
    let dir = t.ckpt();
    write_adapter_checkpoint(&dir, &c).unwrap();
    (dir, c)
}

#[test]
fn round_trip_is_bit_exact() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    let back = read_adapter_checkpoint(&dir, &expected(&c)).unwrap();
    assert_same_bits(&c, &back);
    assert_eq!(back, c);
}

#[test]
fn a_read_returns_entries_in_the_expected_order_not_the_saved_one() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    let mut ex = expected(&c);
    ex.entries.reverse();
    let back = read_adapter_checkpoint(&dir, &ex).unwrap();
    let mut want = c.clone();
    want.entries.reverse();
    assert_same_bits(&want, &back);
}

#[test]
fn the_files_follow_the_documented_layout() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    let a = SafeTensors::open(&dir.join(ADAPTERS_FILE)).unwrap();
    let mut names: Vec<&str> = a.names().collect();
    names.sort_unstable();
    let mut want: Vec<&str> = c.entries.iter().map(|e| e.name.as_str()).collect();
    want.sort_unstable();
    assert_eq!(names, want);
    assert!(a.info(&format!("{MODULE}.lora_A.weight")).is_ok());
    assert!(a.info(&format!("{MODULE}.lora_B.weight")).is_ok());
    for e in &c.entries {
        let info = a.info(&e.name).unwrap();
        assert_eq!(info.dtype, StDtype::F32);
        assert_eq!(
            info.shape,
            e.shape.iter().map(|&d| d as u64).collect::<Vec<_>>()
        );
    }
    let m = SafeTensors::open(&dir.join(MOMENTS_FILE)).unwrap();
    let mut mnames: Vec<String> = m.names().map(str::to_string).collect();
    mnames.sort_unstable();
    let mut mwant: Vec<String> = c
        .entries
        .iter()
        .flat_map(|e| [format!("adam_m.{}", e.name), format!("adam_v.{}", e.name)])
        .collect();
    mwant.sort_unstable();
    assert_eq!(mnames, mwant);

    let md = a.metadata();
    assert_eq!(md[KEY_LAYOUT], ADAPTER_LAYOUT_VERSION.to_string());
    assert_eq!(md[KEY_LAYOUT], "1");
    assert_eq!(md[KEY_STEP], "37");
    assert_eq!(md[KEY_BASE], BASE);
    assert_eq!(
        md["ojas.adapter.base_file"],
        "Qwen3.5-2B-Base-fakequant-bf16.safetensors"
    );
    assert_eq!(md["ojas.adapter.lora_rank"], "16");
    assert_eq!(md["ojas.adapter.lora_alpha"], "16");
    assert_eq!(md["ojas.adapter.rng_state"], "000102feff80");
    assert_eq!(md["ojas.adapter.consumed_digest"], "00ff10aa".repeat(8));
    assert_eq!(md["ojas.adapter.quant.format"], "mlx-affine");
    assert_eq!(md["ojas.adapter.quant.group_size"], "64");
    assert_eq!(md["ojas.adapter.quant.mlx_version"], "0.31.0");
    assert_eq!(m.metadata()[KEY_LAYOUT], "1");
    assert_eq!(m.metadata()[KEY_STEP], "37");
    assert!(m.metadata().contains_key(KEY_PAYLOAD));
    // The recorded payload hash is the documented algorithm's.
    for file in [ADAPTERS_FILE, MOMENTS_FILE] {
        let (raws, mut meta) = load_raw(&dir.join(file));
        let recorded = meta[KEY_PAYLOAD].clone();
        rehash(&raws, &mut meta);
        assert_eq!(meta[KEY_PAYLOAD], recorded, "{file}");
    }
    // Criterion 19: no dropout key anywhere.
    for key in md.keys().chain(m.metadata().keys()) {
        assert!(!key.to_lowercase().contains("dropout"), "{key}");
    }
    for name in a.names().chain(m.names()) {
        assert!(!name.to_lowercase().contains("dropout"), "{name}");
    }
}

#[test]
fn a_checkpoint_without_a_quantizer_round_trips_and_is_not_the_quantised_one() {
    let t = Tmp::new();
    let dir = t.ckpt();
    let mut c = checkpoint(1);
    c.base.quantizer = None;
    write_adapter_checkpoint(&dir, &c).unwrap();
    assert_eq!(read_adapter_checkpoint(&dir, &expected(&c)).unwrap(), c);
    let mut ex = expected(&c);
    ex.base.quantizer = Some(quant());
    assert!(matches!(
        read_adapter_checkpoint(&dir, &ex),
        Err(AdapterError::QuantizerMismatch {
            expected: Some(_),
            found: None
        })
    ));
}

#[test]
fn a_nonzero_dropout_refuses_to_checkpoint_and_writes_nothing() {
    let t = Tmp::new();
    let dir = t.ckpt();
    let mut c = checkpoint(5);
    c.lora = LoraConfig::new(16, 16.0, 0.1).unwrap();
    let err = write_adapter_checkpoint(&dir, &c).unwrap_err();
    assert!(matches!(err, AdapterError::Dropout(p) if p == 0.1), "{err}");
    assert!(!dir.exists());

    // Over an earlier checkpoint it leaves that one readable and unchanged.
    let (dir, good) = saved(&t);
    let err = write_adapter_checkpoint(&dir, &c).unwrap_err();
    assert!(matches!(err, AdapterError::Dropout(_)));
    assert_same_bits(
        &good,
        &read_adapter_checkpoint(&dir, &expected(&good)).unwrap(),
    );
}

#[test]
fn resume_refuses_a_different_frozen_base() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    let mut ex = expected(&c);
    ex.base.sha256 = "f".repeat(64);
    let err = read_adapter_checkpoint(&dir, &ex).unwrap_err();
    assert!(
        matches!(&err, AdapterError::BaseMismatch { expected, found }
            if expected == &"f".repeat(64) && found == BASE),
        "{err}"
    );
    // One flipped digit is a different base.
    let mut ex = expected(&c);
    ex.base.sha256 = format!("{}0", &BASE[..63]);
    assert!(matches!(
        read_adapter_checkpoint(&dir, &ex),
        Err(AdapterError::BaseMismatch { .. })
    ));
}

/// Layout 1 names one base file. A shard or an index is refused at save
/// (nothing written) and as an expectation at resume; a different file name
/// is a different base.
#[test]
fn a_sharded_base_is_refused() {
    for name in [
        "model-00001-of-00004.safetensors",
        "Qwen3.5-2B-Base-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ] {
        let t = Tmp::new();
        let dir = t.ckpt();
        let mut c = checkpoint(1);
        c.base.file_name = name.to_string();
        let err = write_adapter_checkpoint(&dir, &c).unwrap_err();
        assert!(
            matches!(&err, AdapterError::ShardedBase { file_name } if file_name == name),
            "{name}: {err}"
        );
        assert!(!dir.exists());

        let (dir, good) = saved(&t);
        let mut ex = expected(&good);
        ex.base.file_name = name.to_string();
        assert!(matches!(
            read_adapter_checkpoint(&dir, &ex),
            Err(AdapterError::ShardedBase { .. })
        ));
    }
    // A name that merely contains "-of-" is not a shard.
    let t = Tmp::new();
    let mut c = checkpoint(1);
    c.base.file_name = "state-of-the-art.safetensors".to_string();
    write_adapter_checkpoint(&t.ckpt(), &c).unwrap();
    // One file of one is the whole base (the local 2B-Base and 0.8B-Base
    // snapshots are named this way): one sha256 names it, so it is accepted
    // on save and on resume.
    let t = Tmp::new();
    let mut c = checkpoint(1);
    c.base.file_name = "model.safetensors-00001-of-00001.safetensors".to_string();
    let dir = t.ckpt();
    write_adapter_checkpoint(&dir, &c).unwrap();
    let mut ex = expected(&c);
    ex.base.file_name = c.base.file_name.clone();
    read_adapter_checkpoint(&dir, &ex).unwrap();
    // File names that are not one plain name.
    for name in ["", "dir/model.safetensors", "..", "a\\b"] {
        let mut c = checkpoint(1);
        c.base.file_name = name.to_string();
        assert!(matches!(
            write_adapter_checkpoint(&t.0.join("x"), &c),
            Err(AdapterError::InvalidField { .. })
        ));
    }
}

#[test]
fn resume_refuses_a_different_base_file_name() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    let mut ex = expected(&c);
    ex.base.file_name = "other.safetensors".to_string();
    assert!(matches!(
        read_adapter_checkpoint(&dir, &ex),
        Err(AdapterError::BaseFileMismatch { .. })
    ));
}

/// D17 and the span head have the fixed names of `HOST_ENTRY_NAMES`; any
/// other name that is not a LoRA A/B is refused on save and on read.
#[test]
fn an_unknown_host_entry_name_is_refused() {
    let t = Tmp::new();
    let (dir, good) = saved(&t);
    for name in [
        "d17.weight",
        "span_head.weight",
        "span_head.bias",
        "d17",
        ".lora_A.weight",
        "layers.0.lora_C.weight",
        "head",
    ] {
        let mut c = checkpoint(2);
        c.entries[2].name = name.to_string();
        let err = write_adapter_checkpoint(&dir, &c).unwrap_err();
        assert!(
            matches!(&err, AdapterError::UnknownEntryName(n) if n == name),
            "{name}: {err}"
        );
        // The earlier checkpoint is untouched, and an expectation naming it
        // is refused too.
        let mut ex = expected(&good);
        ex.entries[2].name = name.to_string();
        assert!(matches!(
            read_adapter_checkpoint(&dir, &ex),
            Err(AdapterError::UnknownEntryName(_))
        ));
    }
    for name in HOST_ENTRY_NAMES {
        let other = if name == HOST_ENTRY_NAMES[4] {
            HOST_ENTRY_NAMES[3]
        } else {
            HOST_ENTRY_NAMES[4]
        };
        let mut c = checkpoint(3);
        c.entries[2].name = name.to_string();
        c.entries[3].name = other.to_string();
        write_adapter_checkpoint(&t.0.join("ok"), &c).unwrap();
    }
}

#[test]
fn resume_refuses_a_different_quantizer_record() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    let edits: [fn(&mut QuantizerRecord); 3] = [
        |q| q.format = "mlx-mxfp4".to_string(),
        |q| q.group_size = "32".to_string(),
        |q| q.mlx_version = "0.30.9".to_string(),
    ];
    for edit in edits {
        let mut ex = expected(&c);
        edit(ex.base.quantizer.as_mut().unwrap());
        assert!(matches!(
            read_adapter_checkpoint(&dir, &ex),
            Err(AdapterError::QuantizerMismatch { .. })
        ));
    }
    let mut ex = expected(&c);
    ex.base.quantizer = None;
    assert!(matches!(
        read_adapter_checkpoint(&dir, &ex),
        Err(AdapterError::QuantizerMismatch {
            expected: None,
            found: Some(_)
        })
    ));
}

#[test]
fn resume_refuses_a_different_lora_rank_or_alpha() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    for l in [
        LoraConfig::new(8, 16.0, 0.0).unwrap(),
        LoraConfig::new(16, 32.0, 0.0).unwrap(),
    ] {
        let mut ex = expected(&c);
        ex.lora = l;
        assert!(matches!(
            read_adapter_checkpoint(&dir, &ex),
            Err(AdapterError::LoraMismatch { .. })
        ));
    }
}

#[test]
fn resume_refuses_missing_extra_and_misshapen_entries() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);

    let mut ex = expected(&c);
    ex.entries.push(ExpectedEntry {
        name: "span_head.abstain_end".to_string(),
        shape: vec![2],
    });
    let err = read_adapter_checkpoint(&dir, &ex).unwrap_err();
    assert!(
        matches!(&err, AdapterError::MissingEntry(n) if n == "span_head.abstain_end"),
        "{err}"
    );

    let mut ex = expected(&c);
    let dropped = ex.entries.remove(2);
    let err = read_adapter_checkpoint(&dir, &ex).unwrap_err();
    assert!(
        matches!(&err, AdapterError::ExtraEntry(n) if n == &dropped.name),
        "{err}"
    );

    let mut ex = expected(&c);
    ex.entries[1].shape = vec![16, 7];
    let err = read_adapter_checkpoint(&dir, &ex).unwrap_err();
    assert!(
        matches!(&err, AdapterError::LayoutMismatch { name, .. } if name == &lora_b_name(MODULE)),
        "{err}"
    );

    let mut ex = expected(&c);
    ex.entries[0].shape = vec![80];
    assert!(matches!(
        read_adapter_checkpoint(&dir, &ex),
        Err(AdapterError::LayoutMismatch { .. })
    ));

    // A duplicate in the expectation is refused, not merged.
    let mut ex = expected(&c);
    ex.entries.push(ex.entries[0].clone());
    assert!(matches!(
        read_adapter_checkpoint(&dir, &ex),
        Err(AdapterError::InvalidEntry { .. })
    ));
}

#[test]
fn resume_refuses_an_extra_entry_in_the_moments_file() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    tamper(&dir, MOMENTS_FILE, |raws, _| {
        raws.push(Raw {
            name: "adam_m.stray".to_string(),
            dtype: StDtype::F32,
            shape: vec![1],
            data: 1.0f32.to_le_bytes().to_vec(),
        });
    });
    let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
    assert!(
        matches!(&err, AdapterError::ExtraEntry(n) if n == "adam_m.stray"),
        "{err}"
    );
}

#[test]
fn resume_refuses_a_dtype_that_is_not_f32() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    tamper(&dir, ADAPTERS_FILE, |raws, _| {
        let r = raws.iter_mut().find(|r| r.name == "d17.delta").unwrap();
        r.dtype = StDtype::BF16;
        r.data = r.data[..6].to_vec();
    });
    let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
    assert!(
        matches!(&err, AdapterError::LayoutMismatch { name, .. } if name == "d17.delta"),
        "{err}"
    );
}

#[test]
fn resume_refuses_an_unknown_layout_version() {
    for file in [ADAPTERS_FILE, MOMENTS_FILE] {
        for found in ["2", "0", "01", "one"] {
            let t = Tmp::new();
            let (dir, c) = saved(&t);
            tamper(&dir, file, |_, meta| {
                meta.insert(KEY_LAYOUT.to_string(), found.to_string());
            });
            let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
            assert!(
                matches!(&err, AdapterError::UnknownVersion { found: f, supported: 1 } if f == found),
                "{file} {found:?}: {err}"
            );
        }
    }
    // A missing version key is corruption, not a version.
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    tamper(&dir, ADAPTERS_FILE, |_, meta| {
        meta.remove(KEY_LAYOUT);
    });
    assert!(matches!(
        read_adapter_checkpoint(&dir, &expected(&c)),
        Err(AdapterError::Corrupt { .. })
    ));
}

#[test]
fn resume_refuses_missing_or_malformed_metadata() {
    for key in [
        KEY_STEP,
        KEY_BASE,
        "ojas.adapter.lora_rank",
        "ojas.adapter.lora_alpha",
        "ojas.adapter.rng_state",
        "ojas.adapter.consumed_digest",
        KEY_PAYLOAD,
        "ojas.adapter.quant.group_size",
    ] {
        let t = Tmp::new();
        let (dir, c) = saved(&t);
        tamper(&dir, ADAPTERS_FILE, |_, meta| {
            assert!(meta.remove(key).is_some(), "{key}");
        });
        let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
        assert!(matches!(err, AdapterError::Corrupt { .. }), "{key}: {err}");
    }
    for (key, value) in [
        (KEY_STEP, "-1"),
        (KEY_STEP, "1.5"),
        ("ojas.adapter.rng_state", "xyz"),
        ("ojas.adapter.rng_state", "ABCD"),
        ("ojas.adapter.rng_state", "abc"),
        ("ojas.adapter.consumed_digest", "NOTHEX"),
    ] {
        let t = Tmp::new();
        let (dir, c) = saved(&t);
        tamper(&dir, ADAPTERS_FILE, |_, meta| {
            meta.insert(key.to_string(), value.to_string());
        });
        let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
        assert!(
            matches!(err, AdapterError::Corrupt { .. }),
            "{key}={value}: {err}"
        );
    }
}

#[test]
fn resume_refuses_files_written_at_different_steps() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    tamper(&dir, MOMENTS_FILE, |_, meta| {
        meta.insert(KEY_STEP.to_string(), "38".to_string());
    });
    let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
    assert!(
        matches!(
            err,
            AdapterError::StepMismatch {
                adapters: 37,
                moments: 38
            }
        ),
        "{err}"
    );
}

#[test]
fn a_flipped_bit_or_a_non_finite_value_is_corruption() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    tamper(&dir, ADAPTERS_FILE, |raws, _| {
        let r = raws.iter_mut().find(|r| r.name == "d17.delta").unwrap();
        r.data[0] ^= 0x01;
    });
    let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
    assert!(matches!(err, AdapterError::Corrupt { .. }), "{err}");

    // A NaN, an infinity, and a negative second moment, each with the
    // recorded payload hash recomputed so the value alone is what is wrong.
    for (what, bad) in [
        ("NaN", f32::NAN),
        ("infinity", f32::INFINITY),
        ("negative second moment", -1.0),
    ] {
        let t = Tmp::new();
        let (dir, c) = saved(&t);
        tamper(&dir, MOMENTS_FILE, |raws, meta| {
            let r = raws
                .iter_mut()
                .find(|r| r.name == "adam_v.d17.delta")
                .unwrap();
            r.data[..4].copy_from_slice(&bad.to_le_bytes());
            rehash(raws, meta);
        });
        let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
        assert!(matches!(err, AdapterError::Corrupt { .. }), "{what}: {err}");
    }
}

/// The integrity hash covers the metadata too: a flipped hex digit of the
/// rng state or the consumed digest is still valid hex and nothing else
/// compares it, so only the hash can catch it.
#[test]
fn a_flipped_metadata_hex_digit_is_corruption() {
    for key in ["ojas.adapter.rng_state", "ojas.adapter.consumed_digest"] {
        let t = Tmp::new();
        let (dir, c) = saved(&t);
        tamper(&dir, ADAPTERS_FILE, |_, meta| {
            let v = meta.get_mut(key).unwrap();
            let flipped = if v.starts_with('0') { '1' } else { '0' };
            v.replace_range(..1, &flipped.to_string());
        });
        let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
        assert!(matches!(err, AdapterError::Corrupt { .. }), "{key}: {err}");
    }
}

/// Moments from another save at the same step have a valid hash of their
/// own and the same names and shapes; only the binding to the adapters
/// file refuses them.
#[test]
fn moments_from_a_second_checkpoint_at_the_same_step_are_refused() {
    let t = Tmp::new();
    let (dir_a, a) = saved(&t);
    let dir_b = t.0.join("other run");
    let mut b = checkpoint(37);
    b.entries[2].values = pattern(3, 11);
    b.entries[2].moment1 = pattern(3, 12);
    write_adapter_checkpoint(&dir_b, &b).unwrap();
    // Each is readable on its own.
    read_adapter_checkpoint(&dir_b, &expected(&b)).unwrap();
    std::fs::copy(dir_b.join(MOMENTS_FILE), dir_a.join(MOMENTS_FILE)).unwrap();
    let err = read_adapter_checkpoint(&dir_a, &expected(&a)).unwrap_err();
    assert!(matches!(err, AdapterError::PairMismatch { .. }), "{err}");
}

#[test]
fn truncated_and_garbled_files_are_errors_not_panics() {
    let t = Tmp::new();
    let (dir, c) = saved(&t);
    for file in [ADAPTERS_FILE, MOMENTS_FILE] {
        let path = dir.join(file);
        let whole = std::fs::read(&path).unwrap();
        for keep in [0usize, 3, 8, 12, whole.len() / 2, whole.len() - 1] {
            std::fs::write(&path, &whole[..keep]).unwrap();
            let err = read_adapter_checkpoint(&dir, &expected(&c)).unwrap_err();
            assert!(
                matches!(
                    err,
                    AdapterError::Unreadable { .. } | AdapterError::Corrupt { .. }
                ),
                "{file} cut to {keep}: {err}"
            );
        }
        // A header length that points past the file.
        let mut garbled = whole.clone();
        garbled[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&path, &garbled).unwrap();
        assert!(
            read_adapter_checkpoint(&dir, &expected(&c)).is_err(),
            "{file}"
        );
        // Garbage in the JSON header.
        let mut garbled = whole.clone();
        garbled[9] = 0xff;
        garbled[10] = 0xfe;
        std::fs::write(&path, &garbled).unwrap();
        assert!(
            read_adapter_checkpoint(&dir, &expected(&c)).is_err(),
            "{file}"
        );
        std::fs::write(&path, &whole).unwrap();
        assert_same_bits(&c, &read_adapter_checkpoint(&dir, &expected(&c)).unwrap());
    }
    // A missing file, and a missing directory.
    std::fs::remove_file(dir.join(MOMENTS_FILE)).unwrap();
    assert!(matches!(
        read_adapter_checkpoint(&dir, &expected(&c)),
        Err(AdapterError::Unreadable { .. })
    ));
    assert!(matches!(
        read_adapter_checkpoint(&t.0.join("absent"), &expected(&c)),
        Err(AdapterError::Unreadable { .. })
    ));
}

#[test]
fn save_refuses_bad_entries_and_fields_before_touching_the_directory() {
    let t = Tmp::new();
    let (dir, good) = saved(&t);
    let cases: [fn(&mut AdapterCheckpoint); 17] = [
        |c| c.entries[0].values[2] = f32::NAN,
        |c| c.entries[1].moment1[0] = f32::INFINITY,
        |c| c.entries[1].moment2[3] = -1.0,
        |c| {
            c.entries[2].values.pop();
        },
        |c| c.entries[2].moment2.push(0.0),
        |c| c.entries[0].shape = vec![5, 16, 1, 0],
        |c| c.entries[3].name = c.entries[0].name.clone(),
        |c| c.entries[3].name = String::new(),
        |c| c.entries[3].name = "__metadata__".to_string(),
        |c| c.entries.clear(),
        |c| c.data.rng_state.clear(),
        |c| c.data.consumed_digest = "XYZ".to_string(),
        |c| c.data.consumed_digest = "abc".to_string(),
        |c| c.data.consumed_digest.clear(),
        |c| c.base.sha256 = "abc".to_string(),
        |c| c.base.sha256 = "A".repeat(64),
        |c| c.base.quantizer.as_mut().unwrap().format.clear(),
    ];
    for (i, edit) in cases.into_iter().enumerate() {
        let mut c = checkpoint(99);
        edit(&mut c);
        let err = write_adapter_checkpoint(&dir, &c).unwrap_err();
        assert!(
            matches!(
                err,
                AdapterError::InvalidEntry { .. } | AdapterError::InvalidField { .. }
            ),
            "case {i}: {err}"
        );
        // The earlier checkpoint is still whole.
        assert_same_bits(
            &good,
            &read_adapter_checkpoint(&dir, &expected(&good)).unwrap(),
        );
    }
}

#[test]
fn a_second_save_replaces_the_first_whole() {
    let t = Tmp::new();
    let (dir, first) = saved(&t);
    let mut second = checkpoint(38);
    second.entries[2].values = pattern(3, 9);
    second.data.consumed_digest = "ab".repeat(32);
    write_adapter_checkpoint(&dir, &second).unwrap();
    let back = read_adapter_checkpoint(&dir, &expected(&second)).unwrap();
    assert_same_bits(&second, &back);
    assert_ne!(first.step, back.step);
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert_eq!(files, [ADAPTERS_FILE, MOMENTS_FILE]);
    // No staging or backup sibling (dot-prefixed) is left beside it.
    let leftovers: Vec<String> = std::fs::read_dir(&t.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn save_refuses_a_directory_that_is_not_an_adapter_checkpoint() {
    let t = Tmp::new();
    let dir = t.ckpt();
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("notes.txt"), b"keep me").unwrap();
    let err = write_adapter_checkpoint(&dir, &checkpoint(1)).unwrap_err();
    assert!(matches!(err, AdapterError::TargetRefused(_)), "{err}");
    assert_eq!(std::fs::read(dir.join("notes.txt")).unwrap(), b"keep me");
    assert!(!dir.join(ADAPTERS_FILE).exists());

    // An empty directory is fine.
    std::fs::remove_file(dir.join("notes.txt")).unwrap();
    write_adapter_checkpoint(&dir, &checkpoint(1)).unwrap();
}

#[test]
fn unicode_and_spaces_in_the_path_work() {
    let t = Tmp::new();
    let nested = t.0.join("ünï dir").join("日本 語 run");
    std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
    let c = checkpoint(12);
    write_adapter_checkpoint(&nested, &c).unwrap();
    write_adapter_checkpoint(&nested, &c).unwrap();
    assert_same_bits(
        &c,
        &read_adapter_checkpoint(&nested, &expected(&c)).unwrap(),
    );
}

#[test]
fn entries_move_to_and_from_tensors_exactly() {
    let b = Budget::new(1 << 24);
    let e = entry("x.weight".to_string(), &[3, 4], 6);
    let (p, m1, m2) = e.to_tensors(&b).unwrap();
    let back = AdapterEntry::from_tensors("x.weight", &p, &m1, &m2, &b).unwrap();
    assert_eq!(bits(&back.values), bits(&e.values));
    assert_eq!(bits(&back.moment1), bits(&e.moment1));
    assert_eq!(bits(&back.moment2), bits(&e.moment2));
    assert_eq!(back.shape, e.shape);
    let other = ojas_core::Tensor::zeros(&[4, 3], ojas_core::DType::F32, &b).unwrap();
    assert!(matches!(
        AdapterEntry::from_tensors("x.weight", &p, &other, &m2, &b),
        Err(AdapterError::InvalidEntry { .. })
    ));
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use ojas_core::Backend;
    use ojas_metal::MetalBackend;

    /// Device tensors are read back to host entries bit for bit, and the
    /// entries upload again to the same bits.
    #[test]
    fn metal_twin_device_tensors_round_trip_through_an_entry() {
        let m = match MetalBackend::new(Budget::new(1 << 28)) {
            Ok(m) => m,
            Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
                eprintln!("no Metal device: {e:?}");
                return;
            }
            Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
        };
        let b = Budget::new(1 << 24);
        let e = entry("x.weight".to_string(), &[3, 4], 6);
        let (p, m1, m2) = e.to_tensors(&b).unwrap();
        let (dp, dm1, dm2) = (
            m.upload(&p).unwrap(),
            m.upload(&m1).unwrap(),
            m.upload(&m2).unwrap(),
        );
        let back = AdapterEntry::from_tensors("x.weight", &dp, &dm1, &dm2, &b).unwrap();
        assert_eq!(bits(&back.values), bits(&e.values));
        assert_eq!(bits(&back.moment1), bits(&e.moment1));
        assert_eq!(bits(&back.moment2), bits(&e.moment2));

        let t = Tmp::new();
        let mut c = checkpoint(3);
        c.entries[2] = back;
        c.entries[2].name = "d17.delta".to_string();
        c.entries[2].shape = vec![3, 4];
        let dir = t.ckpt();
        write_adapter_checkpoint(&dir, &c).unwrap();
        assert_same_bits(&c, &read_adapter_checkpoint(&dir, &expected(&c)).unwrap());
    }
}
