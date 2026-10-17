//! `load_hf_pinned`: the opt-in pin of a frozen base by file name and sha256,
//! on the tiny fixture (`tests/fixtures/qwen35_tiny`, see `qwen35_common`).
//!
//! A refusal must come before any tensor exists: every refusal here also
//! asserts the budget never held a byte (`live` and `peak` both 0).

mod qwen35_common;

use std::path::{Path, PathBuf};

use ojas_core::{Budget, DType, Tensor};
use ojas_io::sha256::file_sha256_hex;
use ojas_io::SafeTensors;
use ojas_model::qwen35::adapters::{AdapterError, BaseIdentity, QuantizerRecord};
use ojas_model::qwen35::{load_hf, load_hf_pinned, Qwen35Params};
use qwen35_common::*;

const MODEL_FILE: &str = "model.safetensors";

fn model_path() -> PathBuf {
    dir().join(MODEL_FILE)
}

fn identity(file_name: &str, sha256: String) -> BaseIdentity {
    BaseIdentity {
        file_name: file_name.to_string(),
        sha256,
        quantizer: None,
    }
}

fn right() -> BaseIdentity {
    identity(MODEL_FILE, file_sha256_hex(&model_path()).unwrap())
}

fn pinned(
    path: &Path,
    expected: &BaseIdentity,
    budget: &Budget,
) -> Result<Qwen35Params<Tensor>, AdapterError> {
    load_hf_pinned(
        &spec(),
        path,
        expected,
        PREFIX,
        DType::F32,
        DType::F32,
        budget,
    )
}

fn nothing_was_charged(budget: &Budget) {
    assert_eq!(budget.live_bytes().unwrap(), 0, "live bytes after refusal");
    assert_eq!(budget.peak_bytes(), 0, "peak bytes after refusal");
}

fn bits(p: &Qwen35Params<Tensor>) -> Vec<Vec<u32>> {
    p.values()
        .into_iter()
        .map(|t| {
            t.to_f32_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("ojas-pinned-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn a_matching_pin_loads_the_same_bits_as_the_unpinned_load() {
    let b = budget();
    let got = pinned(&model_path(), &right(), &b).unwrap();
    let st = SafeTensors::open(&model_path()).unwrap();
    let want = load_hf(&spec(), &st, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    assert_eq!(bits(&got), bits(&want));
    assert!(b.live_bytes().unwrap() > 0);
}

#[test]
fn a_wrong_sha256_is_refused_with_nothing_charged() {
    let mut expected = right();
    let flipped = if expected.sha256.starts_with('0') {
        '1'
    } else {
        '0'
    };
    expected.sha256.replace_range(0..1, &flipped.to_string());
    let b = budget();
    let err = pinned(&model_path(), &expected, &b)
        .map(|_| ())
        .unwrap_err();
    match err {
        AdapterError::BaseMismatch {
            expected: want,
            found,
        } => {
            assert_eq!(want, expected.sha256);
            assert_eq!(found, right().sha256);
        }
        other => panic!("{other:?}"),
    }
    nothing_was_charged(&b);
}

#[test]
fn a_wrong_file_name_is_refused_with_nothing_charged() {
    let expected = identity("other.safetensors", right().sha256);
    let b = budget();
    let err = pinned(&model_path(), &expected, &b)
        .map(|_| ())
        .unwrap_err();
    match err {
        AdapterError::BaseFileMismatch { expected, found } => {
            assert_eq!(expected, "other.safetensors");
            assert_eq!(found, MODEL_FILE);
        }
        other => panic!("{other:?}"),
    }
    nothing_was_charged(&b);
}

#[test]
fn a_malformed_pin_is_refused_before_the_file_is_read() {
    let b = budget();
    let upper = identity(MODEL_FILE, right().sha256.to_uppercase());
    assert!(matches!(
        pinned(&model_path(), &upper, &b).map(|_| ()),
        Err(AdapterError::InvalidField { .. })
    ));
    let short = identity(MODEL_FILE, "abc".to_string());
    assert!(matches!(
        pinned(&model_path(), &short, &b).map(|_| ()),
        Err(AdapterError::InvalidField { .. })
    ));
    let mut empty_quant = right();
    empty_quant.quantizer = Some(QuantizerRecord {
        format: String::new(),
        group_size: "64".to_string(),
        mlx_version: "0.31.2".to_string(),
    });
    assert!(matches!(
        pinned(&model_path(), &empty_quant, &b).map(|_| ()),
        Err(AdapterError::InvalidField { .. })
    ));
    nothing_was_charged(&b);
}

#[test]
fn a_quantizer_record_rides_along_as_given() {
    let mut expected = right();
    expected.quantizer = Some(QuantizerRecord {
        format: "mlx-affine".to_string(),
        group_size: "64".to_string(),
        mlx_version: "0.31.2".to_string(),
    });
    assert!(pinned(&model_path(), &expected, &budget()).is_ok());
}

#[test]
fn a_missing_file_is_unreadable_with_nothing_charged() {
    let scratch = Scratch::new("missing");
    let path = scratch.0.join(MODEL_FILE);
    let b = budget();
    let err = pinned(&path, &right(), &b).map(|_| ()).unwrap_err();
    assert!(matches!(err, AdapterError::Unreadable { .. }), "{err:?}");
    nothing_was_charged(&b);
}

#[test]
fn a_shard_name_is_refused_and_one_of_one_is_not() {
    let scratch = Scratch::new("shards");
    let sha = right().sha256;
    let b = budget();
    for shard in [
        "model-00001-of-00004.safetensors",
        "model.safetensors.index.json",
    ] {
        let path = scratch.0.join(shard);
        std::fs::copy(model_path(), &path).unwrap();
        let err = pinned(&path, &identity(shard, sha.clone()), &b)
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(err, AdapterError::ShardedBase { .. }), "{err:?}");
        nothing_was_charged(&b);
    }
    let one = "model-00001-of-00001.safetensors";
    let path = scratch.0.join(one);
    std::fs::copy(model_path(), &path).unwrap();
    assert!(pinned(&path, &identity(one, sha), &budget()).is_ok());
}

#[cfg(unix)]
#[test]
fn a_snapshot_link_is_judged_by_its_own_name_and_its_targets_bytes() {
    let scratch = Scratch::new("link");
    let blob = scratch.0.join("blob-928acbf1");
    std::fs::copy(model_path(), &blob).unwrap();
    let link = scratch.0.join(MODEL_FILE);
    std::os::unix::fs::symlink(&blob, &link).unwrap();
    assert!(pinned(&link, &right(), &budget()).is_ok());

    let b = budget();
    let err = pinned(&blob, &right(), &b).map(|_| ()).unwrap_err();
    assert!(
        matches!(err, AdapterError::BaseFileMismatch { .. }),
        "{err:?}"
    );
    nothing_was_charged(&b);
}

#[test]
fn unpinned_load_hf_is_unchanged() {
    let st = SafeTensors::open(&model_path()).unwrap();
    let b = budget();
    let a = load_hf(&spec(), &st, PREFIX, DType::F32, DType::F32, &b).unwrap();
    let again = load_hf(&spec(), &st, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    assert_eq!(bits(&a), bits(&again));
    assert!(b.live_bytes().unwrap() > 0);
}
