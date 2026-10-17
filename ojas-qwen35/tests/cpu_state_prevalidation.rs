//! `load_state` checks every file of every kind before it writes anything,
//! so a malformed state directory is refused with the provider untouched.
//! The checks are `StateIndex::read`'s (the first thing `load_state` does),
//! run here on the CPU against states written by `write_kind`: a cut-short
//! shard of a later kind, a wrong dtype in a later kind, and a file cut short
//! after the index was read (which fails the read, not the process).

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_io::{write_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_qwen35::{
    plan_shards, tower_tensors, write_kind, Kind, Qwen35TextConfig, StateIndex, TensorSpec,
};

fn tiny() -> Qwen35TextConfig {
    Qwen35TextConfig::from_file(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tessl/tests/fixtures/qwen35_train/config.json"),
    )
    .unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ojas-qwen35-preval-{name}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A state at step 3 over a small shard cap, so each kind has several files.
fn state(name: &str) -> (PathBuf, Vec<TensorSpec>, String) {
    let c = tiny();
    let t = tower_tensors(&c);
    let summary = c.canonical_summary();
    let dir = scratch(name);
    let shards = plan_shards(&t, 64 * 1024);
    assert!(shards.len() > 2);
    for kind in Kind::ALL {
        write_kind(&dir, kind, 3, &summary, &t, &shards, |i| {
            Ok(vec![0.5; t[i].numel()])
        })
        .unwrap();
    }
    (dir, t, summary)
}

fn shard_of(dir: &Path, kind: Kind, n: usize) -> PathBuf {
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(&format!("{}-", kind.tag()))
        })
        .collect();
    names.sort();
    names.swap_remove(n)
}

fn cut_short(path: &Path) {
    let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    f.set_len(f.metadata().unwrap().len() - 8).unwrap();
}

#[test]
fn a_cut_short_shard_of_a_later_kind_is_refused_before_any_load() {
    for kind in [Kind::ExpAvg, Kind::ExpAvgSq] {
        let (dir, t, summary) = state(&format!("cut-{}", kind.tag()));
        StateIndex::read(&dir, &t, &summary).unwrap();
        let victim = shard_of(&dir, kind, 1);
        cut_short(&victim);
        let e = StateIndex::read(&dir, &t, &summary)
            .err()
            .unwrap_or_else(|| panic!("{kind:?}: a cut-short shard was accepted"))
            .to_string();
        assert!(
            e.contains(&victim.file_name().unwrap().to_string_lossy().into_owned()),
            "{e}"
        );
    }
}

#[test]
fn a_wrong_dtype_in_a_later_kind_is_refused_before_any_load() {
    let (dir, t, summary) = state("dtype");
    let victim = shard_of(&dir, Kind::ExpAvgSq, 1);
    // The same shard with its first tensor stored as BF16.
    {
        let st = SafeTensors::open(&victim).unwrap();
        let meta: Vec<(String, String)> = st
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let names: Vec<String> = st.names().map(str::to_string).collect();
        let infos: Vec<_> = names.iter().map(|n| st.info(n).unwrap().clone()).collect();
        let bytes: Vec<Vec<u8>> = names
            .iter()
            .enumerate()
            .map(|(k, n)| {
                let raw = st.read_bytes(n).unwrap();
                if k == 0 {
                    raw[..raw.len() / 2].to_vec()
                } else {
                    raw
                }
            })
            .collect();
        let items: Vec<TensorOut<'_>> = names
            .iter()
            .enumerate()
            .map(|(k, n)| TensorOut {
                name: n,
                dtype: if k == 0 {
                    StDtype::BF16
                } else {
                    infos[k].dtype
                },
                shape: &infos[k].shape,
                data: &bytes[k],
            })
            .collect();
        let meta_ref: Vec<(&str, &str)> =
            meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        write_safetensors(&victim, &items, &meta_ref).unwrap();
    }
    let e = StateIndex::read(&dir, &t, &summary)
        .err()
        .expect("a BF16 entry was accepted")
        .to_string();
    assert!(e.contains("expected F32"), "{e}");
}

#[test]
fn a_file_cut_short_after_the_index_was_read_fails_the_read() {
    let (dir, t, summary) = state("after");
    let idx = StateIndex::read(&dir, &t, &summary).unwrap();
    let victim = shard_of(&dir, Kind::ExpAvgSq, 1);
    cut_short(&victim);
    let hit = (0..t.len()).any(|i| idx.read_entry(Kind::ExpAvgSq, i).is_err());
    assert!(hit, "no entry of the cut-short shard failed to read");
    // The other kinds still read.
    idx.read_entry(Kind::Masters, 0).unwrap();
}
