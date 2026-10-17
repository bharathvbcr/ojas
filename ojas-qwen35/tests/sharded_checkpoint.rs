//! A sharded snapshot loads: the tiny fixture split across two shards with a
//! `model.safetensors.index.json`, and the same split laid out as a Hugging
//! Face cache snapshot (symlinks into `blobs/`). The CPU tests check the
//! header pre-flight over every shard; the ignored GPU test shows the sharded
//! load holds exactly the single-file parameters and gives the same step
//! (loss bits and every gradient bit).
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test sharded_checkpoint -- --ignored --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_core::Budget;
use ojas_io::{write_safetensors, SafeTensors, TensorOut};
use ojas_qwen35::{Numerics, Precision, Qwen35Step, Sequence, Snapshot, Which};

fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl/tests/fixtures/qwen35_train")
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ojas-qwen35-shard-{name}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The tiny fixture's tensors split alternately into two shard files written
/// to `blobs_dir` as `shard-0`, `shard-1`; returns the `weight_map` entries
/// (tensor name, shard file name the index uses).
fn write_shards(blobs_dir: &Path) -> Vec<(String, String)> {
    let st = SafeTensors::open(&tiny_dir().join("model.safetensors")).unwrap();
    let names: Vec<String> = st.names().map(str::to_string).collect();
    let mut map = Vec::new();
    for shard in 0..2usize {
        let mine: Vec<&String> = names.iter().skip(shard).step_by(2).collect();
        let data: Vec<(Vec<u8>, ojas_io::TensorInfo)> = mine
            .iter()
            .map(|n| (st.read_bytes(n).unwrap(), st.info(n).unwrap().clone()))
            .collect();
        let items: Vec<TensorOut<'_>> = mine
            .iter()
            .zip(&data)
            .map(|(n, (bytes, info))| TensorOut {
                name: n,
                dtype: info.dtype,
                shape: &info.shape,
                data: bytes,
            })
            .collect();
        write_safetensors(&blobs_dir.join(format!("shard-{shard}")), &items, &[]).unwrap();
        for n in &mine {
            map.push((
                (*n).clone(),
                format!("model-0000{}-of-00002.safetensors", shard + 1),
            ));
        }
    }
    map
}

/// An index over `entries`; `extra` is spliced in as top-level fields.
fn index_json(entries: &[(String, String)], extra: &str) -> String {
    let body: Vec<String> = entries
        .iter()
        .map(|(n, f)| format!("{n:?}: {f:?}"))
        .collect();
    format!(
        "{{{extra}\"metadata\": {{\"total_size\": 1}}, \"weight_map\": {{{}}}}}",
        body.join(", ")
    )
}

/// A plain sharded directory: config, two shard files, the index.
fn plain_sharded(name: &str) -> PathBuf {
    plain_sharded_with(name, |entries| index_json(entries, ""))
}

/// [`plain_sharded`] with the index text made from the true entries by `make`.
fn plain_sharded_with(
    name: &str,
    make: impl FnOnce(&mut Vec<(String, String)>) -> String,
) -> PathBuf {
    let dir = scratch(name);
    let mut entries = write_shards(&dir);
    let index = make(&mut entries);
    for shard in 0..2 {
        std::fs::rename(
            dir.join(format!("shard-{shard}")),
            dir.join(format!("model-0000{}-of-00002.safetensors", shard + 1)),
        )
        .unwrap();
    }
    std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
    std::fs::copy(tiny_dir().join("config.json"), dir.join("config.json")).unwrap();
    dir
}

/// `models--x/blobs/<hash>` plus `snapshots/<rev>/<name> -> ../../blobs/<hash>`.
fn hf_cache_snapshot(name: &str) -> PathBuf {
    let root = scratch(name);
    let blobs = root.join("blobs");
    let snap = root.join("snapshots").join("0123abcd");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    let index = index_json(&write_shards(&blobs), "");
    let link = |file: &str, blob: &str| {
        std::os::unix::fs::symlink(Path::new("../../blobs").join(blob), snap.join(file)).unwrap();
    };
    std::fs::write(blobs.join("idx-blob"), index).unwrap();
    std::fs::copy(tiny_dir().join("config.json"), blobs.join("cfg-blob")).unwrap();
    link("model-00001-of-00002.safetensors", "shard-0");
    link("model-00002-of-00002.safetensors", "shard-1");
    link("model.safetensors.index.json", "idx-blob");
    link("config.json", "cfg-blob");
    snap
}

#[test]
fn a_two_shard_snapshot_passes_the_header_pre_flight() {
    let dir = plain_sharded("plain-cpu");
    let single = Snapshot::from_dir(&tiny_dir()).unwrap();
    let s = Snapshot::from_dir(&dir).unwrap();
    assert_eq!(s.weights_path, dir.join("model.safetensors.index.json"));
    assert_eq!(s.header, single.header);
}

#[test]
fn an_hf_cache_snapshot_with_symlinked_shards_passes_the_header_pre_flight() {
    let snap = hf_cache_snapshot("hf-cpu");
    let s = Snapshot::from_dir(&snap).unwrap();
    assert_eq!(s.header, Snapshot::from_dir(&tiny_dir()).unwrap().header);
}

#[test]
fn a_tensor_in_two_shards_is_refused() {
    let dir = plain_sharded("dup-cpu");
    std::fs::copy(
        dir.join("model-00001-of-00002.safetensors"),
        dir.join("model-00002-of-00002.safetensors"),
    )
    .unwrap();
    refused_with(&dir, &["is in both", "no such tensor"]);
}

/// Everything tessl's `CheckpointView::open` refuses must be refused by the
/// header pre-flight too: it is the one owner of what an index may say.
/// tessl names the first inconsistency it meets, so a test gives every
/// wording that is right for its fault.
fn refused_with(dir: &Path, needles: &[&str]) {
    let e = Snapshot::from_dir(dir).expect_err(needles[0]).to_string();
    assert!(
        needles.iter().any(|n| e.contains(n)),
        "{e:?} lacks any of {needles:?}"
    );
}

#[test]
fn an_index_with_an_unknown_top_level_field_is_refused() {
    let dir = plain_sharded_with("bogus", |e| index_json(e, "\"bogus\": {}, "));
    refused_with(&dir, &["bogus"]);
}

#[test]
fn a_shard_tensor_missing_from_the_weight_map_is_refused() {
    let dir = plain_sharded_with("unlisted", |e| {
        e.remove(0);
        index_json(e, "")
    });
    refused_with(&dir, &["weight_map does not list"]);
}

#[test]
fn a_weight_map_entry_pointing_at_the_wrong_shard_is_refused() {
    let dir = plain_sharded_with("wrong-shard", |e| {
        e[0].1 = if e[0].1.contains("00001") {
            "model-00002-of-00002.safetensors".into()
        } else {
            "model-00001-of-00002.safetensors".into()
        };
        index_json(e, "")
    });
    refused_with(&dir, &["no such tensor", "is in both"]);
}

#[test]
fn a_dangling_index_symlink_beside_model_safetensors_is_refused() {
    let dir = scratch("dangling");
    std::fs::copy(tiny_dir().join("config.json"), dir.join("config.json")).unwrap();
    std::fs::copy(
        tiny_dir().join("model.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    std::os::unix::fs::symlink(
        "no-such-index.json",
        dir.join("model.safetensors.index.json"),
    )
    .unwrap();
    let e = Snapshot::from_dir(&dir).unwrap_err().to_string();
    assert!(e.contains("model.safetensors.index.json"), "{e:?}");
}

fn open(snap: &Snapshot) -> Qwen35Step {
    Qwen35Step::open(
        snap,
        Numerics::ExactF32,
        Precision::F32,
        Budget::new(1 << 30),
    )
    .unwrap()
}

fn param_bits(s: &Qwen35Step) -> Vec<(String, Vec<u32>)> {
    s.read_table(Which::Parameters)
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

fn one_step(s: &mut Qwen35Step) -> (u64, Vec<(String, Vec<u32>)>) {
    let ids: Vec<u32> = (0..12u32).map(|i| (i * 7 + 3) % 64).collect();
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
    let loss = pending.letter_ce_sum().to_bits();
    s.backward(pending, None).unwrap();
    let grads = s
        .read_table(Which::Gradients)
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
        .collect();
    (loss, grads)
}

#[test]
#[ignore]
fn gpu_sharded_load_equals_the_single_file_load() {
    let mut want = open(&Snapshot::from_dir(&tiny_dir()).unwrap());
    let want_params = param_bits(&want);
    let want_step = one_step(&mut want);
    drop(want);
    for (label, dir) in [
        ("plain", plain_sharded("plain-gpu")),
        ("hf-cache", hf_cache_snapshot("hf-gpu")),
    ] {
        let mut got = open(&Snapshot::from_dir(&dir).unwrap());
        assert_eq!(param_bits(&got), want_params, "{label}: parameters differ");
        assert_eq!(one_step(&mut got), want_step, "{label}: step differs");
    }
}
