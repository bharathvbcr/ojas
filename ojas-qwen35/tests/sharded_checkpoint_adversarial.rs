//! Hostile and malformed sharded checkpoints against the CPU header
//! pre-flight (`Snapshot::from_dir` / `from_files`, which open the checkpoint
//! through tessl's `CheckpointView`). Every input is either refused with a
//! typed error that names the fault, or accepted with the single-file result;
//! none may panic, hang or be silently accepted. Nothing here touches the GPU.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_io::{write_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_qwen35::{Qwen35TextConfig, Snapshot};

const SHARD_NAMES: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl/tests/fixtures/qwen35_train")
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ojas-qwen35-adv-{name}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The tiny fixture's tensors split alternately into two shard files' bytes;
/// the first vector says which shard (0 or 1) holds each tensor.
fn split_fixture(name: &str) -> (Vec<(String, usize)>, [Vec<u8>; 2]) {
    let tmp = scratch(&format!("{name}-src"));
    let st = SafeTensors::open(&tiny_dir().join("model.safetensors")).unwrap();
    let names: Vec<String> = st.names().map(str::to_string).collect();
    let mut map = Vec::new();
    let mut bytes: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
    for (shard, slot) in bytes.iter_mut().enumerate() {
        let mine: Vec<&String> = names.iter().skip(shard).step_by(2).collect();
        let data: Vec<(Vec<u8>, ojas_io::TensorInfo)> = mine
            .iter()
            .map(|n| (st.read_bytes(n).unwrap(), st.info(n).unwrap().clone()))
            .collect();
        let items: Vec<TensorOut<'_>> = mine
            .iter()
            .zip(&data)
            .map(|(n, (b, info))| TensorOut {
                name: n,
                dtype: info.dtype,
                shape: &info.shape,
                data: b,
            })
            .collect();
        let p = tmp.join("s");
        write_safetensors(&p, &items, &[]).unwrap();
        *slot = std::fs::read(&p).unwrap();
        for n in &mine {
            map.push(((*n).clone(), shard));
        }
    }
    (map, bytes)
}

/// A JSON string literal for `s` (Rust's `{:?}` is not JSON: it writes `\0`).
fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// An index text mapping each tensor to `files[its shard]`.
fn index_text(map: &[(String, usize)], files: [&str; 2]) -> String {
    let body: Vec<String> = map
        .iter()
        .map(|(n, s)| format!("{}: {}", jstr(n), jstr(files[*s])))
        .collect();
    format!(
        "{{\"metadata\": {{\"total_size\": 1}}, \"weight_map\": {{{}}}}}",
        body.join(", ")
    )
}

/// A sharded snapshot directory whose shard files are named `files` and whose
/// index is `index_text(map, files)`.
fn sharded_as(name: &str, files: [&str; 2]) -> PathBuf {
    let (map, bytes) = split_fixture(name);
    let dir = scratch(name);
    for (f, b) in files.iter().zip(&bytes) {
        std::fs::write(dir.join(f), b).unwrap();
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        index_text(&map, files),
    )
    .unwrap();
    std::fs::copy(tiny_dir().join("config.json"), dir.join("config.json")).unwrap();
    dir
}

fn sharded(name: &str) -> PathBuf {
    sharded_as(name, SHARD_NAMES)
}

/// A sharded snapshot with the index text replaced by `index`.
fn sharded_with_index(name: &str, index: &str) -> PathBuf {
    let dir = sharded(name);
    std::fs::write(dir.join("model.safetensors.index.json"), index).unwrap();
    dir
}

fn refused(dir: &Path) -> String {
    Snapshot::from_dir(dir)
        .expect_err("the hostile checkpoint was accepted")
        .to_string()
}

fn refused_naming(dir: &Path, needles: &[&str]) {
    let e = refused(dir);
    assert!(
        needles.iter().any(|n| e.contains(n)),
        "{e:?} lacks any of {needles:?}"
    );
}

fn single_header() -> ojas_qwen35::HeaderReport {
    Snapshot::from_dir(&tiny_dir()).unwrap().header
}

#[test]
fn a_weight_map_shard_name_that_is_not_a_bare_file_name_is_refused() {
    let (map, _) = split_fixture("badname-src");
    for (label, bad) in [
        ("parent", "../x.safetensors"),
        ("absolute", "/etc/passwd"),
        ("subdir", "sub/x.safetensors"),
        ("backslash", "..\\x.safetensors"),
        ("nul", "x\0.safetensors"),
        ("dot", "."),
        ("dotdot", ".."),
        ("empty", ""),
    ] {
        let dir = sharded_with_index(
            &format!("badname-{label}"),
            &index_text(&map, [bad, SHARD_NAMES[1]]),
        );
        let e = refused(&dir);
        assert!(
            e.contains("not a bare file name"),
            "{label}: {e:?} lacks the bare-name refusal"
        );
    }
}

#[test]
fn a_weight_map_name_with_a_newline_is_refused_not_resolved() {
    let (map, _) = split_fixture("newline-src");
    let dir = sharded_with_index(
        "newline",
        &index_text(&map, ["model\n.safetensors", SHARD_NAMES[1]]),
    );
    let e = refused(&dir);
    assert!(e.contains("model"), "{e:?}");
}

#[test]
fn a_parent_directory_shard_is_not_read_even_when_it_exists() {
    let dir = sharded("escape");
    let outside = dir
        .parent()
        .unwrap()
        .join(format!("escape-out-{}.safetensors", std::process::id()));
    std::fs::copy(dir.join(SHARD_NAMES[0]), &outside).unwrap();
    let (map, _) = split_fixture("escape-src");
    let name = format!("../{}", outside.file_name().unwrap().to_str().unwrap());
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        index_text(&map, [&name, SHARD_NAMES[1]]),
    )
    .unwrap();
    let e = refused(&dir);
    std::fs::remove_file(&outside).unwrap();
    assert!(e.contains("not a bare file name"), "{e:?}");
}

#[test]
fn unusual_but_legal_shard_names_load_and_match_the_single_file() {
    let long = format!("{}.safetensors", "s".repeat(240));
    for (label, files) in [
        (
            "spaces",
            ["model shard one.safetensors", "model shard two.safetensors"],
        ),
        (
            "unicode",
            ["modèle-ö-片-1.safetensors", "modèle-ö-片-2.safetensors"],
        ),
        ("long", [long.as_str(), "b.safetensors"]),
        ("no-extension", ["shard-a", "shard-b"]),
        (".hidden", [".a.safetensors", ".b.safetensors"]),
    ] {
        let dir = sharded_as(&format!("legal-{label}"), files);
        let s = Snapshot::from_dir(&dir).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(s.header, single_header(), "{label}");
    }
}

#[test]
fn a_shard_name_longer_than_the_file_system_allows_is_refused() {
    let long = format!("{}.safetensors", "s".repeat(300));
    let (map, _) = split_fixture("toolong-src");
    let dir = sharded_with_index("toolong", &index_text(&map, [&long, SHARD_NAMES[1]]));
    let e = refused(&dir);
    assert!(e.contains("sss"), "{e:?}");
}

#[test]
fn a_shard_in_the_weight_map_that_does_not_exist_is_refused() {
    let dir = sharded("missing");
    std::fs::remove_file(dir.join(SHARD_NAMES[1])).unwrap();
    refused_naming(&dir, &[SHARD_NAMES[1]]);
}

#[test]
fn a_shard_file_no_entry_refers_to_is_ignored_and_the_result_is_unchanged() {
    let dir = sharded("orphan");
    std::fs::copy(
        dir.join(SHARD_NAMES[0]),
        dir.join("model-00003-of-00002.safetensors"),
    )
    .unwrap();
    std::fs::write(dir.join("junk.safetensors"), b"not a safetensors file").unwrap();
    let s = Snapshot::from_dir(&dir).unwrap();
    assert_eq!(s.header, single_header());
}

#[test]
fn a_tensor_named_twice_in_the_weight_map_is_refused() {
    let (map, _) = split_fixture("dupkey-src");
    let text = index_text(&map, SHARD_NAMES);
    // The first entry's key, repeated with the other shard as its value.
    let first = jstr(&map[0].0);
    let dup = text.replacen(
        "\"weight_map\": {",
        &format!(
            "\"weight_map\": {{{first}: {}, ",
            jstr(SHARD_NAMES[1 - map[0].1])
        ),
        1,
    );
    let dir = sharded_with_index("dupkey", &dup);
    refused_naming(&dir, &["duplicate key"]);
}

#[test]
fn a_duplicate_metadata_key_is_refused() {
    let (map, _) = split_fixture("dupmeta-src");
    let text = index_text(&map, SHARD_NAMES).replacen(
        "\"metadata\": {\"total_size\": 1}",
        "\"metadata\": {\"total_size\": 1, \"total_size\": 2}",
        1,
    );
    refused_naming(&sharded_with_index("dupmeta", &text), &["duplicate key"]);
}

#[test]
fn an_index_that_is_not_a_usable_object_is_refused() {
    let (map, _) = split_fixture("badindex-src");
    let good = index_text(&map, SHARD_NAMES);
    let truncated = good[..good.len() / 2].to_string();
    let cases: Vec<(&str, String)> = vec![
        ("empty-file", String::new()),
        ("not-json", "this is not json".into()),
        ("truncated", truncated),
        ("trailing-garbage", format!("{good} garbage")),
        ("top-level-array", "[]".into()),
        ("top-level-null", "null".into()),
        ("empty-object", "{}".into()),
        ("empty-weight-map", "{\"weight_map\": {}}".into()),
        ("weight-map-array", "{\"weight_map\": []}".into()),
        ("weight-map-string", "{\"weight_map\": \"x\"}".into()),
        ("weight-map-null", "{\"weight_map\": null}".into()),
        ("value-not-string", "{\"weight_map\": {\"a\": 1}}".into()),
        (
            "value-is-array",
            "{\"weight_map\": {\"a\": [\"b\"]}}".into(),
        ),
        (
            "metadata-wrong-type",
            "{\"metadata\": 1, \"weight_map\": {\"a\": \"b\"}}".into(),
        ),
        ("nul-byte-in-text", format!("{good}\0")),
        (
            "deep-nesting",
            format!("{}{}", "[".repeat(5000), "]".repeat(5000)),
        ),
        (
            "lone-surrogate",
            "{\"weight_map\": {\"\\ud800\": \"b\"}}".into(),
        ),
    ];
    for (label, text) in cases {
        let dir = sharded_with_index(&format!("badindex-{label}"), &text);
        let e = refused(&dir);
        assert!(!e.is_empty(), "{label}");
        assert!(
            e.contains("model.safetensors.index.json"),
            "{label}: {e:?} does not name the index"
        );
    }
}

#[test]
fn an_oversized_index_is_refused_without_being_parsed() {
    let dir = sharded("bigindex");
    let f = std::fs::File::create(dir.join("model.safetensors.index.json")).unwrap();
    f.set_len(64 << 20).unwrap();
    refused_naming(&dir, &["index cap"]);
}

#[test]
fn an_index_that_is_a_directory_or_a_non_regular_file_is_refused() {
    let dir = sharded("indexdir");
    std::fs::remove_file(dir.join("model.safetensors.index.json")).unwrap();
    std::fs::create_dir(dir.join("model.safetensors.index.json")).unwrap();
    refused_naming(
        &dir,
        &["not a regular file", "model.safetensors.index.json"],
    );
}

#[test]
fn a_shard_that_is_a_directory_is_refused() {
    let dir = sharded("sharddir");
    std::fs::remove_file(dir.join(SHARD_NAMES[0])).unwrap();
    std::fs::create_dir(dir.join(SHARD_NAMES[0])).unwrap();
    refused_naming(&dir, &["not a regular file"]);
}

/// Rewrites `file` with `edit` applied to the dtype and shape of the first
/// tensor whose name starts with `prefix`; bytes are kept as they were.
fn retype(file: &Path, prefix: &str, edit: impl Fn(StDtype, Vec<u64>) -> (StDtype, Vec<u64>)) {
    let st = SafeTensors::open(file).unwrap();
    let names: Vec<String> = st.names().map(str::to_string).collect();
    let data: Vec<(Vec<u8>, ojas_io::TensorInfo)> = names
        .iter()
        .map(|n| (st.read_bytes(n).unwrap(), st.info(n).unwrap().clone()))
        .collect();
    drop(st);
    let mut done = false;
    let edited: Vec<(StDtype, Vec<u64>)> = names
        .iter()
        .zip(&data)
        .map(|(n, (_, i))| {
            if !done && n.starts_with(prefix) {
                done = true;
                edit(i.dtype, i.shape.clone())
            } else {
                (i.dtype, i.shape.clone())
            }
        })
        .collect();
    assert!(
        done,
        "no tensor starts with {prefix:?} in {}",
        file.display()
    );
    let items: Vec<TensorOut<'_>> = names
        .iter()
        .zip(&data)
        .zip(&edited)
        .map(|((n, (b, _)), (d, s))| TensorOut {
            name: n,
            dtype: *d,
            shape: s,
            data: b,
        })
        .collect();
    write_safetensors(file, &items, &[]).unwrap();
}

/// A tower tensor name (first in sort order) and the shard that holds it.
/// `label` keeps each test's scratch directory its own: tests run in parallel.
fn a_tower_tensor(label: &str) -> (String, usize) {
    let c = Qwen35TextConfig::from_file(&tiny_dir().join("config.json")).unwrap();
    let want = format!("{}layers.0.", c.tower_prefix);
    let (map, _) = split_fixture(&format!("tower-{label}"));
    map.into_iter()
        .find(|(n, _)| n.starts_with(&want))
        .expect("no layer 0 tensor in the fixture")
}

/// An integer dtype of the same width as `d`, so the tensor's bytes still
/// tile and only the dtype lies.
fn same_width_integer(d: StDtype, s: Vec<u64>) -> (StDtype, Vec<u64>) {
    let int = match d {
        StDtype::F32 => StDtype::U32,
        StDtype::BF16 | StDtype::F16 => StDtype::U16,
        other => panic!("the fixture tensor is {other:?}, expected a float"),
    };
    (int, s)
}

#[test]
fn a_tower_tensor_with_an_unloadable_dtype_is_refused_sharded_and_single() {
    let (name, shard) = a_tower_tensor("dtype");
    let dir = sharded("dtype");
    retype(&dir.join(SHARD_NAMES[shard]), &name, same_width_integer);
    refused_naming(&dir, &["not a float", &name]);

    let single = scratch("dtype-single");
    std::fs::copy(tiny_dir().join("config.json"), single.join("config.json")).unwrap();
    std::fs::copy(
        tiny_dir().join("model.safetensors"),
        single.join("model.safetensors"),
    )
    .unwrap();
    retype(&single.join("model.safetensors"), &name, same_width_integer);
    refused_naming(&single, &["not a float", &name]);
}

#[test]
fn a_tower_tensor_with_the_wrong_shape_is_refused_sharded_and_single() {
    let (name, shard) = a_tower_tensor("shape");
    // A trailing dimension of 1 keeps the element count and the dtype, so the
    // bytes still tile and only the shape differs from the config's.
    let flat = |d: StDtype, mut s: Vec<u64>| {
        s.push(1);
        (d, s)
    };
    let dir = sharded("shape");
    retype(&dir.join(SHARD_NAMES[shard]), &name, flat);
    refused_naming(&dir, &["has shape", &name]);

    let single = scratch("shape-single");
    std::fs::copy(tiny_dir().join("config.json"), single.join("config.json")).unwrap();
    std::fs::copy(
        tiny_dir().join("model.safetensors"),
        single.join("model.safetensors"),
    )
    .unwrap();
    retype(&single.join("model.safetensors"), &name, flat);
    refused_naming(&single, &["has shape", &name]);
}

/// A one-tensor-free safetensors file: `header` as the JSON, then `data`.
fn raw_shard(file: &Path, declared_len: u64, header: &[u8], data: &[u8]) {
    let mut b = declared_len.to_le_bytes().to_vec();
    b.extend_from_slice(header);
    b.extend_from_slice(data);
    std::fs::write(file, b).unwrap();
}

#[test]
fn a_shard_header_that_is_truncated_or_lies_about_its_length_is_refused() {
    let (_, bytes) = split_fixture("hdr-src");
    let real_len = u64::from_le_bytes(bytes[0][..8].try_into().unwrap()) as usize;
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty-file", Vec::new()),
        ("three-bytes", vec![1, 2, 3]),
        ("length-only", bytes[0][..8].to_vec()),
        ("mid-header", bytes[0][..8 + real_len / 2].to_vec()),
        ("header-only-no-data", bytes[0][..8 + real_len].to_vec()),
        ("data-truncated", bytes[0][..bytes[0].len() - 1].to_vec()),
    ];
    for (label, declared) in [
        ("length-zero", 0u64),
        ("length-u64-max", u64::MAX),
        ("length-2^40", 1 << 40),
        ("length-over-cap", 100_000_001),
        ("length-past-file", bytes[0].len() as u64),
    ] {
        let mut b = bytes[0].clone();
        b[..8].copy_from_slice(&declared.to_le_bytes());
        cases.push((label, b));
    }
    for (label, b) in cases {
        let dir = sharded(&format!("hdr-{label}"));
        std::fs::write(dir.join(SHARD_NAMES[0]), &b).unwrap();
        let e = refused(&dir);
        assert!(
            e.contains(SHARD_NAMES[0]),
            "{label}: {e:?} does not name the shard"
        );
    }
}

#[test]
fn a_header_declaring_more_bytes_than_the_file_holds_is_refused_before_any_allocation() {
    let dir = sharded("hdr-huge-tensor");
    let header =
        br#"{"t":{"dtype":"F32","shape":[1099511627776],"data_offsets":[0,4398046511104]}}"#;
    raw_shard(
        &dir.join(SHARD_NAMES[0]),
        header.len() as u64,
        header,
        &[0; 4],
    );
    let e = refused(&dir);
    assert!(e.contains(SHARD_NAMES[0]), "{e:?}");

    let overflow =
        br#"{"t":{"dtype":"F32","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,4]}}"#;
    raw_shard(
        &dir.join(SHARD_NAMES[0]),
        overflow.len() as u64,
        overflow,
        &[0; 4],
    );
    let e = refused(&dir);
    assert!(e.contains(SHARD_NAMES[0]), "{e:?}");
}

#[test]
fn a_symlink_loop_as_a_shard_is_refused() {
    let dir = sharded("loop-shard");
    std::fs::remove_file(dir.join(SHARD_NAMES[0])).unwrap();
    std::os::unix::fs::symlink("loop-b", dir.join(SHARD_NAMES[0])).unwrap();
    std::os::unix::fs::symlink(SHARD_NAMES[0], dir.join("loop-b")).unwrap();
    refused_naming(&dir, &[SHARD_NAMES[0]]);
}

#[test]
fn a_symlink_loop_as_the_index_is_refused() {
    let dir = sharded("loop-index");
    let index = dir.join("model.safetensors.index.json");
    std::fs::remove_file(&index).unwrap();
    std::os::unix::fs::symlink("loop-b", &index).unwrap();
    std::os::unix::fs::symlink("model.safetensors.index.json", dir.join("loop-b")).unwrap();
    refused_naming(&dir, &["model.safetensors.index.json"]);
}

#[test]
fn a_dangling_shard_symlink_in_an_hf_cache_snapshot_is_refused() {
    let root = scratch("dangling-shard");
    let blobs = root.join("blobs");
    let snap = root.join("snapshots").join("rev");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&snap).unwrap();
    let (map, bytes) = split_fixture("dangling-shard-src");
    std::fs::write(blobs.join("idx"), index_text(&map, SHARD_NAMES)).unwrap();
    std::fs::write(blobs.join("s0"), &bytes[0]).unwrap();
    std::fs::copy(tiny_dir().join("config.json"), blobs.join("cfg")).unwrap();
    for (link, blob) in [
        ("model.safetensors.index.json", "idx"),
        (SHARD_NAMES[0], "s0"),
        (SHARD_NAMES[1], "s1-never-downloaded"),
        ("config.json", "cfg"),
    ] {
        std::os::unix::fs::symlink(Path::new("../../blobs").join(blob), snap.join(link)).unwrap();
    }
    refused_naming(&snap, &[SHARD_NAMES[1]]);
}

#[test]
fn a_shard_symlink_pointing_outside_the_snapshot_is_followed() {
    // tessl follows symlinks with no containment check (a Hugging Face
    // snapshot's files are all links into `../../blobs`, outside the
    // snapshot directory); the index can name only a bare file name, so only
    // an entry already in the snapshot directory can redirect a read. This
    // pins that policy: an out-of-tree link loads and equals the single file.
    let dir = sharded("outside-link");
    let elsewhere = scratch("outside-link-target");
    let real = elsewhere.join("shard-1.safetensors");
    std::fs::rename(dir.join(SHARD_NAMES[1]), &real).unwrap();
    std::os::unix::fs::symlink(&real, dir.join(SHARD_NAMES[1])).unwrap();
    let s = Snapshot::from_dir(&dir).unwrap();
    assert_eq!(s.header, single_header());
}

#[test]
fn a_shard_symlink_to_a_file_that_is_not_a_safetensors_file_is_refused() {
    let dir = sharded("link-to-junk");
    std::fs::remove_file(dir.join(SHARD_NAMES[1])).unwrap();
    std::os::unix::fs::symlink("/dev/null", dir.join(SHARD_NAMES[1])).unwrap();
    refused_naming(&dir, &[SHARD_NAMES[1]]);
}

#[test]
fn a_shard_symlink_to_a_fifo_is_refused_without_blocking() {
    let dir = sharded("fifo");
    let fifo = dir.join(SHARD_NAMES[1]);
    std::fs::remove_file(&fifo).unwrap();
    let status = std::process::Command::new("/usr/bin/mkfifo")
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(status.success());
    refused_naming(&dir, &["not a regular file"]);
}

#[test]
fn a_directory_an_index_path_and_a_single_file_path_agree() {
    let dir = sharded("three-ways");
    let config = dir.join("config.json");
    let want = single_header();

    let by_dir = Snapshot::from_dir(&dir).unwrap();
    assert_eq!(by_dir.header, want);

    let by_index =
        Snapshot::from_files(&config, &dir.join("model.safetensors.index.json")).unwrap();
    assert_eq!(by_index.header, want);

    let by_file = Snapshot::from_files(&config, &tiny_dir().join("model.safetensors")).unwrap();
    assert_eq!(by_file.header, want);

    // An index under another name still resolves, as any `*.json` is an index.
    let renamed = dir.join("custom-name.json");
    std::fs::copy(dir.join("model.safetensors.index.json"), &renamed).unwrap();
    assert_eq!(
        Snapshot::from_files(&config, &renamed).unwrap().header,
        want
    );
}

#[test]
fn a_directory_with_neither_an_index_nor_model_safetensors_is_refused() {
    let dir = scratch("empty-dir");
    std::fs::copy(tiny_dir().join("config.json"), dir.join("config.json")).unwrap();
    refused_naming(&dir, &["model.safetensors"]);
}

#[test]
fn an_index_beside_a_model_safetensors_wins_and_a_bad_index_is_not_bypassed() {
    let dir = sharded("index-wins");
    std::fs::copy(
        tiny_dir().join("model.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(dir.join("model.safetensors.index.json"), "{}").unwrap();
    refused_naming(&dir, &["model.safetensors.index.json"]);
}
