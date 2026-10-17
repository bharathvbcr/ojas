//! Adversarial tests for `check_gpt2_pre_tokenizer` / `load_hf_gpt2_dir`
//! (card ft-9c4ae3c8). A guard that fails open is a break; a `#[ignore =
//! "BREAK: ..."]` test asserts the behavior the guard should have.

use ojas_data::{check_gpt2_pre_tokenizer, load_hf_gpt2_dir};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;

const CAP: u64 = 32 * 1024 * 1024;
const GPT2_PRE: &str =
    r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}"#;

static N: AtomicU64 = AtomicU64::new(0);

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ojas-adv-pretok-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn with_pre(pre: &str) -> String {
    format!(
        r#"{{"version":"1.0","pre_tokenizer":{pre},"model":{{"type":"BPE","vocab":{{}},"merges":[]}}}}"#
    )
}

fn check_bytes(tag: &str, body: &[u8]) -> Result<(), String> {
    let d = scratch(tag);
    let f = d.join("tokenizer.json");
    std::fs::write(&f, body).unwrap();
    let r = check_gpt2_pre_tokenizer(&f).map_err(|e| e.to_string());
    let _ = std::fs::remove_dir_all(d);
    r
}

#[cfg(unix)]
fn write_tables(dir: &Path) {
    std::fs::write(dir.join("vocab.json"), r#"{"a":0}"#).unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
}

// ---- semantics against Hugging Face tokenizers -------------------------------

#[test]
fn gpt2_family_shapes_are_accepted() {
    // openai-community/gpt2 and distilgpt2 ship exactly GPT2_PRE; older files
    // (before `use_regex` existed) omit it; `trim_offsets` is irrelevant to ids.
    for (tag, pre) in [
        ("gpt2", GPT2_PRE),
        (
            "old",
            r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true}"#,
        ),
        (
            "trimfalse",
            r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true}"#,
        ),
        (
            "spaced",
            r#"{  "type" : "ByteLevel" , "add_prefix_space" : false }"#,
        ),
        (
            "escaped type",
            r#"{"type":"ByteLevel","add_prefix_space":false}"#,
        ),
    ] {
        check_bytes(tag, with_pre(pre).as_bytes()).unwrap_or_else(|e| panic!("{tag}: {e}"));
    }
}

#[test]
fn qwen_llama_gemma_bert_shapes_are_refused() {
    let qwen = r#"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"x"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]}"#;
    let gemma = r#"{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false}"#;
    let llama2 = r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"#;
    let bert = r#"{"type":"BertPreTokenizer"}"#;
    let sp_seq = r#"{"type":"Sequence","pretokenizers":[{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
    for (tag, pre) in [
        ("qwen", qwen),
        ("gemma", gemma),
        ("llama2", llama2),
        ("bert", bert),
        ("spseq", sp_seq),
    ] {
        assert!(check_bytes(tag, with_pre(pre).as_bytes()).is_err(), "{tag}");
    }
}

#[test]
fn add_prefix_space_true_and_the_serde_default_are_refused() {
    // tokenizers' ByteLevel Default is add_prefix_space = true: an absent field
    // prepends a space to the text and changes ids.
    for pre in [
        r#"{"type":"ByteLevel","add_prefix_space":true}"#,
        r#"{"type":"ByteLevel"}"#,
        r#"{"type":"ByteLevel","add_prefix_space":null}"#,
        r#"{"type":"ByteLevel","add_prefix_space":"false"}"#,
    ] {
        assert!(
            check_bytes("aps", with_pre(pre).as_bytes()).is_err(),
            "{pre}"
        );
    }
}

/// A `Sequence` of one `ByteLevel(add_prefix_space false, use_regex true)` is
/// GPT-2's pre-tokenizer in HF semantics (Sequence maps each pre-tokenizer
/// over the splits of the one before). The guard refuses it: conservative, no
/// shipped GPT-2-family file has this shape. Recorded as behavior, not a break.
#[test]
fn a_sequence_of_one_gpt2_byte_level_is_refused_conservatively() {
    let pre = format!(r#"{{"type":"Sequence","pretokenizers":[{GPT2_PRE}]}}"#);
    let err = check_bytes("seq1", with_pre(&pre).as_bytes()).unwrap_err();
    assert!(err.contains("Sequence"), "{err}");
}

/// BREAK (low): the guard reads one key of a many-key contract. A GPT-2
/// pre-tokenizer beside a non-null `normalizer` (Qwen's is NFC) is accepted,
/// though ids differ from `encode_ordinary`'s on non-NFC text.
#[test]
fn a_normalizer_beside_a_gpt2_pre_tokenizer_is_refused() {
    let body = format!(
        r#"{{"normalizer":{{"type":"NFC"}},"pre_tokenizer":{GPT2_PRE},"model":{{"type":"BPE","vocab":{{}},"merges":[]}}}}"#
    );
    assert!(check_bytes("nfc", body.as_bytes()).is_err());
}

#[test]
fn a_non_bpe_model_beside_a_gpt2_pre_tokenizer_is_refused() {
    let body = format!(
        r#"{{"normalizer":null,"pre_tokenizer":{GPT2_PRE},"model":{{"type":"Unigram","vocab":[]}}}}"#
    );
    assert!(check_bytes("unigram", body.as_bytes()).is_err());
}

// ---- hostile JSON ---------------------------------------------------------------

#[test]
fn a_duplicate_pre_tokenizer_is_refused_whichever_comes_first() {
    // serde_json (HF) keeps the last duplicate. The guard refuses any
    // duplicate, which is stricter than HF and so cannot fail open.
    let seq = r#"{"type":"Sequence","pretokenizers":[]}"#;
    for (tag, body) in [
        (
            "first-gpt2",
            format!(r#"{{"pre_tokenizer":{GPT2_PRE},"pre_tokenizer":{seq}}}"#),
        ),
        (
            "first-seq",
            format!(r#"{{"pre_tokenizer":{seq},"pre_tokenizer":{GPT2_PRE}}}"#),
        ),
        (
            "escaped",
            format!(r#"{{"pre_tokenizer":{GPT2_PRE},"pre_tokenizer":{seq}}}"#),
        ),
        (
            "inner",
            r#"{"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":true,"add_prefix_space":false}}"#
                .to_string(),
        ),
    ] {
        let err = check_bytes(tag, body.as_bytes()).unwrap_err();
        assert!(err.contains("duplicate"), "{tag}: {err}");
    }
}

#[test]
fn a_nested_or_non_root_pre_tokenizer_does_not_count() {
    for (tag, body) in [
        (
            "nested",
            format!(r#"{{"model":{{"pre_tokenizer":{GPT2_PRE}}}}}"#),
        ),
        ("array", format!(r#"[{{"pre_tokenizer":{GPT2_PRE}}}]"#)),
        ("string", format!(r#""{}""#, GPT2_PRE.replace('"', "\\\""))),
        ("keycase", format!(r#"{{"Pre_Tokenizer":{GPT2_PRE}}}"#)),
    ] {
        assert!(check_bytes(tag, body.as_bytes()).is_err(), "{tag}");
    }
}

#[test]
fn deep_nesting_is_refused_not_a_stack_overflow() {
    let deep = format!("{}{}", "[".repeat(100_000), "]".repeat(100_000));
    let body = format!(r#"{{"pre_tokenizer":{GPT2_PRE},"x":{deep}}}"#);
    let err = check_bytes("deep", body.as_bytes()).unwrap_err();
    assert!(err.contains("not a tokenizer.json"), "{err}");
    let obj = "{\"a\":".repeat(50_000);
    assert!(check_bytes("deepobj", obj.as_bytes()).is_err());
    assert!(check_bytes("deepopen", "[".repeat(1_000_000).as_bytes()).is_err());
}

#[test]
fn bom_nul_invalid_utf8_and_junk_are_refused() {
    let ok = with_pre(GPT2_PRE);
    let mut bom = vec![0xEF, 0xBB, 0xBF];
    bom.extend_from_slice(ok.as_bytes());
    assert!(
        check_bytes("bom", &bom).is_err(),
        "BOM (serde_json refuses it too)"
    );
    let mut nul = ok.clone().into_bytes();
    nul.push(0);
    assert!(check_bytes("nul-tail", &nul).is_err());
    let nul_in = ok.replacen("ByteLevel", "Byte\0Level", 1);
    assert!(check_bytes("nul-in", nul_in.as_bytes()).is_err());
    let mut bad = ok.clone().into_bytes();
    bad.splice(2..2, [0xFF, 0xFE]);
    assert!(check_bytes("utf8", &bad).is_err());
    for (tag, body) in [
        (
            "surrogate",
            format!(r#"{{"x":"\ud800","pre_tokenizer":{GPT2_PRE}}}"#),
        ),
        ("trailing", format!("{ok} x")),
        ("two", format!("{ok}{ok}")),
        ("nan", format!(r#"{{"x":NaN,"pre_tokenizer":{GPT2_PRE}}}"#)),
        ("comment", format!("/*c*/{ok}")),
    ] {
        assert!(check_bytes(tag, body.as_bytes()).is_err(), "{tag}");
    }
}

#[test]
fn a_huge_string_is_bounded_and_never_a_pass_by_accident() {
    let big = "a".repeat(20 * 1024 * 1024);
    let body = format!(r#"{{"x":"{big}","pre_tokenizer":{GPT2_PRE},"model":{{"type":"BPE"}}}}"#);
    check_bytes("bigstr", body.as_bytes()).unwrap();
    let seq =
        format!(r#"{{"x":"{big}","pre_tokenizer":{{"type":"Sequence","pretokenizers":[]}}}}"#);
    assert!(check_bytes("bigstr-seq", seq.as_bytes()).is_err());
}

#[test]
fn the_cap_is_exact_for_accept_and_one_byte_over_is_refused() {
    let mut at = with_pre(GPT2_PRE).into_bytes();
    at.resize(CAP as usize, b' ');
    check_bytes("at-cap", &at).expect("a file of exactly HF_TEXT_CAP bytes is read");
    at.push(b' ');
    let err = check_bytes("over-cap", &at).unwrap_err();
    assert!(err.contains("exceeds"), "{err}");
}

// ---- hostile filesystem ---------------------------------------------------------

#[cfg(unix)]
#[test]
fn symlinks_directories_and_unreadable_files() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let d = scratch("fs");
    let good = d.join("good.json");
    std::fs::write(&good, with_pre(GPT2_PRE)).unwrap();
    let link = d.join("tokenizer.json");
    symlink(&good, &link).unwrap();
    check_gpt2_pre_tokenizer(&link).unwrap();
    std::fs::remove_file(&link).unwrap();
    // Symlink to /dev/zero: bounded, refused.
    symlink("/dev/zero", &link).unwrap();
    let t0 = std::time::Instant::now();
    assert!(check_gpt2_pre_tokenizer(&link).is_err());
    assert!(t0.elapsed() < Duration::from_secs(20));
    std::fs::remove_file(&link).unwrap();
    // A directory named tokenizer.json, through both entries.
    std::fs::create_dir(&link).unwrap();
    assert!(check_gpt2_pre_tokenizer(&link).is_err());
    write_tables(&d);
    assert!(load_hf_gpt2_dir(&d).is_err());
    std::fs::remove_dir(&link).unwrap();
    // Unreadable file (the assertions are skipped for root, which ignores the mode).
    std::fs::write(&link, with_pre(GPT2_PRE)).unwrap();
    std::fs::set_permissions(&link, std::fs::Permissions::from_mode(0o000)).unwrap();
    if std::fs::File::open(&link).is_err() {
        assert!(check_gpt2_pre_tokenizer(&link).is_err());
        assert!(
            load_hf_gpt2_dir(&d).is_err(),
            "unreadable is an error, not a skip"
        );
    }
    std::fs::set_permissions(&link, std::fs::Permissions::from_mode(0o600)).unwrap();
    let _ = std::fs::remove_dir_all(d);
}

/// BREAK (low): `load_hf_gpt2_dir` takes `NotFound` from `metadata` (which
/// follows symlinks) as "no tokenizer.json" and loads the tables unchecked. A
/// dangling `tokenizer.json` symlink is a present-but-unreachable file.
#[cfg(unix)]
#[test]
fn a_dangling_tokenizer_json_symlink_is_an_error() {
    let d = scratch("dangling");
    write_tables(&d);
    std::os::unix::fs::symlink(
        d.join("not-mounted/tokenizer.json"),
        d.join("tokenizer.json"),
    )
    .unwrap();
    assert!(load_hf_gpt2_dir(&d).is_err());
    let _ = std::fs::remove_dir_all(d);
}

/// BREAK (low, availability): a FIFO named tokenizer.json blocks `open(2)`
/// forever when there is no writer. Nothing is accepted, but a loader on an
/// untrusted directory hangs. The test unblocks its own thread.
#[cfg(unix)]
#[test]
fn a_fifo_in_place_of_tokenizer_json_does_not_hang() {
    let d = scratch("fifo");
    let f = d.join("tokenizer.json");
    let st = std::process::Command::new("mkfifo")
        .arg(&f)
        .status()
        .unwrap();
    assert!(st.success());
    let (tx, rx) = mpsc::channel();
    let path = f.clone();
    let h = std::thread::spawn(move || {
        let _ = tx.send(check_gpt2_pre_tokenizer(&path).is_err());
    });
    let got = rx.recv_timeout(Duration::from_secs(5));
    if got.is_ok() {
        h.join().unwrap();
    } else {
        // Only when the check is stuck in `open(2)`: a writer releases it.
        // (Opening a FIFO for writing blocks until a reader exists, so this is
        // never done when the check returned, or the test would hang itself.)
        let _ = std::fs::OpenOptions::new().write(true).open(&f);
        h.join().unwrap();
    }
    let _ = std::fs::remove_dir_all(d);
    assert_eq!(got, Ok(true), "check hung on a FIFO");
}

// ---- real files on this machine --------------------------------------------------

fn find_tokenizer_jsons(root: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 6 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        // `metadata` follows the snapshot symlinks of the HF cache.
        let Ok(m) = std::fs::metadata(&p) else {
            continue;
        };
        if m.is_dir() {
            find_tokenizer_jsons(&p, out, depth + 1);
        } else if p.file_name().is_some_and(|n| n == "tokenizer.json") {
            out.push(p);
        }
    }
}

/// Every real tokenizer.json in the local HF cache (Qwen3.5, Qwen3.8, Qwen
/// image, Gemma-family, Metaspace, Bert) must be refused for the right
/// reason: a pre-tokenizer mismatch, never a parser-limit error, which would
/// mean a legitimate file does not fit the limits. Prints each verdict. There
/// is no GPT-2-family tokenizer.json on this machine to accept.
#[test]
#[ignore = "reads ~/.cache/huggingface/hub (tens of MB each); run with --ignored --nocapture"]
fn every_real_cached_tokenizer_json_is_refused_by_pre_tokenizer_not_limits() {
    let home = std::env::var_os("HOME").expect("HOME");
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    let mut files = Vec::new();
    find_tokenizer_jsons(&hub, &mut files, 0);
    assert!(
        !files.is_empty(),
        "no tokenizer.json under {}",
        hub.display()
    );
    for f in &files {
        let err = check_gpt2_pre_tokenizer(f).unwrap_err().to_string();
        let found: String = err
            .split("pre_tokenizer is")
            .nth(1)
            .unwrap_or(&err)
            .chars()
            .take(70)
            .collect();
        eprintln!("{} -> {found}", f.display());
        assert!(
            err.contains("pre_tokenizer is"),
            "{}: refused for a reason other than the pre-tokenizer: {err}",
            f.display()
        );
    }
}
