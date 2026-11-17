//! ft-2ac45473 (c1, c2, c7): what `ResumeState::read` will open, and what a resumed run leaves
//! behind.
//!
//! A checkpoint directory is input. `trainer.json` names the provider files to hash and the
//! host state to read; before this change any path it named was read whole (`/dev/zero` read
//! forever, a FIFO blocked, `..` named the checkpoint itself). Now:
//!
//! - a `provider_files` key is a plain relative name the writer could have produced, the listed
//!   set is exactly the regular files under `provider/`, and each is hashed from one capped,
//!   no-follow handle;
//! - `trainer.json` and `host.f32` are read through the capped regular-file open, and `host.f32`
//!   is exactly the size its entries imply;
//! - a resumed run that writes its own checkpoint removes the one it resumed from, so one resume
//!   point stays.
//!
//! Every read here runs on a thread with a 10 s bound, so against code that blocks or reads
//! without end the test fails instead of hanging. Run against the pre-change reader, the
//! `/dev/zero` case grows memory until the bound; that is why it is its own test.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use ojas_io::json::JsonValue;
use ojas_io::sha256::sha256_hex;
use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::pyjson::{CANONICAL, dumps};
use ojas_train::lappi::recipe::OptimizerRecipe;
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::LrSchedule;
use ojas_train::lappi::step::ParamSpec;
use ojas_train::lappi::trainer::{
    BatchCounts, CheckpointPolicy, ConsumedBatch, Hooks, HostParams, MAX_TRAINER_JSON_BYTES,
    MicroBatchPlan, MicroLoss, Objective, ResumeState, SequenceWork, TrainConfig, TrainError,
    train,
};

// ---------------------------------------------------------------- the run that writes a checkpoint

struct Batch(u64);

impl ConsumedBatch for Batch {
    fn index(&self) -> u64 {
        self.0
    }

    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        prefix.fold(&[&self.0.to_be_bytes()]);
        Ok(())
    }
}

/// Next-token loss with `values.len()` host tensors, each getting a small gradient at `close`.
struct Obj {
    values: Vec<Vec<f32>>,
    grads: Vec<Option<Vec<f32>>>,
}

impl Obj {
    fn new(values: Vec<Vec<f32>>) -> Self {
        let grads = vec![None; values.len()];
        Self { values, grads }
    }
}

impl HostParams for Obj {
    fn entries(&self) -> Vec<ParamSpec> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, v)| ParamSpec::new(format!("host.{i}"), &[v.len()]))
            .collect()
    }

    fn zero_grads(&mut self) {
        self.grads.iter_mut().for_each(|g| *g = None);
    }

    fn grads(&self) -> Vec<Option<&[f32]>> {
        self.grads.iter().map(|g| g.as_deref()).collect()
    }

    fn values(&self) -> Vec<&[f32]> {
        self.values.iter().map(Vec::as_slice).collect()
    }

    fn values_and_grads(&mut self) -> Vec<(&mut [f32], Option<&[f32]>)> {
        self.values
            .iter_mut()
            .zip(self.grads.iter())
            .map(|(v, g)| (v.as_mut_slice(), g.as_deref()))
            .collect()
    }
}

impl Objective for Obj {
    type Batch = Batch;

    fn run_kind(&self) -> &'static str {
        "ft"
    }

    fn plan(&mut self, batch: &Batch) -> Result<MicroBatchPlan, TrainError> {
        let tokens: Vec<u32> = (0..5).map(|k| ((batch.0 * 3 + k * 2) % 8) as u32).collect();
        Ok(MicroBatchPlan {
            sequences: vec![SequenceWork {
                positions: (0..4).collect(),
                targets: tokens[1..].to_vec(),
                tokens,
                hidden_at: Vec::new(),
            }],
            row_scale: 0.25,
            counts: BatchCounts::default(),
        })
    }

    fn hidden_grad(
        &mut self,
        _seq: usize,
        _positions: &[u32],
        _hidden: &[f32],
        _width: usize,
    ) -> Result<Vec<f32>, TrainError> {
        Err(TrainError::Refused("no outside loss".into()))
    }

    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        for (g, v) in self.grads.iter_mut().zip(&self.values) {
            *g = Some(vec![0.05; v.len()]);
        }
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * 0.25,
            channels: Vec::new(),
        })
    }

    fn host(&mut self) -> Option<&mut dyn HostParams> {
        Some(self)
    }

    fn host_ref(&self) -> Option<&dyn HostParams> {
        Some(self)
    }
}

fn scratch(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("ojas-resume-reads-{tag}-{}", std::process::id()));
    if root.exists() {
        std::fs::remove_dir_all(&root).expect("clear scratch");
    }
    root
}

fn cfg(dir: PathBuf) -> TrainConfig {
    let schedule = LrSchedule::new(1e-2, 4, 1, 1e-3).unwrap();
    TrainConfig {
        schedule,
        cap: WallClockCap::new(3600.0).unwrap(),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: schedule.peak_lr(),
            beta2: 0.999,
            lower_layers: None,
        },
        max_grad_norm: 1.0,
        epoch: 0,
        epochs: 1,
        seed: 7,
        checkpoint: Some(CheckpointPolicy { dir, every: 2 }),
        eta: None,
    }
}

/// Two of the schedule's four steps, one checkpoint at step 2; returns that directory.
fn checkpoint_of(host: Vec<Vec<f32>>, dir: PathBuf) -> PathBuf {
    let mut provider = ToyProvider::new(8, 4, 2, 42).unwrap();
    train(
        &mut provider,
        &mut Obj::new(host),
        (0..2).map(|n| Ok::<_, TrainError>(Batch(n))),
        &cfg(dir),
        &MonotonicClock::new(),
        None,
        &mut Hooks::default(),
    )
    .unwrap_or_else(|e| panic!("train: {e}"))
    .final_checkpoint
    .expect("a checkpoint")
}

fn plain_checkpoint(tag: &str) -> (PathBuf, PathBuf) {
    let root = scratch(tag);
    let dir = checkpoint_of(Vec::new(), root.join("run"));
    (root, dir)
}

fn host_checkpoint(tag: &str) -> (PathBuf, PathBuf) {
    let root = scratch(tag);
    let dir = checkpoint_of(vec![vec![0.5; 3]], root.join("run"));
    (root, dir)
}

// ---------------------------------------------------------------- reading under a bound

/// `ResumeState::read` on a thread: a reader that blocks (a FIFO) or never ends (`/dev/zero`)
/// fails the test after 10 s instead of hanging it.
fn read_within(dir: &Path) -> Result<ResumeState, TrainError> {
    let (tx, rx) = mpsc::channel();
    let d = dir.to_path_buf();
    std::thread::spawn(move || {
        let _ = tx.send(ResumeState::read(&d));
    });
    rx.recv_timeout(Duration::from_secs(10))
        .expect("ResumeState::read did not return within 10 s: it opened a FIFO or device")
}

fn refusal(r: Result<ResumeState, TrainError>, why: &str) -> String {
    match r {
        Err(TrainError::Refused(m)) => m,
        Err(e) => panic!("{why}: expected a refusal, got {e:?}"),
        Ok(_) => panic!("{why}: expected a refusal, but the checkpoint was accepted"),
    }
}

// ---------------------------------------------------------------- editing trainer.json

fn json_text(s: &str) -> String {
    dumps(&JsonValue::from(s), CANONICAL).unwrap()
}

/// Replace the value of the top-level member `"<key>":` (an object without nested braces, or a
/// string) in `trainer.json`.
fn set_member(dir: &Path, key: &str, value: &str) {
    let path = dir.join("trainer.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let marker = format!("\"{key}\":");
    let start = text
        .find(&marker)
        .unwrap_or_else(|| panic!("no {key} member"))
        + marker.len();
    let end = match text.as_bytes()[start] {
        b'{' => start + text[start..].find('}').unwrap() + 1,
        b'"' => start + 1 + text[start + 1..].find('"').unwrap() + 1,
        other => panic!("{key} starts with {}", other as char),
    };
    std::fs::write(
        &path,
        format!("{}{}{}", &text[..start], value, &text[end..]),
    )
    .unwrap();
}

fn provider_file(dir: &Path) -> PathBuf {
    dir.join("provider").join("toy-state.f32")
}

fn provider_digest(dir: &Path) -> String {
    sha256_hex(&std::fs::read(provider_file(dir)).unwrap())
}

fn list_provider_files(dir: &Path, entries: &[(&str, &str)]) {
    let body: Vec<String> = entries
        .iter()
        .map(|(k, d)| format!("{}:{}", json_text(k), json_text(d)))
        .collect();
    set_member(dir, "provider_files", &format!("{{{}}}", body.join(",")));
}

#[cfg(unix)]
fn mkfifo(path: &Path) {
    let made = std::process::Command::new("/usr/bin/mkfifo")
        .arg(path)
        .status()
        .expect("mkfifo could not run");
    assert!(made.success(), "mkfifo {path:?} failed");
}

// ---------------------------------------------------------------- c1: provider_files keys

/// The untouched checkpoint reads, so every refusal below is the tampering's.
#[test]
fn an_untouched_checkpoint_reads() {
    let (root, dir) = plain_checkpoint("clean");
    read_within(&dir).unwrap();
    let (root2, dir2) = host_checkpoint("clean-host");
    let state = read_within(&dir2).unwrap();
    assert_eq!(state.host_entries.len(), 1);
    assert_eq!(state.host_values.len(), 1);
    assert_eq!(state.host_values[0].len(), 3);
    std::fs::remove_dir_all(root).ok();
    std::fs::remove_dir_all(root2).ok();
}

/// Expected: each key is refused as "not a plain relative name", whatever it points at.
///
/// Pre-change, `..` was `fs::read` of the checkpoint directory (an `Io` error, not a refusal),
/// `a/./b` and `a//b` and `a/` named a file that does not exist (`Io`), a key with NUL failed
/// inside the OS (`Io`), and the empty key read the provider directory (`Io`): none was a
/// `Refused`, so the match below fails on the old code.
#[test]
fn provider_keys_that_are_not_plain_relative_names_are_refused() {
    let (root, dir) = plain_checkpoint("keys");
    let digest = provider_digest(&dir);
    for key in [
        "..",
        "../run",
        "../../etc/passwd",
        ".",
        "./toy-state.f32",
        "provider/..",
        "toy-state.f32/",
        "toy-state.f32/.",
        "a/./b",
        "a//b",
        "",
        "a\u{0}b",
        "toy-state.f32\u{0}",
    ] {
        list_provider_files(&dir, &[(key, &digest)]);
        let m = refusal(read_within(&dir), &format!("key {key:?}"));
        assert!(m.contains("plain relative name"), "key {key:?}: {m}");
    }
    std::fs::remove_dir_all(root).ok();
}

/// Expected: an absolute key is refused. `Path::join` of an absolute path replaces the base, so
/// pre-change this read `/dev/zero` itself, a regular-file read of an endless device: memory
/// grows until the 10 s bound fails the test.
#[cfg(unix)]
#[test]
fn an_absolute_key_naming_dev_zero_is_refused_unread() {
    let (root, dir) = plain_checkpoint("devzero");
    let digest = provider_digest(&dir);
    for key in ["/dev/zero", "/etc/passwd", "/"] {
        list_provider_files(&dir, &[(key, &digest)]);
        let m = refusal(read_within(&dir), &format!("key {key:?}"));
        assert!(m.contains("plain relative name"), "key {key:?}: {m}");
    }
    std::fs::remove_dir_all(root).ok();
}

/// Expected: a FIFO in `provider/`, listed or not, is refused without being opened.
/// Pre-change `fs::read` of a listed FIFO blocks for a writer that never comes (the 10 s bound
/// fails the test); an unlisted one was never looked at.
#[cfg(unix)]
#[test]
fn a_fifo_in_the_provider_directory_is_refused_not_opened() {
    let (root, dir) = plain_checkpoint("fifo");
    let digest = provider_digest(&dir);
    mkfifo(&dir.join("provider").join("pipe"));
    list_provider_files(&dir, &[("toy-state.f32", &digest), ("pipe", &digest)]);
    let m = refusal(read_within(&dir), "a listed FIFO");
    assert!(m.contains("not a regular file"), "{m}");

    list_provider_files(&dir, &[("toy-state.f32", &digest)]);
    let m = refusal(read_within(&dir), "an unlisted FIFO");
    assert!(m.contains("not a regular file"), "{m}");

    // The real file replaced by a FIFO of the same name.
    std::fs::remove_file(dir.join("provider").join("pipe")).unwrap();
    std::fs::remove_file(provider_file(&dir)).unwrap();
    mkfifo(&provider_file(&dir));
    let m = refusal(read_within(&dir), "the provider file itself a FIFO");
    assert!(m.contains("not a regular file"), "{m}");
    std::fs::remove_dir_all(root).ok();
}

/// Expected: a symbolic link in `provider/` is refused even when it points at a file with the
/// recorded bytes (the digest would match). Pre-change `fs::read` followed it and the checkpoint
/// was accepted.
#[cfg(unix)]
#[test]
fn a_symlinked_provider_file_is_refused() {
    let (root, dir) = plain_checkpoint("symlink");
    let real = provider_file(&dir);
    let elsewhere = root.join("elsewhere.f32");
    std::fs::rename(&real, &elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &real).unwrap();
    let m = refusal(read_within(&dir), "a symlinked provider file");
    assert!(m.contains("symbolic link"), "{m}");

    // A symlinked directory holding the file is refused the same way.
    std::fs::remove_file(&real).unwrap();
    let hidden = root.join("hidden");
    std::fs::create_dir(&hidden).unwrap();
    std::fs::rename(&elsewhere, hidden.join("toy-state.f32")).unwrap();
    std::os::unix::fs::symlink(&hidden, dir.join("provider").join("sub")).unwrap();
    let digest = sha256_hex(&std::fs::read(hidden.join("toy-state.f32")).unwrap());
    list_provider_files(&dir, &[("sub/toy-state.f32", &digest)]);
    let m = refusal(read_within(&dir), "a symlinked provider directory");
    assert!(m.contains("symbolic link"), "{m}");
    std::fs::remove_dir_all(root).ok();
}

/// Expected: `provider/` itself may not be a symbolic link. Without the check the walk would
/// list, and the reader hash, a whole tree from wherever the link points. Pre-change it was
/// followed and accepted.
#[cfg(unix)]
#[test]
fn a_symlinked_provider_directory_itself_is_refused() {
    let (root, dir) = plain_checkpoint("symlink-top");
    let provider = dir.join("provider");
    let moved = root.join("provider-elsewhere");
    std::fs::rename(&provider, &moved).unwrap();
    std::os::unix::fs::symlink(&moved, &provider).unwrap();
    let m = refusal(read_within(&dir), "a symlinked provider directory");
    assert!(m.contains("not a plain directory"), "{m}");
    std::fs::remove_dir_all(root).ok();
}

/// Expected: the listed set must equal the files under `provider/` (the provider loads from that
/// directory). An empty map with a file present, an extra file, and a listed file that is gone
/// are each refused. Pre-change none of the first two was noticed, and the third was an `Io`.
#[test]
fn the_listed_set_must_equal_the_provider_directory() {
    let (root, dir) = plain_checkpoint("set");
    let digest = provider_digest(&dir);

    list_provider_files(&dir, &[]);
    let m = refusal(read_within(&dir), "an empty provider_files");
    assert!(m.contains("unlisted") && m.contains("toy-state.f32"), "{m}");

    list_provider_files(&dir, &[("toy-state.f32", &digest)]);
    std::fs::write(dir.join("provider").join("extra.bin"), b"loaded unchecked").unwrap();
    let m = refusal(read_within(&dir), "an extra file");
    assert!(m.contains("unlisted") && m.contains("extra.bin"), "{m}");
    std::fs::remove_file(dir.join("provider").join("extra.bin")).unwrap();

    list_provider_files(&dir, &[("toy-state.f32", &digest), ("gone.bin", &digest)]);
    let m = refusal(read_within(&dir), "a listed file that is absent");
    assert!(m.contains("absent") && m.contains("gone.bin"), "{m}");

    list_provider_files(&dir, &[("toy-state.f32", &digest)]);
    read_within(&dir).unwrap();
    std::fs::remove_dir_all(root).ok();
}

/// A recorded digest that does not match the file is still refused (unchanged behavior).
#[test]
fn a_wrong_digest_is_still_refused() {
    let (root, dir) = plain_checkpoint("digest");
    let wrong = "0".repeat(64);
    list_provider_files(&dir, &[("toy-state.f32", &wrong)]);
    let m = refusal(read_within(&dir), "a wrong digest");
    assert!(m.contains("does not match its recorded sha256"), "{m}");
    std::fs::remove_dir_all(root).ok();
}

/// A nested key is legitimate: the adapter provider writes `provider/adapters.safetensors`
/// under the provider directory it is given, so the writer records a key with a `/`. The reader
/// accepts what the writer produces. (Acceptance case; the old reader accepted it too.)
#[test]
fn a_nested_key_the_writer_could_produce_is_accepted() {
    let (root, dir) = plain_checkpoint("nested");
    let digest = provider_digest(&dir);
    let nested = dir.join("provider").join("provider");
    std::fs::create_dir(&nested).unwrap();
    std::fs::rename(provider_file(&dir), nested.join("toy-state.f32")).unwrap();
    list_provider_files(&dir, &[("provider/toy-state.f32", &digest)]);
    read_within(&dir).unwrap();
    std::fs::remove_dir_all(root).ok();
}

/// A duplicate key is refused. The JSON parser already refuses duplicate object keys, so this
/// passes on the old code too: it pins the behavior, it does not discriminate.
#[test]
fn a_duplicate_key_is_refused() {
    let (root, dir) = plain_checkpoint("dup");
    let digest = provider_digest(&dir);
    let d = json_text(&digest);
    set_member(
        &dir,
        "provider_files",
        &format!("{{\"toy-state.f32\":{d},\"toy-state.f32\":{d}}}"),
    );
    assert!(read_within(&dir).is_err());
    std::fs::remove_dir_all(root).ok();
}

// ---------------------------------------------------------------- c2: capped reads

/// Expected: a `trainer.json` over the cap is refused naming the cap, from its length, before it
/// is read. Pre-change `read_to_string` read all of it (an `Io` error from the parser's own
/// limit), so the refusal text and the `Refused` kind are what fail on the old code.
#[test]
fn an_over_cap_trainer_json_is_refused_by_name() {
    let (root, dir) = plain_checkpoint("bigjson");
    let path = dir.join("trainer.json");
    let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.set_len(MAX_TRAINER_JSON_BYTES + 1).unwrap();
    drop(f);
    let m = refusal(read_within(&dir), "an over-cap trainer.json");
    assert!(m.contains("trainer.json byte cap"), "{m}");
    assert!(
        m.contains(&format!("{}", MAX_TRAINER_JSON_BYTES + 1)),
        "{m}"
    );
    std::fs::remove_dir_all(root).ok();
}

/// Expected: `trainer.json` that is a directory or a FIFO is refused without blocking.
/// Pre-change a FIFO blocked in `read_to_string`.
#[cfg(unix)]
#[test]
fn a_trainer_json_that_is_not_a_regular_file_is_refused() {
    let (root, dir) = plain_checkpoint("jsonfifo");
    let path = dir.join("trainer.json");
    std::fs::remove_file(&path).unwrap();
    mkfifo(&path);
    let m = refusal(read_within(&dir), "trainer.json as a FIFO");
    assert!(m.contains("not a regular file"), "{m}");
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    let m = refusal(read_within(&dir), "trainer.json as a directory");
    assert!(m.contains("not a regular file"), "{m}");
    std::fs::remove_dir_all(root).ok();
}

fn set_host_sha(dir: &Path, bytes: &[u8]) {
    set_member(dir, "host_sha256", &json_text(&sha256_hex(bytes)));
}

/// Expected: `host.f32` must be exactly the size its entries imply, judged from the handle's
/// length. With the recorded digest made consistent with the tampered file (so the digest cannot
/// be what refuses it), one byte short and one byte long are both refused. Pre-change one byte
/// LONG was accepted: the trailing byte fell off `as_chunks::<4>()` and every entry still
/// found its floats. (One byte short was refused before too, as "shorter than its entries".)
#[test]
fn host_f32_one_byte_short_or_long_is_refused() {
    let (root, dir) = host_checkpoint("hostsize");
    let path = dir.join("host.f32");
    let good = std::fs::read(&path).unwrap();
    assert_eq!(good.len(), 3 * 3 * 4, "value, m and v of a 3-float entry");

    let mut long = good.clone();
    long.push(0);
    std::fs::write(&path, &long).unwrap();
    set_host_sha(&dir, &long);
    let m = refusal(read_within(&dir), "host.f32 one byte long");
    assert!(m.contains("longer than its entries"), "{m}");

    let short = &good[..good.len() - 1];
    std::fs::write(&path, short).unwrap();
    set_host_sha(&dir, short);
    let m = refusal(read_within(&dir), "host.f32 one byte short");
    assert!(m.contains("shorter than its entries"), "{m}");

    std::fs::write(&path, &good).unwrap();
    set_host_sha(&dir, &good);
    read_within(&dir).unwrap();
    std::fs::remove_dir_all(root).ok();
}

/// Expected: a symlinked or FIFO `host.f32` is refused, the FIFO without blocking. Pre-change
/// the symlink was followed and the FIFO blocked.
#[cfg(unix)]
#[test]
fn host_f32_that_is_a_link_or_a_fifo_is_refused() {
    let (root, dir) = host_checkpoint("hostlink");
    let path = dir.join("host.f32");
    let elsewhere = root.join("host-elsewhere.f32");
    std::fs::rename(&path, &elsewhere).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &path).unwrap();
    let m = refusal(read_within(&dir), "host.f32 as a symlink");
    assert!(m.contains("symbolic link"), "{m}");

    std::fs::remove_file(&path).unwrap();
    mkfifo(&path);
    let m = refusal(read_within(&dir), "host.f32 as a FIFO");
    assert!(m.contains("not a regular file"), "{m}");
    std::fs::remove_dir_all(root).ok();
}

/// A host shape whose byte size overflows `u64` is refused as such, not wrapped or allocated.
#[test]
fn host_entries_whose_sizes_overflow_are_refused() {
    let (root, dir) = host_checkpoint("hostoverflow");
    let path = dir.join("trainer.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let patched = text.replace("\"shape\":[3]", "\"shape\":[18446744073709551615]");
    assert_ne!(text, patched, "the shape member was not found");
    std::fs::write(&path, patched).unwrap();
    let m = refusal(read_within(&dir), "an overflowing host shape");
    assert!(m.contains("overflow"), "{m}");
    std::fs::remove_dir_all(root).ok();
}

// ---------------------------------------------------------------- c7: one resume point

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    v.sort();
    v
}

fn resume(ckpt: &Path, dir: PathBuf) -> Result<PathBuf, TrainError> {
    let state = ResumeState::read(ckpt)?;
    let mut provider = ToyProvider::new(8, 4, 2, 42).unwrap();
    train(
        &mut provider,
        &mut Obj::new(Vec::new()),
        (0..4).map(|n| Ok::<_, TrainError>(Batch(n))),
        &cfg(dir),
        &MonotonicClock::new(),
        Some(state),
        &mut Hooks::default(),
    )
    .map(|r| r.final_checkpoint.expect("a checkpoint"))
}

/// Bug: `last_checkpoint` starts as `None` in every call, so the first checkpoint a resumed run
/// writes passed `previous = None` and the checkpoint it resumed from was never removed: the
/// policy directory held two resume points, the old one and the new one, though the writer's
/// contract is one. Expected: after a resumed run writes its checkpoint, only that one is left.
/// Pre-change both `ckpt-00000002` and `ckpt-00000004` were there.
#[test]
fn a_resumed_run_leaves_one_resume_point() {
    let root = scratch("onepoint");
    let dir = root.join("run");
    let first = checkpoint_of(Vec::new(), dir.clone());
    assert_eq!(names(&dir), ["ckpt-00000002"]);
    let last = resume(&first, dir.clone()).unwrap();
    assert_eq!(last, dir.join("ckpt-00000004"));
    assert_eq!(names(&dir), ["ckpt-00000004"], "two resume points remain");
    // The new checkpoint is whole and resumable.
    ResumeState::read(&last).unwrap();
    std::fs::remove_dir_all(root).ok();
}

/// A checkpoint resumed from outside the policy directory (an archive, a copy) is not this
/// call's to delete: both it and the new checkpoint stay.
#[test]
fn a_checkpoint_resumed_from_elsewhere_is_never_removed() {
    let root = scratch("archive");
    let archive = checkpoint_of(Vec::new(), root.join("archive"));
    let live = root.join("live");
    let last = resume(&archive, live.clone()).unwrap();
    assert_eq!(names(&live), ["ckpt-00000004"]);
    assert!(
        archive.join("trainer.json").is_file(),
        "the archive was removed"
    );
    ResumeState::read(&last).unwrap();
    std::fs::remove_dir_all(root).ok();
}
