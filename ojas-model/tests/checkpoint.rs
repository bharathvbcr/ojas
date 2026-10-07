//! The checkpoint directory (`docs/framework-design.md` §4) and G9: a run
//! saved, dropped and resumed is bit for bit the run that never stopped.
//! Every malformed directory is refused with nothing applied.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use common::{snapshot, token_bin, Fault, Probe, Resident, TempBin, TempDir};
use ojas_core::{Backend, Budget, DataCursor, Numerics, OjasError, Tensor, READBACK_CHUNK_BYTES};
use ojas_cpu::{CosineSchedule, CpuBackend, LrSchedule};
use ojas_data::{SamplerRngState, TokenBin};
use ojas_io::{
    encode_f32_as, read_checkpoint, write_checkpoint, write_safetensors, SafeTensors, StDtype,
    TensorOut,
};
use ojas_model::{
    init_params, load_model, ModelSpec, TrainConfig, TrainState, Trainer, MODEL_FILE, OPTIM_FILE,
    RUN_METADATA_KEY, SPEC_METADATA_KEY, STATE_FILE, STEP_METADATA_KEY,
};

const SEQ: usize = 32;
const BATCH: usize = 2;
const ACCUM: usize = 2;

fn exact(cap: u64) -> CpuBackend {
    CpuBackend::new(Budget::new(cap)).with_numerics(Numerics::Exact)
}

fn config() -> TrainConfig {
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    TrainConfig {
        tokenizer_hash: [7; 32],
        git_sha: [9; 20],
        ..TrainConfig::nanolab(BATCH, SEQ, ACCUM, 1337, schedule)
    }
}

fn host_params(seed: u64) -> Vec<Tensor> {
    init_params(&ModelSpec::tiny(), seed, &Budget::new(1 << 30)).unwrap()
}

fn reopen(bin: &TempBin) -> TokenBin {
    TokenBin::open_headerless(&bin.path).unwrap()
}

/// Every entry under `root` (not following links) with its file bytes.
fn tree(root: &Path) -> Vec<(PathBuf, Option<Vec<u8>>)> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        let kind = fs::symlink_metadata(&path).unwrap();
        if kind.is_dir() {
            out.push((path.clone(), None));
            out.extend(tree(&path));
        } else if kind.file_type().is_symlink() {
            out.push((
                path.clone(),
                Some(
                    fs::read_link(&path)
                        .unwrap()
                        .into_os_string()
                        .into_encoded_bytes(),
                ),
            ));
        } else {
            out.push((path.clone(), Some(fs::read(&path).unwrap())));
        }
    }
    out
}

/// A trainer after `steps` steps from init seed `seed`, saved to `dir`.
fn saved(dir: &Path, seed: u64, steps: usize) -> (TempBin, String) {
    let (tmp, bin) = token_bin(20_000, 256);
    let mut t = Trainer::new(
        exact(1 << 30),
        ModelSpec::tiny(),
        &host_params(seed),
        bin,
        config(),
    )
    .unwrap();
    for _ in 0..steps {
        t.step().unwrap();
    }
    t.save(dir).unwrap();
    (tmp, t.run_id().to_string())
}

/// `resume_from` refuses, the directory tree is untouched, and nothing
/// stays charged to the backend's budget.
fn refused(root: &Path, dir: &Path, bin: &TempBin, cfg: TrainConfig) -> OjasError {
    let before = tree(root);
    let budget = Budget::new(1 << 30);
    let backend = CpuBackend::new(budget.clone()).with_numerics(Numerics::Exact);
    let err = match Trainer::resume_from(backend, dir, reopen(bin), cfg) {
        Ok(_) => panic!("resume_from accepted"),
        Err(err) => err,
    };
    assert_eq!(
        budget.live_bytes().unwrap(),
        0,
        "{err}: memory left charged"
    );
    assert!(tree(root) == before, "{err}: the directory changed");
    err
}

// ------------------------------------------------------- safetensors edits

struct Entry {
    name: String,
    dtype: StDtype,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn read_file(path: &Path) -> (Vec<Entry>, Vec<(String, String)>) {
    let file = SafeTensors::open(path).unwrap();
    let mut entries: Vec<Entry> = file
        .names()
        .map(|name| {
            let info = file.info(name).unwrap();
            Entry {
                name: name.to_string(),
                dtype: info.dtype,
                shape: info.shape.clone(),
                data: file.read_bytes(name).unwrap(),
            }
        })
        .collect();
    entries.sort_by_key(|e| file.info(&e.name).unwrap().begin);
    let meta = file
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    (entries, meta)
}

fn write_file(path: &Path, entries: &[Entry], meta: &[(String, String)]) {
    let items: Vec<TensorOut<'_>> = entries
        .iter()
        .map(|e| TensorOut {
            name: &e.name,
            dtype: e.dtype,
            shape: &e.shape,
            data: &e.data,
        })
        .collect();
    let meta: Vec<(&str, &str)> = meta.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    write_safetensors(path, &items, &meta).unwrap();
}

/// Rewrite one safetensors file of `dir` through `edit`.
fn edit(path: &Path, edit: impl FnOnce(&mut Vec<Entry>, &mut Vec<(String, String)>)) {
    let (mut entries, mut meta) = read_file(path);
    edit(&mut entries, &mut meta);
    write_file(path, &entries, &meta);
}

fn set_meta(meta: &mut [(String, String)], key: &str, value: &str) {
    let slot = meta.iter_mut().find(|(k, _)| k == key).unwrap();
    slot.1 = value.to_string();
}

// --------------------------------------------------------------------- G9

#[test]
fn g9_forty_steps_equal_twenty_saved_dropped_resumed_and_twenty_more() {
    let (tmp, bin) = token_bin(20_000, 256);
    let report = |r: ojas_model::StepReport| (r.loss.to_bits(), r.grad_norm.to_bits(), r.step);

    let mut straight = Trainer::new(
        exact(1 << 30),
        ModelSpec::tiny(),
        &host_params(5),
        bin,
        config(),
    )
    .unwrap();
    let want: Vec<_> = (0..40).map(|_| report(straight.step().unwrap())).collect();

    let dir = TempDir::new();
    let mut first = Trainer::new(
        exact(1 << 30),
        ModelSpec::tiny(),
        &host_params(5),
        reopen(&tmp),
        config(),
    )
    .unwrap();
    assert_eq!(
        first.run_id(),
        straight.run_id(),
        "the run id is a pure function"
    );
    let mut got: Vec<_> = (0..20).map(|_| report(first.step().unwrap())).collect();
    let at_save = snapshot(&first);
    let run = first.run_id().to_string();
    first.save(&dir.ckpt()).unwrap();
    drop(first);
    // The resume below reads its sampler key back from this v1 rng_state.
    let state = read_checkpoint(&dir.ckpt().join(STATE_FILE)).unwrap();
    assert_eq!(
        SamplerRngState::decode(&state.rng_state).unwrap(),
        SamplerRngState {
            seed: config().data_seed
        }
    );

    let mut resumed =
        Trainer::resume_from(exact(1 << 30), &dir.ckpt(), reopen(&tmp), config()).unwrap();
    assert_eq!(snapshot(&resumed), at_save, "the resume restores every bit");
    assert_eq!(resumed.run_id(), run);
    assert_eq!(resumed.state(), TrainState::Ready);
    got.extend((0..20).map(|_| report(resumed.step().unwrap())));

    assert_eq!(got, want, "loss and grad-norm bits");
    assert_eq!(
        snapshot(&resumed),
        snapshot(&straight),
        "params, moments, step, cursor"
    );
    // The cursor is the sampler's: 40 steps of K=2 draws have moved it.
    assert_ne!(resumed.cursor(), DataCursor::default());
}

#[test]
fn the_directory_holds_the_three_files_as_documented() {
    let dir = TempDir::new();
    let (tmp, bin) = token_bin(20_000, 256);
    let mut t = Trainer::new(
        exact(1 << 30),
        ModelSpec::tiny(),
        &host_params(5),
        bin,
        config(),
    )
    .unwrap();
    for _ in 0..3 {
        t.step().unwrap();
    }
    t.save(&dir.ckpt()).unwrap();
    drop(tmp);

    let mut names: Vec<String> = fs::read_dir(dir.ckpt())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, [MODEL_FILE, OPTIM_FILE, STATE_FILE]);
    let parent: Vec<_> = fs::read_dir(&dir.path).unwrap().collect();
    assert_eq!(parent.len(), 1, "no stage or backup is left beside it");

    // The weights load as a model file: nanolab names, the spec, the step.
    let model = SafeTensors::open(&dir.ckpt().join(MODEL_FILE)).unwrap();
    let (spec, params) = load_model(&model, &Budget::new(1 << 30)).unwrap();
    assert_eq!(spec, ModelSpec::tiny());
    for ((info, value), host) in t.params().zip(&params) {
        assert_eq!(
            common::bits(t.backend(), value),
            common::f32_bits(&host.to_f32_vec().unwrap()),
            "{}",
            info.name
        );
    }
    let meta = model.metadata();
    assert_eq!(meta.len(), 3);
    assert_eq!(meta[STEP_METADATA_KEY], "3");
    assert_eq!(meta[RUN_METADATA_KEY], t.run_id());
    assert_eq!(
        meta[SPEC_METADATA_KEY],
        ModelSpec::tiny().to_json().unwrap()
    );

    // Moments under their documented names; the frozen parameter has none.
    let optim = SafeTensors::open(&dir.ckpt().join(OPTIM_FILE)).unwrap();
    let names: Vec<&str> = optim.names().collect();
    assert!(
        names.contains(&"muon.blocks.0.mixer.q_proj.weight"),
        "{names:?}"
    );
    assert!(names.contains(&"adam_m.tok_emb.weight"), "{names:?}");
    assert!(names.contains(&"adam_v.tok_emb.weight"), "{names:?}");
    assert!(
        !names
            .iter()
            .any(|n| n.ends_with("blocks.0.mixer.vr_lambda")),
        "{names:?}"
    );
    assert_eq!(optim.metadata().len(), 2);
    assert_eq!(optim.metadata()[STEP_METADATA_KEY], "3");

    // The state file is checkpoint v1 with empty tensor sections.
    let state = read_checkpoint(&dir.ckpt().join(STATE_FILE)).unwrap();
    assert_eq!(state.step, 3);
    assert_eq!(state.data_cursor, t.cursor());
    assert_eq!(state.tokenizer_hash, [7; 32]);
    assert_eq!(state.git_sha, [9; 20]);
    assert!(state.weights.is_empty());
    // rng_state is the sampler's v1 record, byte for byte.
    assert_eq!(
        state.rng_state,
        SamplerRngState {
            seed: config().data_seed
        }
        .encode()
    );
    let config_text = String::from_utf8(state.config).unwrap();
    assert!(
        config_text.starts_with("{\"format\":\"ojas-train-v1\",\"run\":"),
        "{config_text}"
    );
}

#[test]
fn a_second_save_replaces_the_first_whole() {
    let dir = TempDir::new();
    let (tmp, bin) = token_bin(20_000, 256);
    let mut t = Trainer::new(
        exact(1 << 30),
        ModelSpec::tiny(),
        &host_params(5),
        bin,
        config(),
    )
    .unwrap();
    t.step().unwrap();
    t.save(&dir.ckpt()).unwrap();
    t.step().unwrap();
    t.save(&dir.ckpt()).unwrap();
    assert_eq!(fs::read_dir(&dir.path).unwrap().count(), 1);
    let resumed =
        Trainer::resume_from(exact(1 << 30), &dir.ckpt(), reopen(&tmp), config()).unwrap();
    assert_eq!(snapshot(&resumed), snapshot(&t));
}

// ------------------------------------------------- save: memory and reads

/// Weight and moment tensors of the tiny trainer: every parameter, one
/// Muon momentum per matrix, two AdamW moments per other trained one.
fn tensor_count<B: Backend>(t: &Trainer<B>) -> usize {
    t.params()
        .map(|(info, _)| match t.moments(&info.name).unwrap() {
            ojas_model::MomentsRef::Muon { .. } => 2,
            ojas_model::MomentsRef::AdamW { .. } => 3,
            ojas_model::MomentsRef::Frozen => 1,
        })
        .sum()
}

#[test]
fn save_reads_each_tensor_once_and_holds_one_at_a_time() {
    let dir = TempDir::new();
    let (_tmp, bin) = token_bin(20_000, 256);
    let probe = Probe::new(Resident::new(1 << 30));
    let mut t = Trainer::new(&probe, ModelSpec::tiny(), &host_params(5), bin, config()).unwrap();
    t.step().unwrap();
    probe.take_download_live();
    let downloads = probe.count("download");
    let budget = probe.budget().clone();
    let baseline = budget.live_bytes().unwrap();
    let (calls, _) = budget.device_readbacks();

    t.save(&dir.ckpt()).unwrap();

    let n = tensor_count(&t);
    assert_eq!(
        probe.count("download") - downloads,
        n,
        "one download per tensor"
    );
    assert_eq!(budget.device_readbacks().0 - calls, n as u64);
    let live = probe.take_download_live();
    assert_eq!(live.len(), n);
    assert!(
        live.iter().all(|&l| l == baseline),
        "a host copy outlived its write: {live:?} vs {baseline}"
    );
    assert_eq!(budget.live_bytes().unwrap(), baseline);
}

#[test]
fn resume_holds_one_host_tensor_at_a_time() {
    let dir = TempDir::new();
    let (tmp, bin) = token_bin(20_000, 256);
    let mut t = Trainer::new(
        Resident::new(1 << 30),
        ModelSpec::tiny(),
        &host_params(5),
        bin,
        config(),
    )
    .unwrap();
    t.step().unwrap();
    t.save(&dir.ckpt()).unwrap();
    let n = tensor_count(&t);
    let want = snapshot(&t);
    drop(t);

    let probe = Probe::new(Resident::new(1 << 30));
    let base = probe.budget().live_bytes().unwrap();
    let resumed = Trainer::resume_from(&probe, &dir.ckpt(), reopen(&tmp), config()).unwrap();
    let uploads = probe.take_upload_live();
    // The checkpoint's tensors, then the RoPE table.
    assert!(
        uploads.len() > n,
        "{} uploads for {n} tensors",
        uploads.len()
    );
    let mut held = base;
    for (i, &(live, bytes)) in uploads[..n].iter().enumerate() {
        assert!(
            live <= held + bytes,
            "upload {i}: {live} live bytes, {held} held plus {bytes} for this tensor"
        );
        held += bytes;
    }
    assert_eq!(snapshot(&resumed), want);
}

#[test]
fn save_fits_in_one_tensor_plus_one_readback_piece_of_host_headroom() {
    // Measure what a fresh trainer holds, then rebuild it under a cap that
    // leaves exactly the largest tensor (the embedding) free, plus the one
    // readback piece `Tensor::to_host` charges while it decodes the device
    // bytes into typed host storage (at most READBACK_CHUNK_BYTES; for the
    // tiny spec the whole 64 KiB embedding is one piece).
    let spec = ModelSpec::tiny();
    let embedding = (spec.vocab * spec.n_embd * 4) as u64;
    let largest = embedding + embedding.min(READBACK_CHUNK_BYTES as u64);
    let held = {
        let (_tmp, bin) = token_bin(20_000, 256);
        let t = Trainer::new(Resident::new(1 << 30), spec, &host_params(5), bin, config()).unwrap();
        t.backend().budget().live_bytes().unwrap()
    };
    for (slack, fits) in [(0u64, true), (4, false)] {
        let cap = held + largest - slack;
        let dir = TempDir::new();
        let (_tmp, bin) = token_bin(20_000, 256);
        let t = Trainer::new(Resident::new(cap), spec, &host_params(5), bin, config()).unwrap();
        assert_eq!(t.backend().budget().live_bytes().unwrap(), held);
        let result = t.save(&dir.ckpt());
        if fits {
            result.unwrap();
        } else {
            let err = result.unwrap_err();
            assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
            assert!(!dir.ckpt().exists(), "a failed save left a directory");
            assert_eq!(
                fs::read_dir(&dir.path).unwrap().count(),
                0,
                "a stage was left"
            );
        }
    }
}

#[test]
fn a_poisoned_trainer_refuses_to_save_and_keeps_the_last_checkpoint() {
    let dir = TempDir::new();
    let (tmp, bin) = token_bin(20_000, 256);
    let probe = Probe::new(exact(1 << 30));
    let mut t = Trainer::new(&probe, ModelSpec::tiny(), &host_params(5), bin, config()).unwrap();
    t.step().unwrap();
    t.save(&dir.ckpt()).unwrap();
    let before = tree(&dir.path);
    probe.fail("muon_ns5_step", 0, Fault::Backend);
    t.step().unwrap_err();
    assert_eq!(t.state(), TrainState::Poisoned);
    let err = t.save(&dir.ckpt()).unwrap_err();
    assert!(matches!(err, OjasError::Poisoned), "{err:?}");
    assert!(
        tree(&dir.path) == before,
        "the last good checkpoint changed"
    );
    let other = dir.path.join("other");
    assert!(matches!(t.save(&other), Err(OjasError::Poisoned)));
    assert!(!other.exists());
    let resumed =
        Trainer::resume_from(exact(1 << 30), &dir.ckpt(), reopen(&tmp), config()).unwrap();
    assert_eq!(resumed.step_count(), 1);
}

#[test]
fn dropping_a_trainer_adds_no_wait() {
    let (_tmp, bin) = token_bin(20_000, 256);
    let probe = Probe::new(exact(1 << 30));
    let mut t = Trainer::new(&probe, ModelSpec::tiny(), &host_params(5), bin, config()).unwrap();
    t.step().unwrap();
    let syncs = probe.count("sync");
    let downloads = probe.count("download");
    drop(t);
    assert_eq!(probe.count("sync"), syncs, "drop synchronised the backend");
    assert_eq!(probe.count("download"), downloads, "drop read back");
}

// ------------------------------------------------------- resume refusals

#[test]
fn a_step_that_disagrees_across_the_files_is_refused() {
    type Edit = fn(&Path);
    let cases: [(&str, Edit); 5] = [
        ("model step", |d| {
            edit(&d.join(MODEL_FILE), |_, m| {
                set_meta(m, STEP_METADATA_KEY, "4")
            })
        }),
        ("optim step", |d| {
            edit(&d.join(OPTIM_FILE), |_, m| {
                set_meta(m, STEP_METADATA_KEY, "2")
            })
        }),
        ("leading zero", |d| {
            edit(&d.join(MODEL_FILE), |_, m| {
                set_meta(m, STEP_METADATA_KEY, "03")
            });
            edit(&d.join(OPTIM_FILE), |_, m| {
                set_meta(m, STEP_METADATA_KEY, "03")
            });
        }),
        ("state step", |d| {
            let path = d.join(STATE_FILE);
            let mut state = read_checkpoint(&path).unwrap();
            state.step += 1;
            write_checkpoint(&path, &state).unwrap();
        }),
        ("no step key", |d| {
            edit(&d.join(OPTIM_FILE), |_, m| {
                m.retain(|(k, _)| k != STEP_METADATA_KEY)
            })
        }),
    ];
    for (what, apply) in cases {
        let root = TempDir::new();
        let (bin, _) = saved(&root.ckpt(), 5, 3);
        apply(&root.ckpt());
        let err = refused(&root.path, &root.ckpt(), &bin, config());
        assert!(
            matches!(err, OjasError::OutOfRange { .. }),
            "{what}: {err:?}"
        );
    }
}

#[test]
fn files_of_another_run_are_refused() {
    for file in [MODEL_FILE, OPTIM_FILE] {
        let root = TempDir::new();
        let (bin, run) = saved(&root.ckpt(), 5, 3);
        let other = root.path.join("other");
        let (_bin2, other_run) = saved(&other, 6, 3);
        assert_ne!(run, other_run, "init seeds 5 and 6 give different runs");
        fs::copy(other.join(file), root.ckpt().join(file)).unwrap();
        let err = refused(&root.path, &root.ckpt(), &bin, config());
        assert!(
            matches!(err, OjasError::OutOfRange { .. }),
            "{file}: {err:?}"
        );
        assert!(err.to_string().contains(&other_run), "{file}: {err}");
    }
}

#[test]
fn a_different_config_or_tokenizer_is_refused_and_git_sha_is_not_compared() {
    let root = TempDir::new();
    let (bin, _) = saved(&root.ckpt(), 5, 3);
    let base = config();
    for cfg in [
        TrainConfig {
            data_seed: 1338,
            ..base
        },
        TrainConfig { accum: 1, ..base },
        TrainConfig {
            matrix_lr: 0.02,
            ..base
        },
        TrainConfig {
            tokenizer_hash: [8; 32],
            ..base
        },
    ] {
        let err = refused(&root.path, &root.ckpt(), &bin, cfg);
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err:?}");
    }
    let t = Trainer::resume_from(
        exact(1 << 30),
        &root.ckpt(),
        reopen(&bin),
        TrainConfig {
            git_sha: [1; 20],
            ..base
        },
    )
    .unwrap();
    assert_eq!(
        t.config().git_sha,
        [1; 20],
        "the next save records the new sha"
    );
}

#[test]
fn a_truncated_file_is_refused_at_every_cut() {
    for file in [OPTIM_FILE, MODEL_FILE, STATE_FILE] {
        let root = TempDir::new();
        let (bin, _) = saved(&root.ckpt(), 5, 3);
        let path = root.ckpt().join(file);
        let whole = fs::read(&path).unwrap();
        let len = whole.len();
        let header_end = if file == STATE_FILE {
            12
        } else {
            8 + u64::from_le_bytes(whole[..8].try_into().unwrap()) as usize
        };
        let mut cuts = vec![0, 4, 8, header_end / 2, header_end - 1, header_end];
        cuts.extend([header_end + 1, (header_end + len) / 2, len - 4, len - 1]);
        cuts.sort_unstable();
        cuts.dedup();
        for cut in cuts {
            fs::write(&path, &whole[..cut]).unwrap();
            let err = refused(&root.path, &root.ckpt(), &bin, config());
            assert!(
                matches!(err, OjasError::OutOfRange { .. }),
                "{file} at {cut}: {err:?}"
            );
        }
        fs::write(&path, &whole).unwrap();
        Trainer::resume_from(exact(1 << 30), &root.ckpt(), reopen(&bin), config()).unwrap();
    }
}

#[test]
fn a_missing_extra_retyped_or_reshaped_tensor_is_refused() {
    type Edit = fn(&mut Vec<Entry>);
    let cases: [(&str, &str, Edit); 8] = [
        ("missing weight", MODEL_FILE, |es| {
            es.retain(|e| e.name != "norm_f.weight");
        }),
        ("missing moment", OPTIM_FILE, |es| {
            es.retain(|e| e.name != "adam_v.tok_emb.weight");
        }),
        ("extra weight", MODEL_FILE, |es| {
            let e = &es[0];
            let copy = Entry {
                name: "lm_head.weight".to_string(),
                dtype: e.dtype,
                shape: e.shape.clone(),
                data: e.data.clone(),
            };
            es.push(copy);
        }),
        ("extra moment", OPTIM_FILE, |es| {
            es.push(Entry {
                name: "muon.blocks.0.mixer.vr_lambda".to_string(),
                dtype: StDtype::F32,
                shape: vec![1],
                data: vec![0; 4],
            });
        }),
        ("renamed moment", OPTIM_FILE, |es| {
            let i = es.iter().position(|e| e.name.starts_with("muon.")).unwrap();
            es[i].name = es[i].name.replacen("muon.", "adam_m.", 1);
        }),
        ("bf16 weight", MODEL_FILE, |es| {
            let e = es.iter_mut().find(|e| e.name == "norm_f.weight").unwrap();
            let values: Vec<f32> = e
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect();
            e.data = encode_f32_as(StDtype::BF16, &values).unwrap();
            e.dtype = StDtype::BF16;
        }),
        ("flattened weight", MODEL_FILE, |es| {
            let e = es.iter_mut().find(|e| e.name == "tok_emb.weight").unwrap();
            e.shape = vec![e.shape.iter().product()];
        }),
        ("transposed moment", OPTIM_FILE, |es| {
            let e = es
                .iter_mut()
                .find(|e| e.name == "adam_m.tok_emb.weight")
                .unwrap();
            e.shape.reverse();
        }),
    ];
    for (what, file, apply) in cases {
        let root = TempDir::new();
        let (bin, _) = saved(&root.ckpt(), 5, 3);
        edit(&root.ckpt().join(file), |es, _| apply(es));
        let err = refused(&root.path, &root.ckpt(), &bin, config());
        assert!(matches!(err, OjasError::Shape { .. }), "{what}: {err:?}");
    }
}

#[test]
fn metadata_that_is_not_exactly_the_documented_set_is_refused() {
    let other_spec = ModelSpec {
        max_seq: 64,
        ..ModelSpec::tiny()
    }
    .to_json()
    .unwrap();
    type Edit = Box<dyn Fn(&Path)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "spec",
            Box::new(move |d: &Path| {
                edit(&d.join(MODEL_FILE), |_, m| {
                    set_meta(m, SPEC_METADATA_KEY, &other_spec)
                })
            }),
        ),
        (
            "extra key",
            Box::new(|d: &Path| {
                edit(&d.join(OPTIM_FILE), |_, m| {
                    m.push(("ojas.note".to_string(), "x".to_string()))
                })
            }),
        ),
        (
            "no run",
            Box::new(|d: &Path| {
                edit(&d.join(MODEL_FILE), |_, m| {
                    m.retain(|(k, _)| k != RUN_METADATA_KEY)
                })
            }),
        ),
        (
            "rng state",
            Box::new(|d: &Path| {
                let path = d.join(STATE_FILE);
                let mut state = read_checkpoint(&path).unwrap();
                state.rng_state = vec![1];
                write_checkpoint(&path, &state).unwrap();
            }),
        ),
        (
            "empty rng state, as written before the v1 layout",
            Box::new(|d: &Path| {
                let path = d.join(STATE_FILE);
                let mut state = read_checkpoint(&path).unwrap();
                state.rng_state = Vec::new();
                write_checkpoint(&path, &state).unwrap();
            }),
        ),
        (
            "rng state version",
            Box::new(|d: &Path| {
                let path = d.join(STATE_FILE);
                let mut state = read_checkpoint(&path).unwrap();
                state.rng_state[0..4].copy_from_slice(&2u32.to_le_bytes());
                write_checkpoint(&path, &state).unwrap();
            }),
        ),
        (
            "rng state generator",
            Box::new(|d: &Path| {
                let path = d.join(STATE_FILE);
                let mut state = read_checkpoint(&path).unwrap();
                state.rng_state[4..8].copy_from_slice(&2u32.to_le_bytes());
                write_checkpoint(&path, &state).unwrap();
            }),
        ),
        (
            "rng state seed",
            Box::new(|d: &Path| {
                let path = d.join(STATE_FILE);
                let mut state = read_checkpoint(&path).unwrap();
                state.rng_state = SamplerRngState {
                    seed: config().data_seed + 1,
                }
                .encode()
                .to_vec();
                write_checkpoint(&path, &state).unwrap();
            }),
        ),
        (
            "state config",
            Box::new(|d: &Path| {
                let path = d.join(STATE_FILE);
                let mut state = read_checkpoint(&path).unwrap();
                state.config = b"{}".to_vec();
                write_checkpoint(&path, &state).unwrap();
            }),
        ),
    ];
    for (what, apply) in cases {
        let root = TempDir::new();
        let (bin, _) = saved(&root.ckpt(), 5, 3);
        apply(&root.ckpt());
        let err = refused(&root.path, &root.ckpt(), &bin, config());
        assert!(
            matches!(err, OjasError::OutOfRange { .. }),
            "{what}: {err:?}"
        );
    }
}

#[test]
fn an_oversized_state_file_is_refused_before_it_is_decoded() {
    let root = TempDir::new();
    let (bin, _) = saved(&root.ckpt(), 5, 3);
    let path = root.ckpt().join(STATE_FILE);
    let mut state = read_checkpoint(&path).unwrap();
    state.config = vec![b' '; 1 << 20];
    write_checkpoint(&path, &state).unwrap();
    let err = refused(&root.path, &root.ckpt(), &bin, config());
    assert!(err.to_string().contains("state cap"), "{err}");
}

#[test]
fn a_cursor_the_sampler_refuses_is_refused() {
    let root = TempDir::new();
    let (bin, _) = saved(&root.ckpt(), 5, 3);
    let path = root.ckpt().join(STATE_FILE);
    let mut state = read_checkpoint(&path).unwrap();
    state.data_cursor.token_index = u64::MAX;
    write_checkpoint(&path, &state).unwrap();
    refused(&root.path, &root.ckpt(), &bin, config());
}

#[test]
fn symlinks_are_refused_for_the_directory_and_its_files() {
    let root = TempDir::new();
    let (bin, _) = saved(&root.ckpt(), 5, 3);
    let link = root.path.join("link");
    std::os::unix::fs::symlink(root.ckpt(), &link).unwrap();
    refused(&root.path, &link, &bin, config());

    let elsewhere = root.path.join("elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    for file in [MODEL_FILE, OPTIM_FILE, STATE_FILE] {
        let real = elsewhere.join(file);
        fs::rename(root.ckpt().join(file), &real).unwrap();
        std::os::unix::fs::symlink(&real, root.ckpt().join(file)).unwrap();
        refused(&root.path, &root.ckpt(), &bin, config());
        fs::remove_file(root.ckpt().join(file)).unwrap();
        fs::rename(&real, root.ckpt().join(file)).unwrap();
    }
    Trainer::resume_from(exact(1 << 30), &root.ckpt(), reopen(&bin), config()).unwrap();
}

#[test]
fn a_crash_between_the_two_renames_is_recovered() {
    let root = TempDir::new();
    let (bin, _) = saved(&root.ckpt(), 5, 3);
    let want = snapshot(
        &Trainer::resume_from(exact(1 << 30), &root.ckpt(), reopen(&bin), config()).unwrap(),
    );
    // The first rename moved the old directory aside; the second never ran.
    let backup = root.path.join(".ckpt.4242.7.old");
    fs::rename(root.ckpt(), &backup).unwrap();
    let t = Trainer::resume_from(exact(1 << 30), &root.ckpt(), reopen(&bin), config()).unwrap();
    assert_eq!(snapshot(&t), want);
    assert!(
        !backup.exists() && root.ckpt().is_dir(),
        "the backup was restored"
    );

    // A stale stage from a crash before the swap is swept.
    let stage = root.path.join(".ckpt.4242.9.stage");
    fs::create_dir(&stage).unwrap();
    fs::write(stage.join(MODEL_FILE), b"partial").unwrap();
    let t = Trainer::resume_from(exact(1 << 30), &root.ckpt(), reopen(&bin), config()).unwrap();
    assert_eq!(snapshot(&t), want);
    assert!(!stage.exists());

    // Two backups and no target: nothing says which is newer.
    let other = root.path.join("other");
    saved(&other, 5, 4);
    fs::rename(root.ckpt(), root.path.join(".ckpt.4242.7.old")).unwrap();
    fs::rename(&other, root.path.join(".ckpt.4242.8.old")).unwrap();
    refused(&root.path, &root.ckpt(), &bin, config());
}

#[test]
fn a_missing_directory_is_refused() {
    let root = TempDir::new();
    let (tmp, _) = token_bin(20_000, 256);
    refused(&root.path, &root.ckpt(), &tmp, config());
}

#[test]
fn a_resumed_run_saves_and_resumes_again() {
    let root = TempDir::new();
    let (bin, run) = saved(&root.ckpt(), 5, 2);
    let mut t = Trainer::resume_from(exact(1 << 30), &root.ckpt(), reopen(&bin), config()).unwrap();
    t.step().unwrap();
    t.save(&root.ckpt()).unwrap();
    let again = Trainer::resume_from(exact(1 << 30), &root.ckpt(), reopen(&bin), config()).unwrap();
    assert_eq!(again.run_id(), run);
    assert_eq!(snapshot(&again), snapshot(&t));
}
