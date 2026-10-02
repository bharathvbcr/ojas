//! The checkpoint directory (`docs/framework-design.md` §4).
//!
//! | File | Content |
//! | :--- | :--- |
//! | [`MODEL_FILE`] | `F32` weights with nanolab names, in [`param_table`] order. Metadata: `ojas.spec`, `ojas.step`, `ojas.run`. [`crate::load_model`] reads it. |
//! | [`OPTIM_FILE`] | `muon.<name>`, `adam_m.<name>`, `adam_v.<name>` for every trained parameter, in table order. Metadata: `ojas.step`, `ojas.run`. |
//! | [`STATE_FILE`] | Checkpoint v1 with empty tensor sections and an empty `rng_state` (the sampler is a keyed permutation; its key is the config's `data_seed`). `config` is the JSON object `{"format","run","spec","train"}`. |
//!
//! [`Trainer::save`] builds the three files in a staging directory and
//! swaps it in with [`ojas_io::replace_dir_with`]. Each tensor is read back
//! with one download and written before the next is read, so the host
//! holds one tensor at a time.
//!
//! [`Trainer::resume_from`] runs [`ojas_io::recover_replaced_dir`], reads
//! the state file, opens both safetensors files, checks that step and run
//! agree across the three, that the saved train config and tokenizer hash
//! equal the caller's, and that the tensor names, dtypes and shapes are
//! exactly the expected ones. Only then are tensors streamed in, one at a
//! time, and the trainer built. A refusal leaves nothing behind.

use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::path::Path;

use ojas_core::{Backend, CheckpointV1, DType, OjasError, OptimizerCheckpoint, Tensor};
use ojas_cpu::OptimGroup;
use ojas_data::TokenBin;
use ojas_io::{
    read_checkpoint_from, recover_replaced_dir, replace_dir_with, write_checkpoint, IoError,
    SafeTensors, SafeTensorsWriter, StDtype, TensorSpec,
};

use crate::json::{flat_object, quote};
use crate::names::{param_table, ParamInfo};
use crate::spec::{ModelSpec, SPEC_METADATA_KEY};
use crate::trainer::{Moments, Slot, TrainConfig, Trainer};

/// Weights file of a checkpoint directory.
pub const MODEL_FILE: &str = "model.safetensors";
/// Optimizer-moment file of a checkpoint directory.
pub const OPTIM_FILE: &str = "optim.safetensors";
/// Checkpoint v1 state file of a checkpoint directory.
pub const STATE_FILE: &str = "state.ojck";
/// Safetensors metadata key holding the completed step count, in decimal.
pub const STEP_METADATA_KEY: &str = "ojas.step";
/// Safetensors metadata key holding the run id.
pub const RUN_METADATA_KEY: &str = "ojas.run";
/// `format` of the state file's config object.
pub const STATE_FORMAT: &str = "ojas-train-v1";
/// Keys of the state file's config object, all strings, all required.
const STATE_KEYS: [&str; 4] = ["format", "run", "spec", "train"];

/// The state file holds no tensors; anything larger is refused unread.
const MAX_STATE_BYTES: u64 = 1 << 20;
/// Bytes converted to little-endian per write.
const LE_CHUNK: usize = 64 * 1024;

fn refuse(detail: String) -> OjasError {
    OjasError::OutOfRange {
        op: "Trainer::resume_from",
        detail,
    }
}

fn layout_error(detail: String) -> OjasError {
    OjasError::Shape {
        op: "Trainer::resume_from",
        detail,
    }
}

fn save_error(detail: String) -> OjasError {
    OjasError::OutOfRange {
        op: "Trainer::save",
        detail,
    }
}

/// The optimizer tensors of one parameter, by its table row. The trainer
/// builds its moments from the same `(trains, group)` pair.
enum MomentNames {
    Frozen,
    Muon(String),
    AdamW(String, String),
}

impl MomentNames {
    fn of(info: &ParamInfo) -> Self {
        match (info.trains, info.group) {
            (false, _) => Self::Frozen,
            (true, OptimGroup::MuonMatrix) => Self::Muon(format!("muon.{}", info.name)),
            (true, OptimGroup::AdamEmbedding | OptimGroup::AdamVector) => Self::AdamW(
                format!("adam_m.{}", info.name),
                format!("adam_v.{}", info.name),
            ),
        }
    }

    fn names(&self) -> Vec<&str> {
        match self {
            Self::Frozen => Vec::new(),
            Self::Muon(n) => vec![n],
            Self::AdamW(m, v) => vec![m, v],
        }
    }
}

/// The state file's `config`: the run id, the spec and the train config.
fn state_config(run: &str, spec_json: &str, cfg: &TrainConfig) -> String {
    format!(
        "{{\"format\":{},\"run\":{},\"spec\":{},\"train\":{}}}",
        quote(STATE_FORMAT),
        quote(run),
        quote(spec_json),
        quote(&cfg.to_json())
    )
}

/// A parsed [`state_config`].
#[derive(Debug, PartialEq)]
struct SavedConfig {
    run: String,
    spec: String,
    train: String,
}

impl SavedConfig {
    /// Exactly the four string keys of [`state_config`], `format` equal to
    /// [`STATE_FORMAT`] and a well-formed run id.
    fn parse(bytes: &[u8]) -> Result<Self, OjasError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| refuse(format!("state config is not UTF-8: {e}")))?;
        let io = |e: IoError| refuse(format!("state config: {}", e.detail()));
        let root = flat_object(text, bytes.len(), &STATE_KEYS).map_err(io)?;
        let take = |key: &str| root.get_str(key).map(str::to_string).map_err(io);
        let format = take("format")?;
        let run = take("run")?;
        let spec = take("spec")?;
        let train = take("train")?;
        if format != STATE_FORMAT {
            return Err(refuse(format!(
                "state format {format:?}, expected {STATE_FORMAT:?}"
            )));
        }
        if !is_run_id(&run) {
            return Err(refuse(format!(
                "run id {run:?} is not 16 lowercase hex digits"
            )));
        }
        Ok(Self { run, spec, train })
    }
}

fn is_run_id(run: &str) -> bool {
    run.len() == 16 && run.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Open `path` read-only without following a final-component symlink
/// ([`ojas_io::open_nofollow`]: also refuses anything but a regular file).
fn open_nofollow(path: &Path) -> Result<File, OjasError> {
    ojas_io::open_nofollow(path).map_err(|e| refuse(e.detail().to_string()))
}

/// The state file, decoded, with every field the directory layout leaves
/// empty checked empty. The state cap is checked on the open file's length
/// before anything is read.
fn read_state(path: &Path) -> Result<CheckpointV1, OjasError> {
    let what = path.display();
    let file = open_nofollow(path)?;
    let len = file
        .metadata()
        .map_err(|e| refuse(format!("{what}: {e}")))?
        .len();
    if len > MAX_STATE_BYTES {
        return Err(refuse(format!(
            "{what}: larger than the {MAX_STATE_BYTES}-byte state cap"
        )));
    }
    let state =
        read_checkpoint_from(file).map_err(|e| refuse(format!("{what}: {}", e.detail())))?;
    let opt = &state.optimizer;
    let sections = [
        ("weights", state.weights.len()),
        ("muon_momentum", opt.muon_momentum.len()),
        ("adamw_first_moment", opt.adamw_first_moment.len()),
        ("adamw_second_moment", opt.adamw_second_moment.len()),
        ("rng_state", state.rng_state.len()),
    ];
    if let Some((name, n)) = sections.iter().find(|(_, n)| *n != 0) {
        return Err(refuse(format!(
            "{what}: {name} has {n} entries; the directory layout keeps it empty"
        )));
    }
    Ok(state)
}

fn open_tensors(path: &Path) -> Result<SafeTensors<'static>, OjasError> {
    let file = open_nofollow(path)?;
    SafeTensors::from_file(file).map_err(|e| refuse(format!("{}: {}", path.display(), e.detail())))
}

/// `file`'s metadata is exactly `expected`.
fn check_metadata(
    what: &str,
    file: &SafeTensors<'_>,
    expected: &[(&str, &str)],
) -> Result<(), OjasError> {
    let found = file.metadata();
    for (key, want) in expected {
        match found.get(*key) {
            Some(got) if got == want => {}
            Some(got) => {
                return Err(refuse(format!(
                    "{what}: {key} is {got:?}, the state file says {want:?}"
                )))
            }
            None => return Err(refuse(format!("{what}: no {key:?} in the metadata"))),
        }
    }
    if found.len() != expected.len() {
        let extra: Vec<&String> = found
            .keys()
            .filter(|k| !expected.iter().any(|(e, _)| e == k))
            .collect();
        return Err(refuse(format!("{what}: unexpected metadata {extra:?}")));
    }
    Ok(())
}

/// `file` holds exactly the tensors `expected`, each `F32` with its shape.
fn check_layout(
    what: &str,
    file: &SafeTensors<'_>,
    expected: &[(&str, &[usize])],
) -> Result<(), OjasError> {
    for (name, shape) in expected {
        let info = file
            .info(name)
            .map_err(|_| layout_error(format!("{what}: missing tensor {name:?}")))?;
        if info.dtype != StDtype::F32 {
            return Err(layout_error(format!(
                "{what}: {name} is {:?}, expected F32",
                info.dtype
            )));
        }
        let same = info.shape.len() == shape.len()
            && info
                .shape
                .iter()
                .zip(shape.iter())
                .all(|(&a, &b)| a == b as u64);
        if !same {
            return Err(layout_error(format!(
                "{what}: {name} has shape {:?}, expected {shape:?}",
                info.shape
            )));
        }
    }
    let count = file.names().count();
    if count != expected.len() {
        let extra: Vec<&str> = file
            .names()
            .filter(|n| !expected.iter().any(|(e, _)| e == n))
            .collect();
        return Err(layout_error(format!(
            "{what}: unexpected tensors {extra:?}"
        )));
    }
    Ok(())
}

/// Write `tensors` as `F32` safetensors at `path`: each one downloaded,
/// written in little-endian pieces of [`LE_CHUNK`] bytes, and dropped
/// before the next is downloaded.
fn write_tensors<B: Backend + ?Sized>(
    backend: &B,
    path: &Path,
    tensors: &[(&str, &Tensor)],
    metadata: &[(&str, &str)],
) -> Result<(), OjasError> {
    let what = path.display();
    let io = |e: IoError| save_error(format!("{what}: {}", e.detail()));
    let shapes: Vec<Vec<u64>> = tensors
        .iter()
        .map(|(_, t)| t.shape().iter().map(|&d| d as u64).collect())
        .collect();
    let specs: Vec<TensorSpec<'_>> = tensors
        .iter()
        .zip(&shapes)
        .map(|((name, _), shape)| TensorSpec {
            name,
            dtype: StDtype::F32,
            shape,
        })
        .collect();
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| save_error(format!("{what}: {e}")))?;
    let mut writer = SafeTensorsWriter::new(BufWriter::new(file), &specs, metadata).map_err(io)?;
    let mut le = Vec::with_capacity(LE_CHUNK);
    for (name, tensor) in tensors {
        if tensor.dtype() != DType::F32 {
            return Err(OjasError::Dtype {
                op: "Trainer::save",
                expected: DType::F32,
                got: tensor.dtype(),
            });
        }
        let host = backend.download(tensor)?;
        for piece in host.f32_slice()?.chunks(LE_CHUNK / 4) {
            le.clear();
            for x in piece {
                le.extend_from_slice(&x.to_le_bytes());
            }
            writer.write(name, &le).map_err(io)?;
        }
        drop(host);
    }
    writer
        .finish()
        .map_err(io)?
        .into_inner()
        .map_err(|e| save_error(format!("{what}: {}", e.error())))?;
    Ok(())
}

impl<B: Backend> Trainer<B> {
    /// Write the checkpoint directory `dir`, replacing any earlier one
    /// whole (see the module docs). A [`crate::TrainState::Poisoned`]
    /// trainer is refused with [`OjasError::Poisoned`]: its parameters are
    /// partly updated. On any error the previous `dir` is left as it was.
    pub fn save(&self, dir: &Path) -> Result<(), OjasError> {
        self.ready()?;
        let backend = self.tape.backend();
        let step = self.step.to_string();
        let spec_json = self.spec.to_json()?;
        let model: Vec<(&str, &Tensor)> = self
            .slots
            .iter()
            .map(|s| (s.info.name.as_str(), &s.value))
            .collect();
        let mut moment_names = Vec::new();
        for slot in &self.slots {
            moment_names.push(MomentNames::of(&slot.info));
        }
        let mut optim: Vec<(&str, &Tensor)> = Vec::new();
        for (slot, names) in self.slots.iter().zip(&moment_names) {
            let tensors: Vec<&Tensor> = match &slot.moments {
                Moments::Frozen => Vec::new(),
                Moments::Muon { momentum } => vec![momentum],
                Moments::AdamW { m, v } => vec![m, v],
            };
            let names = names.names();
            if names.len() != tensors.len() {
                return Err(save_error(format!(
                    "{}: optimizer state disagrees with its table row",
                    slot.info.name
                )));
            }
            optim.extend(names.into_iter().zip(tensors));
        }
        let state = CheckpointV1 {
            config: state_config(&self.run, &spec_json, &self.cfg).into_bytes(),
            tokenizer_hash: self.cfg.tokenizer_hash,
            git_sha: self.cfg.git_sha,
            weights: Vec::new(),
            optimizer: OptimizerCheckpoint::default(),
            step: self.step,
            rng_state: Vec::new(),
            data_cursor: self.cursor,
        };
        let model_meta = [
            (SPEC_METADATA_KEY, spec_json.as_str()),
            (STEP_METADATA_KEY, step.as_str()),
            (RUN_METADATA_KEY, self.run.as_str()),
        ];
        let optim_meta = [
            (STEP_METADATA_KEY, step.as_str()),
            (RUN_METADATA_KEY, self.run.as_str()),
        ];
        // `replace_dir_with` takes an `IoError`; the trainer's own error
        // is kept here and returned as it was.
        let mut failed: Option<OjasError> = None;
        let swapped = replace_dir_with(dir, |stage| {
            let written = write_tensors(backend, &stage.join(MODEL_FILE), &model, &model_meta)
                .and_then(|()| write_tensors(backend, &stage.join(OPTIM_FILE), &optim, &optim_meta))
                .and_then(|()| {
                    write_checkpoint(&stage.join(STATE_FILE), &state)
                        .map_err(|e| save_error(format!("{STATE_FILE}: {}", e.detail())))
                });
            written.map_err(|e| {
                let detail = e.to_string();
                failed = Some(e);
                IoError::new(detail)
            })
        });
        match (swapped, failed) {
            (Ok(()), _) => Ok(()),
            (Err(_), Some(err)) => Err(err),
            (Err(e), None) => Err(save_error(e.detail().to_string())),
        }
    }

    /// A trainer restored from the checkpoint directory `dir` written by
    /// [`Self::save`], continuing the same run on `backend` over `bin`.
    ///
    /// `cfg` must equal the saved train config field for field
    /// ([`TrainConfig::to_json`]), and its `tokenizer_hash` the saved one;
    /// `git_sha` may differ. The spec comes from the directory. Every check
    /// in the module docs runs before the first tensor is read, and the
    /// trainer is built only once all tensors are in place; on a refusal
    /// nothing of the directory stays allocated.
    pub fn resume_from(
        backend: B,
        dir: &Path,
        bin: TokenBin,
        cfg: TrainConfig,
    ) -> Result<Self, OjasError> {
        recover_replaced_dir(dir).map_err(|e| refuse(e.detail().to_string()))?;
        let state = read_state(&dir.join(STATE_FILE))?;
        let saved = SavedConfig::parse(&state.config)?;
        let train = cfg.to_json();
        if saved.train != train {
            return Err(refuse(format!(
                "the saved train config {} differs from the caller's {train}",
                saved.train
            )));
        }
        if state.tokenizer_hash != cfg.tokenizer_hash {
            return Err(refuse(
                "the saved tokenizer hash differs from the caller's".to_string(),
            ));
        }
        let spec = ModelSpec::from_json(&saved.spec)?;
        let model = open_tensors(&dir.join(MODEL_FILE))?;
        let optim = open_tensors(&dir.join(OPTIM_FILE))?;
        let step = state.step.to_string();
        check_metadata(
            MODEL_FILE,
            &model,
            &[
                (SPEC_METADATA_KEY, &saved.spec),
                (STEP_METADATA_KEY, &step),
                (RUN_METADATA_KEY, &saved.run),
            ],
        )?;
        check_metadata(
            OPTIM_FILE,
            &optim,
            &[(STEP_METADATA_KEY, &step), (RUN_METADATA_KEY, &saved.run)],
        )?;
        let sampler = Self::check_setup(&spec, &cfg, &bin, state.data_cursor)?;
        let table = param_table(&spec)?;
        let moment_names: Vec<MomentNames> = table.iter().map(MomentNames::of).collect();
        let weights: Vec<(&str, &[usize])> = table
            .iter()
            .map(|i| (i.name.as_str(), i.shape.as_slice()))
            .collect();
        let moments: Vec<(&str, &[usize])> = table
            .iter()
            .zip(&moment_names)
            .flat_map(|(i, n)| n.names().into_iter().map(|n| (n, i.shape.as_slice())))
            .collect();
        check_layout(MODEL_FILE, &model, &weights)?;
        check_layout(OPTIM_FILE, &optim, &moments)?;

        let budget = backend.budget().clone();
        // One host tensor per parameter: the file is read in bounded chunks
        // and decoded straight into its typed storage (dtype and shape were
        // checked above, so the byte lengths agree), then it is uploaded and
        // dropped. On the CPU the upload of a host tensor shares it, and once
        // `host` is dropped the trainer's copy is the only owner.
        let read = |file: &SafeTensors<'_>, what: &str, name: &str, shape: &[usize]| {
            let host = Tensor::from_le_reader(shape, DType::F32, &budget, |offset, chunk| {
                file.read_into(name, offset, chunk)
                    .map_err(|e| refuse(format!("{what}: {}", e.detail())))
            })?;
            backend.upload(&host)
        };
        let mut slots = Vec::with_capacity(table.len());
        for (info, names) in table.into_iter().zip(moment_names) {
            let value = read(&model, MODEL_FILE, &info.name, &info.shape)?;
            let moments = match names {
                MomentNames::Frozen => Moments::Frozen,
                MomentNames::Muon(n) => Moments::Muon {
                    momentum: read(&optim, OPTIM_FILE, &n, &info.shape)?,
                },
                MomentNames::AdamW(m, v) => Moments::AdamW {
                    m: read(&optim, OPTIM_FILE, &m, &info.shape)?,
                    v: read(&optim, OPTIM_FILE, &v, &info.shape)?,
                },
            };
            slots.push(Slot::new(info, value, moments, &cfg));
        }
        Self::assemble(
            backend,
            spec,
            cfg,
            bin,
            sampler,
            slots,
            state.step,
            state.data_cursor,
            saved.run,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_cpu::{CosineSchedule, LrSchedule, WsdSchedule};

    fn cfg() -> TrainConfig {
        TrainConfig::nanolab(
            4,
            16,
            2,
            7,
            LrSchedule::Wsd(WsdSchedule::new(2, 40, 0.2).unwrap()),
        )
    }

    #[test]
    fn the_train_config_is_flat_json_that_tells_every_field_apart() {
        let base = cfg();
        let root = ojas_io::parse_json(&base.to_json()).unwrap();
        let fields = root.as_object().unwrap();
        let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            keys,
            [
                "accum",
                "adam_lr",
                "batch",
                "chunk_cols",
                "chunk_rows",
                "data_seed",
                "decay_frac",
                "grad_clip",
                "ignore",
                "ignore_index",
                "matrix_lr",
                "on_nonfinite",
                "schedule",
                "seq_len",
                "total_steps",
                "warmup_steps"
            ]
        );
        let cosine = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
        let variants = [
            TrainConfig { accum: 3, ..base },
            TrainConfig {
                adam_lr: 6.000000000000001e-4,
                ..base
            },
            TrainConfig { batch: 5, ..base },
            TrainConfig {
                chunk: ojas_core::CeChunk {
                    rows: 1024,
                    cols: 8191,
                },
                ..base
            },
            TrainConfig {
                chunk: ojas_core::CeChunk {
                    rows: 1023,
                    cols: 8192,
                },
                ..base
            },
            TrainConfig {
                data_seed: 8,
                ..base
            },
            TrainConfig {
                schedule: LrSchedule::Wsd(WsdSchedule::new(2, 40, 0.25).unwrap()),
                ..base
            },
            TrainConfig {
                grad_clip: 0.1,
                ..base
            },
            TrainConfig {
                ignore_index: Some(0),
                ..base
            },
            TrainConfig {
                ignore_index: Some(1),
                ..base
            },
            TrainConfig {
                matrix_lr: 0.02,
                ..base
            },
            TrainConfig {
                on_nonfinite: crate::NonFinitePolicy::SkipBatch,
                ..base
            },
            TrainConfig {
                schedule: cosine,
                ..base
            },
            TrainConfig { seq_len: 8, ..base },
            TrainConfig {
                schedule: LrSchedule::Wsd(WsdSchedule::new(2, 41, 0.2).unwrap()),
                ..base
            },
            TrainConfig {
                schedule: LrSchedule::Wsd(WsdSchedule::new(3, 40, 0.2).unwrap()),
                ..base
            },
        ];
        let mut texts = vec![base.to_json()];
        for v in variants {
            let text = v.to_json();
            ojas_io::parse_json(&text).unwrap();
            assert!(!texts.contains(&text), "{text}");
            texts.push(text);
        }
        // Not part of the text: they have their own state-file fields.
        let other = TrainConfig {
            tokenizer_hash: [1; 32],
            git_sha: [2; 20],
            ..base
        };
        assert_eq!(other.to_json(), base.to_json());
        // `f32` printed as itself, not widened.
        assert!(TrainConfig {
            grad_clip: 0.1,
            ..base
        }
        .to_json()
        .contains("\"grad_clip\":0.1,"));
    }

    #[test]
    fn the_state_config_round_trips_and_refuses_anything_else() {
        let spec = ModelSpec::tiny().to_json().unwrap();
        let text = state_config("0123456789abcdef", &spec, &cfg());
        let saved = SavedConfig::parse(text.as_bytes()).unwrap();
        assert_eq!(
            saved,
            SavedConfig {
                run: "0123456789abcdef".to_string(),
                spec: spec.clone(),
                train: cfg().to_json(),
            }
        );
        let q = |s: &str| quote(s);
        let bad = [
            state_config("0123456789ABCDEF", &spec, &cfg()),
            state_config("0123456789abcde", &spec, &cfg()),
            state_config("0123456789abcdef0", &spec, &cfg()),
            state_config("0123456789abcdeg", &spec, &cfg()),
            text.replace(STATE_FORMAT, "ojas-train-v2"),
            format!(
                "{{\"format\":{},\"run\":{},\"spec\":{}}}",
                q(STATE_FORMAT),
                q("0123456789abcdef"),
                q(&spec)
            ),
            format!("{},\"extra\":\"x\"}}", &text[..text.len() - 1]),
            text.replace("\"format\"", "\"formats\""),
            format!(
                "{{\"format\":{},\"run\":{},\"spec\":{},\"train\":7}}",
                q(STATE_FORMAT),
                q("0123456789abcdef"),
                q(&spec)
            ),
        ];
        for b in bad {
            assert!(SavedConfig::parse(b.as_bytes()).is_err(), "{b}");
        }
        assert!(SavedConfig::parse(&[0xff, 0xfe]).is_err());
    }
}
