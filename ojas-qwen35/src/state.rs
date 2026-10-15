//! Masters and AdamW moments to and from disk, and host copies of any table.
//!
//! A state directory holds three kinds of sharded f32 safetensors (written
//! and read with ojas-io), each tensor under transformers' name below the
//! tower prefix and in transformers' layout (`[out, in]` for a linear layer),
//! so a torch oracle reads them as they are:
//!
//! ```text
//! <dir>/masters-00000-of-00009.safetensors     the parameters
//! <dir>/exp_avg-00000-of-00009.safetensors     AdamW's first moment
//! <dir>/exp_avg_sq-00000-of-00009.safetensors  AdamW's second moment
//! ```
//!
//! Every file's `__metadata__` carries the format, its kind and shard, the
//! AdamW step count and the config's [`crate::config::Qwen35TextConfig::canonical_summary`];
//! a directory whose files disagree on any of these, or that lacks or adds a
//! tensor, is refused before anything is written into the model.
//!
//! Costs on the 2B: tessl's per-entry host-mapped reads and writes
//! (`read_parameter_host`, `write_parameter_host`, `read_gradient_host`,
//! `read_adamw_moment_host`, `write_adamw_moment_host`) move one entry at a
//! time straight between its device buffer and the host, so a save or a load
//! allocates nothing on the device. On the host a save holds one entry
//! at a time: [`write_kind`] streams each entry into its shard through
//! ojas-io's `SafeTensorsWriter` in 1 MiB pieces, so the peak is the largest
//! entry (the 2 GB embedding), not a shard. A load reads one entry at a
//! time. Shards are [`SHARD_BYTES`] unless one tensor is larger (the
//! embedding). A resume opens the base snapshot (tessl's loader is the
//! model's only constructor) and then writes these over it.
//!
//! A [`Precision::Bf16`] provider refuses both: its masters are AdamW's f32
//! copies, which tessl's `read_parameters` does not return (it reads the
//! bf16 weights), so a save would drop them.

use std::fs::File;
use std::io::BufWriter;
use std::ops::Range;
use std::path::{Path, PathBuf};

use ojas_core::Tensor;
use ojas_io::{replace_dir_with, IoError, SafeTensors, SafeTensorsWriter, StDtype};
use tessl::qwen35_adamw::Moment;

use crate::error::{invalid, tessl, Qwen35Error, Result};
use crate::names::TensorSpec;
use crate::step::{BankState, Precision, Qwen35Step};

/// Target bytes per shard file.
pub const SHARD_BYTES: u64 = 1 << 30;

/// The format tag in every state file.
pub const STATE_FORMAT: &str = "ojas-qwen35-state/1";

/// What a state file (or a host copy) holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Masters,
    /// torch's `exp_avg`.
    ExpAvg,
    /// torch's `exp_avg_sq`.
    ExpAvgSq,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Masters, Kind::ExpAvg, Kind::ExpAvgSq];

    pub fn tag(self) -> &'static str {
        match self {
            Kind::Masters => "masters",
            Kind::ExpAvg => "exp_avg",
            Kind::ExpAvgSq => "exp_avg_sq",
        }
    }
}

/// Which table [`Qwen35Step::read_table`] copies to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Parameters,
    Gradients,
    ExpAvg,
    ExpAvgSq,
}

/// One table entry on the host, in transformers' name and layout.
pub struct NamedTensor {
    pub name: String,
    pub tensor: Tensor,
}

/// Consecutive table ranges of about `cap_bytes` of f32 each; an entry larger
/// than the cap gets a shard of its own.
pub fn plan_shards(table: &[TensorSpec], cap_bytes: u64) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut bytes) = (0usize, 0u64);
    for (i, t) in table.iter().enumerate() {
        let b = t.numel() as u64 * 4;
        if i > start && bytes + b > cap_bytes {
            out.push(start..i);
            start = i;
            bytes = 0;
        }
        bytes += b;
    }
    if start < table.len() {
        out.push(start..table.len());
    }
    out
}

/// tessl's storage layout (`[in, out]` for an entry it stores transposed) to
/// transformers' (`[out, in]`), or back.
pub fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if rows.checked_mul(cols) != Some(data.len()) {
        return Err(invalid(
            "transpose_2d",
            format!("{} values for [{rows}, {cols}]", data.len()),
        ));
    }
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    Ok(out)
}

/// Storage values of `spec` (read from tessl) in transformers' layout.
fn storage_to_hf(spec: &TensorSpec, transposed: bool, data: Vec<f32>) -> Result<Vec<f32>> {
    if transposed {
        // tessl holds [in, out] = [shape[1], shape[0]].
        transpose_2d(&data, spec.shape[1], spec.shape[0])
    } else {
        Ok(data)
    }
}

fn hf_to_storage(spec: &TensorSpec, transposed: bool, data: Vec<f32>) -> Result<Vec<f32>> {
    if transposed {
        transpose_2d(&data, spec.shape[0], spec.shape[1])
    } else {
        Ok(data)
    }
}

fn shard_name(kind: Kind, i: usize, n: usize) -> String {
    format!("{}-{i:05}-of-{n:05}.safetensors", kind.tag())
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> Qwen35Error + '_ {
    move |e| Qwen35Error::Io(format!("{}: {e}", path.display()))
}

/// f32 values per piece handed to the writer (1 MiB).
const PIECE_VALUES: usize = 1 << 18;

/// Write one kind's shards into `dir` as new files (an existing file is
/// refused), each entry's values (transformers' layout) from `values(i)`,
/// streamed one entry at a time. The host work of a save, with no device in
/// it. Files are flushed, not synced: [`Qwen35Step::save_state`] writes into
/// an ojas-io `replace_dir_with` stage, which syncs the tree before the swap.
/// A shard that fails part-way is removed.
pub fn write_kind(
    dir: &Path,
    kind: Kind,
    step: u64,
    config_summary: &str,
    table: &[TensorSpec],
    shards: &[Range<usize>],
    mut values: impl FnMut(usize) -> Result<Vec<f32>>,
) -> Result<()> {
    let n = shards.len();
    let (step_s, n_s, entries_s) = (step.to_string(), n.to_string(), table.len().to_string());
    for (si, range) in shards.iter().enumerate() {
        let shapes: Vec<Vec<u64>> = range
            .clone()
            .map(|i| table[i].shape.iter().map(|&d| d as u64).collect())
            .collect();
        let specs: Vec<ojas_io::TensorSpec<'_>> = range
            .clone()
            .zip(&shapes)
            .map(|(i, s)| ojas_io::TensorSpec {
                name: &table[i].name,
                dtype: StDtype::F32,
                shape: s,
            })
            .collect();
        let si_s = si.to_string();
        let meta = [
            ("format", STATE_FORMAT),
            ("kind", kind.tag()),
            ("shard", si_s.as_str()),
            ("shards", n_s.as_str()),
            ("step", step_s.as_str()),
            ("entries", entries_s.as_str()),
            ("config", config_summary),
        ];
        let path = dir.join(shard_name(kind, si, n));
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(io(&path))?;
        let written = (|| -> Result<()> {
            let mut w = SafeTensorsWriter::new(BufWriter::new(file), &specs, &meta)?;
            let mut bytes = Vec::with_capacity(PIECE_VALUES * 4);
            for i in range.clone() {
                let v = values(i)?;
                if v.len() != table[i].numel() {
                    return Err(invalid(
                        "write_kind",
                        format!(
                            "{}: {} values for shape {:?}",
                            table[i].name,
                            v.len(),
                            table[i].shape
                        ),
                    ));
                }
                for piece in v.chunks(PIECE_VALUES) {
                    bytes.clear();
                    bytes.extend(piece.iter().flat_map(|x| x.to_le_bytes()));
                    w.write(&table[i].name, &bytes)?;
                }
            }
            w.finish()?.into_inner().map_err(|e| {
                Qwen35Error::Io(format!("{}: flush: {}", path.display(), e.error()))
            })?;
            Ok(())
        })();
        if let Err(e) = written {
            return Err(match std::fs::remove_file(&path) {
                Ok(()) => e,
                Err(c) => Qwen35Error::Io(format!(
                    "{e}; removing the partial shard {} also failed: {c}",
                    path.display()
                )),
            });
        }
    }
    Ok(())
}

/// Refuse a device left with more than its recommended working set
/// allocated (`MTLDevice::recommendedMaxWorkingSetSize`) once `staging`
/// bytes of tables are allocated. Past that line, Metal has to page the
/// resident set; on the 2B the next command buffer timed out
/// (`MTL4CommandQueueErrorTimeout`) and poisoned the runtime. Exactly the
/// working set passes.
///
/// `allocated` is the device's figure measured after the staging is
/// allocated, not a projection: tessl's allocations of 7.53 GB of staging
/// grew the device by 8.76-8.85 GB on the 2B, so "before + staging" would
/// have passed the run that faulted (51.12 GB projected, 52.44 GB actual).
/// Allocation is host-side work only, so measuring after it is still
/// before any GPU work. `staging` is reported, not compared.
pub fn check_device_working_set(
    op: &'static str,
    allocated: u64,
    staging: u64,
    recommended: u64,
) -> Result<()> {
    if allocated > recommended {
        return Err(Qwen35Error::Unsupported {
            what: format!(
                "{op}: staging a whole table ({staging} B requested) left {allocated} B \
                 allocated on the device, over its recommended working set of {recommended} B; \
                 past it a command buffer times out and poisons the runtime"
            ),
            needs: "per-entry or host-mapped table reads in tessl, so a read-back never stages a \
                    whole table, or less device memory held by the step"
                .to_string(),
        });
    }
    Ok(())
}

/// Refuse a [`Qwen35Step::save_state`] target that is present in any form.
/// A symbolic link is named as one and refused whether or not it dangles
/// (ojas-io's single-file replace would replace the link itself, and a
/// directory swap through a link would write somewhere the caller did not
/// name). Anything else present is refused because a state directory is
/// never overwritten. Only an absent path with a final component passes.
pub fn check_new_state_dir(dir: &Path) -> Result<()> {
    const OP: &str = "Qwen35Step::save_state";
    if dir.file_name().is_none() {
        return Err(invalid(
            OP,
            format!("{} has no final component", dir.display()),
        ));
    }
    match std::fs::symlink_metadata(dir) {
        Ok(m) if m.file_type().is_symlink() => Err(invalid(
            OP,
            format!(
                "{} is a symbolic link; a state directory is never written through one",
                dir.display()
            ),
        )),
        Ok(_) => Err(invalid(
            OP,
            format!(
                "{} exists; a state directory is never overwritten",
                dir.display()
            ),
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Qwen35Error::Io(format!("{}: {e}", dir.display()))),
    }
}

/// A state directory whose headers have all been checked against a table.
pub struct StateIndex {
    pub step: u64,
    /// For each kind, for each table entry, the index in `shards` of the
    /// file that holds it.
    files: Vec<Vec<usize>>,
    /// Every state file, opened once and held: a header that passed is the
    /// file that is later read, so a file replaced after [`Self::read`]
    /// cannot swap in a different entry.
    shards: Vec<(PathBuf, SafeTensors<'static>)>,
    table: Vec<TensorSpec>,
}

impl StateIndex {
    /// Read and check every header in `dir` against `table` and the config
    /// summary. Nothing is loaded and no device is touched.
    pub fn read(dir: &Path, table: &[TensorSpec], config_summary: &str) -> Result<Self> {
        let mut by_kind: Vec<Vec<(String, PathBuf)>> = vec![Vec::new(); 3];
        for entry in std::fs::read_dir(dir).map_err(io(dir))? {
            let entry = entry.map_err(io(dir))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = Kind::ALL.iter().position(|k| {
                name.starts_with(&format!("{}-", k.tag())) && name.ends_with(".safetensors")
            });
            let Some(k) = kind else {
                let why = "is not a state file; refusing a directory with unknown contents";
                return Err(Qwen35Error::Io(format!(
                    "{}: {name:?} {why}",
                    dir.display()
                )));
            };
            by_kind[k].push((name, entry.path()));
        }
        let mut step: Option<u64> = None;
        let mut files = Vec::with_capacity(3);
        let mut held: Vec<(PathBuf, SafeTensors<'static>)> = Vec::new();
        for (k, kind) in Kind::ALL.iter().enumerate() {
            let mut shards = std::mem::take(&mut by_kind[k]);
            shards.sort();
            if shards.is_empty() {
                return Err(Qwen35Error::Io(format!(
                    "{}: no {} files",
                    dir.display(),
                    kind.tag()
                )));
            }
            let n = shards.len();
            let mut where_: Vec<Option<usize>> = vec![None; table.len()];
            let index: std::collections::HashMap<&str, usize> = table
                .iter()
                .enumerate()
                .map(|(i, t)| (t.name.as_str(), i))
                .collect();
            for (si, (fname, path)) in shards.iter().enumerate() {
                if *fname != shard_name(*kind, si, n) {
                    return Err(Qwen35Error::Io(format!(
                        "{}: expected {} as shard {si} of {n}, found {fname}",
                        dir.display(),
                        shard_name(*kind, si, n)
                    )));
                }
                let st = SafeTensors::open(path)
                    .map_err(|e| Qwen35Error::Io(format!("{}: {e}", path.display())))?;
                let meta = st.metadata();
                let want = |k: &str, v: &str| -> Result<()> {
                    match meta.get(k) {
                        Some(got) if got == v => Ok(()),
                        got => Err(Qwen35Error::Io(format!(
                            "{}: metadata {k} is {got:?}, expected {v:?}",
                            path.display()
                        ))),
                    }
                };
                want("format", STATE_FORMAT)?;
                want("kind", kind.tag())?;
                want("shard", &si.to_string())?;
                want("shards", &n.to_string())?;
                want("entries", &table.len().to_string())?;
                want("config", config_summary)?;
                let s: u64 = meta
                    .get("step")
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| {
                        Qwen35Error::Io(format!(
                            "{}: metadata step is missing or not a u64",
                            path.display()
                        ))
                    })?;
                match step {
                    None => step = Some(s),
                    Some(prev) if prev == s => {}
                    Some(prev) => {
                        return Err(Qwen35Error::Io(format!(
                            "{}: step {s}, but another state file says {prev}",
                            path.display()
                        )))
                    }
                }
                for tname in st.names() {
                    let i = *index.get(tname).ok_or_else(|| {
                        Qwen35Error::Io(format!(
                            "{}: {tname} is not a parameter of this tower",
                            path.display()
                        ))
                    })?;
                    let info = st.info(tname)?;
                    let want_shape: Vec<u64> = table[i].shape.iter().map(|&d| d as u64).collect();
                    if info.dtype != StDtype::F32 || info.shape != want_shape {
                        return Err(Qwen35Error::Io(format!(
                            "{}: {tname} is {:?} {:?}, expected F32 {want_shape:?}",
                            path.display(),
                            info.dtype,
                            info.shape
                        )));
                    }
                    if where_[i].replace(held.len()).is_some() {
                        return Err(Qwen35Error::Io(format!(
                            "{}: {tname} appears in two {} files",
                            dir.display(),
                            kind.tag()
                        )));
                    }
                }
                held.push((path.clone(), st));
            }
            let mut kind_files = Vec::with_capacity(table.len());
            for (i, w) in where_.into_iter().enumerate() {
                kind_files.push(w.ok_or_else(|| {
                    Qwen35Error::Io(format!(
                        "{}: no {} value for {}",
                        dir.display(),
                        kind.tag(),
                        table[i].name
                    ))
                })?);
            }
            files.push(kind_files);
        }
        Ok(Self {
            step: step
                .ok_or_else(|| Qwen35Error::Io(format!("{}: no state files", dir.display())))?,
            files,
            shards: held,
            table: table.to_vec(),
        })
    }

    /// Entry `i` of `kind`, in transformers' layout.
    pub fn read_entry(&self, kind: Kind, i: usize) -> Result<Vec<f32>> {
        let k = match kind {
            Kind::Masters => 0,
            Kind::ExpAvg => 1,
            Kind::ExpAvgSq => 2,
        };
        let (path, st) = &self.shards[self.files[k][i]];
        let (_, values) = st
            .read_f32(&self.table[i].name)
            .map_err(|e| Qwen35Error::Io(format!("{}: {e}", path.display())))?;
        Ok(values)
    }
}

impl Qwen35Step {
    /// Entry `i` of `which` on the host, in tessl's storage layout (the
    /// transposed `[in, out]` order for a linear layer), through tessl's
    /// per-entry host-mapped reads: no device tensor is allocated for it, so a
    /// read-back's device peak is the model's own, at any model size. Each
    /// read waits for prior GPU work and is refused as "runtime busy" while
    /// another host mapping is live.
    fn read_host(&self, which: Which, i: usize) -> Result<Vec<f32>> {
        let mut dst = vec![0f32; self.table[i].numel()];
        match which {
            Which::Parameters => self
                .model
                .read_parameter_host(i, &mut dst)
                .map_err(tessl("read_parameter_host"))?,
            Which::Gradients => self
                .model
                .read_gradient_host(&self.bank, i, &mut dst)
                .map_err(tessl("read_gradient_host"))?,
            Which::ExpAvg => self
                .model
                .read_adamw_moment_host(&self.adamw, Moment::First, i, &mut dst)
                .map_err(tessl("read_adamw_moment_host"))?,
            Which::ExpAvgSq => self
                .model
                .read_adamw_moment_host(&self.adamw, Moment::Second, i, &mut dst)
                .map_err(tessl("read_adamw_moment_host"))?,
        }
        Ok(dst)
    }

    /// Entry `i` of `which` on the host in transformers' layout.
    fn read_hf(&self, which: Which, i: usize) -> Result<Vec<f32>> {
        storage_to_hf(
            &self.table[i],
            self.transposed[i],
            self.read_host(which, i)?,
        )
    }

    /// Every entry of `which` on the host, in transformers' names and layout,
    /// charged to the provider's host budget (8 GB per table on the 2B). Each
    /// entry is read from its device buffer directly; nothing is staged on
    /// the device.
    pub fn read_table(&self, which: Which) -> Result<Vec<NamedTensor>> {
        let all: Vec<&str> = self.table.iter().map(|t| t.name.as_str()).collect();
        self.read_entries(which, &all)
    }

    /// [`Self::read_table`] for the named entries only (in the order given).
    /// Only those entries are read from the device.
    pub fn read_entries(&self, which: Which, names: &[&str]) -> Result<Vec<NamedTensor>> {
        const OP: &str = "Qwen35Step::read_entries";
        self.check_live()?;
        if which == Which::Gradients && !matches!(self.bank_state, BankState::Holds(_)) {
            return Err(invalid(
                OP,
                format!("the gradient bank is {:?}", self.bank_state),
            ));
        }
        let picks: Vec<usize> = names
            .iter()
            .map(|n| {
                self.table
                    .iter()
                    .position(|t| t.name == *n)
                    .ok_or_else(|| invalid(OP, format!("{n} is not a parameter of this tower")))
            })
            .collect::<Result<_>>()?;
        let mut out = Vec::with_capacity(picks.len());
        for i in picks {
            let spec = &self.table[i];
            let v = self.read_hf(which, i)?;
            out.push(NamedTensor {
                name: spec.name.clone(),
                tensor: Tensor::from_f32(&v, &spec.shape, &self.budget)?,
            });
        }
        Ok(out)
    }

    /// Write the masters, both AdamW moments and the step count into a new
    /// directory `dir`, through ojas-io's `replace_dir_with`: the files go
    /// into a staging directory beside `dir`, the tree is synced, and only
    /// then is it renamed into place; on any failure the stage is removed and
    /// nothing appears at `dir`.
    ///
    /// Refused before anything is written ([`check_new_state_dir`]): a `dir`
    /// that exists in any form, which is never overwritten, and a `dir` that
    /// is a symbolic link, dangling or not. `replace_dir_with` also refuses a
    /// parent directory that is a symbolic link. Two limits are ojas-io's:
    /// writers into one parent directory are serialized by an `flock` on it
    /// with no timeout, and a directory created at `dir` by someone else
    /// between the check and the swap would be replaced.
    pub fn save_state(&self, dir: &Path) -> Result<()> {
        const OP: &str = "Qwen35Step::save_state";
        self.check_live()?;
        self.require_f32_state(OP)?;
        // The checkpoint holds masters and moments, not the gradient bank, and
        // `load_state` empties the bank. A checkpoint cut mid-accumulation
        // would resume without those gradients and give no sign of it.
        match &self.bank_state {
            BankState::Empty => {}
            held => {
                return Err(invalid(
                    OP,
                    format!(
                        "the gradient bank is {held:?}; the checkpoint does not hold it and \
                         load_state empties the bank, so those gradients would be lost on \
                         resume. Take the AdamW step, or discard_gradients, before saving"
                    ),
                ))
            }
        }
        check_new_state_dir(dir)?;
        let step = self.step_count();
        let summary = self.cfg.canonical_summary();
        let shards = plan_shards(&self.table, SHARD_BYTES);
        // The provider's own error, kept typed; ojas-io only carries text.
        let mut failure: Option<Qwen35Error> = None;
        let swapped = replace_dir_with(dir, |stage| {
            let run = || -> Result<()> {
                for (kind, which) in [
                    (Kind::Masters, Which::Parameters),
                    (Kind::ExpAvg, Which::ExpAvg),
                    (Kind::ExpAvgSq, Which::ExpAvgSq),
                ] {
                    write_kind(stage, kind, step, &summary, &self.table, &shards, |i| {
                        self.read_hf(which, i)
                    })?;
                }
                Ok(())
            };
            run().map_err(|e| {
                let text = IoError::new(e.to_string());
                failure = Some(e);
                text
            })
        });
        match (swapped, failure) {
            (Ok(()), _) => Ok(()),
            (Err(_), Some(e)) => Err(e),
            (Err(e), None) => Err(e.into()),
        }
    }

    /// Restore masters, both moments and the step count from a directory
    /// [`Self::save_state`] wrote for this config. Every header is checked
    /// first; a failure after the first device write poisons the provider
    /// (the model would hold a mix of two states). The bank is emptied.
    pub fn load_state(&mut self, dir: &Path) -> Result<()> {
        self.check_live()?;
        self.require_f32_state("Qwen35Step::load_state")?;
        let idx = StateIndex::read(dir, &self.table, &self.cfg.canonical_summary())?;
        self.load_state_from_index(&idx)
    }

    /// [`Self::load_state`] from a [`StateIndex`] already read for this
    /// tower. Every header, shape, dtype and byte length of every kind was
    /// checked when the index was read, so a malformed directory is refused
    /// before the first write and leaves the provider usable. Only a failure
    /// after a write has landed (a file cut short underneath the held index,
    /// a device error) poisons it: a refused first write moved nothing.
    pub fn load_state_from_index(&mut self, idx: &StateIndex) -> Result<()> {
        self.check_live()?;
        self.require_f32_state("Qwen35Step::load_state")?;
        if idx.table != self.table {
            return Err(invalid(
                "Qwen35Step::load_state",
                "the state index was read for a different parameter table",
            ));
        }
        let mut wrote = false;
        for kind in Kind::ALL {
            for i in 0..self.table.len() {
                let result = idx
                    .read_entry(kind, i)
                    .and_then(|v| hf_to_storage(&self.table[i], self.transposed[i], v))
                    .and_then(|v| self.write_host(kind, i, &v));
                match result {
                    Ok(()) => wrote = true,
                    Err(e) => {
                        if wrote {
                            self.poisoned = Some(format!(
                                "load_state failed part-way ({} entry {i}): {e}",
                                kind.tag()
                            ));
                        }
                        return Err(e);
                    }
                }
            }
        }
        self.adamw.set_step_count(idx.step);
        self.bank_state = BankState::Empty;
        self.weights_version += 1;
        Ok(())
    }

    /// The state format holds f32 masters as the weights: a
    /// [`Precision::Bf16`] provider's are AdamW's f32 copies, which tessl
    /// does not read out, so its saves and loads are refused.
    fn require_f32_state(&self, op: &'static str) -> Result<()> {
        if self.precision == Precision::Bf16 {
            return Err(Qwen35Error::Unsupported {
                what: format!("{op} on a Precision::Bf16 provider"),
                needs: "a read and write of AdamW's f32 masters (tessl reads only the bf16 \
                        weights), so a save would drop them"
                    .into(),
            });
        }
        Ok(())
    }

    /// Entry `i` of `kind`, in tessl's storage layout, written into the
    /// model through tessl's per-entry host-mapped writes. A parameter write
    /// bumps tessl's parameter generation before its first byte lands.
    fn write_host(&mut self, kind: Kind, i: usize, v: &[f32]) -> Result<()> {
        match kind {
            Kind::Masters => self
                .model
                .write_parameter_host(i, v)
                .map_err(tessl("write_parameter_host")),
            Kind::ExpAvg => self
                .model
                .write_adamw_moment_host(&mut self.adamw, Moment::First, i, v)
                .map_err(tessl("write_adamw_moment_host")),
            Kind::ExpAvgSq => self
                .model
                .write_adamw_moment_host(&mut self.adamw, Moment::Second, i, v)
                .map_err(tessl("write_adamw_moment_host")),
        }
    }
}
