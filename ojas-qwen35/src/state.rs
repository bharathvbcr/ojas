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
//! Costs on the 2B, from tessl's API rather than this format: tessl's
//! `read_parameters` / `write_parameters` / `*_adamw_moment` move every
//! tensor of a kind in one call, so a save or a load stages one kind (8 GB of
//! f32) in extra device tensors at a time, besides one shard on the host
//! (ojas-io's writer encodes a whole file in memory). Shards are
//! [`SHARD_BYTES`] unless one tensor is larger (the 2 GB embedding).
//! A resume opens the base snapshot (tessl's loader is the model's only
//! constructor) and then writes these over it.

use std::ops::Range;
use std::path::{Path, PathBuf};

use ojas_core::Tensor;
use ojas_io::{write_safetensors, SafeTensors, StDtype, TensorOut};
use tessl::qwen35_adamw::Moment;


use crate::error::{invalid, tessl, Qwen35Error, Result};
use crate::names::TensorSpec;
use crate::step::{BankState, Qwen35Step};

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
        return Err(invalid("transpose_2d", format!("{} values for [{rows}, {cols}]", data.len())));
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

/// Write one kind's shards into `dir`, each entry's values (transformers'
/// layout) from `values(i)`. The host work of a save, with no device in it.
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
        let mut data: Vec<Vec<u8>> = Vec::with_capacity(range.len());
        let mut shapes: Vec<Vec<u64>> = Vec::with_capacity(range.len());
        for i in range.clone() {
            let v = values(i)?;
            if v.len() != table[i].numel() {
                return Err(invalid(
                    "write_kind",
                    format!("{}: {} values for shape {:?}", table[i].name, v.len(), table[i].shape),
                ));
            }
            data.push(v.iter().flat_map(|x| x.to_le_bytes()).collect());
            shapes.push(table[i].shape.iter().map(|&d| d as u64).collect());
        }
        let items: Vec<TensorOut<'_>> = range
            .clone()
            .zip(data.iter().zip(&shapes))
            .map(|(i, (d, s))| TensorOut {
                name: &table[i].name,
                dtype: StDtype::F32,
                shape: s,
                data: d,
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
        write_safetensors(&dir.join(shard_name(kind, si, n)), &items, &meta)?;
    }
    Ok(())
}

/// A state directory whose headers have all been checked against a table.
pub struct StateIndex {
    pub step: u64,
    /// For each kind, for each table entry, the file that holds it.
    files: Vec<Vec<PathBuf>>,
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
            let kind = Kind::ALL
                .iter()
                .position(|k| name.starts_with(&format!("{}-", k.tag())) && name.ends_with(".safetensors"));
            match kind {
                Some(k) => by_kind[k].push((name, entry.path())),
                None => {
                    return Err(Qwen35Error::Io(format!(
                        "{}: {name:?} is not a state file; refusing a directory with unknown contents",
                        dir.display()
                    )))
                }
            }
        }
        let mut step: Option<u64> = None;
        let mut files = Vec::with_capacity(3);
        for (k, kind) in Kind::ALL.iter().enumerate() {
            let mut shards = std::mem::take(&mut by_kind[k]);
            shards.sort();
            if shards.is_empty() {
                return Err(Qwen35Error::Io(format!("{}: no {} files", dir.display(), kind.tag())));
            }
            let n = shards.len();
            let mut where_: Vec<Option<PathBuf>> = vec![None; table.len()];
            let index: std::collections::HashMap<&str, usize> =
                table.iter().enumerate().map(|(i, t)| (t.name.as_str(), i)).collect();
            for (si, (fname, path)) in shards.iter().enumerate() {
                if *fname != shard_name(*kind, si, n) {
                    return Err(Qwen35Error::Io(format!(
                        "{}: expected {} as shard {si} of {n}, found {fname}",
                        dir.display(),
                        shard_name(*kind, si, n)
                    )));
                }
                let st = SafeTensors::open(path).map_err(|e| Qwen35Error::Io(format!("{}: {e}", path.display())))?;
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
                    .ok_or_else(|| Qwen35Error::Io(format!("{}: metadata step is missing or not a u64", path.display())))?;
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
                        Qwen35Error::Io(format!("{}: {tname} is not a parameter of this tower", path.display()))
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
                    if where_[i].replace(path.clone()).is_some() {
                        return Err(Qwen35Error::Io(format!(
                            "{}: {tname} appears in two {} files",
                            dir.display(),
                            kind.tag()
                        )));
                    }
                }
            }
            let mut kind_files = Vec::with_capacity(table.len());
            for (i, w) in where_.into_iter().enumerate() {
                kind_files.push(w.ok_or_else(|| {
                    Qwen35Error::Io(format!("{}: no {} value for {}", dir.display(), kind.tag(), table[i].name))
                })?);
            }
            files.push(kind_files);
        }
        Ok(Self {
            step: step.ok_or_else(|| Qwen35Error::Io(format!("{}: no state files", dir.display())))?,
            files,
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
        let path = &self.files[k][i];
        let st = SafeTensors::open(path).map_err(|e| Qwen35Error::Io(format!("{}: {e}", path.display())))?;
        let (shape, values) = st.read_f32(&self.table[i].name)?;
        let want: Vec<u64> = self.table[i].shape.iter().map(|&d| d as u64).collect();
        if shape != want {
            return Err(Qwen35Error::Io(format!(
                "{}: {} changed shape since the index was read",
                path.display(),
                self.table[i].name
            )));
        }
        Ok(values)
    }
}

impl Qwen35Step {
    /// One kind of table staged in fresh device tensors (tessl's storage
    /// layout, every entry at once, as tessl's API requires).
    fn device_copies(&self, which: Which) -> Result<Vec<tessl::Tensor>> {
        let ts: Vec<tessl::Tensor> = self
            .table
            .iter()
            .zip(&self.transposed)
            .map(|(t, &tr)| {
                let shape = if tr { vec![t.shape[1], t.shape[0]] } else { t.shape.clone() };
                self.rt.alloc_tensor_f32(&shape).map_err(tessl("alloc staging tensor"))
            })
            .collect::<Result<_>>()?;
        match which {
            Which::Parameters => self.model.read_parameters(&ts).map_err(tessl("read_parameters"))?,
            Which::Gradients => self.model.read_gradients(&self.bank, &ts).map_err(tessl("read_gradients"))?,
            Which::ExpAvg => self
                .model
                .read_adamw_moment(&self.adamw, Moment::First, &ts)
                .map_err(tessl("read_adamw_moment"))?,
            Which::ExpAvgSq => self
                .model
                .read_adamw_moment(&self.adamw, Moment::Second, &ts)
                .map_err(tessl("read_adamw_moment"))?,
        }
        Ok(ts)
    }

    /// Every entry of `which` on the host, in transformers' names and layout,
    /// charged to the provider's host budget (8 GB per table on the 2B, plus
    /// the same again staged on the device while it is read).
    pub fn read_table(&self, which: Which) -> Result<Vec<NamedTensor>> {
        let all: Vec<&str> = self.table.iter().map(|t| t.name.as_str()).collect();
        self.read_entries(which, &all)
    }

    /// [`Self::read_table`] for the named entries only (in the order given).
    /// The device side still stages the whole table (tessl's copies take
    /// every entry at once); only the host copies shrink.
    pub fn read_entries(&self, which: Which, names: &[&str]) -> Result<Vec<NamedTensor>> {
        const OP: &str = "Qwen35Step::read_entries";
        self.check_live()?;
        if which == Which::Gradients && !matches!(self.bank_state, BankState::Holds(_)) {
            return Err(invalid(OP, format!("the gradient bank is {:?}", self.bank_state)));
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
        let ts = self.device_copies(which)?;
        let mut out = Vec::with_capacity(picks.len());
        for i in picks {
            let spec = &self.table[i];
            let v = storage_to_hf(spec, self.transposed[i], ts[i].read_f32().map_err(tessl("read staging tensor"))?)?;
            out.push(NamedTensor {
                name: spec.name.clone(),
                tensor: Tensor::from_f32(&v, &spec.shape, &self.budget)?,
            });
        }
        Ok(out)
    }

    /// Write the masters, both AdamW moments and the step count into a new
    /// directory `dir`. It is written as a sibling `<dir>.partial-<pid>` and
    /// renamed into place only when every file is written; an existing `dir`
    /// is refused, never overwritten. On failure the partial directory is
    /// left for inspection and named in the error.
    pub fn save_state(&self, dir: &Path) -> Result<()> {
        const OP: &str = "Qwen35Step::save_state";
        self.check_live()?;
        if dir.exists() {
            return Err(invalid(OP, format!("{} exists; a state directory is never overwritten", dir.display())));
        }
        let leaf = dir
            .file_name()
            .ok_or_else(|| invalid(OP, format!("{} has no final component", dir.display())))?;
        let tmp = dir.with_file_name(format!("{}.partial-{}", leaf.to_string_lossy(), std::process::id()));
        if tmp.exists() {
            return Err(invalid(OP, format!("{} exists from an earlier save", tmp.display())));
        }
        std::fs::create_dir_all(&tmp).map_err(io(&tmp))?;
        let step = self.step_count();
        let summary = self.cfg.canonical_summary();
        let shards = plan_shards(&self.table, SHARD_BYTES);
        let run = || -> Result<()> {
            for (kind, which) in [
                (Kind::Masters, Which::Parameters),
                (Kind::ExpAvg, Which::ExpAvg),
                (Kind::ExpAvgSq, Which::ExpAvgSq),
            ] {
                let ts = self.device_copies(which)?;
                write_kind(&tmp, kind, step, &summary, &self.table, &shards, |i| {
                    storage_to_hf(&self.table[i], self.transposed[i], ts[i].read_f32().map_err(tessl("read staging tensor"))?)
                })?;
            }
            std::fs::rename(&tmp, dir).map_err(io(dir))
        };
        run().map_err(|e| Qwen35Error::Io(format!("{e} (the partial save is left at {})", tmp.display())))
    }

    /// Restore masters, both moments and the step count from a directory
    /// [`Self::save_state`] wrote for this config. Every header is checked
    /// first; a failure after the first device write poisons the provider
    /// (the model would hold a mix of two states). The bank is emptied.
    pub fn load_state(&mut self, dir: &Path) -> Result<()> {
        self.check_live()?;
        let idx = StateIndex::read(dir, &self.table, &self.cfg.canonical_summary())?;
        let mut wrote = false;
        for kind in Kind::ALL {
            let staged = self.stage(&idx, kind);
            let result = staged.and_then(|ts| {
                wrote = true;
                match kind {
                    Kind::Masters => self.model.write_parameters(&ts).map_err(tessl("write_parameters")),
                    Kind::ExpAvg => self
                        .model
                        .write_adamw_moment(&mut self.adamw, Moment::First, &ts)
                        .map_err(tessl("write_adamw_moment")),
                    Kind::ExpAvgSq => self
                        .model
                        .write_adamw_moment(&mut self.adamw, Moment::Second, &ts)
                        .map_err(tessl("write_adamw_moment")),
                }
            });
            if let Err(e) = result {
                if wrote {
                    self.poisoned = Some(format!("load_state from {} failed part-way: {e}", dir.display()));
                }
                return Err(e);
            }
        }
        self.adamw.set_step_count(idx.step);
        self.bank_state = BankState::Empty;
        self.weights_version += 1;
        Ok(())
    }

    /// One kind read from disk into device tensors in tessl's layout.
    fn stage(&self, idx: &StateIndex, kind: Kind) -> Result<Vec<tessl::Tensor>> {
        let mut ts = Vec::with_capacity(self.table.len());
        for (i, (spec, &tr)) in self.table.iter().zip(&self.transposed).enumerate() {
            let shape = if tr { vec![spec.shape[1], spec.shape[0]] } else { spec.shape.clone() };
            let t = self.rt.alloc_tensor_f32(&shape).map_err(tessl("alloc staging tensor"))?;
            t.write_f32(&hf_to_storage(spec, tr, idx.read_entry(kind, i)?)?)
                .map_err(tessl("write staging tensor"))?;
            ts.push(t);
        }
        Ok(ts)
    }
}
