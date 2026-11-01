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
//! **Host-neutral since the tessl step shell was retired.** The provider that
//! wrote and loaded these directories (`Qwen35Step::save_state` / `load_state`,
//! over tessl's per-entry host-mapped reads) is gone; what stays is the
//! format, its shard planner, the streaming writer ([`write_kind`]) and the
//! checked reader ([`StateIndex`]), which touch no device. On the host a write
//! holds one entry at a time (the 2 GB embedding on the 2B), streamed into its
//! shard through ojas-io's `SafeTensorsWriter` in 1 MiB pieces; a read loads
//! one entry at a time. Shards are [`SHARD_BYTES`] unless one tensor is larger.

use std::fs::File;
use std::io::BufWriter;
use std::ops::Range;
use std::path::{Path, PathBuf};

use ojas_io::{SafeTensors, SafeTensorsWriter, StDtype};

use crate::error::{invalid, Qwen35Error, Result};
use crate::names::TensorSpec;

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
/// it. Files are flushed, not synced: a state save writes into
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

/// Refuse a state-save target that is present in any form.
/// A symbolic link is named as one and refused whether or not it dangles
/// (ojas-io's single-file replace would replace the link itself, and a
/// directory swap through a link would write somewhere the caller did not
/// name). Anything else present is refused because a state directory is
/// never overwritten. Only an absent path with a final component passes.
pub fn check_new_state_dir(dir: &Path) -> Result<()> {
    const OP: &str = "check_new_state_dir";
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
