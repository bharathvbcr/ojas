//! Paths under the configured root; Inspect, Load and New.
//!
//! Every path the C ABI accepts is relative to the model root
//! ([`crate::set_model_root`]) and resolved by [`resolve_under_root`] or
//! [`resolve_dir_under_root`]: no `..`, no absolute path, and every existing
//! component's canonical form must stay under the root. A file is then
//! opened once with `O_NOFOLLOW` ([`ojas_io::open_nofollow`]) and the open
//! file is checked to still be the entry the path names
//! ([`confirm_open_identity`]).

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Component, Path, PathBuf};

use ojas_core::{Backend, Numerics, OjasError};
use ojas_io::SafeTensors;
use ojas_model::ModelSpec;

use crate::gate::Check;
use crate::model::{self, Build, Model, Placement};
use crate::session::{self, DeviceKind};
use crate::wire::{field, required, tag, Fields, Kind, Reader};

/// Gusset copies at most this many inline input bytes (R16).
pub const INLINE_PATH_MAX: usize = 4096;

/// Largest `threads` a [`DEVICE_CPU_PARALLEL`] session accepts. Larger
/// values are refused, not clamped. The session's
/// [`ojas_cpu::CpuBackend::with_threads`] pool splits independent output
/// rows across up to this many workers.
pub const MAX_CPU_THREADS: u32 = 256;

pub const DEVICE_CPU: u32 = 0;
pub const DEVICE_CPU_PARALLEL: u32 = 1;
pub const DEVICE_METAL: u32 = 2;
pub const DEVICE_WGPU: u32 = 3;

/// A session's byte budget when the caller names none: 1 GiB. Every
/// session budget is drawn from the process ceiling
/// ([`session::DEFAULT_MEMORY_CEILING_BYTES`], also 1 GiB until
/// [`session::set_memory_ceiling`] changes it), so a budget above the
/// ceiling is refused at load and all sessions together never pass it. A
/// 124M training session needs several GiB: raise the ceiling, then ask.
pub const DEFAULT_BUDGET_BYTES: u64 = 1 << 30;

/// What a resolved path must be.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Want {
    File,
    Dir,
    /// A directory that may not exist yet; its parent must.
    NewOrDir,
}

pub fn resolve_under_root(root: &Path, raw: &str) -> Result<PathBuf, String> {
    resolve(root, raw, Want::File)
}

/// A checkpoint directory under `root`. With `must_exist` it must be a
/// directory now (Resume). Without, it may be missing, and then its parent
/// must be an existing directory under the root (Save); if it exists it
/// must be a directory.
pub fn resolve_dir_under_root(root: &Path, raw: &str, must_exist: bool) -> Result<PathBuf, String> {
    resolve(
        root,
        raw,
        if must_exist {
            Want::Dir
        } else {
            Want::NewOrDir
        },
    )
}

fn resolve(root: &Path, raw: &str, want: Want) -> Result<PathBuf, String> {
    if raw.len() > INLINE_PATH_MAX {
        return Err(format!("path exceeds {INLINE_PATH_MAX} bytes"));
    }
    if raw.is_empty() || raw.contains('\0') {
        return Err("path is empty or contains NUL".to_string());
    }
    // Path::components drops a trailing '/', which would let "file/" open a
    // regular file that POSIX open() refuses with ENOTDIR.
    if raw.ends_with('/') {
        return Err("path names a directory".to_string());
    }
    let path = Path::new(raw);
    if path.is_absolute() {
        return Err("path must be relative".to_string());
    }
    let root = root
        .canonicalize()
        .map_err(|err| format!("model root: {err}"))?;
    if !root.is_dir() {
        return Err(format!("model root is not a directory: {}", root.display()));
    }
    let mut acc = root.clone();
    let mut saw_name = false;
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(name) => {
                saw_name = true;
                acc.push(name);
                if acc.symlink_metadata().is_ok() {
                    let canon = acc.canonicalize().map_err(|err| format!("path: {err}"))?;
                    if !canon.starts_with(&root) {
                        return Err("path escapes model root".to_string());
                    }
                    acc = canon;
                }
            }
            Component::ParentDir => return Err("path must not contain ..".to_string()),
            Component::RootDir | Component::Prefix(_) => {
                return Err("path must be relative".to_string())
            }
        }
    }
    if !saw_name {
        return Err("path is empty".to_string());
    }
    let exists = acc.symlink_metadata().is_ok();
    match want {
        Want::File if !acc.is_file() => return Err(format!("missing file: {}", acc.display())),
        Want::Dir if !acc.is_dir() => return Err(format!("missing directory: {}", acc.display())),
        Want::NewOrDir if exists && !acc.is_dir() => {
            return Err(format!("not a directory: {}", acc.display()))
        }
        Want::NewOrDir if !exists => {
            let parent = acc
                .parent()
                .filter(|p| p.is_dir())
                .ok_or_else(|| format!("missing parent directory: {}", acc.display()))?;
            let parent = parent
                .canonicalize()
                .map_err(|err| format!("path: {err}"))?;
            if !parent.starts_with(&root) {
                return Err("path escapes model root".to_string());
            }
            return Ok(acc);
        }
        _ => {}
    }
    let canon = acc
        .canonicalize()
        .map_err(|err| format!("missing file: {err}"))?;
    if !canon.starts_with(&root) || (want != Want::File && canon == root) {
        return Err("path escapes model root".to_string());
    }
    Ok(canon)
}

/// The opened file must still be the directory entry named by `path`.
///
/// Compares `fstat` of `file` with `lstat` of `path`. The path is not
/// canonicalized and not opened again, so a symlink swapped into that entry
/// is a different inode rather than a second open that follows it.
pub fn confirm_open_identity(file: &File, path: &Path, root: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let opened = file
        .metadata()
        .map_err(|err| format!("model file: {err}"))?;
    let root = root
        .canonicalize()
        .map_err(|err| format!("model root: {err}"))?;
    let listed = path
        .symlink_metadata()
        .map_err(|err| format!("path: {err}"))?;
    if listed.file_type().is_symlink()
        || listed.dev() != opened.dev()
        || listed.ino() != opened.ino()
    {
        return Err("model file changed while opening".to_string());
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent
        .canonicalize()
        .map_err(|err| format!("path: {err}"))?;
    if !parent.starts_with(&root) {
        return Err("path escapes model root".to_string());
    }
    Ok(())
}

/// One read-only open through [`ojas_io::open_nofollow`]: a final-component
/// symlink that appeared after the path was resolved is refused instead of
/// followed, and so is anything but a regular file.
fn open_model(path: &Path) -> Result<File, String> {
    ojas_io::open_nofollow(path).map_err(|err| format!("missing file: {}", err.detail()))
}

/// A file under the root, opened once and confirmed.
pub struct Verified {
    pub path: PathBuf,
    pub file: File,
}

impl Verified {
    /// `raw` resolved under the configured root, opened with `O_NOFOLLOW`,
    /// and confirmed to be the entry the path names.
    pub fn open(raw: &str) -> Result<Self, String> {
        let root = session::root()?;
        let path = resolve_under_root(&root, raw)?;
        let file = open_model(&path)?;
        confirm_open_identity(&file, &path, &root)?;
        Ok(Self { path, file })
    }

    /// A path naming this open file, for a loader that takes a path
    /// (`ojas_data::TokenBin`, `ojas_data::load_hf_gpt2`). Opening it opens
    /// the file this descriptor holds, never whatever the original path names
    /// now, so the loader's own open cannot follow a symlink swapped in after
    /// this one. The descriptor stays open while `self` lives.
    pub fn fd_path(&self) -> PathBuf {
        PathBuf::from(format!("/dev/fd/{}", self.file.as_raw_fd()))
    }

    /// `detail` from a loader given [`Self::fd_path`], naming the real path.
    pub fn explain(&self, detail: &str) -> String {
        let fd = self.fd_path();
        detail.replace(&*fd.to_string_lossy(), &self.path.to_string_lossy())
    }
}

/// Number of tensors in a safetensors file, from its header alone. The
/// header is validated by [`SafeTensors::from_file`] (the official 100 MB
/// header cap and every per-tensor check); no tensor bytes are read.
pub fn inspect(raw: &str) -> Result<u32, String> {
    let file = Verified::open(raw)?;
    let st = SafeTensors::from_file(file.file)
        .map_err(|e| format!("not a safetensors file: {}", e.detail()))?;
    let n = st.names().count();
    if n == 0 {
        return Err("not a safetensors file: header has no tensors".to_string());
    }
    u32::try_from(n).map_err(|_| "not a safetensors file: tensor count overflows".to_string())
}

/// The placement fields every session-creating record accepts.
pub(crate) const PLACEMENT_FIELDS: [(u32, &str, Kind); 4] = [
    (tag::DEVICE, "device", Kind::U32),
    (tag::THREADS, "threads", Kind::U32),
    (tag::BUDGET, "budget", Kind::U64),
    (tag::NUMERICS, "numerics", Kind::U32),
];

/// Device 0 CPU, 1 CPU parallel (`threads` 1..=256), 2 Metal, 3 wgpu;
/// default CPU. `threads` is read only for CPU parallel. `budget` (bytes,
/// non-zero) defaults to [`DEFAULT_BUDGET_BYTES`]; one above the process
/// ceiling is refused when the device opens. `numerics` 1 Exact or
/// 2 Fast is CPU only; absent keeps the backend's default.
pub(crate) fn placement(f: &Fields<'_>) -> Result<Placement, String> {
    let device = match f.u32(tag::DEVICE).unwrap_or(DEVICE_CPU) {
        DEVICE_CPU => DeviceKind::Cpu { threads: 1 },
        DEVICE_CPU_PARALLEL => {
            let threads = f.u32(tag::THREADS).unwrap_or(0);
            if threads == 0 {
                return Err("load: thread count is 0".to_string());
            }
            if threads > MAX_CPU_THREADS {
                return Err(format!(
                    "load: thread count {threads} exceeds {MAX_CPU_THREADS}"
                ));
            }
            let threads = usize::try_from(threads).map_err(|_| "load: thread count")?;
            DeviceKind::Cpu { threads }
        }
        DEVICE_METAL => DeviceKind::Metal,
        DEVICE_WGPU => DeviceKind::Wgpu,
        other => return Err(format!("load: unknown device {other}")),
    };
    let budget_bytes = f.u64(tag::BUDGET).unwrap_or(DEFAULT_BUDGET_BYTES);
    if budget_bytes == 0 {
        return Err("load: budget is 0 bytes".to_string());
    }
    let numerics = match f.u32(tag::NUMERICS) {
        None => None,
        Some(1) => Some(Numerics::Exact),
        Some(2) => Some(Numerics::Fast),
        Some(other) => return Err(format!("load: unknown numerics {other}")),
    };
    if numerics.is_some() && !matches!(device, DeviceKind::Cpu { .. }) {
        return Err("load: numerics is fixed by the Metal and wgpu backends".to_string());
    }
    Ok(Placement {
        device,
        budget_bytes,
        numerics,
    })
}

/// Parameters read from a checked safetensors file.
struct FromFile {
    spec: ModelSpec,
    file: SafeTensors<'static>,
}

impl Build for FromFile {
    fn build<B: Backend + Clone>(self, backend: B) -> Result<Model<B>, OjasError> {
        let host = ojas_model::load_params(&self.spec, &self.file, backend.budget())?;
        Model::resident(backend, self.spec, host)
    }
}

/// Fresh nanolab init.
struct Fresh {
    spec: ModelSpec,
    seed: u64,
}

impl Build for Fresh {
    fn build<B: Backend + Clone>(self, backend: B) -> Result<Model<B>, OjasError> {
        let host = ojas_model::init_params(&self.spec, self.seed, backend.budget())?;
        Model::resident(backend, self.spec, host)
    }
}

/// Open the device, build the model on it, and add the session. Nothing
/// is added unless every step succeeds.
pub(crate) fn create(
    path: Option<PathBuf>,
    placement: &Placement,
    mut check: Check,
    build: impl Build,
) -> Result<session::Session, String> {
    let (opened, lease) = model::open(placement, &mut check)?;
    let state = model::state_on(opened, lease, check, build)?;
    let tensors = state.engine.tensors()?;
    session::insert(path, tensors, placement.device, state)
}

/// LOAD: `{path, device?, threads?, budget?, numerics?}`.
///
/// The file is opened and its header and spec checked before the device
/// opens, so a bad file does not cost a device thread. Then every parameter
/// is read, checked against the spec (`ojas_model::load_params`), and
/// uploaded to the session's device.
pub fn load_request(bytes: &[u8], mut check: Check) -> Result<session::Session, String> {
    check()?;
    let mut r = Reader::new(bytes);
    let mut allowed = PLACEMENT_FIELDS.to_vec();
    allowed.push(field(tag::PATH, Kind::Str));
    let f = Fields::read(&mut r, &allowed)?;
    r.finish()?;
    let placement = placement(&f)?;
    let raw = required(f.str(tag::PATH), tag::PATH)?;
    session::check_room()?;
    let verified = Verified::open(raw)?;
    let path = verified.path.clone();
    let file = SafeTensors::from_file(verified.file)
        .map_err(|e| format!("not a safetensors file: {}", e.detail()))?;
    let spec = ojas_model::load_spec(&file).map_err(|e| crate::ojas_error("load", &e))?;
    create(Some(path), &placement, check, FromFile { spec, file })
}

/// The spec fields of NEW. Every one is required; the tied head, QK-norm,
/// the gate and the value residual are always on.
const SPEC_FIELDS: [(u32, &str, Kind); 10] = [
    (tag::VOCAB, "vocab", Kind::U32),
    (tag::N_EMBD, "n_embd", Kind::U32),
    (tag::N_LAYER, "n_layer", Kind::U32),
    (tag::N_HEAD, "n_head", Kind::U32),
    (tag::N_KV_HEAD, "n_kv_head", Kind::U32),
    (tag::HEAD_DIM, "head_dim", Kind::U32),
    (tag::HIDDEN, "hidden", Kind::U32),
    (tag::MAX_SEQ, "max_seq", Kind::U32),
    (tag::ROPE_BASE, "rope_base", Kind::F64),
    (tag::RMS_EPS, "rms_eps", Kind::F64),
];

fn spec_from(f: &Fields<'_>) -> Result<ModelSpec, String> {
    let size = |t: u32| -> Result<usize, String> {
        usize::try_from(required(f.u32(t), t)?).map_err(|_| "new: size exceeds usize".to_string())
    };
    let spec = ModelSpec {
        vocab: size(tag::VOCAB)?,
        n_embd: size(tag::N_EMBD)?,
        n_layer: size(tag::N_LAYER)?,
        n_head: size(tag::N_HEAD)?,
        n_kv_head: size(tag::N_KV_HEAD)?,
        head_dim: size(tag::HEAD_DIM)?,
        hidden: size(tag::HIDDEN)?,
        max_seq: size(tag::MAX_SEQ)?,
        rope_base: required(f.f64(tag::ROPE_BASE), tag::ROPE_BASE)?,
        rms_eps: required(f.f64(tag::RMS_EPS), tag::RMS_EPS)?,
        tie_embeddings: true,
    };
    spec.validate().map_err(|e| crate::ojas_error("new", &e))?;
    Ok(spec)
}

/// NEW: the spec fields, `seed`, and the placement fields. A fresh nanolab
/// init (`ojas_model::init_params`) on the session's device.
pub fn new_request(bytes: &[u8], mut check: Check) -> Result<session::Session, String> {
    check()?;
    let mut r = Reader::new(bytes);
    let mut allowed = PLACEMENT_FIELDS.to_vec();
    allowed.extend(SPEC_FIELDS);
    allowed.push(field(tag::SEED, Kind::U64));
    let f = Fields::read(&mut r, &allowed)?;
    r.finish()?;
    let placement = placement(&f)?;
    let spec = spec_from(&f)?;
    let seed = required(f.u64(tag::SEED), tag::SEED)?;
    session::check_room()?;
    create(None, &placement, check, Fresh { spec, seed })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn replaced_file_fails_the_identity_check() {
        let dir = std::env::temp_dir().join(format!("ojas-root-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, b"first").unwrap();
        let file = File::open(&path).unwrap();
        confirm_open_identity(&file, &path, &dir).unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut replaced = File::create(&path).unwrap();
        replaced.write_all(b"second").unwrap();
        let err = confirm_open_identity(&file, &path, &dir).unwrap_err();
        assert!(err.contains("changed"), "{err}");
        drop(file);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_symlink_swapped_in_is_not_followed() {
        let dir = std::env::temp_dir().join(format!("ojas-root-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, b"first").unwrap();
        let file = File::open(&path).unwrap();
        let outside = std::env::temp_dir().join(format!("ojas-outside-{}", std::process::id()));
        std::fs::write(&outside, b"secret").unwrap();
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&outside, &path).unwrap();
        let err = confirm_open_identity(&file, &path, &dir).unwrap_err();
        assert!(err.contains("changed"), "{err}");
        drop(file);
        let _ = std::fs::remove_file(&outside);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_model_refuses_symlinks_and_opens_plain_files() {
        let dir = std::env::temp_dir().join(format!("ojas-io-open-model-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("model.safetensors");
        std::fs::write(&real, b"x").unwrap();
        assert!(open_model(&real).is_ok());
        let link = dir.join("link.safetensors");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let err = open_model(&link).unwrap_err();
        assert!(
            err.starts_with("missing file:") && err.contains("symbolic link"),
            "{err}"
        );
        let dangling = dir.join("dangling.safetensors");
        std::os::unix::fs::symlink(dir.join("nowhere"), &dangling).unwrap();
        assert!(open_model(&dangling).unwrap_err().contains("symbolic link"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A checkpoint directory resolves under the root like a file does, and
    /// one that would land outside it, on the root itself, or on a file is
    /// refused.
    #[test]
    fn checkpoint_directories_stay_under_the_root() {
        let dir = std::env::temp_dir().join(format!("ojas-root-dirs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("runs/a")).unwrap();
        std::fs::write(dir.join("file"), b"x").unwrap();
        let outside = std::env::temp_dir().join(format!("ojas-outside-dir-{}", std::process::id()));
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("escape")).unwrap();
        let canon = dir.canonicalize().unwrap();

        assert_eq!(
            resolve_dir_under_root(&dir, "runs/a", true).unwrap(),
            canon.join("runs/a")
        );
        assert_eq!(
            resolve_dir_under_root(&dir, "runs/new", false).unwrap(),
            canon.join("runs/new")
        );
        let refusals = [
            ("runs/new", true, "missing directory"),
            ("missing/new", false, "missing parent"),
            ("file", false, "not a directory"),
            ("file", true, "missing directory"),
            ("escape", true, "escapes"),
            ("escape/new", false, "escapes"),
            ("../x", false, ".."),
            ("/tmp/x", false, "relative"),
            (".", true, "empty"),
            ("runs/a/", true, "names a directory"),
        ];
        for (raw, exist, want) in refusals {
            let err = resolve_dir_under_root(&dir, raw, exist).unwrap_err();
            assert!(err.contains(want), "{raw} {exist}: {err}");
        }
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    /// The descriptor path opens the file the descriptor holds, even after
    /// the original name is replaced by a symlink to a file outside the root.
    #[test]
    fn the_descriptor_path_reopens_the_verified_file() {
        let dir = std::env::temp_dir().join(format!("ojas-fd-path-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tokens.bin");
        std::fs::write(&path, b"inside").unwrap();
        let verified = Verified {
            path: path.clone(),
            file: open_model(&path).unwrap(),
        };
        let outside = std::env::temp_dir().join(format!("ojas-fd-outside-{}", std::process::id()));
        std::fs::write(&outside, b"outside").unwrap();
        std::fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&outside, &path).unwrap();
        assert_eq!(std::fs::read(verified.fd_path()).unwrap(), b"inside");
        let fd = verified.fd_path();
        let detail = format!("{}: bad", fd.display());
        assert_eq!(
            verified.explain(&detail),
            format!("{}: bad", path.display())
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&outside);
    }
}
