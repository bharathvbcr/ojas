//! Relative paths under a configured root, and a short safetensors header check.

use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use crate::session;

/// Gusset copies at most this many inline input bytes (R16).
pub const INLINE_PATH_MAX: usize = 4096;

/// Official safetensors header-length cap.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// This loader reads the header to count `"dtype"` entries. Longer headers
/// are refused rather than allocated.
const HEADER_READ_CAP: u64 = 1 << 20;

pub fn resolve_under_root(root: &Path, raw: &str) -> Result<PathBuf, String> {
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
    if !acc.is_file() {
        return Err(format!("missing file: {}", acc.display()));
    }
    let canon = acc
        .canonicalize()
        .map_err(|err| format!("missing file: {err}"))?;
    if !canon.starts_with(&root) {
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

/// One read-only open. `O_NOFOLLOW` refuses a final-component symlink that
/// appeared after the path was resolved, instead of following it.
fn open_model(path: &Path) -> Result<File, String> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true);
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    opts.custom_flags(0x0000_0100);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    opts.custom_flags(0x0002_0000);
    opts.open(path)
        .map_err(|err| format!("missing file: {err}"))
}

/// Header length plus a count of `"dtype"` keys. Not a full safetensors parser.
fn inspect_file(file: &mut File) -> Result<u32, String> {
    let file_len = file
        .metadata()
        .map_err(|err| format!("missing file: {err}"))?
        .len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|_| "not a safetensors file: short header length".to_string())?;
    let n = u64::from_le_bytes(prefix);
    if n == 0 || n > MAX_HEADER_BYTES {
        return Err(format!("not a safetensors file: header length {n}"));
    }
    if n > HEADER_READ_CAP {
        return Err(format!(
            "not a safetensors file: header length {n} exceeds the 1 MiB this loader reads"
        ));
    }
    if file_len < 8u64.saturating_add(n) {
        return Err("not a safetensors file: truncated header".to_string());
    }
    let n_us =
        usize::try_from(n).map_err(|_| "not a safetensors file: header length".to_string())?;
    let mut header = vec![0u8; n_us];
    file.read_exact(&mut header)
        .map_err(|_| "not a safetensors file: truncated header".to_string())?;
    let text = std::str::from_utf8(&header)
        .map_err(|_| "not a safetensors file: header is not utf-8".to_string())?;
    if !text.trim_start().starts_with('{') {
        return Err("not a safetensors file: header is not a JSON object".to_string());
    }
    let tensors = text.matches("\"dtype\"").count();
    if tensors == 0 {
        return Err("not a safetensors file: header has no tensors".to_string());
    }
    u32::try_from(tensors).map_err(|_| "not a safetensors file: tensor count overflows".to_string())
}

/// Largest `threads` a [`DEVICE_CPU_PARALLEL`] load accepts. Larger values
/// are refused, not clamped. Each step builds a
/// [`ojas_cpu::CpuBackend::with_threads`] backend whose persistent worker
/// pool (spawned on the first op large enough to use it, shared by the
/// backend's clones) splits independent output rows across up to this many
/// workers.
pub const MAX_CPU_THREADS: u32 = 256;

pub const DEVICE_CPU: u32 = 0;
pub const DEVICE_CPU_PARALLEL: u32 = 1;
pub const DEVICE_METAL: u32 = 2;
pub const DEVICE_WGPU: u32 = 3;

pub fn load_path(raw: &str) -> Result<session::Session, String> {
    load_on(raw, session::Compute::Cpu { threads: 1 })
}

/// `OJDV` + kind u32 + threads u32 + relative path, or a bare relative path
/// for the CPU. Kind 0 CPU, 1 CPU parallel, 2 Metal, 3 wgpu.
///
/// CPU kinds run step and generate on [`ojas_cpu::CpuBackend`]. Metal checks
/// the file, then opens an [`ojas_metal::MetalBackend`] that the session
/// keeps, so its step and generate run on that device; an open failure is
/// the load's error, never a CPU session. wgpu does the same with an
/// [`ojas_wgpu::WgpuBackend`]: no adapter is the load's error and no session
/// is created. `check` is polled while a device open is pending; `threads`
/// is ignored for Metal and wgpu.
pub fn load_request(
    bytes: &[u8],
    check: impl FnMut() -> Result<(), String>,
) -> Result<session::Session, String> {
    let Some(rest) = bytes.strip_prefix(b"OJDV") else {
        let path = std::str::from_utf8(bytes).map_err(|_| "load: path is not utf-8")?;
        return load_path(path);
    };
    let (kind, rest) = split_u32(rest).ok_or("load: device header is short")?;
    let (threads, rest) = split_u32(rest).ok_or("load: device header is short")?;
    let path = std::str::from_utf8(rest).map_err(|_| "load: path is not utf-8")?;
    match kind {
        DEVICE_CPU => load_on(path, session::Compute::Cpu { threads: 1 }),
        DEVICE_CPU_PARALLEL => {
            if threads == 0 {
                return Err("load: thread count is 0".to_string());
            }
            if threads > MAX_CPU_THREADS {
                return Err(format!(
                    "load: thread count {threads} exceeds {MAX_CPU_THREADS}"
                ));
            }
            let threads = usize::try_from(threads).map_err(|_| "load: thread count")?;
            load_on(path, session::Compute::Cpu { threads })
        }
        DEVICE_METAL => load_metal(path, check),
        DEVICE_WGPU => load_wgpu(path, check),
        _ => Err(format!("load: unknown device {kind}")),
    }
}

fn split_u32(bytes: &[u8]) -> Option<(u32, &[u8])> {
    let (head, tail) = bytes.split_first_chunk::<4>()?;
    Some((u32::from_le_bytes(*head), tail))
}

fn checked_file(raw: &str) -> Result<(PathBuf, u32), String> {
    let root = session::root()?;
    let path = resolve_under_root(&root, raw)?;
    let mut file = open_model(&path)?;
    confirm_open_identity(&file, &path, &root)?;
    let tensors = inspect_file(&mut file)?;
    Ok((path, tensors))
}

fn load_on(raw: &str, compute: session::Compute) -> Result<session::Session, String> {
    let (path, tensors) = checked_file(raw)?;
    session::load_model_on(path, tensors, compute)
}

/// The file is checked before the device opens, so a bad path does not cost
/// a device thread. The backend charges the shared step process budget.
#[cfg(target_os = "macos")]
fn load_metal(
    raw: &str,
    check: impl FnMut() -> Result<(), String>,
) -> Result<session::Session, String> {
    let (path, tensors) = checked_file(raw)?;
    let backend = crate::owner::open_metal(crate::step::step_budget(), check)?;
    session::load_model_on(path, tensors, session::Compute::Metal(backend))
}

/// As [`load_metal`]: the file is checked first, and the backend charges the
/// shared step process budget.
fn load_wgpu(
    raw: &str,
    check: impl FnMut() -> Result<(), String>,
) -> Result<session::Session, String> {
    let (path, tensors) = checked_file(raw)?;
    let backend = crate::owner::open_wgpu(crate::step::step_budget(), check)?;
    session::load_model_on(
        path,
        tensors,
        session::Compute::Wgpu(std::sync::Arc::new(backend)),
    )
}

#[cfg(not(target_os = "macos"))]
fn load_metal(
    raw: &str,
    check: impl FnMut() -> Result<(), String>,
) -> Result<session::Session, String> {
    checked_file(raw)?;
    crate::owner::open_metal(check)?;
    Err("metal: Metal requires macOS".to_string())
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
}
