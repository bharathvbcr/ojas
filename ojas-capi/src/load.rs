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

/// Header length plus a count of `"dtype"` keys. Not a full safetensors parser.
pub fn inspect_safetensors(path: &Path) -> Result<u32, String> {
    let mut file = File::open(path).map_err(|err| format!("missing file: {err}"))?;
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

pub fn load_path(raw: &str) -> Result<session::Session, String> {
    let root = session::root()?;
    let path = resolve_under_root(&root, raw)?;
    let tensors = inspect_safetensors(&path)?;
    session::load_model(path, tensors)
}
