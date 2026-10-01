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

/// The opened file must still be the canonical file under `root`.
///
/// A path component swapped between resolving and opening changes `(dev, ino)`
/// or lands outside the root, and both are refused.
pub fn confirm_open_identity(file: &File, path: &Path, root: &Path) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    let opened = file
        .metadata()
        .map_err(|err| format!("model file: {err}"))?;
    let root = root
        .canonicalize()
        .map_err(|err| format!("model root: {err}"))?;
    let fresh = path.canonicalize().map_err(|err| format!("path: {err}"))?;
    if !fresh.starts_with(&root) {
        return Err("path escapes model root".to_string());
    }
    let again = File::open(&fresh)
        .map_err(|err| format!("path: {err}"))?
        .metadata()
        .map_err(|err| format!("path: {err}"))?;
    if again.dev() != opened.dev() || again.ino() != opened.ino() {
        return Err("model file changed while opening".to_string());
    }
    Ok(())
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

pub fn load_path(raw: &str) -> Result<session::Session, String> {
    load_on(raw, 1, false, false)
}

/// `OJDV` + kind u32 + threads u32 + relative path.
/// kind 0 CPU, 1 CPU parallel, 2 Metal, 3 wgpu.
pub fn load_request(bytes: &[u8]) -> Result<session::Session, String> {
    if bytes.starts_with(b"OJDV") {
        if bytes.len() < 12 {
            return Err("load: device header is short".to_string());
        }
        let kind = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let threads = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let path = std::str::from_utf8(&bytes[12..]).map_err(|_| "load: path is not utf-8")?;
        return match kind {
            0 => load_on(path, 1, false, false),
            1 => load_on(path, threads as usize, false, false),
            2 => load_on(path, 1, true, false),
            3 => load_on(path, 1, false, true),
            _ => Err(format!("load: unknown device {kind}")),
        };
    }
    let path = std::str::from_utf8(bytes).map_err(|_| "load: path is not utf-8")?;
    load_path(path)
}

fn load_on(
    raw: &str,
    threads: usize,
    metal: bool,
    wgpu_device: bool,
) -> Result<session::Session, String> {
    let root = session::root()?;
    let path = resolve_under_root(&root, raw)?;
    let mut file = File::open(&path).map_err(|err| format!("missing file: {err}"))?;
    confirm_open_identity(&file, &path, &root)?;
    let tensors = inspect_file(&mut file)?;
    if metal {
        let owner = crate::owner::MetalOwner::spawn()?;
        let session = session::load_model_with_threads(path, tensors, 1)?;
        crate::owner::retain(session.id, owner)?;
        return Ok(session);
    }
    if wgpu_device {
        ojas_wgpu::WgpuContext::open().map_err(|err| format!("wgpu: {err}"))?;
    }
    session::load_model_with_threads(path, tensors, threads)
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
}
