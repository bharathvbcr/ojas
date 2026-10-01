//! Whole-file replacement. The bytes go to a sibling temp file, which is
//! synced and then renamed over the target. A reader that already holds the
//! target keeps the old bytes, and a failed or interrupted write never leaves
//! a partial target behind. Replacing a symlink replaces the link itself.
//!
//! A crashed writer leaves `.{name}.{pid}.{seq}.tmp` beside the target. The
//! next replace of that same name removes only those leftovers. Other names,
//! and files that are not that three-part temp pattern, stay.

use crate::error::IoError;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

pub(crate) fn replace_file(path: &Path, bytes: &[u8]) -> Result<(), IoError> {
    replace_with(path, |file| {
        file.write_all(bytes)
            .map_err(|e| IoError::new(e.to_string()))
    })
}

pub(crate) fn replace_with<F>(path: &Path, write: F) -> Result<(), IoError>
where
    F: FnOnce(&mut File) -> Result<(), IoError>,
{
    let what = path.display();
    let name = path
        .file_name()
        .ok_or_else(|| IoError::new(format!("{what}: path has no file name")))?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    sweep_orphan_temps(dir, name)?;
    let mut tmp_name = std::ffi::OsString::from(".");
    tmp_name.push(name);
    tmp_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp = dir.join(tmp_name);
    let staged = (|| -> Result<(), IoError> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| IoError::new(e.to_string()))?;
        write(&mut file)?;
        file.sync_all().map_err(|e| IoError::new(e.to_string()))?;
        fs::rename(&tmp, path).map_err(|e| IoError::new(e.to_string()))?;
        Ok(())
    })();
    if let Err(e) = staged {
        return Err(match fs::remove_file(&tmp) {
            Err(c) if c.kind() != std::io::ErrorKind::NotFound => IoError::new(format!(
                "{what}: {e}; removing temp file {} also failed: {c}",
                tmp.display()
            )),
            _ => IoError::new(format!("{what}: {e}")),
        });
    }
    sync_dir(dir).map_err(|e| {
        IoError::new(format!(
            "{what}: written, but syncing its directory failed: {e}"
        ))
    })
}

/// Delete `.{name}.{pid}.{seq}.tmp` files in `dir`. `pid` and `seq` are
/// decimal. A directory with that name, or any other spelling, is left alone.
fn sweep_orphan_temps(dir: &Path, name: &OsStr) -> Result<(), IoError> {
    let entries = fs::read_dir(dir).map_err(|e| {
        IoError::new(format!(
            "{}: cannot scan temp files before replace: {e}",
            dir.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| {
            IoError::new(format!(
                "{}: cannot read a directory entry before replace: {e}",
                dir.display()
            ))
        })?;
        if !is_orphan_temp(name, &entry.file_name()) {
            continue;
        }
        let kind = entry.file_type().map_err(|e| {
            IoError::new(format!(
                "{}: cannot stat {} before replace: {e}",
                dir.display(),
                entry.path().display()
            ))
        })?;
        if kind.is_dir() {
            continue;
        }
        fs::remove_file(entry.path()).map_err(|e| {
            IoError::new(format!(
                "{}: cannot remove orphan temp {}: {e}",
                dir.display(),
                entry.path().display()
            ))
        })?;
    }
    Ok(())
}

fn is_orphan_temp(name: &OsStr, fname: &OsStr) -> bool {
    let file = fname.as_encoded_bytes();
    let name = name.as_encoded_bytes();
    let mut prefix = Vec::with_capacity(name.len() + 2);
    prefix.push(b'.');
    prefix.extend_from_slice(name);
    prefix.push(b'.');
    let Some(rest) = file.strip_prefix(prefix.as_slice()) else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(b".tmp") else {
        return false;
    };
    let mut parts = rest.split(|b| *b == b'.');
    let (Some(pid), Some(seq)) = (parts.next(), parts.next()) else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    is_digits(pid) && is_digits(seq)
}

fn is_digits(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(|b| b.is_ascii_digit())
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    fs::File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}
