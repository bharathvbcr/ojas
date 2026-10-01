//! Whole-file replacement. The bytes go to a sibling temp file, which is
//! synced and then renamed over the target. A reader that already holds the
//! target keeps the old bytes, and a failed or interrupted write never leaves
//! a partial target behind. Replacing a symlink replaces the link itself.

use crate::error::IoError;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

pub(crate) fn replace_file(path: &Path, bytes: &[u8]) -> Result<(), IoError> {
    let what = path.display();
    let name = path
        .file_name()
        .ok_or_else(|| IoError::new(format!("{what}: path has no file name")))?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let mut tmp_name = OsString::from(".");
    tmp_name.push(name);
    tmp_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let tmp = dir.join(tmp_name);
    let staged = write_synced(&tmp, bytes).and_then(|()| fs::rename(&tmp, path));
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

fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> std::io::Result<()> {
    fs::File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> std::io::Result<()> {
    Ok(())
}
