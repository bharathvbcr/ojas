//! Whole-file and whole-directory replacement.
//!
//! **Files** ([`replace_with`]). The bytes go to a sibling temp file, which is
//! synced and then renamed over the target. A reader that already holds the
//! target keeps the old bytes, and a failed or interrupted write never leaves
//! a partial target behind. Replacing a symlink replaces the link itself.
//!
//! A crashed writer leaves `.{name}.{pid}.{seq}.tmp` beside the target. The
//! next replace of that same name removes only those leftovers. Other names,
//! and files that are not that three-part temp pattern, stay.
//!
//! **Directories** ([`replace_dir_with`], unix only): a staging directory, two
//! renames through a backup, and [`recover_replaced_dir`] for the one crash
//! window that leaves no target. The function docs carry the details.

use crate::error::IoError;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::thread::ThreadId;

static SEQ: AtomicU64 = AtomicU64::new(0);

pub(crate) fn replace_with<F>(path: &Path, write: F) -> Result<(), IoError>
where
    F: FnOnce(&mut File) -> Result<(), IoError>,
{
    let what = path.display();
    let (dir, name) = split(path)?;
    sweep_orphan_temps(dir, name)?;
    let tmp = dir.join(temp_name(name, SEQ.fetch_add(1, Ordering::Relaxed), "tmp"));
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

/// The parent directory (`.` when there is none) and the file name.
fn split(path: &Path) -> Result<(&Path, &OsStr), IoError> {
    let name = path
        .file_name()
        .ok_or_else(|| IoError::new(format!("{}: path has no file name", path.display())))?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    Ok((dir, name))
}

/// `.{name}.{pid}.{seq}.{suffix}`.
fn temp_name(name: &OsStr, seq: u64, suffix: &str) -> OsString {
    let mut out = OsString::from(".");
    out.push(name);
    out.push(format!(".{}.{seq}.{suffix}", std::process::id()));
    out
}

/// Delete `.{name}.{pid}.{seq}.tmp` files in `dir`. `pid` and `seq` are
/// decimal. A directory with that name, or any other spelling, is left alone.
fn sweep_orphan_temps(dir: &Path, name: &OsStr) -> Result<(), IoError> {
    for (path, kind) in temps_of(dir, name, "tmp")? {
        if kind.is_dir() {
            continue;
        }
        fs::remove_file(&path).map_err(|e| {
            IoError::new(format!(
                "{}: cannot remove orphan temp {}: {e}",
                dir.display(),
                path.display()
            ))
        })?;
    }
    Ok(())
}

/// Entries of `dir` named `.{name}.{pid}.{seq}.{suffix}`, with their types.
/// The type is the entry's own: a symlink is a symlink, not its target.
fn temps_of(
    dir: &Path,
    name: &OsStr,
    suffix: &str,
) -> Result<Vec<(PathBuf, fs::FileType)>, IoError> {
    let entries = fs::read_dir(dir).map_err(|e| {
        IoError::new(format!(
            "{}: cannot scan temp files before replace: {e}",
            dir.display()
        ))
    })?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| {
            IoError::new(format!(
                "{}: cannot read a directory entry before replace: {e}",
                dir.display()
            ))
        })?;
        if !is_temp_name(name, &entry.file_name(), suffix) {
            continue;
        }
        let kind = entry.file_type().map_err(|e| {
            IoError::new(format!(
                "{}: cannot stat {} before replace: {e}",
                dir.display(),
                entry.path().display()
            ))
        })?;
        out.push((entry.path(), kind));
    }
    Ok(out)
}

fn is_temp_name(name: &OsStr, fname: &OsStr, suffix: &str) -> bool {
    let file = fname.as_encoded_bytes();
    let name = name.as_encoded_bytes();
    let mut prefix = Vec::with_capacity(name.len() + 2);
    prefix.push(b'.');
    prefix.extend_from_slice(name);
    prefix.push(b'.');
    let Some(rest) = file.strip_prefix(prefix.as_slice()) else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(suffix.as_bytes()) else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(b".") else {
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

/// Deepest staged tree [`replace_dir_with`] walks to fsync. A checkpoint is flat.
const MAX_STAGE_DEPTH: usize = 16;

/// Staging names tried before giving up. A name is never reused: an existing
/// entry with it is skipped, not written into.
const STAGE_ATTEMPTS: usize = 8;

/// Replace directory `target` with one `fill` writes.
///
/// `fill` gets a new, empty sibling staging directory
/// `.{name}.{pid}.{seq}.stage`. When it returns `Ok`, every file and
/// directory in the stage is fsynced (`F_FULLFSYNC` on Apple targets, which
/// is what std's `sync_all` issues there), then the stage is swapped in.
/// POSIX `rename` cannot replace a non-empty directory, and the atomic
/// exchange calls (`renameat2(RENAME_EXCHANGE)` on Linux,
/// `renamex_np(RENAME_SWAP)` on macOS) need FFI this crate forbids. So an
/// existing target is renamed to a backup `.{name}.{pid}.{seq}.old`, the
/// stage is renamed to the target, the parent is fsynced, and the backup is
/// removed and the parent fsynced again. A new target is one rename and one
/// parent fsync.
///
/// What a reader of `target` sees if the writer dies at each point:
/// 1. Before the first rename (filling or syncing the stage): the old
///    directory, whole. The partial stage is removed by the next call.
/// 2. Between the two renames: **no target.** The backup holds the old
///    directory, whole. [`recover_replaced_dir`], which this function also
///    runs first, renames it back. The finished stage is discarded, because
///    nothing records that it was complete.
/// 3. After the second rename: the new directory, whole. The backup, possibly
///    partly deleted, is removed by the next call.
///
/// Without a crash, a reader that looks between the two renames also finds no
/// target; that window is two `rename` calls long. A reader that opens two
/// files of the target separately can straddle a swap and get one old file
/// and one new one, so files that must agree need something to cross-check
/// (the checkpoint directory's step and run). A reader that already holds an
/// open file keeps its bytes.
///
/// Writers to targets in one parent directory are serialized by an exclusive
/// `flock` on that parent, so the leftover sweep never removes another
/// writer's live stage or backup. The lock blocks with no timeout; it is
/// released when the holder returns or unwinds, or its process dies. Writers
/// that bypass this function are not excluded.
///
/// Refused before anything is touched: a target or parent directory that is
/// a symbolic link (only the immediate parent is checked, since system temp
/// directories often sit under symlinked ancestors), a target that exists and
/// is not a directory, and a nested call for the same parent on the same
/// thread, which would otherwise deadlock on the lock. Inside the stage, a
/// symbolic link or anything but a regular file or directory is refused at
/// sync time. On any error, or a panic in `fill`, the stage is removed and the
/// old target is left as it was. Errors after the second rename say that the
/// target was replaced. Other platforms get an error: the swap relies on unix
/// rename and directory fsync.
pub fn replace_dir_with<F>(target: &Path, fill: F) -> Result<(), IoError>
where
    F: FnOnce(&Path) -> Result<(), IoError>,
{
    if cfg!(not(unix)) {
        return Err(IoError::new(format!(
            "{}: replace_dir_with needs unix rename and directory fsync",
            target.display()
        )));
    }
    let what = target.display();
    let mut site = Site::lock(target)?;
    site.recover()?;
    site.sweep_stages()?;
    let mut stage = Stage {
        path: create_stage(site.parent, site.name, &SEQ)?,
    };
    let staged = fill(&stage.path).and_then(|()| sync_tree(&stage.path, 0));
    if let Err(e) = staged {
        return Err(match stage.remove() {
            Ok(()) => IoError::new(format!("{what}: {}", e.detail())),
            Err(c) => IoError::new(format!(
                "{what}: {}; removing stage {} also failed: {c}",
                e.detail(),
                stage.path.display()
            )),
        });
    }
    site.swap(stage)
}

/// Undo a [`replace_dir_with`] that died between its two renames. If
/// `target` is missing and exactly one backup of it is beside it, the backup
/// is renamed back and `true` is returned. If `target` exists, leftover
/// backups are removed. Stale stages are removed either way. More than one
/// backup with no target is refused, since nothing says which is newer. Takes
/// the same lock as [`replace_dir_with`] and refuses the same symlinks.
pub fn recover_replaced_dir(target: &Path) -> Result<bool, IoError> {
    if cfg!(not(unix)) {
        return Err(IoError::new(format!(
            "{}: recover_replaced_dir needs unix",
            target.display()
        )));
    }
    let mut site = Site::lock(target)?;
    let restored = site.recover()?;
    site.sweep_stages()?;
    Ok(restored)
}

/// Parents locked by a live [`Site`] in this process, with the locking thread.
static HELD: Mutex<Vec<(PathBuf, ThreadId)>> = Mutex::new(Vec::new());

/// A target whose parent directory this thread holds the replace lock on.
struct Site<'a> {
    target: &'a Path,
    parent: &'a Path,
    name: &'a OsStr,
    /// `true` when the target existed (as a directory) once the lock was held.
    exists: bool,
    _lock: ParentLock,
}

struct ParentLock {
    key: PathBuf,
    /// Holds the `flock`; closing it releases the lock.
    _file: Option<File>,
}

impl Drop for ParentLock {
    fn drop(&mut self) {
        let me = std::thread::current().id();
        let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = held.iter().position(|(k, t)| *k == self.key && *t == me) {
            held.swap_remove(i);
        }
    }
}

impl<'a> Site<'a> {
    fn lock(target: &'a Path) -> Result<Self, IoError> {
        let what = target.display();
        let (parent, name) = split(target)?;
        let parent_kind = fs::symlink_metadata(parent)
            .map_err(|e| IoError::new(format!("{what}: parent {}: {e}", parent.display())))?;
        if parent_kind.file_type().is_symlink() {
            return Err(IoError::new(format!(
                "{what}: parent {} is a symbolic link",
                parent.display()
            )));
        }
        if !parent_kind.is_dir() {
            return Err(IoError::new(format!(
                "{what}: parent {} is not a directory",
                parent.display()
            )));
        }
        let key = fs::canonicalize(parent)
            .map_err(|e| IoError::new(format!("{what}: parent {}: {e}", parent.display())))?;
        let me = std::thread::current().id();
        let mut lock = {
            let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
            if held.iter().any(|(k, t)| *k == key && *t == me) {
                return Err(IoError::new(format!(
                    "{what}: a replace in {} is already running on this thread; \
                     a nested one would deadlock",
                    parent.display()
                )));
            }
            held.push((key.clone(), me));
            ParentLock { key, _file: None }
        };
        let file = File::open(parent)
            .map_err(|e| IoError::new(format!("{what}: parent {}: {e}", parent.display())))?;
        file.lock().map_err(|e| {
            IoError::new(format!(
                "{what}: cannot lock parent {}: {e}",
                parent.display()
            ))
        })?;
        lock._file = Some(file);
        let exists = match fs::symlink_metadata(target) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err(IoError::new(format!("{what}: is a symbolic link")));
            }
            Ok(m) if !m.is_dir() => {
                return Err(IoError::new(format!(
                    "{what}: exists and is not a directory"
                )));
            }
            Ok(_) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(IoError::new(format!("{what}: {e}"))),
        };
        Ok(Site {
            target,
            parent,
            name,
            exists,
            _lock: lock,
        })
    }

    /// Restore a lone backup over a missing target, or remove backups beside
    /// a present one. Returns whether a backup was restored.
    fn recover(&mut self) -> Result<bool, IoError> {
        let what = self.target.display();
        let backups: Vec<PathBuf> = temps_of(self.parent, self.name, "old")?
            .into_iter()
            .filter(|(_, kind)| kind.is_dir())
            .map(|(p, _)| p)
            .collect();
        if backups.is_empty() {
            return Ok(false);
        }
        if self.exists {
            for b in &backups {
                fs::remove_dir_all(b).map_err(|e| {
                    IoError::new(format!(
                        "{what}: cannot remove old backup {}: {e}",
                        b.display()
                    ))
                })?;
            }
            self.sync_parent("removed old backups")?;
            return Ok(false);
        }
        let [backup] = backups.as_slice() else {
            return Err(IoError::new(format!(
                "{what}: missing, with {} backups beside it; restore one by hand",
                backups.len()
            )));
        };
        fs::rename(backup, self.target).map_err(|e| {
            IoError::new(format!(
                "{what}: missing; restoring backup {} failed: {e}",
                backup.display()
            ))
        })?;
        self.exists = true;
        self.sync_parent("restored the backup")?;
        Ok(true)
    }

    /// Remove stale `.{name}.{pid}.{seq}.stage` directories. Under the lock
    /// no live writer owns one. Entries of other types are left alone.
    fn sweep_stages(&self) -> Result<(), IoError> {
        for (path, kind) in temps_of(self.parent, self.name, "stage")? {
            if !kind.is_dir() {
                continue;
            }
            fs::remove_dir_all(&path).map_err(|e| {
                IoError::new(format!(
                    "{}: cannot remove stale stage {}: {e}",
                    self.target.display(),
                    path.display()
                ))
            })?;
        }
        Ok(())
    }

    fn sync_parent(&self, done: &str) -> Result<(), IoError> {
        sync_dir(self.parent).map_err(|e| {
            IoError::new(format!(
                "{}: {done}, but syncing {} failed: {e}",
                self.target.display(),
                self.parent.display()
            ))
        })
    }

    /// Swap a synced `stage` in for the target. The site was locked before the
    /// stage was made, so `exists` still describes the target.
    fn swap(self, mut stage: Stage) -> Result<(), IoError> {
        let what = self.target.display();
        let fail = |stage: &mut Stage, msg: String| match stage.remove() {
            Ok(()) => IoError::new(msg),
            Err(c) => IoError::new(format!(
                "{msg}; removing stage {} also failed: {c}",
                stage.path.display()
            )),
        };
        if !self.exists {
            if let Err(e) = fs::rename(&stage.path, self.target) {
                return Err(fail(&mut stage, format!("{what}: {e}")));
            }
            stage.disarm();
            return self.sync_parent("replaced");
        }
        let backup = self.parent.join(temp_name(
            self.name,
            SEQ.fetch_add(1, Ordering::Relaxed),
            "old",
        ));
        // No fsync between the two renames: every state the parent can
        // persist (neither, the first, both) is one `recover` handles, and
        // the window in which a live reader finds no target stays short.
        if let Err(e) = fs::rename(self.target, &backup) {
            return Err(fail(&mut stage, format!("{what}: moving it aside: {e}")));
        }
        if let Err(e) = fs::rename(&stage.path, self.target) {
            let msg = format!("{what}: {e}");
            return Err(match fs::rename(&backup, self.target) {
                Ok(()) => fail(&mut stage, msg),
                Err(r) => fail(
                    &mut stage,
                    format!(
                        "{msg}; putting the old directory back failed ({r}); it is at {} and \
                         recover_replaced_dir restores it",
                        backup.display()
                    ),
                ),
            });
        }
        stage.disarm();
        self.sync_parent("replaced")?;
        fs::remove_dir_all(&backup).map_err(|e| {
            IoError::new(format!(
                "{what}: replaced, but removing the old directory {} failed: {e}; \
                 the next replace removes it",
                backup.display()
            ))
        })?;
        self.sync_parent("replaced and removed the old directory")
    }
}

/// A staging directory, removed on drop unless it was swapped in. Drop is the
/// cleanup for a panicking `fill`; error paths call [`Stage::remove`] so a
/// failed removal is reported.
struct Stage {
    path: PathBuf,
}

impl Stage {
    /// On failure the stage stays armed: the path is still there for the
    /// caller's message, and drop tries once more.
    fn remove(&mut self) -> std::io::Result<()> {
        match fs::remove_dir_all(&self.path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => {
                self.disarm();
                Ok(())
            }
        }
    }

    fn disarm(&mut self) {
        self.path = PathBuf::new();
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if !self.path.as_os_str().is_empty() {
            // Unwinding: no caller to report to. The next replace of this
            // target sweeps anything left.
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

/// Create a new, empty `.{name}.{pid}.{seq}.stage` in `parent`, taking `seq`
/// from `counter`. A name that already exists, as anything, is skipped.
fn create_stage(parent: &Path, name: &OsStr, counter: &AtomicU64) -> Result<PathBuf, IoError> {
    for _ in 0..STAGE_ATTEMPTS {
        let path = parent.join(temp_name(
            name,
            counter.fetch_add(1, Ordering::Relaxed),
            "stage",
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(IoError::new(format!(
                    "cannot create stage {}: {e}",
                    path.display()
                )))
            }
        }
    }
    Err(IoError::new(format!(
        "{}: {STAGE_ATTEMPTS} stage names in a row already exist",
        parent.display()
    )))
}

/// fsync every regular file and directory under `dir`, then `dir` itself.
fn sync_tree(dir: &Path, depth: usize) -> Result<(), IoError> {
    if depth > MAX_STAGE_DEPTH {
        return Err(IoError::new(format!(
            "{}: staged tree deeper than {MAX_STAGE_DEPTH}",
            dir.display()
        )));
    }
    let entries = fs::read_dir(dir).map_err(|e| IoError::new(format!("{}: {e}", dir.display())))?;
    for entry in entries {
        let entry = entry.map_err(|e| IoError::new(format!("{}: {e}", dir.display())))?;
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|e| IoError::new(format!("{}: {e}", path.display())))?;
        if kind.is_symlink() {
            return Err(IoError::new(format!(
                "staged {} is a symbolic link",
                path.display()
            )));
        } else if kind.is_dir() {
            sync_tree(&path, depth + 1)?;
        } else if kind.is_file() {
            File::open(&path)
                .and_then(|f| f.sync_all())
                .map_err(|e| IoError::new(format!("syncing {}: {e}", path.display())))?;
        } else {
            return Err(IoError::new(format!(
                "staged {} is not a regular file or directory",
                path.display()
            )));
        }
    }
    sync_dir(dir).map_err(|e| IoError::new(format!("syncing {}: {e}", dir.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::tmp_dir;

    #[test]
    fn stage_names_that_exist_are_skipped_never_reused() {
        let dir = tmp_dir("stage-collide");
        let pid = std::process::id();
        let taken_dir = dir.0.join(format!(".ckpt.{pid}.0.stage"));
        fs::create_dir(&taken_dir).unwrap();
        fs::write(taken_dir.join("theirs"), b"keep").unwrap();
        let taken_file = dir.0.join(format!(".ckpt.{pid}.1.stage"));
        fs::write(&taken_file, b"keep").unwrap();
        let counter = AtomicU64::new(0);
        let got = create_stage(&dir.0, OsStr::new("ckpt"), &counter).unwrap();
        assert_eq!(got, dir.0.join(format!(".ckpt.{pid}.2.stage")));
        assert!(fs::read_dir(&got).unwrap().next().is_none());
        assert_eq!(fs::read(taken_dir.join("theirs")).unwrap(), b"keep");
        assert_eq!(fs::read(&taken_file).unwrap(), b"keep");

        let full = AtomicU64::new(100);
        for seq in 100..100 + STAGE_ATTEMPTS as u64 {
            fs::write(dir.0.join(format!(".ckpt.{pid}.{seq}.stage")), b"x").unwrap();
        }
        let err = create_stage(&dir.0, OsStr::new("ckpt"), &full).unwrap_err();
        assert!(err.detail().contains("already exist"), "{err}");
    }

    #[test]
    fn temp_names_need_the_exact_suffix_and_two_decimal_fields() {
        let n = OsStr::new("ckpt");
        let yes = |f: &str, s: &str| is_temp_name(n, OsStr::new(f), s);
        assert!(yes(".ckpt.12.0.stage", "stage"));
        assert!(yes(".ckpt.12.0.old", "old"));
        assert!(yes(".ckpt.12.0.tmp", "tmp"));
        assert!(!yes(".ckpt.12.0.stage", "old"));
        assert!(!yes(".ckpt.12.0xstage", "stage"));
        assert!(!yes(".ckpt.12.stage", "stage"));
        assert!(!yes(".ckpt.1.2.3.stage", "stage"));
        assert!(!yes(".ckpt.a.2.stage", "stage"));
        assert!(!yes(".ckpt..2.stage", "stage"));
        assert!(!yes(".ckpt2.1.2.stage", "stage"));
        assert!(!yes("ckpt.1.2.stage", "stage"));
    }
}
