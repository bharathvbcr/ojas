//! Positioned reads: the file cursor is neither used nor moved, so a `File`
//! handed in by a caller reads the same whatever its cursor says.

use std::fs::File;
use std::io::{ErrorKind, Read};

/// One positioned read of up to `buf.len()` bytes at `at`.
#[cfg(unix)]
pub(crate) fn read_at(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, at)
}

#[cfg(windows)]
pub(crate) fn read_at(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, at)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn read_at(_file: &File, _buf: &mut [u8], _at: u64) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        ErrorKind::Unsupported,
        "positioned read needs unix or windows",
    ))
}

/// Fill `buf` from `at`; a short file is `UnexpectedEof`.
pub(crate) fn read_exact_at(file: &File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    let mut reader = ReadAt { file, pos: at };
    reader.read_exact(buf)
}

/// `Read` over positioned reads, starting at `pos`.
pub(crate) struct ReadAt<'a> {
    pub(crate) file: &'a File,
    pub(crate) pos: u64,
}

impl Read for ReadAt<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = read_at(self.file, buf, self.pos)?;
        let step =
            u64::try_from(n).map_err(|_| std::io::Error::other("read length exceeds u64"))?;
        self.pos = self
            .pos
            .checked_add(step)
            .ok_or_else(|| std::io::Error::new(ErrorKind::InvalidData, "file offset overflows"))?;
        Ok(n)
    }
}
