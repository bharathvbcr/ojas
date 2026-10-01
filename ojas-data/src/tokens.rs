//! Headerless little-endian `u16` token bins, and the FineWeb 256×`i32`
//! header (`magic` 20240520 at word 0, version at word 1, token count at word 2).
//!
//! Reads are positioned. A shard is not loaded with `fs::read`.

use crate::error::DataError;
use std::fs::File;
use std::path::Path;

pub const FINEWEB_MAGIC: i32 = 20240520;
pub const FINEWEB_VERSION: i32 = 1;
pub const FINEWEB_HEADER_BYTES: u64 = 256 * 4;

#[derive(Debug)]
pub struct TokenBin {
    file: File,
    /// Byte offset of token 0.
    data_offset: u64,
    len_tokens: u64,
}

impl TokenBin {
    /// Headerless `u16` little-endian stream. An odd file length is an error.
    /// The first even prefix is not returned.
    pub fn open_headerless(path: &Path) -> Result<Self, DataError> {
        let (file, len) = open_len(path)?;
        if len % 2 != 0 {
            return Err(DataError::new(format!(
                "{}: truncated token bin, {len} bytes is not a multiple of 2",
                path.display()
            )));
        }
        Ok(Self {
            file,
            data_offset: 0,
            len_tokens: len / 2,
        })
    }

    /// FineWeb header: 256 little-endian `i32`s, then `u16` tokens.
    /// A magic other than [`FINEWEB_MAGIC`] is an error. The byte length must
    /// be exactly the header plus two bytes per token. A short file is truncated.
    pub fn open_fineweb(path: &Path) -> Result<Self, DataError> {
        let (file, len) = open_len(path)?;
        if len < FINEWEB_HEADER_BYTES {
            return Err(DataError::new(format!(
                "{}: truncated FineWeb header, {len} bytes",
                path.display()
            )));
        }
        let mut header = [0u8; FINEWEB_HEADER_BYTES as usize];
        read_exact_at(&file, &mut header, 0)
            .map_err(|e| DataError::new(format!("{}: header: {e}", path.display())))?;
        let magic = read_i32(&header, 0);
        let version = read_i32(&header, 1);
        let count = read_i32(&header, 2);
        if magic != FINEWEB_MAGIC {
            return Err(DataError::new(format!(
                "{}: magic {magic} != {FINEWEB_MAGIC}",
                path.display()
            )));
        }
        if version != FINEWEB_VERSION {
            return Err(DataError::new(format!(
                "{}: version {version} != {FINEWEB_VERSION}",
                path.display()
            )));
        }
        if count < 0 {
            return Err(DataError::new(format!(
                "{}: token count {count} is negative",
                path.display()
            )));
        }
        let count_u = count as u64;
        let payload = count_u.checked_mul(2).ok_or_else(|| {
            DataError::new(format!("{}: token byte length overflows", path.display()))
        })?;
        let expect = FINEWEB_HEADER_BYTES
            .checked_add(payload)
            .ok_or_else(|| DataError::new(format!("{}: file length overflows", path.display())))?;
        if len != expect {
            return Err(DataError::new(format!(
                "{}: size mismatch, expected {expect} bytes, got {len}",
                path.display()
            )));
        }
        Ok(Self {
            file,
            data_offset: FINEWEB_HEADER_BYTES,
            len_tokens: count_u,
        })
    }

    pub fn len(&self) -> u64 {
        self.len_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.len_tokens == 0
    }

    /// Little-endian `u16` tokens starting at `index`, into `dst`.
    /// Does not read the rest of the file. A range past the end is an error.
    pub fn read_into(&self, index: u64, dst: &mut [u16]) -> Result<(), DataError> {
        let n =
            u64::try_from(dst.len()).map_err(|_| DataError::new("window length exceeds u64"))?;
        let end = index
            .checked_add(n)
            .ok_or_else(|| DataError::new("token index overflows"))?;
        if end > self.len_tokens {
            return Err(DataError::new(format!(
                "token window [{index}, {end}) past length {}",
                self.len_tokens
            )));
        }
        if dst.is_empty() {
            return Ok(());
        }
        let byte_off = index
            .checked_mul(2)
            .and_then(|b| b.checked_add(self.data_offset))
            .ok_or_else(|| DataError::new("token byte offset overflows"))?;
        let nbytes = dst
            .len()
            .checked_mul(2)
            .ok_or_else(|| DataError::new("window byte length overflows"))?;
        let mut raw = Vec::new();
        raw.try_reserve_exact(nbytes)
            .map_err(|_| DataError::new(format!("allocation of {nbytes} bytes refused")))?;
        raw.resize(nbytes, 0);
        read_exact_at(&self.file, &mut raw, byte_off)
            .map_err(|e| DataError::new(format!("token read: {e}")))?;
        for (i, slot) in dst.iter_mut().enumerate() {
            let o = i * 2;
            *slot = u16::from_le_bytes([raw[o], raw[o + 1]]);
        }
        Ok(())
    }
}

fn open_len(path: &Path) -> Result<(File, u64), DataError> {
    let file = File::open(path).map_err(|e| DataError::new(format!("{}: {e}", path.display())))?;
    let len = file
        .metadata()
        .map_err(|e| DataError::new(format!("{}: {e}", path.display())))?
        .len();
    Ok((file, len))
}

fn read_i32(header: &[u8], index: usize) -> i32 {
    let o = index * 4;
    i32::from_le_bytes([header[o], header[o + 1], header[o + 2], header[o + 3]])
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut rest = buf;
        let mut at = offset;
        while !rest.is_empty() {
            let n = file.seek_read(rest, at)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read",
                ));
            }
            at = at.saturating_add(n as u64);
            rest = &mut rest[n..];
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positioned read needs unix or windows",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::CounterRng;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    struct Tmp(std::path::PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    fn tmp(tag: &str) -> Tmp {
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        Tmp(std::env::temp_dir().join(format!("ojas-bin-{}-{tag}-{n}", std::process::id())))
    }

    fn write(path: &std::path::Path, bytes: &[u8]) {
        let mut f = File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    #[test]
    fn headerless_window_and_odd_length() {
        let path = tmp("raw");
        let mut bytes = Vec::new();
        for t in [1u16, 2, 3, 4] {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        write(&path.0, &bytes);
        let bin = TokenBin::open_headerless(&path.0).unwrap();
        assert_eq!(bin.len(), 4);
        let mut one = [0u16; 1];
        bin.read_into(2, &mut one).unwrap();
        assert_eq!(one, [3]);
        assert!(bin.read_into(3, &mut [0, 0]).is_err());

        let odd = tmp("odd");
        write(&odd.0, &[1, 2, 3]);
        assert!(TokenBin::open_headerless(&odd.0).is_err());
    }

    #[test]
    fn fineweb_magic_count_and_mismatch() {
        let path = tmp("fw");
        let mut bytes = vec![0u8; FINEWEB_HEADER_BYTES as usize];
        bytes[0..4].copy_from_slice(&FINEWEB_MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&FINEWEB_VERSION.to_le_bytes());
        bytes[8..12].copy_from_slice(&3i32.to_le_bytes());
        for t in [9u16, 8, 7] {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        write(&path.0, &bytes);
        let bin = TokenBin::open_fineweb(&path.0).unwrap();
        let mut got = [0u16; 3];
        bin.read_into(0, &mut got).unwrap();
        assert_eq!(got, [9, 8, 7]);

        let bad = tmp("magic");
        let mut wrong = bytes.clone();
        wrong[0..4].copy_from_slice(&1i32.to_le_bytes());
        write(&bad.0, &wrong);
        let err = TokenBin::open_fineweb(&bad.0).unwrap_err();
        assert!(err.detail().contains("magic"), "{err}");

        let short = tmp("short");
        write(&short.0, &bytes[..bytes.len() - 2]);
        assert!(TokenBin::open_fineweb(&short.0)
            .unwrap_err()
            .detail()
            .contains("size"));
    }

    fn fineweb(magic: i32, version: i32, count: i32, tokens: &[u16]) -> Vec<u8> {
        let mut bytes = vec![0u8; FINEWEB_HEADER_BYTES as usize];
        bytes[0..4].copy_from_slice(&magic.to_le_bytes());
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        bytes[8..12].copy_from_slice(&count.to_le_bytes());
        for t in tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        bytes
    }

    /// Every window of `bin` reads `tokens` when in range and errs otherwise.
    fn check_windows(bin: &TokenBin, tokens: &[u16], rng: &mut CounterRng) {
        assert_eq!(bin.len(), tokens.len() as u64);
        assert_eq!(bin.is_empty(), tokens.is_empty());
        for _ in 0..8 {
            let index = match rng.next_u64() % 4 {
                0 => u64::MAX - rng.next_u64() % 3,
                _ => rng.next_u64() % (tokens.len() as u64 + 3),
            };
            let mut dst = vec![0u16; (rng.next_u64() % 6) as usize];
            let fits = index
                .checked_add(dst.len() as u64)
                .is_some_and(|end| end <= tokens.len() as u64);
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bin.read_into(index, &mut dst)
            }))
            .unwrap_or_else(|_| panic!("read_into({index}, {}) panicked", dst.len()));
            assert_eq!(got.is_ok(), fits, "window {index}+{}", dst.len());
            if fits {
                let i = index as usize;
                assert_eq!(dst, tokens[i..i + dst.len()]);
            }
        }
    }

    #[test]
    fn headerless_every_length_and_window() {
        let mut rng = CounterRng::new(11);
        let path = tmp("every");
        for len in 0..80usize {
            let bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
            write(&path.0, &bytes);
            let opened = TokenBin::open_headerless(&path.0);
            assert_eq!(opened.is_ok(), len.is_multiple_of(2), "{len}");
            if let Ok(bin) = opened {
                let tokens: Vec<u16> = bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect();
                check_windows(&bin, &tokens, &mut rng);
            }
        }
    }

    #[test]
    fn mutated_fineweb_headers_never_panic_and_only_exact_files_open() {
        let mut rng = CounterRng::new(0xF1E);
        let path = tmp("fw-fuzz");
        let mut opened = 0;
        for _ in 0..600 {
            let n = (rng.next_u64() % 8) as usize;
            let tokens: Vec<u16> = (0..n).map(|_| rng.next_u64() as u16).collect();
            let pick = |rng: &mut CounterRng, good: i32| match rng.next_u64() % 4 {
                0 => rng.next_u64() as i32,
                1 => [i32::MIN, -1, 0, i32::MAX][(rng.next_u64() % 4) as usize],
                _ => good,
            };
            let magic = pick(&mut rng, FINEWEB_MAGIC);
            let version = pick(&mut rng, FINEWEB_VERSION);
            let count = pick(&mut rng, n as i32);
            let mut bytes = fineweb(magic, version, count, &tokens);
            if rng.next_u64().is_multiple_of(4) {
                let cut = (rng.next_u64() % (bytes.len() as u64 + 1)) as usize;
                bytes.truncate(cut);
            }
            write(&path.0, &bytes);
            let got = std::panic::catch_unwind(|| TokenBin::open_fineweb(&path.0))
                .unwrap_or_else(|_| panic!("open_fineweb panicked on {magic} {version} {count}"));
            let exact = magic == FINEWEB_MAGIC
                && version == FINEWEB_VERSION
                && count == n as i32
                && bytes.len() == FINEWEB_HEADER_BYTES as usize + 2 * n;
            assert_eq!(
                got.is_ok(),
                exact,
                "{magic} {version} {count} {}",
                bytes.len()
            );
            if let Ok(bin) = got {
                check_windows(&bin, &tokens, &mut rng);
                opened += 1;
            }
        }
        assert!(opened > 30, "opened {opened}");
        write(&path.0, &[]);
        assert!(TokenBin::open_fineweb(&path.0).is_err());
        assert!(TokenBin::open_headerless(&path.0).unwrap().is_empty());
    }

    #[test]
    fn odd_length_is_not_a_truncated_prefix() {
        for len in [1usize, 3, 5, 255] {
            let path = tmp("odd-len");
            let bytes: Vec<u8> = (0..len).map(|i| i as u8).collect();
            write(&path.0, &bytes);
            let opened = std::panic::catch_unwind(|| TokenBin::open_headerless(&path.0))
                .unwrap_or_else(|_| panic!("open panicked on {len} bytes"));
            let err = opened.expect_err("odd file was opened");
            assert!(
                err.detail().contains("multiple of 2") || err.detail().contains("truncated"),
                "{len}: {err}"
            );
        }
        let path = tmp("one-tok");
        write(&path.0, &0xBEEFu16.to_le_bytes());
        let bin = TokenBin::open_headerless(&path.0).unwrap();
        let mut got = [0u16; 1];
        bin.read_into(0, &mut got).unwrap();
        assert_eq!(got, [0xBEEF]);
    }

    #[test]
    fn fineweb_wrong_magic_and_count_that_disagrees_with_the_payload() {
        let token = 0xBEEFu16;
        let mut payload = token.to_le_bytes().to_vec();
        payload.extend_from_slice(&0x1111u16.to_le_bytes());

        let bad_magic = tmp("bad-magic");
        write(&bad_magic.0, &fineweb(1, FINEWEB_VERSION, 1, &[token]));
        let err = TokenBin::open_fineweb(&bad_magic.0).unwrap_err();
        assert!(err.detail().contains("magic"), "{err}");

        let negative = tmp("neg");
        write(
            &negative.0,
            &fineweb(FINEWEB_MAGIC, FINEWEB_VERSION, -1, &[]),
        );
        let err = TokenBin::open_fineweb(&negative.0).unwrap_err();
        assert!(err.detail().contains("negative"), "{err}");

        let short = tmp("count-high");
        write(
            &short.0,
            &fineweb(FINEWEB_MAGIC, FINEWEB_VERSION, 2, &[token]),
        );
        let err = TokenBin::open_fineweb(&short.0).unwrap_err();
        assert!(err.detail().contains("size"), "{err}");

        let extra = tmp("count-low");
        write(
            &extra.0,
            &fineweb(FINEWEB_MAGIC, FINEWEB_VERSION, 1, &[token, 0x1111]),
        );
        let err = TokenBin::open_fineweb(&extra.0).unwrap_err();
        assert!(err.detail().contains("size"), "{err}");

        let ok = tmp("count-ok");
        // Header's first words are the magic, not the token. A reader that
        // starts at byte 0 returns the magic instead of 0xBEEF.
        write(
            &ok.0,
            &fineweb(FINEWEB_MAGIC, FINEWEB_VERSION, 2, &[token, 0x1111]),
        );
        let bin = TokenBin::open_fineweb(&ok.0).unwrap();
        let mut got = [0u16; 2];
        bin.read_into(0, &mut got).unwrap();
        assert_eq!(got, [token, 0x1111]);
        assert_ne!(got[0], FINEWEB_MAGIC as u16);
    }
}
