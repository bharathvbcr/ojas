//! SHA-256 (FIPS 180-4), streaming and one-shot, and a bounded-memory digest
//! of a file.
//!
//! Hand-rolled because ojas carries no `sha2` dependency. It is for artifact
//! and checkpoint digests (integrity pins), not for anything secret: it makes
//! no constant-time claim.

use crate::error::IoError;
use crate::posread::ReadAt;
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::Path;

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const BLOCK: usize = 64;

/// Streaming SHA-256 state.
#[derive(Clone, Debug)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; BLOCK],
    buf_len: usize,
    total_bytes: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

fn compress(state: &mut [u32; 8], block: &[u8; BLOCK]) {
    let mut w = [0u32; 64];
    for (i, word) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            state: INITIAL,
            buf: [0; BLOCK],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    /// Absorbs `data`. Any split of a message across calls gives the same
    /// digest.
    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (BLOCK - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len < BLOCK {
                return;
            }
            let block = self.buf;
            compress(&mut self.state, &block);
            self.buf_len = 0;
        }
        let mut blocks = data.chunks_exact(BLOCK);
        for block in &mut blocks {
            let block: &[u8; BLOCK] = block.try_into().expect("chunks_exact yields 64 bytes");
            compress(&mut self.state, block);
        }
        let rest = blocks.remainder();
        self.buf[..rest.len()].copy_from_slice(rest);
        self.buf_len = rest.len();
    }

    /// Pads, absorbs the bit length and returns the digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.update(&[0x80]);
        let zeros = [0u8; BLOCK];
        let pad = if self.buf_len <= 56 {
            56 - self.buf_len
        } else {
            120 - self.buf_len
        };
        self.update(&zeros[..pad]);
        self.update(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// The SHA-256 digest of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

/// Lowercase hex of `bytes`.
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 0xf) as usize] as char);
    }
    s
}

/// The SHA-256 digest of `data` as 64 lowercase hex characters.
pub fn sha256_hex(data: &[u8]) -> String {
    to_hex(&sha256(data))
}

/// Bytes read per call by [`file_sha256_hex`] and [`handle_sha256_hex`]: the
/// only buffer they hold, whatever the file's size.
pub const FILE_CHUNK_BYTES: usize = 1 << 20;

/// `path` for an error message. A symbolic link also names where it
/// resolves, so a refusal about a snapshot link says which blob it read.
/// The resolved path is for the message only; no identity is taken from it.
fn describe(path: &Path) -> String {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => match std::fs::canonicalize(path) {
            Ok(resolved) => format!("{} (resolves to {})", path.display(), resolved.display()),
            Err(_) => format!("{} (symbolic link that does not resolve)", path.display()),
        },
        _ => path.display().to_string(),
    }
}

/// The SHA-256 of the regular file `file` is open on, as 64 lowercase hex
/// characters, read from the start to end of file `FILE_CHUNK_BYTES` at a
/// time with positioned reads (the cursor is neither used nor moved, so a
/// caller can keep using the same handle). Anything but a regular file is
/// refused before a byte is read, so a FIFO cannot block. A zero-length file
/// is the digest of the empty message.
///
/// Hashing the handle a caller then parses closes the window between "hash
/// the path" and "open the path" in which the file could be swapped.
pub fn handle_sha256_hex(file: &File) -> Result<String, IoError> {
    let meta = file
        .metadata()
        .map_err(|e| IoError::new(format!("metadata: {e}")))?;
    if !meta.is_file() {
        return Err(IoError::new("not a regular file"));
    }
    let mut chunk = Vec::new();
    chunk
        .try_reserve_exact(FILE_CHUNK_BYTES)
        .map_err(|_| IoError::new(format!("read buffer of {FILE_CHUNK_BYTES} bytes refused")))?;
    chunk.resize(FILE_CHUNK_BYTES, 0u8);
    let mut reader = ReadAt { file, pos: 0 };
    let mut hash = Sha256::new();
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => hash.update(&chunk[..n]),
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => {
                return Err(IoError::new(format!("read at byte {}: {e}", reader.pos)));
            }
        }
    }
    Ok(to_hex(&hash.finalize()))
}

/// The SHA-256 of the file at `path` as 64 lowercase hex characters.
///
/// Memory is bounded by [`FILE_CHUNK_BYTES`], never the file's size. A
/// symbolic link is followed (a Hugging Face snapshot file is one); an error
/// about a link names the path it resolves to. A missing path, a directory
/// and any other non-regular file are errors; a zero-length file hashes to
/// the empty message's digest. The whole file is read once: at this module's
/// measured throughput that is seconds per GB (see `bench/sha256_throughput.rs`).
pub fn file_sha256_hex(path: &Path) -> Result<String, IoError> {
    open_sha256_hex(path).map(|(_, hex)| hex)
}

/// [`file_sha256_hex`] that also hands back the handle it hashed, so a caller
/// that goes on to parse the file reads the very bytes that were hashed (a
/// second `open` of the path could see a different file). Same errors.
pub fn open_sha256_hex(path: &Path) -> Result<(File, String), IoError> {
    let file = File::open(path).map_err(|e| IoError::new(format!("{}: {e}", describe(path))))?;
    let hex = handle_sha256_hex(&file)
        .map_err(|e| IoError::new(format!("{}: {}", describe(path), e.detail())))?;
    Ok((file, hex))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const TWO_BLOCK: &str = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
    const MILLION_A: &str = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";

    #[test]
    fn fips_empty_string() {
        assert_eq!(sha256_hex(b""), EMPTY);
    }

    #[test]
    fn fips_abc() {
        assert_eq!(sha256_hex(b"abc"), ABC);
    }

    #[test]
    fn fips_448_bit_message() {
        let msg = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(msg.len(), 56);
        assert_eq!(sha256_hex(msg), TWO_BLOCK);
    }

    #[test]
    fn fips_one_million_a() {
        let mut h = Sha256::new();
        let chunk = [b'a'; 1000];
        for _ in 0..1000 {
            h.update(&chunk);
        }
        assert_eq!(to_hex(&h.finalize()), MILLION_A);
        assert_eq!(sha256_hex(&vec![b'a'; 1_000_000]), MILLION_A);
    }

    #[test]
    fn hex_is_lowercase_and_padded() {
        assert_eq!(to_hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        assert_eq!(to_hex(&[]), "");
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(3))
            .collect()
    }

    #[test]
    fn split_at_every_boundary_equals_one_shot() {
        let data = pattern(200);
        let want = sha256(&data);
        for split in 0..=data.len() {
            let mut h = Sha256::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), want, "split at {split}");
        }
    }

    #[test]
    fn chunked_equals_one_shot_for_lengths_0_to_300() {
        for len in 0..=300 {
            let data = pattern(len);
            let want = sha256(&data);
            for chunk in [1usize, 3, 7, 31, 55, 56, 63, 64, 65, 100] {
                let mut h = Sha256::new();
                for piece in data.chunks(chunk) {
                    h.update(piece);
                }
                assert_eq!(h.finalize(), want, "len {len}, chunk {chunk}");
            }
        }
    }

    #[test]
    fn empty_updates_change_nothing() {
        let mut h = Sha256::new();
        h.update(b"");
        h.update(b"abc");
        h.update(b"");
        assert_eq!(to_hex(&h.finalize()), ABC);
    }

    /// FIPS 180-4 encodes the length mod 2^64 bits: a byte count that is
    /// 2^61 larger must pad identically (no panic, no saturation).
    #[test]
    fn bit_length_wraps_mod_2_pow_64() {
        for tail in [0usize, 1, 55, 56, 63] {
            let data = pattern(tail);
            let mut a = Sha256::new();
            a.update(&data);
            let mut b = Sha256::new();
            b.update(&data);
            b.total_bytes = b.total_bytes.wrapping_add(1u64 << 61);
            assert_eq!(a.finalize(), b.finalize(), "tail {tail}");
        }
    }

    #[test]
    fn padding_boundaries_differ_from_each_other() {
        let digests: Vec<[u8; 32]> = (54..=66).map(|n| sha256(&pattern(n))).collect();
        for i in 0..digests.len() {
            for j in i + 1..digests.len() {
                assert_ne!(digests[i], digests[j], "lengths {} and {}", 54 + i, 54 + j);
            }
        }
    }

    use crate::test_util::{tmp, tmp_dir};

    #[test]
    fn file_digest_equals_one_shot_around_the_chunk_size() {
        const MIB: usize = 1 << 20;
        assert_eq!(FILE_CHUNK_BYTES, MIB);
        for len in [0, 1, MIB - 1, MIB, MIB + 1, 3 * MIB + 17] {
            let data = pattern(len);
            let file = tmp("sha-file");
            std::fs::write(&file.0, &data).unwrap();
            assert_eq!(
                file_sha256_hex(&file.0).unwrap(),
                sha256_hex(&data),
                "len {len}"
            );
        }
    }

    #[test]
    fn zero_length_file_is_the_empty_digest() {
        let file = tmp("sha-empty");
        std::fs::write(&file.0, b"").unwrap();
        assert_eq!(file_sha256_hex(&file.0).unwrap(), EMPTY);
    }

    #[test]
    fn missing_path_and_directory_are_refused() {
        let dir = tmp_dir("sha-dir");
        let missing = dir.0.join("missing");
        let err = file_sha256_hex(&missing).unwrap_err();
        assert!(err.detail().contains("missing"), "{err}");
        let err = file_sha256_hex(&dir.0).unwrap_err();
        assert!(err.detail().contains("not a regular file"), "{err}");
    }

    #[test]
    fn handle_digest_ignores_and_keeps_the_cursor() {
        let data = pattern(5000);
        let file = tmp("sha-cursor");
        std::fs::write(&file.0, &data).unwrap();
        let mut open = File::open(&file.0).unwrap();
        let mut skip = [0u8; 100];
        std::io::Read::read_exact(&mut open, &mut skip).unwrap();
        assert_eq!(handle_sha256_hex(&open).unwrap(), sha256_hex(&data));
        let mut next = [0u8; 4];
        std::io::Read::read_exact(&mut open, &mut next).unwrap();
        assert_eq!(next, data[100..104]);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_is_followed_and_errors_name_the_target() {
        let dir = tmp_dir("sha-link");
        let real = dir.0.join("blob");
        let data = pattern(3000);
        std::fs::write(&real, &data).unwrap();
        let link = dir.0.join("snapshot");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert_eq!(file_sha256_hex(&link).unwrap(), sha256_hex(&data));

        let dir_link = dir.0.join("dir-link");
        std::os::unix::fs::symlink(&dir.0, &dir_link).unwrap();
        let err = file_sha256_hex(&dir_link).unwrap_err();
        assert!(err.detail().contains("resolves to"), "{err}");
        assert!(err.detail().contains("not a regular file"), "{err}");

        let dangling = dir.0.join("dangling");
        std::os::unix::fs::symlink(dir.0.join("nowhere"), &dangling).unwrap();
        let err = file_sha256_hex(&dangling).unwrap_err();
        assert!(err.detail().contains("does not resolve"), "{err}");
    }
}
