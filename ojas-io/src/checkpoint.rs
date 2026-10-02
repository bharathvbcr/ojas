//! Checkpoint v1 file body, matching `docs/checkpoint-v1.md`.
//!
//! The 12-byte prefix is [`ojas_core::prefix_bytes`]. Sections after it are
//! length-prefixed except the fixed hash, SHA, step, and cursor fields.
//! A short file or a length that runs past the end is an error.

use crate::error::IoError;
use crate::posread::ReadAt;
use crate::replace::replace_with;
use ojas_core::{read_prefix, CheckpointV1, DType, DataCursor, NamedBlob, OptimizerCheckpoint};
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

/// The target is replaced by rename, so a failed write leaves the previous
/// checkpoint intact. A checkpoint the reader would refuse is not written.
pub fn write_checkpoint(path: &Path, ckpt: &CheckpointV1) -> Result<(), IoError> {
    let n = prepare(ckpt)?;
    replace_with(path, |file| {
        write_to(file, ckpt)?;
        let pos = file
            .stream_position()
            .map_err(|e| IoError::new(format!("checkpoint write: {e}")))?;
        if pos != n {
            return Err(IoError::new(format!(
                "checkpoint length mismatch: wrote {pos}, expected {n}"
            )));
        }
        Ok(())
    })
}

/// Refused before `read_checkpoint` reserves a buffer.
pub const MAX_CHECKPOINT_BYTES: u64 = 1 << 30;

/// `n == cap` is allowed. One byte past `cap` is not.
///
/// Production callers pass [`MAX_CHECKPOINT_BYTES`]. A test can pass a
/// smaller cap and check the same compare without allocating a gibibyte.
fn exceeds_checkpoint_cap(n: u64, cap: u64) -> bool {
    n > cap
}

pub fn read_checkpoint(path: &Path) -> Result<CheckpointV1, IoError> {
    let what = path.display();
    let file = File::open(path).map_err(|e| IoError::new(format!("{what}: {e}")))?;
    read_checkpoint_from(file).map_err(|e| IoError::new(format!("{what}: {}", e.detail())))
}

/// Decode a checkpoint from a file the caller already opened, for example
/// with [`crate::open_nofollow`]. Same checks and cap as [`read_checkpoint`],
/// which calls this. Anything but a regular file is refused before a byte is
/// read. Reads are positioned from byte 0: the file's cursor is neither used
/// nor moved.
pub fn read_checkpoint_from(file: File) -> Result<CheckpointV1, IoError> {
    let meta = file.metadata().map_err(|e| IoError::new(e.to_string()))?;
    if !meta.is_file() {
        return Err(IoError::new("not a regular file"));
    }
    let file_len = meta.len();
    if exceeds_checkpoint_cap(file_len, MAX_CHECKPOINT_BYTES) {
        return Err(IoError::new(format!(
            "{file_len} bytes exceeds checkpoint cap {MAX_CHECKPOINT_BYTES}"
        )));
    }
    // Stream records into the decoded checkpoint. The file bytes are not held
    // beside a second copy of every payload.
    let mut at = ReadAt {
        file: &file,
        pos: 0,
    };
    decode_reader(&mut (&mut at).take(file_len), file_len)
}

pub fn encode_checkpoint(ckpt: &CheckpointV1) -> Result<Vec<u8>, IoError> {
    let n = prepare(ckpt)?;
    let n_us = usize::try_from(n).map_err(|_| IoError::new("checkpoint length exceeds usize"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(n_us)
        .map_err(|_| IoError::new(format!("checkpoint allocation of {n_us} bytes refused")))?;
    write_to(&mut out, ckpt)?;
    if out.len() as u64 != n {
        return Err(IoError::new(format!(
            "checkpoint length mismatch: encoded {}, expected {n}",
            out.len()
        )));
    }
    Ok(out)
}

pub fn decode_checkpoint(bytes: &[u8]) -> Result<CheckpointV1, IoError> {
    let len =
        u64::try_from(bytes.len()).map_err(|_| IoError::new("checkpoint length exceeds u64"))?;
    decode_reader(&mut std::io::Cursor::new(bytes), len)
}

fn prepare(ckpt: &CheckpointV1) -> Result<u64, IoError> {
    validate_list("weights", &ckpt.weights)?;
    validate_list("muon_momentum", &ckpt.optimizer.muon_momentum)?;
    validate_list("adamw_first_moment", &ckpt.optimizer.adamw_first_moment)?;
    validate_list("adamw_second_moment", &ckpt.optimizer.adamw_second_moment)?;
    let n = encoded_len(ckpt)?;
    if exceeds_checkpoint_cap(n, MAX_CHECKPOINT_BYTES) {
        return Err(IoError::new(format!(
            "checkpoint of {n} bytes exceeds cap {MAX_CHECKPOINT_BYTES}"
        )));
    }
    Ok(n)
}

fn encoded_len(ckpt: &CheckpointV1) -> Result<u64, IoError> {
    let mut n = 12u64;
    n = add(n, lp_len(&ckpt.config)?)?;
    n = add(n, 32 + 20 + 8 + 8 + 8)?;
    n = add(n, lp_len(&ckpt.rng_state)?)?;
    n = add(n, section_len(&ckpt.weights)?)?;
    n = add(n, section_len(&ckpt.optimizer.muon_momentum)?)?;
    n = add(n, section_len(&ckpt.optimizer.adamw_first_moment)?)?;
    n = add(n, section_len(&ckpt.optimizer.adamw_second_moment)?)?;
    Ok(n)
}

fn add(n: u64, m: u64) -> Result<u64, IoError> {
    n.checked_add(m)
        .ok_or_else(|| IoError::new("checkpoint length overflows"))
}

fn u64_len(bytes: &[u8]) -> Result<u64, IoError> {
    u64::try_from(bytes.len()).map_err(|_| IoError::new("section length exceeds u64"))
}

fn lp_len(bytes: &[u8]) -> Result<u64, IoError> {
    add(8, u64_len(bytes)?)
}

fn section_len(blobs: &[NamedBlob]) -> Result<u64, IoError> {
    let mut body = 8u64;
    for blob in blobs {
        body = add(body, record_len(blob)?)?;
    }
    add(8, body)
}

fn record_len(b: &NamedBlob) -> Result<u64, IoError> {
    let name = u64::try_from(b.name.len()).map_err(|_| IoError::new("name length exceeds u64"))?;
    let rank = u64::try_from(b.shape.len())
        .map_err(|_| IoError::new(format!("{:?}: rank exceeds u64", b.name)))?;
    let shape = rank
        .checked_mul(8)
        .ok_or_else(|| IoError::new(format!("{:?}: shape byte length overflows", b.name)))?;
    let payload =
        u64::try_from(b.bytes.len()).map_err(|_| IoError::new("payload length exceeds u64"))?;
    let mut n = add(8, name)?;
    n = add(n, 4 + 4)?;
    n = add(n, shape)?;
    n = add(n, 8 + 8)?;
    add(n, payload)
}

fn write_to(out: &mut impl Write, ckpt: &CheckpointV1) -> Result<(), IoError> {
    put(out, &ojas_core::prefix_bytes())?;
    write_lp(out, &ckpt.config)?;
    put(out, &ckpt.tokenizer_hash)?;
    put(out, &ckpt.git_sha)?;
    put(out, &ckpt.step.to_le_bytes())?;
    put(out, &ckpt.data_cursor.shard.to_le_bytes())?;
    put(out, &ckpt.data_cursor.token_index.to_le_bytes())?;
    write_lp(out, &ckpt.rng_state)?;
    write_blob_section(out, &ckpt.weights)?;
    write_blob_section(out, &ckpt.optimizer.muon_momentum)?;
    write_blob_section(out, &ckpt.optimizer.adamw_first_moment)?;
    write_blob_section(out, &ckpt.optimizer.adamw_second_moment)?;
    Ok(())
}

fn decode_reader(reader: &mut impl Read, len: u64) -> Result<CheckpointV1, IoError> {
    if exceeds_checkpoint_cap(len, MAX_CHECKPOINT_BYTES) {
        return Err(IoError::new(format!(
            "{len} bytes exceeds checkpoint cap {MAX_CHECKPOINT_BYTES}"
        )));
    }
    let mut c = In {
        r: reader,
        left: len,
    };
    let prefix = c.take_vec(12)?;
    read_prefix(&prefix).map_err(|e| IoError::new(e.to_string()))?;
    let config = c.take_lp()?;
    let mut tokenizer_hash = [0u8; 32];
    tokenizer_hash.copy_from_slice(&c.take_vec(32)?);
    let mut git_sha = [0u8; 20];
    git_sha.copy_from_slice(&c.take_vec(20)?);
    let step = c.u64()?;
    let data_cursor = DataCursor {
        shard: c.u64()?,
        token_index: c.u64()?,
    };
    let rng_state = c.take_lp()?;
    let weights = c.blob_section("weights")?;
    let muon_momentum = c.blob_section("muon_momentum")?;
    let adamw_first_moment = c.blob_section("adamw_first_moment")?;
    let adamw_second_moment = c.blob_section("adamw_second_moment")?;
    if c.left != 0 {
        return Err(IoError::new("trailing bytes after checkpoint"));
    }
    Ok(CheckpointV1 {
        config,
        tokenizer_hash,
        git_sha,
        weights,
        optimizer: OptimizerCheckpoint {
            muon_momentum,
            adamw_first_moment,
            adamw_second_moment,
        },
        step,
        rng_state,
        data_cursor,
    })
}

fn validate_list(what: &str, blobs: &[NamedBlob]) -> Result<(), IoError> {
    let mut seen = std::collections::BTreeSet::new();
    for b in blobs {
        if !seen.insert(b.name.as_str()) {
            return Err(IoError::new(format!("{what}: duplicate key {:?}", b.name)));
        }
        window_ok(b)?;
    }
    Ok(())
}

fn window_ok(b: &NamedBlob) -> Result<(), IoError> {
    let mut n = 1u64;
    for &d in &b.shape {
        if d == 0 {
            n = 0;
            break;
        }
        n = n
            .checked_mul(d)
            .ok_or_else(|| IoError::new(format!("{:?}: shape product overflows", b.name)))?;
    }
    let elem = u64::try_from(b.dtype.size())
        .map_err(|_| IoError::new(format!("{:?}: dtype width exceeds u64", b.name)))?;
    let nbytes = n
        .checked_mul(elem)
        .ok_or_else(|| IoError::new(format!("{:?}: byte size overflows", b.name)))?;
    let end = b
        .byte_offset
        .checked_add(nbytes)
        .ok_or_else(|| IoError::new(format!("{:?}: byte_offset window overflows", b.name)))?;
    let len = u64::try_from(b.bytes.len())
        .map_err(|_| IoError::new(format!("{:?}: payload length exceeds u64", b.name)))?;
    if end > len {
        return Err(IoError::new(format!(
            "{:?}: byte_offset {} plus {nbytes} bytes exceeds payload {len}",
            b.name, b.byte_offset
        )));
    }
    Ok(())
}

fn put(out: &mut impl Write, bytes: &[u8]) -> Result<(), IoError> {
    out.write_all(bytes)
        .map_err(|e| IoError::new(format!("checkpoint write: {e}")))
}

fn write_lp(out: &mut impl Write, bytes: &[u8]) -> Result<(), IoError> {
    let n = u64_len(bytes)?;
    put(out, &n.to_le_bytes())?;
    put(out, bytes)
}

fn write_blob_section(out: &mut impl Write, blobs: &[NamedBlob]) -> Result<(), IoError> {
    let body = section_len(blobs)? - 8;
    put(out, &body.to_le_bytes())?;
    let count = u64::try_from(blobs.len()).map_err(|_| IoError::new("record count exceeds u64"))?;
    put(out, &count.to_le_bytes())?;
    for b in blobs {
        write_record(out, b)?;
    }
    Ok(())
}

fn write_record(out: &mut impl Write, b: &NamedBlob) -> Result<(), IoError> {
    let name = b.name.as_bytes();
    let name_len =
        u64::try_from(name.len()).map_err(|_| IoError::new("name length exceeds u64"))?;
    put(out, &name_len.to_le_bytes())?;
    put(out, name)?;
    put(out, &b.dtype.tag().to_le_bytes())?;
    let rank = u32::try_from(b.shape.len())
        .map_err(|_| IoError::new(format!("{:?}: rank exceeds u32", b.name)))?;
    put(out, &rank.to_le_bytes())?;
    for &d in &b.shape {
        put(out, &d.to_le_bytes())?;
    }
    put(out, &b.byte_offset.to_le_bytes())?;
    let payload_len =
        u64::try_from(b.bytes.len()).map_err(|_| IoError::new("payload length exceeds u64"))?;
    put(out, &payload_len.to_le_bytes())?;
    put(out, &b.bytes)
}

/// Name length, dtype tag, rank, byte_offset, payload length: a rank-0 record
/// with an empty name and payload.
const MIN_RECORD_BYTES: u64 = 8 + 4 + 4 + 8 + 8;

struct In<'a, R: Read> {
    r: &'a mut R,
    left: u64,
}

impl<R: Read> In<'_, R> {
    fn take_vec(&mut self, n: u64) -> Result<Vec<u8>, IoError> {
        if n > self.left {
            return Err(IoError::new("truncated checkpoint"));
        }
        let n_us = usize::try_from(n).map_err(|_| IoError::new("section length exceeds usize"))?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(n_us)
            .map_err(|_| IoError::new(format!("checkpoint allocation of {n_us} bytes refused")))?;
        buf.resize(n_us, 0);
        self.r
            .read_exact(&mut buf)
            .map_err(|e| IoError::new(format!("truncated checkpoint: {e}")))?;
        self.left -= n;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32, IoError> {
        let s = self.take_vec(4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(&s);
        Ok(u32::from_le_bytes(a))
    }

    fn u64(&mut self) -> Result<u64, IoError> {
        let s = self.take_vec(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(&s);
        Ok(u64::from_le_bytes(a))
    }

    fn take_lp(&mut self) -> Result<Vec<u8>, IoError> {
        let n = self.u64()?;
        self.take_vec(n)
    }

    fn blob_section(&mut self, what: &str) -> Result<Vec<NamedBlob>, IoError> {
        let body_len = self.u64()?;
        if body_len > self.left {
            return Err(IoError::new("truncated checkpoint"));
        }
        let resume = self.left - body_len;
        self.left = body_len;
        let out = self.blob_body(what)?;
        if self.left != 0 {
            return Err(IoError::new(format!("{what}: trailing bytes in section")));
        }
        self.left = resume;
        Ok(out)
    }

    fn blob_body(&mut self, what: &str) -> Result<Vec<NamedBlob>, IoError> {
        let count = self.u64()?;
        let room = self.left;
        if MIN_RECORD_BYTES == 0 || count > room / MIN_RECORD_BYTES {
            return Err(IoError::new(format!(
                "{what}: record count {count} exceeds what {room} section bytes can hold"
            )));
        }
        let count_us = usize::try_from(count)
            .map_err(|_| IoError::new(format!("{what}: count exceeds usize")))?;
        // Grow one validated record at a time. A count that fits the minimum
        // record size is not a reservation for that many empty slots.
        let mut out = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..count_us {
            let blob = self.record()?;
            if !seen.insert(blob.name.clone()) {
                return Err(IoError::new(format!(
                    "{what}: duplicate key {:?}",
                    blob.name
                )));
            }
            window_ok(&blob)?;
            out.try_reserve(1)
                .map_err(|_| IoError::new(format!("{what}: record allocation refused")))?;
            out.push(blob);
        }
        Ok(out)
    }

    fn record(&mut self) -> Result<NamedBlob, IoError> {
        let name_len = self.u64()?;
        if name_len > self.left {
            return Err(IoError::new("truncated checkpoint"));
        }
        let name_bytes = self.take_vec(name_len)?;
        let name = std::str::from_utf8(&name_bytes)
            .map_err(|e| IoError::new(format!("tensor name is not UTF-8: {e}")))?
            .to_string();
        let dtype = DType::from_tag(self.u32()?).map_err(|e| IoError::new(e.to_string()))?;
        let rank = self.u32()?;
        let rank_us = usize::try_from(rank).map_err(|_| IoError::new("rank exceeds usize"))?;
        let shape_bytes = u64::from(rank)
            .checked_mul(8)
            .ok_or_else(|| IoError::new("shape byte length overflows"))?;
        if shape_bytes > self.left {
            return Err(IoError::new(format!(
                "{name:?}: rank {rank} needs {shape_bytes} shape bytes, {} remain",
                self.left
            )));
        }
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(rank_us)
            .map_err(|_| IoError::new("shape allocation refused"))?;
        for _ in 0..rank_us {
            shape.push(self.u64()?);
        }
        let byte_offset = self.u64()?;
        let payload = self.take_lp()?;
        Ok(NamedBlob {
            name,
            dtype,
            shape,
            byte_offset,
            bytes: payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{tmp, tmp_dir, Mix};

    fn sample() -> CheckpointV1 {
        let mut bytes = vec![0xAB, 0xCD];
        bytes.extend_from_slice(&1.25f32.to_ne_bytes());
        CheckpointV1 {
            config: b"cfg".to_vec(),
            tokenizer_hash: [7u8; 32],
            git_sha: [9u8; 20],
            weights: vec![NamedBlob {
                name: "tok_emb".into(),
                dtype: DType::F32,
                shape: vec![1],
                byte_offset: 2,
                bytes,
            }],
            optimizer: OptimizerCheckpoint {
                muon_momentum: vec![],
                adamw_first_moment: vec![NamedBlob {
                    name: "ln.weight".into(),
                    dtype: DType::F32,
                    shape: vec![1],
                    byte_offset: 0,
                    bytes: 0.5f32.to_ne_bytes().to_vec(),
                }],
                adamw_second_moment: vec![],
            },
            step: 4,
            rng_state: vec![1, 2, 3, 4, 5, 6, 7, 8],
            data_cursor: DataCursor {
                shard: 2,
                token_index: 99,
            },
        }
    }

    #[test]
    fn checkpoint_round_trip_and_truncated_file() {
        let ckpt = sample();
        let bytes = encode_checkpoint(&ckpt).unwrap();
        assert_eq!(decode_checkpoint(&bytes).unwrap(), ckpt);
        let path = tmp("ckpt");
        write_checkpoint(&path.0, &ckpt).unwrap();
        let got = read_checkpoint(&path.0).unwrap();
        assert_eq!(got, ckpt);

        let mut short = bytes.clone();
        short.pop();
        assert!(decode_checkpoint(&short)
            .unwrap_err()
            .detail()
            .contains("truncated"));
        let mut huge = ojas_core::prefix_bytes().to_vec();
        huge.extend_from_slice(&u64::MAX.to_le_bytes());
        let err =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode_checkpoint(&huge)));
        let parsed = err.expect("huge section length panicked");
        assert!(parsed.is_err());
    }

    #[test]
    fn byte_window_past_the_payload_is_an_error() {
        let mut ckpt = sample();
        ckpt.weights[0].byte_offset = 100;
        assert!(encode_checkpoint(&ckpt).is_err());
    }

    /// A checkpoint with empty fields whose weights section body is `raw`.
    fn with_weights_section(raw: &[u8]) -> Vec<u8> {
        let mut out = ojas_core::prefix_bytes().to_vec();
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&[0u8; 32 + 20 + 24]);
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
        out.extend_from_slice(raw);
        for _ in 0..3 {
            out.extend_from_slice(&8u64.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
        }
        out
    }

    #[test]
    fn record_count_is_bounded_by_the_section_before_allocating() {
        assert!(decode_checkpoint(&with_weights_section(&0u64.to_le_bytes())).is_ok());
        for count in [10_000_000u64, u64::MAX] {
            let err = decode_checkpoint(&with_weights_section(&count.to_le_bytes())).unwrap_err();
            assert!(err.detail().contains("record count"), "{count}: {err}");
        }
    }

    #[test]
    fn rank_is_bounded_by_the_section_before_allocating() {
        let mut raw = 1u64.to_le_bytes().to_vec();
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&DType::F32.tag().to_le_bytes());
        raw.extend_from_slice(&u32::MAX.to_le_bytes());
        raw.extend_from_slice(&[0u8; 64]);
        let err = decode_checkpoint(&with_weights_section(&raw)).unwrap_err();
        assert!(err.detail().contains("shape bytes"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn rewrite_leaves_an_open_reader_on_the_old_file() {
        let path = tmp("ckpt-swap");
        let first = sample();
        write_checkpoint(&path.0, &first).unwrap();
        let mut old = File::open(&path.0).unwrap();
        let mut second = sample();
        second.step = 5;
        write_checkpoint(&path.0, &second).unwrap();
        let mut held = Vec::new();
        old.read_to_end(&mut held).unwrap();
        assert_eq!(decode_checkpoint(&held).unwrap(), first);
        assert_eq!(read_checkpoint(&path.0).unwrap(), second);
    }

    fn random_blobs(rng: &mut Mix, tag: &str) -> Vec<NamedBlob> {
        let dtypes = [DType::F32, DType::Bf16, DType::F16, DType::U32];
        (0..rng.below(4))
            .map(|i| {
                let dtype = dtypes[rng.below(dtypes.len())];
                let shape: Vec<u64> = (0..rng.below(4))
                    .map(|_| match rng.below(4) {
                        0 => 0,
                        _ => 1 + rng.below(3) as u64,
                    })
                    .collect();
                let window = shape.iter().product::<u64>() as usize * dtype.size();
                let byte_offset = rng.below(3);
                let len = byte_offset + window + rng.below(3);
                NamedBlob {
                    name: if i == 0 && rng.below(4) == 0 {
                        String::new()
                    } else {
                        format!("{tag}.{i}.é")
                    },
                    dtype,
                    shape,
                    byte_offset: byte_offset as u64,
                    bytes: (0..len).map(|_| rng.next() as u8).collect(),
                }
            })
            .collect()
    }

    fn random_checkpoint(rng: &mut Mix) -> CheckpointV1 {
        fn bytes(rng: &mut Mix, n: usize) -> Vec<u8> {
            (0..n).map(|_| rng.next() as u8).collect()
        }
        let n = rng.below(16);
        let config = bytes(rng, n);
        let n = rng.below(16);
        let rng_state = bytes(rng, n);
        let mut tokenizer_hash = [0u8; 32];
        tokenizer_hash.copy_from_slice(&bytes(rng, 32));
        let mut git_sha = [0u8; 20];
        git_sha.copy_from_slice(&bytes(rng, 20));
        CheckpointV1 {
            config,
            tokenizer_hash,
            git_sha,
            weights: random_blobs(rng, "w"),
            optimizer: OptimizerCheckpoint {
                muon_momentum: random_blobs(rng, "m"),
                adamw_first_moment: random_blobs(rng, "a1"),
                adamw_second_moment: random_blobs(rng, "a2"),
            },
            step: rng.next(),
            rng_state,
            data_cursor: DataCursor {
                shard: rng.next(),
                token_index: rng.next(),
            },
        }
    }

    #[test]
    fn replace_sweeps_orphan_temps_for_this_name_only() {
        let dir = tmp_dir("sweep");
        let path = dir.0.join("model.bin");
        let orphan = dir
            .0
            .join(format!(".model.bin.{}.7.tmp", std::process::id()));
        std::fs::write(&orphan, b"stale").unwrap();
        let unrelated = dir.0.join(format!(".other.{}.1.tmp", std::process::id()));
        std::fs::write(&unrelated, b"keep").unwrap();
        std::fs::write(dir.0.join(".model.bin.tmp"), b"keep").unwrap();
        std::fs::write(dir.0.join(".model.bin.pid.seq.tmp"), b"keep").unwrap();
        std::fs::write(dir.0.join(".model.bin.1.2.3.tmp"), b"keep").unwrap();
        std::fs::write(dir.0.join("model.bin.1.2.tmp"), b"keep").unwrap();
        write_checkpoint(&path, &sample()).unwrap();
        assert!(
            !orphan.exists(),
            "orphan temp survived a successful replace"
        );
        assert!(unrelated.exists());
        assert!(dir.0.join(".model.bin.tmp").exists());
        assert!(dir.0.join(".model.bin.pid.seq.tmp").exists());
        assert!(dir.0.join(".model.bin.1.2.3.tmp").exists());
        assert!(dir.0.join("model.bin.1.2.tmp").exists());
        assert!(path.is_file());
    }

    #[test]
    fn encoded_length_matches_the_bytes_and_the_writer_shares_the_read_cap() {
        let ckpt = sample();
        let bytes = encode_checkpoint(&ckpt).unwrap();
        assert_eq!(encoded_len(&ckpt).unwrap() as usize, bytes.len());
        assert!(bytes.len() as u64 <= MAX_CHECKPOINT_BYTES);

        let mut over = ckpt;
        let base = encoded_len(&over).unwrap();
        let old = over.weights[0].bytes.len() as u64;
        let mut new_len = (MAX_CHECKPOINT_BYTES + 1) - (base - old);
        let rem = new_len % 4;
        if rem != 0 {
            new_len += 4 - rem;
        }
        over.weights[0].dtype = DType::F32;
        over.weights[0].shape = vec![new_len / 4];
        over.weights[0].byte_offset = 0;
        over.weights[0].bytes = vec![0u8; usize::try_from(new_len).unwrap()];
        let err = encode_checkpoint(&over).unwrap_err();
        assert!(err.detail().contains("cap"), "{err}");
        let path = tmp("ckpt-over-cap");
        let err = write_checkpoint(&path.0, &over).unwrap_err();
        assert!(err.detail().contains("cap"), "{err}");
        assert!(!path.0.exists());
    }

    /// The 1 GiB boundary is the compare. Building a payload of that size is
    /// skipped; a smaller injected cap uses the same function the writer calls.
    #[test]
    fn checkpoint_cap_allows_the_exact_byte_and_refuses_one_past() {
        assert!(!exceeds_checkpoint_cap(
            MAX_CHECKPOINT_BYTES,
            MAX_CHECKPOINT_BYTES
        ));
        assert!(exceeds_checkpoint_cap(
            MAX_CHECKPOINT_BYTES + 1,
            MAX_CHECKPOINT_BYTES
        ));
        let small = 64u64;
        assert!(!exceeds_checkpoint_cap(small, small));
        assert!(exceeds_checkpoint_cap(small + 1, small));
        let ckpt = sample();
        let n = encoded_len(&ckpt).unwrap();
        assert!(n < MAX_CHECKPOINT_BYTES);
        assert!(!exceeds_checkpoint_cap(n, MAX_CHECKPOINT_BYTES));
        encode_checkpoint(&ckpt).unwrap();
    }

    #[test]
    fn oversized_checkpoint_is_refused_from_metadata() {
        let path = tmp("ckpt-cap");
        let file = std::fs::File::create(&path.0).unwrap();
        file.set_len(MAX_CHECKPOINT_BYTES + 1).unwrap();
        let err = read_checkpoint(&path.0).unwrap_err();
        drop(file);
        assert!(err.to_string().contains("cap"), "{err}");
    }

    #[test]
    fn random_checkpoints_round_trip_through_bytes_and_files() {
        let mut rng = Mix::new(0xC0);
        for i in 0..500 {
            let ckpt = random_checkpoint(&mut rng);
            let bytes = encode_checkpoint(&ckpt).unwrap();
            assert_eq!(decode_checkpoint(&bytes).unwrap(), ckpt);
            if i % 25 == 0 {
                let path = tmp("ckpt-prop");
                write_checkpoint(&path.0, &ckpt).unwrap();
                assert_eq!(std::fs::read(&path.0).unwrap(), bytes);
                assert_eq!(read_checkpoint(&path.0).unwrap(), ckpt);
            }
        }
    }

    fn mutate(rng: &mut Mix, base: &[u8], other: &[u8]) -> Vec<u8> {
        let mut v = base.to_vec();
        let op = if v.len() < 8 { 4 } else { rng.below(6) };
        match op {
            0 => {
                for _ in 0..1 + rng.below(4) {
                    let i = rng.below(v.len());
                    v[i] ^= 1 << rng.below(8);
                }
            }
            1 => v.truncate(rng.below(v.len() + 1)),
            2 => {
                let picks = [
                    0,
                    1,
                    7,
                    1 << 20,
                    1 << 32,
                    1 << 40,
                    u64::MAX / 32,
                    u64::MAX,
                    v.len() as u64,
                    rng.next(),
                ];
                let at = rng.below(v.len() - 7);
                v[at..at + 8].copy_from_slice(&picks[rng.below(picks.len())].to_le_bytes());
            }
            3 => {
                let picks = [0u32, 4, 5, 255, 1 << 16, u32::MAX, rng.next() as u32];
                let at = rng.below(v.len() - 3);
                v[at..at + 4].copy_from_slice(&picks[rng.below(picks.len())].to_le_bytes());
            }
            4 => {
                let a = rng.below(other.len() + 1);
                let b = a + rng.below(other.len() - a + 1);
                let at = rng.below(v.len() + 1);
                let end = at + rng.below(v.len() - at + 1);
                v.splice(at..end, other[a..b].iter().copied());
            }
            _ => {
                let at = rng.below(v.len() + 1);
                v.insert(at, rng.next() as u8);
            }
        }
        v
    }

    fn decode_checked(bytes: &[u8]) -> bool {
        std::panic::catch_unwind(|| match decode_checkpoint(bytes) {
            Ok(ckpt) => {
                assert_eq!(encode_checkpoint(&ckpt).unwrap(), bytes);
                true
            }
            Err(_) => false,
        })
        .unwrap_or_else(|_| panic!("panicked on {bytes:?}"))
    }

    /// The layout has no slack, so anything that decodes must re-encode to
    /// the same bytes.
    #[test]
    fn mutated_checkpoints_never_panic_and_accepted_ones_re_encode_exactly() {
        let mut rng = Mix::new(0xFACE);
        let seeds: Vec<Vec<u8>> = (0..4)
            .map(|_| encode_checkpoint(&random_checkpoint(&mut rng)).unwrap())
            .chain([encode_checkpoint(&sample()).unwrap()])
            .collect();
        let mut truncations = 0;
        for s in &seeds {
            assert!(decode_checked(s));
            for n in 0..s.len() {
                assert!(!decode_checked(&s[..n]));
                truncations += 1;
            }
        }
        let (mut accepted, mut via_file) = (0, 0);
        for i in 0..8000 {
            let base = &seeds[rng.below(seeds.len())];
            let other = &seeds[rng.below(seeds.len())];
            let mut v = mutate(&mut rng, base, other);
            for _ in 0..rng.below(3) {
                v = mutate(&mut rng, &v, other);
            }
            let ok = decode_checked(&v);
            accepted += usize::from(ok);
            if i % 80 == 0 {
                let path = tmp("ckpt-fuzz");
                std::fs::write(&path.0, &v).unwrap();
                let read = std::panic::catch_unwind(|| read_checkpoint(&path.0))
                    .unwrap_or_else(|_| panic!("read panicked on {v:?}"));
                assert_eq!(read.is_ok(), ok);
                via_file += 1;
            }
        }
        assert_eq!(truncations, seeds.iter().map(Vec::len).sum::<usize>());
        assert_eq!(via_file, 100);
        assert!(
            accepted > 50 && accepted < 8000,
            "accepted {accepted} of 8000"
        );
    }

    #[test]
    fn read_from_an_open_file_ignores_and_keeps_the_cursor() {
        let path = tmp("ckpt-from");
        write_checkpoint(&path.0, &sample()).unwrap();
        let mut file = File::open(&path.0).unwrap();
        file.seek(std::io::SeekFrom::Start(7)).unwrap();
        let got = read_checkpoint_from(file.try_clone().unwrap()).unwrap();
        assert_eq!(got, sample());
        assert_eq!(file.stream_position().unwrap(), 7, "cursor moved");
        file.seek(std::io::SeekFrom::End(0)).unwrap();
        assert_eq!(read_checkpoint_from(file).unwrap(), sample());

        let dir = tmp_dir("ckpt-from-dir");
        let err = read_checkpoint_from(File::open(&dir.0).unwrap()).unwrap_err();
        assert!(err.detail().contains("not a regular file"), "{err}");
        let err = read_checkpoint(&dir.0).unwrap_err();
        assert!(err.detail().contains("not a regular file"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn read_from_a_nofollow_open() {
        let dir = tmp_dir("ckpt-nofollow");
        let real = dir.0.join("state.ojck");
        write_checkpoint(&real, &sample()).unwrap();
        let link = dir.0.join("link.ojck");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        assert!(crate::open_nofollow(&link).is_err());
        let file = crate::open_nofollow(&real).unwrap();
        assert_eq!(read_checkpoint_from(file).unwrap(), sample());
    }
}
