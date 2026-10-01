//! Checkpoint v1 file body, matching `docs/checkpoint-v1.md`.
//!
//! The 12-byte prefix is [`ojas_core::prefix_bytes`]. Sections after it are
//! length-prefixed except the fixed hash, SHA, step, and cursor fields.
//! A short file or a length that runs past the end is an error.

use crate::error::IoError;
use crate::replace::replace_file;
use ojas_core::{read_prefix, CheckpointV1, DType, DataCursor, NamedBlob, OptimizerCheckpoint};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// The target is replaced by rename, so a failed write leaves the previous
/// checkpoint intact.
pub fn write_checkpoint(path: &Path, ckpt: &CheckpointV1) -> Result<(), IoError> {
    replace_file(path, &encode_checkpoint(ckpt)?)
}

pub fn read_checkpoint(path: &Path) -> Result<CheckpointV1, IoError> {
    let mut file =
        File::open(path).map_err(|e| IoError::new(format!("{}: {e}", path.display())))?;
    let file_len = file
        .metadata()
        .map_err(|e| IoError::new(format!("{}: {e}", path.display())))?
        .len();
    let mut buf = Vec::new();
    let n =
        usize::try_from(file_len).map_err(|_| IoError::new("checkpoint length exceeds usize"))?;
    buf.try_reserve_exact(n)
        .map_err(|_| IoError::new(format!("checkpoint allocation of {n} bytes refused")))?;
    buf.resize(n, 0);
    file.read_exact(&mut buf)
        .map_err(|e| IoError::new(format!("{}: truncated checkpoint: {e}", path.display())))?;
    decode_checkpoint(&buf)
}

pub fn encode_checkpoint(ckpt: &CheckpointV1) -> Result<Vec<u8>, IoError> {
    validate_list("weights", &ckpt.weights)?;
    validate_list("muon_momentum", &ckpt.optimizer.muon_momentum)?;
    validate_list("adamw_first_moment", &ckpt.optimizer.adamw_first_moment)?;
    validate_list("adamw_second_moment", &ckpt.optimizer.adamw_second_moment)?;
    let mut out = Vec::new();
    out.extend_from_slice(&ojas_core::prefix_bytes());
    write_bytes(&mut out, &ckpt.config)?;
    out.extend_from_slice(&ckpt.tokenizer_hash);
    out.extend_from_slice(&ckpt.git_sha);
    out.extend_from_slice(&ckpt.step.to_le_bytes());
    out.extend_from_slice(&ckpt.data_cursor.shard.to_le_bytes());
    out.extend_from_slice(&ckpt.data_cursor.token_index.to_le_bytes());
    write_bytes(&mut out, &ckpt.rng_state)?;
    write_blob_section(&mut out, &ckpt.weights)?;
    write_blob_section(&mut out, &ckpt.optimizer.muon_momentum)?;
    write_blob_section(&mut out, &ckpt.optimizer.adamw_first_moment)?;
    write_blob_section(&mut out, &ckpt.optimizer.adamw_second_moment)?;
    Ok(out)
}

pub fn decode_checkpoint(bytes: &[u8]) -> Result<CheckpointV1, IoError> {
    let mut c = Cursor::new(bytes);
    let prefix = c.take(12)?;
    read_prefix(prefix).map_err(|e| IoError::new(e.to_string()))?;
    let config = c.take_lp()?.to_vec();
    let mut tokenizer_hash = [0u8; 32];
    tokenizer_hash.copy_from_slice(c.take(32)?);
    let mut git_sha = [0u8; 20];
    git_sha.copy_from_slice(c.take(20)?);
    let step = c.u64()?;
    let data_cursor = DataCursor {
        shard: c.u64()?,
        token_index: c.u64()?,
    };
    let rng_state = c.take_lp()?.to_vec();
    let weights = c.blob_section("weights")?;
    let muon_momentum = c.blob_section("muon_momentum")?;
    let adamw_first_moment = c.blob_section("adamw_first_moment")?;
    let adamw_second_moment = c.blob_section("adamw_second_moment")?;
    if c.i != c.b.len() {
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

fn write_bytes(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), IoError> {
    let n = u64::try_from(bytes.len()).map_err(|_| IoError::new("section length exceeds u64"))?;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn write_blob_section(out: &mut Vec<u8>, blobs: &[NamedBlob]) -> Result<(), IoError> {
    let mut payload = Vec::new();
    let count = u64::try_from(blobs.len()).map_err(|_| IoError::new("record count exceeds u64"))?;
    payload.extend_from_slice(&count.to_le_bytes());
    for b in blobs {
        write_record(&mut payload, b)?;
    }
    write_bytes(out, &payload)
}

fn write_record(out: &mut Vec<u8>, b: &NamedBlob) -> Result<(), IoError> {
    let name = b.name.as_bytes();
    let name_len =
        u64::try_from(name.len()).map_err(|_| IoError::new("name length exceeds u64"))?;
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(&b.dtype.tag().to_le_bytes());
    let rank = u32::try_from(b.shape.len())
        .map_err(|_| IoError::new(format!("{:?}: rank exceeds u32", b.name)))?;
    out.extend_from_slice(&rank.to_le_bytes());
    for &d in &b.shape {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&b.byte_offset.to_le_bytes());
    let payload_len = u64::try_from(b.bytes.len())
        .map_err(|_| IoError::new(format!("{:?}: payload length exceeds u64", b.name)))?;
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&b.bytes);
    Ok(())
}

/// Name length, dtype tag, rank, byte_offset, payload length: a rank-0 record
/// with an empty name and payload.
const MIN_RECORD_BYTES: usize = 8 + 4 + 4 + 8 + 8;

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }

    fn remaining(&self) -> usize {
        self.b.len().saturating_sub(self.i)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], IoError> {
        let end = self
            .i
            .checked_add(n)
            .ok_or_else(|| IoError::new("truncated checkpoint"))?;
        let s = self
            .b
            .get(self.i..end)
            .ok_or_else(|| IoError::new("truncated checkpoint"))?;
        self.i = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, IoError> {
        let s = self.take(4)?;
        let mut a = [0u8; 4];
        a.copy_from_slice(s);
        Ok(u32::from_le_bytes(a))
    }

    fn u64(&mut self) -> Result<u64, IoError> {
        let s = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    fn take_lp(&mut self) -> Result<&'a [u8], IoError> {
        let n = self.u64()?;
        let n_us = usize::try_from(n).map_err(|_| IoError::new("section length exceeds usize"))?;
        self.take(n_us)
    }

    fn blob_section(&mut self, what: &str) -> Result<Vec<NamedBlob>, IoError> {
        let raw = self.take_lp()?;
        let mut inner = Cursor::new(raw);
        let count = inner.u64()?;
        let count_us = usize::try_from(count)
            .map_err(|_| IoError::new(format!("{what}: count exceeds usize")))?;
        let room = inner.remaining();
        if count_us > room / MIN_RECORD_BYTES {
            return Err(IoError::new(format!(
                "{what}: record count {count} exceeds what {room} section bytes can hold"
            )));
        }
        let mut out = Vec::new();
        out.try_reserve_exact(count_us)
            .map_err(|_| IoError::new(format!("{what}: record allocation refused")))?;
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..count_us {
            let blob = inner.record()?;
            if !seen.insert(blob.name.clone()) {
                return Err(IoError::new(format!(
                    "{what}: duplicate key {:?}",
                    blob.name
                )));
            }
            window_ok(&blob)?;
            out.push(blob);
        }
        if inner.i != inner.b.len() {
            return Err(IoError::new(format!("{what}: trailing bytes in section")));
        }
        Ok(out)
    }

    fn record(&mut self) -> Result<NamedBlob, IoError> {
        let name_len = self.u64()?;
        let name_us =
            usize::try_from(name_len).map_err(|_| IoError::new("name length exceeds usize"))?;
        let name_bytes = self.take(name_us)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|e| IoError::new(format!("tensor name is not UTF-8: {e}")))?
            .to_string();
        let dtype = DType::from_tag(self.u32()?).map_err(|e| IoError::new(e.to_string()))?;
        let rank = self.u32()?;
        let rank_us = usize::try_from(rank).map_err(|_| IoError::new("rank exceeds usize"))?;
        let shape_bytes = rank_us
            .checked_mul(8)
            .ok_or_else(|| IoError::new("shape byte length overflows"))?;
        if shape_bytes > self.remaining() {
            return Err(IoError::new(format!(
                "{name:?}: rank {rank} needs {shape_bytes} shape bytes, {} remain",
                self.remaining()
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
        let payload = self.take_lp()?.to_vec();
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
    use crate::test_util::{tmp, Mix};

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
}
