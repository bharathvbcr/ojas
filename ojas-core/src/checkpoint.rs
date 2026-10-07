//! Checkpoint schema v1.
//!
//! The on-disk layout is little-endian and is specified in `docs/checkpoint-v1.md`.
//! This module holds that schema as types plus the 12-byte prefix. It does not
//! read or write weights, optimizer state, or files.

use crate::dtype::DType;
use crate::OjasError;

/// `b"OJAS0001"`. Eight bytes.
pub const CHECKPOINT_MAGIC: [u8; 8] = *b"OJAS0001";

pub const CHECKPOINT_VERSION: u32 = 1;

/// Magic and version at the front of a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointPrefix {
    pub version: u32,
}

/// Twelve-byte prefix: magic, then version as `u32` little-endian.
pub fn prefix_bytes() -> [u8; 12] {
    let mut out = [0u8; 12];
    out[..8].copy_from_slice(&CHECKPOINT_MAGIC);
    out[8..].copy_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
    out
}

/// Read [`prefix_bytes`]. A short buffer, a wrong magic, or a version other
/// than 1 is [`OjasError::OutOfRange`].
pub fn read_prefix(bytes: &[u8]) -> Result<CheckpointPrefix, OjasError> {
    let (magic, rest) = bytes
        .split_at_checked(8)
        .ok_or_else(|| OjasError::OutOfRange {
            op: "read_prefix",
            detail: format!("need 12 bytes, got {}", bytes.len()),
        })?;
    if magic != CHECKPOINT_MAGIC {
        return Err(OjasError::OutOfRange {
            op: "read_prefix",
            detail: "checkpoint magic mismatch".to_string(),
        });
    }
    let (ver, _) = rest
        .split_at_checked(4)
        .ok_or_else(|| OjasError::OutOfRange {
            op: "read_prefix",
            detail: format!("need 12 bytes, got {}", bytes.len()),
        })?;
    let buf: [u8; 4] = ver.try_into().map_err(|_| OjasError::OutOfRange {
        op: "read_prefix",
        detail: "version bytes are not 4 long".to_string(),
    })?;
    let version = u32::from_le_bytes(buf);
    if version != CHECKPOINT_VERSION {
        return Err(OjasError::OutOfRange {
            op: "read_prefix",
            detail: format!("checkpoint version {version} != {CHECKPOINT_VERSION}"),
        });
    }
    Ok(CheckpointPrefix { version })
}

/// One named tensor payload in the checkpoint.
///
/// `byte_offset` is the start of element 0 inside `bytes`. A consumer that
/// ignores it reads the wrong columns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedBlob {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub byte_offset: u64,
    pub bytes: Vec<u8>,
}

/// Optimizer tensors kept beside the weights.
///
/// Muon keeps one momentum buffer per matrix. AdamW keeps the first and
/// second moments. Gate weights are Muon matrices; gate bias, `vr_lambda`,
/// and norm weights are AdamW parameters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OptimizerCheckpoint {
    pub muon_momentum: Vec<NamedBlob>,
    pub adamw_first_moment: Vec<NamedBlob>,
    pub adamw_second_moment: Vec<NamedBlob>,
}

/// Position in the training stream. Units belong to the data loader.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DataCursor {
    pub shard: u64,
    pub token_index: u64,
}

/// In-memory checkpoint v1.
///
/// `config` is an opaque byte string (the model config the training step
/// used). `tokenizer_hash` is 32 bytes. `git_sha` is 20 bytes, the raw SHA-1,
/// not hex. `rng_state` is an opaque byte string to this schema; its layout
/// is its producer's (the trainer writes `ojas_data::SamplerRngState`, v1,
/// specified in `docs/checkpoint-v1.md`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointV1 {
    pub config: Vec<u8>,
    pub tokenizer_hash: [u8; 32],
    pub git_sha: [u8; 20],
    pub weights: Vec<NamedBlob>,
    pub optimizer: OptimizerCheckpoint,
    pub step: u64,
    pub rng_state: Vec<u8>,
    pub data_cursor: DataCursor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_round_trips_and_ignores_trailing_bytes() {
        let prefix = prefix_bytes();
        assert_eq!(&prefix[..8], b"OJAS0001");
        assert_eq!(
            read_prefix(&prefix).unwrap(),
            CheckpointPrefix { version: 1 }
        );
        let mut longer = prefix.to_vec();
        longer.extend_from_slice(&[0xFF; 64]);
        assert_eq!(read_prefix(&longer).unwrap().version, 1);
    }

    #[test]
    fn every_truncation_and_corruption_is_refused() {
        let prefix = prefix_bytes();
        for len in 0..12 {
            let err = read_prefix(&prefix[..len]).unwrap_err();
            assert!(matches!(err, OjasError::OutOfRange { .. }), "{len}: {err}");
        }
        for byte in 0..12 {
            for flip in [0x01u8, 0x80, 0xFF] {
                let mut bad = prefix;
                bad[byte] ^= flip;
                assert!(read_prefix(&bad).is_err(), "byte {byte} flip {flip:#x}");
            }
        }
        for version in [0u32, 2, u32::MAX] {
            let mut bad = prefix;
            bad[8..].copy_from_slice(&version.to_le_bytes());
            assert!(read_prefix(&bad).is_err());
        }
    }
}
