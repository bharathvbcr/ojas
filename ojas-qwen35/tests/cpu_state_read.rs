//! `StateIndex::read_entry` (the 2B resume path of `load_state`) reads the
//! values a shard holds, bit for bit, on the CPU: no device, no tessl runtime.
//!
//! The values include negative zero, a subnormal, an infinity, a NaN with a
//! payload, and an entry longer than several of `ojas-io`'s read chunks with a
//! ragged tail, so a decode that drops, reorders or canonicalizes any of them
//! fails.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use ojas_qwen35::state::{plan_shards, write_kind};
use ojas_qwen35::{Kind, StateIndex, TensorSpec};

const SUMMARY: &str = "cpu-state-read-test";

fn table() -> Vec<TensorSpec> {
    let spec = |name: &str, shape: &[usize]| TensorSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
    };
    vec![
        spec("a.small", &[2, 3]),
        spec("b.long", &[7, 10_007]),
        spec("c.vector", &[5]),
    ]
}

/// Entry `i` of `kind`: a fixed pattern, with the awkward bit patterns first.
fn values(kind: usize, i: usize, n: usize) -> Vec<f32> {
    let special = [
        0x8000_0000u32, // -0.0
        0x0000_0001,    // smallest subnormal
        0x7F80_0000,    // +inf
        0x7FC0_1234,    // NaN with a payload
        0xFF7F_FFFF,    // -f32::MAX
    ];
    (0..n)
        .map(|j| match special.get(j) {
            Some(&bits) => f32::from_bits(bits),
            None => f32::from_bits(
                (j as u32)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(((kind * 31 + i) as u32).wrapping_mul(0x9E37_79B9))
                    & 0x7EFF_FFFF,
            ),
        })
        .collect()
}

fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-qwen35-state-read-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn read_entry_returns_every_bit_that_write_kind_stored() {
    let dir = scratch_dir();
    let table = table();
    // One shard, then three (a cap below the long entry forces its own shard).
    for cap in [u64::MAX, 4 * 70_000] {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let shards = plan_shards(&table, cap);
        for (k, kind) in Kind::ALL.iter().enumerate() {
            write_kind(&dir, *kind, 42, SUMMARY, &table, &shards, |i| {
                Ok(values(k, i, table[i].numel()))
            })
            .unwrap();
        }
        let idx = StateIndex::read(&dir, &table, SUMMARY).unwrap();
        assert_eq!(idx.step, 42);
        for (k, kind) in Kind::ALL.iter().enumerate() {
            for (i, spec) in table.iter().enumerate() {
                let got = idx.read_entry(*kind, i).unwrap();
                let want = values(k, i, spec.numel());
                assert_eq!(got.len(), want.len(), "{} {}", kind.tag(), spec.name);
                let same = got
                    .iter()
                    .zip(&want)
                    .all(|(g, w)| g.to_bits() == w.to_bits());
                assert!(same, "{} {}: bits differ", kind.tag(), spec.name);
            }
        }
        let first = idx.read_entry(Kind::Masters, 0).unwrap();
        let bits: Vec<u32> = first.iter().take(5).map(|v| v.to_bits()).collect();
        assert_eq!(
            bits,
            [
                0x8000_0000,
                0x0000_0001,
                0x7F80_0000,
                0x7FC0_1234,
                0xFF7F_FFFF
            ]
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
