//! `gather_embedding_rows` against `copy_from_slice`.
//!
//! A row is a whole number of 64-float blocks, so the kernel has no tail:
//! one block (64), nanolab's twelve (768) and Qwen3.5's thirty-two (2048).
//! The last lane is still checked. A refusal does not grow `dst`.

#![cfg(all(target_arch = "aarch64", target_feature = "neon"))]

use ojas_simd::{gather_embedding_rows, SimdError};

const ROWS: [usize; 3] = [64, 768, 2048];

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().copied().map(f32::to_bits).collect()
}

fn copy_rows(table: &[f32], row: usize, ids: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * row];
    for (dst, &id) in out.chunks_exact_mut(row).zip(ids) {
        let start = id as usize * row;
        dst.copy_from_slice(&table[start..start + row]);
    }
    out
}

fn xorshift(state: &mut u64) -> u64 {
    let mut z = *state;
    z ^= z << 13;
    z ^= z >> 7;
    z ^= z << 17;
    *state = z;
    z
}

#[test]
fn rows_match_copy_from_slice_including_negative_zero_and_duplicate_ids() {
    for row in ROWS {
        let mut state = 0x7680_u64 ^ row as u64;
        let vocab = 19usize;
        let mut table = vec![0.0f32; vocab * row];
        for slot in &mut table {
            let raw = xorshift(&mut state);
            *slot = f32::from_bits((raw as u32) & 0x7f80_0000 | (raw as u32) & 0x007f_ffff);
        }
        table[0] = -0.0;
        table[7] = -0.0;
        table[63] = f32::from_bits(0x8000_0001);
        table[row - 1] = -0.0;
        table[row] = -0.0;
        let last = table.len() - 1;
        table[last] = -0.0;
        table[3] = f32::from_bits(0x7fc0_0001);

        let mut ids = Vec::new();
        for _ in 0..40 {
            ids.push((xorshift(&mut state) as usize % vocab) as u32);
        }
        ids.push(ids[0]);
        ids.push(ids[1]);
        ids.push(0);
        ids.push((vocab - 1) as u32);
        ids.push(ids[0]);

        let expect = copy_rows(&table, row, &ids);
        let mut dst = Vec::new();
        dst.reserve_exact(ids.len() * row);
        gather_embedding_rows(&table, row, &ids, &mut dst).unwrap();
        assert_eq!(dst.len(), ids.len() * row, "row {row}");
        assert_eq!(bits(&dst), bits(&expect), "row {row}");
        let row0 = ids.iter().position(|&id| id == 0).unwrap() * row;
        let last_row = ids
            .iter()
            .rposition(|&id| id == (vocab - 1) as u32)
            .unwrap()
            * row;
        assert_eq!(dst[row0 + row - 1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(dst[last_row + row - 1].to_bits(), (-0.0f32).to_bits());
    }
}

#[test]
fn a_refusal_does_not_publish_a_partial_buffer() {
    for row in ROWS {
        let table = vec![0.25f32; 4 * row];
        let mut dst = Vec::new();
        dst.reserve_exact(row);
        let err = gather_embedding_rows(&table, row, &[0, 1], &mut dst).unwrap_err();
        assert!(matches!(err, SimdError::OutputLength { .. }), "{err:?}");
        assert_eq!(dst.len(), 0);

        dst.reserve_exact(3 * row);
        let err = gather_embedding_rows(&table, row, &[0, 9, 1], &mut dst).unwrap_err();
        assert!(matches!(err, SimdError::BufferTooShort { .. }), "{err:?}");
        assert_eq!(dst.len(), 0);

        let short = vec![0.5f32; row + 3];
        let err = gather_embedding_rows(&short, row, &[0], &mut dst).unwrap_err();
        assert!(matches!(err, SimdError::BufferTooShort { .. }), "{err:?}");
        assert_eq!(dst.len(), 0);

        gather_embedding_rows(&table, row, &[1, 1, 0], &mut dst).unwrap();
        assert_eq!(bits(&dst), bits(&copy_rows(&table, row, &[1, 1, 0])));
    }
}

#[test]
fn a_width_that_is_not_whole_blocks_is_refused_before_any_write() {
    let table = vec![1.0f32; 4 * 96];
    let mut dst = Vec::with_capacity(96);
    for row in [0usize, 32, 96, 760] {
        let err = gather_embedding_rows(&table, row, &[0], &mut dst).unwrap_err();
        assert!(
            matches!(err, SimdError::RowWidth { row: got } if got == row),
            "{err:?}"
        );
        assert_eq!(dst.len(), 0);
    }
}
