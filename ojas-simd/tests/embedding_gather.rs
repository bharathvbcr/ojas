//! `gather_embedding_rows_768` against `copy_from_slice`.
//!
//! A row is 768 floats: twelve blocks of 64, so the kernel has no tail.
//! The last lane is still checked. A refusal does not grow `dst`.

#![cfg(all(target_arch = "aarch64", target_feature = "neon"))]

use ojas_simd::{gather_embedding_rows_768, SimdError};

const ROW: usize = 768;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().copied().map(f32::to_bits).collect()
}

fn copy_rows(table: &[f32], ids: &[u32]) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * ROW];
    for (dst, &id) in out.as_chunks_mut::<ROW>().0.iter_mut().zip(ids) {
        let start = id as usize * ROW;
        dst.copy_from_slice(&table[start..start + ROW]);
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
    assert_eq!(ROW % 64, 0, "the 768 kernel has no short tail");
    let mut state = 0x7680_u64;
    let vocab = 19usize;
    let mut table = vec![0.0f32; vocab * ROW];
    for slot in &mut table {
        let raw = xorshift(&mut state);
        *slot = f32::from_bits((raw as u32) & 0x7f80_0000 | (raw as u32) & 0x007f_ffff);
    }
    table[0] = -0.0;
    table[7] = -0.0;
    table[63] = f32::from_bits(0x8000_0001);
    table[767] = -0.0;
    table[ROW] = -0.0;
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

    let expect = copy_rows(&table, &ids);
    let mut dst = Vec::new();
    dst.reserve_exact(ids.len() * ROW);
    gather_embedding_rows_768(&table, &ids, &mut dst).unwrap();
    assert_eq!(dst.len(), ids.len() * ROW);
    assert_eq!(bits(&dst), bits(&expect));
    let row0 = ids.iter().position(|&id| id == 0).unwrap() * ROW;
    let last_row = ids
        .iter()
        .rposition(|&id| id == (vocab - 1) as u32)
        .unwrap()
        * ROW;
    assert_eq!(dst[row0 + 767].to_bits(), (-0.0f32).to_bits());
    assert_eq!(dst[last_row + ROW - 1].to_bits(), (-0.0f32).to_bits());
}

#[test]
fn a_refusal_does_not_publish_a_partial_buffer() {
    let table = vec![0.25f32; 4 * ROW];
    let mut dst = Vec::new();
    dst.reserve_exact(ROW);
    let err = gather_embedding_rows_768(&table, &[0, 1], &mut dst).unwrap_err();
    assert!(matches!(err, SimdError::OutputLength { .. }), "{err:?}");
    assert_eq!(dst.len(), 0);

    dst.reserve_exact(3 * ROW);
    let err = gather_embedding_rows_768(&table, &[0, 9, 1], &mut dst).unwrap_err();
    assert!(matches!(err, SimdError::BufferTooShort { .. }), "{err:?}");
    assert_eq!(dst.len(), 0);

    let short = vec![0.5f32; ROW + 3];
    let err = gather_embedding_rows_768(&short, &[0], &mut dst).unwrap_err();
    assert!(matches!(err, SimdError::BufferTooShort { .. }), "{err:?}");
    assert_eq!(dst.len(), 0);

    gather_embedding_rows_768(&table, &[1, 1, 0], &mut dst).unwrap();
    assert_eq!(bits(&dst), bits(&copy_rows(&table, &[1, 1, 0])));
}
