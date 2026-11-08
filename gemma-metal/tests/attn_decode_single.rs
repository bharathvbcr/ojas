//! `decode_attn_single_pass` (one dispatch, `kernels/flash_attn_decode_single.metal`) against
//! tessl's partial + reduce passes (`tessl::nn::flash_attn_decode_values`) at the E4B decode
//! shapes: bit-identical outputs whenever every live key fits one chunk, and a refusal past it.

use gemma_metal::kernels::decode_attn_single_pass;
use gemma_metal::tessl::nn::{flash_attn_decode_values, AttnDims, DecodeScratch};
use gemma_metal::GemmaGpu;

fn values(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

#[test]
fn single_pass_decode_attention_matches_partial_plus_reduce() {
    let gpu = match GemmaGpu::new() {
        Ok(g) => g,
        Err(e) => panic!("Metal runtime unavailable (not a pass): {e}"),
    };
    // (head_dim, heads, kv_heads, window, capacity, tkv, q_pos, kv_pos_offset, out_bf16)
    let cases: &[(u32, u32, u32, u32, usize, u32, u32, u32, bool)] = &[
        (256, 8, 2, 512, 520, 1, 0, 0, true),
        (256, 8, 2, 512, 520, 37, 36, 0, true),
        (256, 8, 2, 512, 520, 128, 127, 0, false),
        (256, 8, 2, 16, 520, 100, 99, 0, true), // window masks the oldest keys
        (256, 8, 2, 512, 520, 64, 700, 637, true), // kv positions offset (after a wrap)
        (256, 8, 2, 512, 520, 20, 3, 10, true), // every key in the future: zeros
        (512, 8, 2, 0, 256, 1, 0, 0, true),
        (512, 8, 2, 0, 256, 77, 76, 0, false),
        (512, 8, 1, 0, 256, 128, 127, 0, true),
    ];
    for &(d, h, hkv, window, cap, tkv, q_pos, kv_off, out_bf16) in cases {
        let case = format!("d={d} h={h} hkv={hkv} win={window} tkv={tkv} q={q_pos} kv_off={kv_off} bf16={out_bf16}");
        let buf = |v: &[f32]| {
            let b = gpu.rt.alloc_buffer(v.len() * 4).expect("alloc");
            b.write_f32(v);
            b
        };
        let q = buf(&values((h * d) as usize, 1, 2.0));
        let k = buf(&values(cap * (hkv * d) as usize, 2, 2.0));
        let v = buf(&values(cap * (hkv * d) as usize, 3, 2.0));
        let o_len = (h * d) as usize;
        let sentinel = vec![f32::from_bits(0x7f80_0001); o_len];
        let o_ref = buf(&sentinel);
        let o_one = buf(&sentinel);
        let scratch = DecodeScratch::new(&gpu.rt, 1, h, cap, d).expect("scratch");
        let dims = AttnDims {
            batch: 1,
            tq: 1,
            heads: h,
            heads_kv: hkv,
            window,
            scale: 1.0,
        };
        flash_attn_decode_values(
            &gpu.rt, &q, &k, &v, &o_ref, &scratch, tkv, q_pos, kv_off, dims, d, cap, out_bf16,
        )
        .expect("partial + reduce");
        gpu.barrier().expect("barrier");
        let ran = decode_attn_single_pass(
            &gpu, &q, &k, &v, &o_one, 1, tkv, h, hkv, d, window, 1.0, q_pos, kv_off, cap, out_bf16,
        )
        .expect("single pass");
        assert!(ran, "single pass refused: {case}");
        gpu.synchronize().expect("complete");
        let words = if out_bf16 { o_len / 2 } else { o_len };
        let a: Vec<u32> = o_ref.read_f32()[..words]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let b: Vec<u32> = o_one.read_f32()[..words]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert!(
            a.iter().all(|&w| w != 0x7f80_0001),
            "reference left sentinel: {case}"
        );
        assert_eq!(a, b, "outputs differ: {case}");
    }

    // Past one chunk the helper declines, so the caller takes the two-pass path.
    let o = gpu.rt.alloc_buffer(8 * 256 * 4).expect("alloc");
    let kv = gpu.rt.alloc_buffer(520 * 2 * 256 * 4).expect("alloc");
    let ran = decode_attn_single_pass(
        &gpu, &o, &kv, &kv, &o, 1, 129, 8, 2, 256, 512, 1.0, 128, 0, 520, true,
    )
    .expect("decline");
    assert!(!ran, "129 keys must take the two-pass path");
}

/// The verify instance over M query rows: row `i` (position `q_pos0 + i`) must equal the
/// decode instance run on that row alone, bit for bit, over the same K/V.
#[test]
fn verify_block_attention_matches_decode_row_by_row() {
    use gemma_metal::kernels::verify_attn_single_pass;
    let gpu =
        GemmaGpu::new().unwrap_or_else(|e| panic!("Metal runtime unavailable (not a pass): {e}"));
    // (head_dim, heads, kv_heads, window, capacity, m, tkv, q_pos0, kv_pos_offset, out_bf16)
    let cases: &[(u32, u32, u32, u32, usize, u32, u32, u32, u32, bool)] = &[
        (256, 8, 2, 512, 520, 2, 12, 10, 0, true),
        (256, 8, 2, 512, 520, 4, 40, 36, 0, false),
        (256, 8, 2, 16, 520, 5, 100, 95, 0, true),
        (256, 8, 2, 512, 520, 8, 128, 120, 0, true),
        (512, 8, 2, 0, 256, 3, 30, 27, 0, true),
        (512, 8, 1, 0, 256, 8, 128, 120, 0, false),
    ];
    for &(d, h, hkv, window, cap, m, tkv, q_pos0, kv_off, out_bf16) in cases {
        let case = format!(
            "d={d} h={h} hkv={hkv} win={window} m={m} tkv={tkv} q0={q_pos0} bf16={out_bf16}"
        );
        let buf = |v: &[f32]| {
            let b = gpu.rt.alloc_buffer(v.len() * 4).expect("alloc");
            b.write_f32(v);
            b
        };
        let row = (h * d) as usize;
        let qs: Vec<f32> = values(m as usize * row, 11, 2.0);
        let q = buf(&qs);
        let k = buf(&values(cap * (hkv * d) as usize, 12, 2.0));
        let v = buf(&values(cap * (hkv * d) as usize, 13, 2.0));
        let sentinel = vec![f32::from_bits(0x7f80_0001); m as usize * row];
        let o_blk = buf(&sentinel);
        let ran = verify_attn_single_pass(
            &gpu, &q, &k, &v, &o_blk, m, tkv, h, hkv, d, window, 1.0, q_pos0, kv_off, cap, out_bf16,
        )
        .expect("verify pass");
        assert!(ran, "verify pass refused: {case}");
        gpu.synchronize().expect("complete");
        let blk = o_blk.read_f32();
        let words = if out_bf16 { row / 2 } else { row };
        for i in 0..m as usize {
            let qi = buf(&qs[i * row..(i + 1) * row]);
            let oi = buf(&vec![f32::from_bits(0x7f80_0001); row]);
            let ran = decode_attn_single_pass(
                &gpu,
                &qi,
                &k,
                &v,
                &oi,
                1,
                tkv,
                h,
                hkv,
                d,
                window,
                1.0,
                q_pos0 + i as u32,
                kv_off,
                cap,
                out_bf16,
            )
            .expect("decode pass");
            assert!(ran, "decode pass refused: {case}");
            gpu.synchronize().expect("complete");
            let one = oi.read_f32();
            let base = if out_bf16 { i * row / 2 } else { i * row };
            for w in 0..words {
                assert_eq!(
                    blk[base + w].to_bits(),
                    one[w].to_bits(),
                    "{case} row {i} word {w}"
                );
            }
        }
    }
}
