//! `rms_norm_residual_add_emit_bf16` against the dispatches it replaces in the decode tails:
//! `rms_norm_residual_add_f32_scaled` followed by `rms_norm_bf16` (normed next input) or
//! `cast_f32_to_bf16` (raw next input). Same lane count and order, so bit-identical.

use gemma_metal::kernels::{
    cast_f32_to_bf16, rms_norm_bf16, rms_norm_residual_add_emit_bf16,
    rms_norm_residual_add_f32_scaled,
};
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

/// bf16 elements of a buffer read back through its f32 view (little-endian pairs).
fn bf16_bits(words: &[f32], n: usize) -> Vec<u16> {
    words
        .iter()
        .flat_map(|w| {
            let b = w.to_bits();
            [b as u16, (b >> 16) as u16]
        })
        .take(n)
        .collect()
}

#[test]
fn fused_residual_tail_matches_the_separate_dispatches() {
    let gpu = match GemmaGpu::new() {
        Ok(g) => g,
        Err(e) => panic!("Metal runtime unavailable (not a pass): {e}"),
    };
    let eps = 1e-6;
    for &(dim, layer_scale, normalize) in &[
        (2560usize, 1.0f32, true),
        (2560, 0.71, true),
        (2560, 1.0, false),
        (2560, 0.71, false),
        (100, 1.3, true),
        (100, 1.0, false),
    ] {
        let x = values(dim, 1, 40.0);
        let w = values(dim, 2, 2.0);
        let w_next = values(dim, 3, 2.0);
        let r0 = values(dim, 4, 8.0);
        let buf = |v: &[f32]| {
            let b = gpu.rt.alloc_buffer(v.len() * 4).expect("alloc");
            b.write_f32(v);
            b
        };
        let (bx, bw, bw_next) = (buf(&x), buf(&w), buf(&w_next));
        let (r_ref, r_fused) = (buf(&r0), buf(&r0));
        let out_ref = gpu.rt.alloc_buffer(dim * 2 + 4).expect("alloc");
        let out_fused = gpu.rt.alloc_buffer(dim * 2 + 4).expect("alloc");

        rms_norm_residual_add_f32_scaled(&gpu, &bx, &bw, &r_ref, 1, dim as u32, eps, layer_scale)
            .expect("reference residual");
        gpu.barrier().expect("barrier");
        if normalize {
            rms_norm_bf16(&gpu, &r_ref, &bw_next, &out_ref, 1, dim as u32, eps)
                .expect("reference norm");
        } else {
            cast_f32_to_bf16(&gpu, &r_ref, &out_ref, dim as u32).expect("reference cast");
        }
        rms_norm_residual_add_emit_bf16(
            &gpu,
            &bx,
            &bw,
            &r_fused,
            normalize.then_some(&bw_next),
            &out_fused,
            1,
            dim as u32,
            eps,
            layer_scale,
        )
        .expect("fused tail");
        gpu.synchronize().expect("complete");

        let case = format!("dim={dim} scale={layer_scale} normalize={normalize}");
        let (a, b) = (r_ref.read_f32(), r_fused.read_f32());
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "residual differs: {case}"
        );
        assert_ne!(a, r0, "residual unchanged: {case}");
        let (oa, ob) = (
            bf16_bits(&out_ref.read_f32(), dim),
            bf16_bits(&out_fused.read_f32(), dim),
        );
        assert_eq!(oa, ob, "bf16 output differs: {case}");
        assert!(oa.iter().any(|&h| h != 0), "empty output: {case}");
    }
}
