//! `gelu_mul_rows_bf16` (one dispatch for the rows of a verify block, `up` at a row stride)
//! against `mlp_gelu_tanh_bf16_up_at` run once per row, bit for bit.

use gemma_metal::kernels::{gelu_mul_rows_bf16, mlp_gelu_tanh_bf16_up_at};
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
fn rows_kernel_matches_the_per_row_dispatch() {
    let gpu = GemmaGpu::new().unwrap_or_else(|e| panic!("Metal unavailable (not a pass): {e}"));
    // (n, rows, layers, layer): E4B's per-layer input shape and an odd one.
    for &(n, rows, layers, layer) in &[
        (256usize, 4usize, 42usize, 17usize),
        (256, 8, 42, 41),
        (100, 3, 5, 0),
    ] {
        let buf = |v: &[f32]| {
            let b = gpu.rt.alloc_buffer(v.len() * 4).expect("alloc");
            b.write_f32(v);
            b
        };
        let gate = buf(&values(rows * n, 1, 12.0));
        let up = buf(&values(rows * layers * n, 2, 3.0));
        let out_rows = gpu.rt.alloc_buffer(rows * n * 2).expect("alloc");
        let out_each = gpu.rt.alloc_buffer(rows * n * 2).expect("alloc");
        gelu_mul_rows_bf16(
            &gpu,
            &gate,
            &up,
            layer * n,
            layers * n,
            &out_rows,
            n as u32,
            rows as u32,
        )
        .expect("rows");
        for r in 0..rows {
            mlp_gelu_tanh_bf16_up_at(
                &gpu,
                &gate,
                r * n,
                &up,
                (r * layers + layer) * n,
                &out_each,
                r * n,
                n as u32,
            )
            .expect("per row");
        }
        gpu.synchronize().expect("sync");
        let a: Vec<u32> = out_rows.read_f32().iter().map(|x| x.to_bits()).collect();
        let b: Vec<u32> = out_each.read_f32().iter().map(|x| x.to_bits()).collect();
        assert_eq!(a, b, "n={n} rows={rows} layer={layer}");
    }
}
