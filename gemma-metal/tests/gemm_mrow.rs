//! `gemm_q4_mlx_mrow` (compile-time M) against the single-row GEMV and tessl's runtime-M
//! `gemm_q4_mlx_simd` on E4B projection shapes; and (ignored) their timing at M = 2..8.
//!
//! `cargo test --release --test gemm_mrow -- --include-ignored --nocapture`

use gemma_metal::kernels::{gemm_q4_mlx_mrow, gemm_q4_mlx_simd, upload_quant_hot, HotQuantBanks};
use gemma_metal::quant::{quantize_affine_f32, QuantScheme};
use gemma_metal::tessl::tensor::GpuBuffer;
use gemma_metal::GemmaGpu;
use std::time::Instant;

fn values(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

fn bf16_words(v: &[f32]) -> Vec<f32> {
    v.chunks(2)
        .map(|p| {
            let lo = (p[0].to_bits() >> 16) as u32;
            let hi = (p.get(1).copied().unwrap_or(0.0).to_bits() >> 16) as u32;
            f32::from_bits(lo | (hi << 16))
        })
        .collect()
}

fn bank(gpu: &GemmaGpu, rows: usize, cols: usize) -> HotQuantBanks {
    let w = quantize_affine_f32(
        rows,
        cols,
        &values(rows * cols, 7, 0.1),
        QuantScheme::q4_mlx_default(),
    )
    .expect("quantize");
    upload_quant_hot(gpu, &w).expect("upload")
}

fn x_rows(gpu: &GemmaGpu, m: usize, cols: usize) -> (GpuBuffer, Vec<Vec<f32>>) {
    let rows: Vec<Vec<f32>> = (0..m).map(|i| values(cols, 100 + i as u32, 4.0)).collect();
    let flat: Vec<f32> = rows.concat();
    let b = gpu.rt.alloc_buffer(m * cols * 2).expect("alloc");
    b.write_f32(&bf16_words(&flat));
    (b, rows)
}

#[test]
fn mrow_gemm_matches_the_gemv_and_the_simd_gemm() {
    let gpu = GemmaGpu::new().unwrap_or_else(|e| panic!("Metal unavailable (not a pass): {e}"));
    let mut mismatches = 0usize;
    for &(rows, cols) in &[
        (2048usize, 2560usize),
        (256, 2560),
        (2560, 10240),
        (300, 512),
        (2560, 256),
    ] {
        let b = bank(&gpu, rows, cols);
        assert!(b.can_gemm_mrow(2), "{rows}x{cols} must take the mrow GEMM");
        for m in 2..=8usize {
            let (x, host_rows) = x_rows(&gpu, m, cols);
            let y_new = gpu.rt.alloc_buffer(m * rows * 4).expect("alloc");
            let y_old = gpu.rt.alloc_buffer(m * rows * 4).expect("alloc");
            gemm_q4_mlx_mrow(
                &gpu,
                &b.packed,
                &b.scales,
                &b.zeros,
                &x,
                &y_new,
                rows as u32,
                cols as u32,
                b.group_size,
                m as u32,
            )
            .expect("mrow");
            gemm_q4_mlx_simd(
                &gpu,
                &b.packed,
                &b.scales,
                &b.zeros,
                &x,
                &y_old,
                rows as u32,
                cols as u32,
                b.group_size,
                m as u32,
                false,
            )
            .expect("simd gemm");
            gpu.synchronize().expect("sync");
            let new = y_new.read_f32();
            let old = y_old.read_f32();
            let (mut d_simd, mut d_gemv, mut simd_vs_gemv) = (0usize, 0usize, 0usize);
            let mut worst = 0f32;
            for mi in 0..m {
                let (xr, _) = x_rows_one(&gpu, &host_rows[mi]);
                let y1 = gpu.rt.alloc_buffer(rows * 4).expect("alloc");
                b.gemv_bf16_x(&gpu, &xr, &y1).expect("gemv");
                gpu.synchronize().expect("sync");
                let gemv = y1.read_f32();
                let scale = gemv.iter().fold(0f32, |a, v| a.max(v.abs()));
                for r in 0..rows {
                    worst = worst.max((new[mi * rows + r] - gemv[r]).abs() / scale);
                    let i = mi * rows + r;
                    d_simd += (new[i].to_bits() != old[i].to_bits()) as usize;
                    d_gemv += (new[i].to_bits() != gemv[r].to_bits()) as usize;
                    simd_vs_gemv += (old[i].to_bits() != gemv[r].to_bits()) as usize;
                }
            }
            println!(
                "{rows}x{cols} M={m}: differ from gemv: mrow {d_gemv}, simd gemm {simd_vs_gemv}; \
                 mrow vs simd gemm {d_simd} (of {}); worst |mrow - gemv| / max|gemv| {worst:.1e}",
                m * rows
            );
            // Neither GEMM is bit-identical to the GEMV (the compiler contracts the block sums
            // differently); both differ in the last bits of ~2% of outputs. Bound the size.
            assert!(
                worst < 1e-6,
                "{rows}x{cols} M={m}: mrow off the gemv by {worst:e}"
            );
            mismatches += d_gemv;
        }
    }
    println!("{mismatches} outputs differ from the gemv in the last bits");
}

fn x_rows_one(gpu: &GemmaGpu, row: &[f32]) -> (GpuBuffer, ()) {
    let b = gpu.rt.alloc_buffer(row.len() * 2).expect("alloc");
    b.write_f32(&bf16_words(row));
    (b, ())
}

#[test]
#[ignore = "timing only"]
fn mrow_gemm_timing() {
    let gpu = GemmaGpu::new().expect("metal");
    let reps = 40;
    for &(name, rows, cols) in &[
        ("q_proj", 2048usize, 2560usize),
        ("gate", 10240, 2560),
        ("down", 2560, 10240),
    ] {
        let b = bank(&gpu, rows, cols);
        let (x, _) = x_rows(&gpu, 8, cols);
        let y = gpu.rt.alloc_buffer(8 * rows * 4).expect("alloc");
        let mb = (rows * cols / 2 + rows * (cols / 64) * 4) as f64 / 1e6;
        let time = |f: &dyn Fn()| {
            f();
            gpu.synchronize().unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t0 = Instant::now();
                for _ in 0..reps {
                    f();
                    gpu.barrier().unwrap();
                }
                gpu.synchronize().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
            }
            best
        };
        let gemv = time(&|| b.gemv_bf16_x(&gpu, &x, &y).unwrap());
        let mut line = format!(
            "{name} {rows}x{cols} {mb:.1}MB gemv={gemv:.0}us({:.0}GB/s)",
            mb / gemv * 1e3
        );
        for m in 2..=8u32 {
            let old = time(&|| {
                gemm_q4_mlx_simd(
                    &gpu,
                    &b.packed,
                    &b.scales,
                    &b.zeros,
                    &x,
                    &y,
                    rows as u32,
                    cols as u32,
                    b.group_size,
                    m,
                    false,
                )
                .unwrap()
            });
            let new = time(&|| {
                gemm_q4_mlx_mrow(
                    &gpu,
                    &b.packed,
                    &b.scales,
                    &b.zeros,
                    &x,
                    &y,
                    rows as u32,
                    cols as u32,
                    b.group_size,
                    m,
                )
                .unwrap()
            });
            line += &format!(" | M{m} simd={:.2}x mrow={:.2}x", old / gemv, new / gemv);
        }
        println!("{line}");
    }
}
