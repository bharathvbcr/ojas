// Throwaway: X·Xᵀ (r×c times its transpose) as one cblas_sgemm, as two row
// bands on two threads (ojas's split when k < 2m), and as cblas_ssyrk plus a
// mirror pass. Interleaved rounds, min of N.
use std::time::Instant;
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(o: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, al: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, be: f32, c: *mut f32, ldc: i32);
    fn cblas_ssyrk(o: i32, up: i32, t: i32, n: i32, k: i32, al: f32, a: *const f32, lda: i32, be: f32, c: *mut f32, ldc: i32);
}
const ROW: i32 = 101; const NT: i32 = 111; const T: i32 = 112; const UP: i32 = 121;
fn gemm(x: &[f32], rows: usize, r: usize, c: usize, off: usize, out: &mut [f32]) {
    unsafe { cblas_sgemm(ROW, NT, T, rows as i32, r as i32, c as i32, 1.0, x.as_ptr().add(off * c), c as i32, x.as_ptr(), c as i32, 0.0, out.as_mut_ptr(), r as i32) }
}
fn main() {
    let shapes = [(768usize, 768usize), (768, 2048), (768, 3072), (2048, 2048), (2048, 6144)];
    let rounds = 11;
    for (r, c) in shapes {
        let x: Vec<f32> = (0..r * c).map(|i| ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / 16777216.0 - 0.5).collect();
        let (mut g, mut g2, mut s) = (vec![0f32; r * r], vec![0f32; r * r], vec![0f32; r * r]);
        let mut best = [f64::MAX; 3];
        for round in 0..rounds {
            let order = if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
            for which in order {
                let t0 = Instant::now();
                match which {
                    0 => gemm(&x, r, r, c, 0, &mut g),
                    1 => {
                        let h = r / 2;
                        let (top, bot) = g2.split_at_mut(h * r);
                        let xr = &x;
                        std::thread::scope(|sc| {
                            sc.spawn(move || gemm(xr, h, r, c, 0, top));
                            gemm(xr, r - h, r, c, h, bot);
                        });
                    }
                    _ => {
                        unsafe { cblas_ssyrk(ROW, UP, NT, r as i32, c as i32, 1.0, x.as_ptr(), c as i32, 0.0, s.as_mut_ptr(), r as i32) };
                        for i in 0..r { for j in 0..i { s[i * r + j] = s[j * r + i]; } }
                    }
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                best[which] = best[which].min(ms);
            }
        }
        let eq = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        println!("XXt r={r} c={c} gemm={:.3} gemm_2band={:.3} syrk_mirror={:.3} syrk/2band={:.2} syrk/gemm={:.2} bits_syrk_eq_gemm={}/{} bits_2band_eq_gemm={}",
            best[0], best[1], best[2], best[2] / best[1], best[2] / best[0], eq(&g, &s), r * r, eq(&g, &g2));
    }
}
