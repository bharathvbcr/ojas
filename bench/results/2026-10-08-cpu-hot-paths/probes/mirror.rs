// Throwaway: lower-from-upper mirror of an n×n row-major matrix, the serial
// column walk ssyrk_accelerate shipped with against a 32×32 tiled walk.
use std::time::Instant;
fn naive(c: &mut [f32], n: usize) {
    for i in 1..n { let (above, row) = c.split_at_mut(i * n); for (j, v) in row[..i].iter_mut().enumerate() { *v = above[j * n + i]; } }
}
fn tiled<const T: usize>(c: &mut [f32], n: usize) {
    for ib in (0..n).step_by(T) {
        let ie = (ib + T).min(n);
        for jb in (0..=ib).step_by(T) {
            for j in jb..(jb + T).min(ie) {
                for i in (ib.max(j + 1))..ie { c[i * n + j] = c[j * n + i]; }
            }
        }
    }
}
fn main() {
    for n in [768usize, 2048] {
        let src: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let fs: [(&str, fn(&mut [f32], usize)); 5] = [("serial", naive), ("t8", tiled::<8>), ("t16", tiled::<16>), ("t32", tiled::<32>), ("t64", tiled::<64>)];
        let mut best = [f64::MAX; 5];
        let mut outs: Vec<Vec<f32>> = (0..5).map(|_| src.clone()).collect();
        for r in 0..21 {
            for w in 0..5 {
                let w = if r % 2 == 0 { w } else { 4 - w };
                let t = Instant::now();
                (fs[w].1)(&mut outs[w], n);
                best[w] = best[w].min(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let same = outs.iter().all(|o| *o == outs[0]);
        print!("mirror n={n} same={same}");
        for w in 0..5 { print!(" {}={:.3}", fs[w].0, best[w]); }
        println!();
    }
}
