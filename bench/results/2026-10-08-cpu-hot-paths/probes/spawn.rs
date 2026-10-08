// Throwaway: cost of one std::thread::scope with N-1 spawned threads doing
// nothing, as ojas-cpu's scoped ops pay per split. Min and median of 2000.
use std::time::Instant;
fn main() {
    for threads in [2usize, 4, 6] {
        let mut v = Vec::new();
        for _ in 0..2000 {
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for _ in 1..threads {
                    s.spawn(|| std::hint::black_box(0));
                }
            });
            v.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        v.sort_by(f64::total_cmp);
        println!("scope threads={threads} spawned={} min_us={:.1} med_us={:.1}", threads - 1, v[0], v[v.len() / 2]);
    }
}
