// Throwaway diagnostic (this lane): where does Fast's grad_q differ from
// Exact's at T = 257? Deleted after use.

use ojas_core::{Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn diag() {
    let (t, d) = (257usize, 64usize);
    let shape = [1, 1, t, d];
    let host = Budget::new(u64::MAX);
    let mk = |seed| Tensor::from_f32(&values(t * d, seed), &shape, &host).unwrap();
    let (q, k, v, g) = (mk(1), mk(2), mk(3), mk(4));
    let be = |n| CpuBackend::with_threads(Budget::new(1 << 32), 2).unwrap().with_numerics(n);
    let (fast, exact) = (be(Numerics::Fast), be(Numerics::Exact));
    let (yf, lf) = fast.causal_sdpa_forward(&q, &k, &v, None).unwrap();
    let (ye, le) = exact.causal_sdpa_forward(&q, &k, &v, None).unwrap();
    let f32s = |x: &Tensor| x.to_f32_vec().unwrap();
    let maxdiff = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    println!("y diff {:e} lse diff {:e}", maxdiff(&f32s(&yf), &f32s(&ye)), maxdiff(&f32s(&lf), &f32s(&le)));
    let ge = exact.causal_sdpa_backward(&q, &k, &v, &ye, &le, &g, None).unwrap();
    let scale_q = f32s(&ge.0).iter().fold(0f32, |m, x| m.max(x.abs()));
    for (name, y, l) in [("fast saved", &yf, &lf), ("exact saved", &ye, &le), ("fast y exact lse", &yf, &le), ("exact y fast lse", &ye, &lf)] {
        let gf = fast.causal_sdpa_backward(&q, &k, &v, y, l, &g, None).unwrap();
        let (a, b) = (f32s(&gf.0), f32s(&ge.0));
        let mut rows: Vec<(f32, usize)> = (0..t).map(|r| (maxdiff(&a[r * d..(r + 1) * d], &b[r * d..(r + 1) * d]), r)).collect();
        rows.sort_by(|x, y| y.0.total_cmp(&x.0));
        println!("{name}: grad_q rel {:e}; worst rows {:?}", rows[0].0 / scale_q, &rows[..5]);
        let gx = exact.causal_sdpa_backward(&q, &k, &v, y, l, &g, None).unwrap();
        println!("   exact bwd on same saved: grad_q rel {:e}", maxdiff(&f32s(&gx.0), &b) / scale_q);
    }
}
