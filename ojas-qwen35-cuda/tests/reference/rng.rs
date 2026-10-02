//! splitmix64, ported from tessl `tests/common/mod.rs:69-120` (`SplitMix`,
//! `random_f32`), so the cases here draw operands the same way tessl's tests
//! do and a failing draw reproduces from its seed.

pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        SplitMix(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1), on the 2^-23 grid tessl's `unit` produces. tessl
    /// forms it in f32; every step of that is exact (a 24-bit integer, a
    /// power-of-two division, and a subtraction that needs at most 24
    /// significant bits), so forming it in f64 and narrowing is the same value.
    /// The narrowing is checked rather than assumed.
    pub fn unit(&mut self) -> f32 {
        let top = u32::try_from(self.next_u64() >> 40).expect("a 24-bit value fits u32");
        let wide = f64::from(top) / 8_388_608.0 - 1.0;
        let narrow = wide as f32;
        assert!(
            f64::from(narrow) == wide,
            "unit() {wide} is not exact in f32"
        );
        narrow
    }
}

/// `n` draws of [`SplitMix::unit`] from `seed`.
pub fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix::new(seed);
    (0..n).map(|_| rng.unit()).collect()
}

/// [`random_f32`] widened to f64 (exact).
pub fn random_f64(n: usize, seed: u64) -> Vec<f64> {
    random_f32(n, seed).into_iter().map(f64::from).collect()
}
