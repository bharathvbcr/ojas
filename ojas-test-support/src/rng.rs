//! splitmix64, so randomized tests need no dependency.

/// Advance `state` by one splitmix64 step and return the output word.
pub fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic splitmix64 stream; the field is the seed and then the state.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        splitmix64_next(&mut self.0)
    }

    /// Uniform in `0..n`. Panics when `n` is 0.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform in `[-1, 1)`, from the top 24 bits.
    pub fn unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        2.0 * mantissa - 1.0
    }

    /// `n` values uniform in `[-scale, scale)`.
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| scale * self.unit()).collect()
    }

    pub fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
