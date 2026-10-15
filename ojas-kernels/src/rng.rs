//! ojas's one SplitMix64 (Steele, Lea and Flood, 2014): the parity harness,
//! ojas-cuda's test inputs, ojas-autograd's tiny initialiser, ojas-infer's
//! sampler and ojas-data's `CounterRng` all draw from this generator, so one
//! seed is one stream everywhere.

/// SplitMix64 with the state as an explicit counter. Two values built with
/// the same seed produce the same stream; there is no global seed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The counter. The whole stream from here is a function of it.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Replace the counter. The next [`Self::next_u64`] continues from `state`.
    pub fn set_state(&mut self, state: u64) {
        self.state = state;
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[-1, 1)` from the top 24 bits, which are exact in f32, so
    /// `2 * m - 1` rounds nowhere.
    pub fn next_signed_f32(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        2.0 * mantissa - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference vectors ojas-data's `CounterRng` and ojas-infer's
    /// sampler pinned before they shared this generator.
    #[test]
    fn the_stream_is_the_reference_splitmix64() {
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(r.next_u64(), 0x06C4_5D18_8009_454F);
        let mut top = SplitMix64::new(u64::MAX);
        assert_eq!(top.next_u64(), 0xE4D9_7177_1B65_2C20);
        assert_eq!(top.state(), 0x9E37_79B9_7F4A_7C14);
    }

    #[test]
    fn set_state_replays_and_the_derived_draws_use_the_top_bits() {
        let mut a = SplitMix64::new(42);
        let saved = a.state();
        let n = a.next_u64();
        a.set_state(saved);
        assert_eq!(a.next_u64(), n);

        let u = SplitMix64::new(0).next_f64();
        assert_eq!(
            u,
            (0xE220_A839_7B1D_CDAFu64 >> 11) as f64 / (1u64 << 53) as f64
        );
        // Bits from the copies this replaced (ojas-autograd's tiny `unit`,
        // ojas-cuda's `splitmix_f32` at scale 1).
        assert_eq!(SplitMix64::new(0).next_signed_f32().to_bits(), 0x3f44_4150);
        let mut s = SplitMix64::new(0);
        for _ in 0..10_000 {
            let x = s.next_signed_f32();
            assert!((-1.0..1.0).contains(&x), "{x}");
        }
    }
}
