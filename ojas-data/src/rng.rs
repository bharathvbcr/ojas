//! SplitMix64 with an explicit counter state.
//!
//! This is not `libc` `rand`, and it has no process-global seed. Two values
//! constructed with the same seed produce the same stream. Lappi's
//! `CounterRng` hashes a domain string with SHA-256. That `sha2` dependency is
//! not in this workspace, so this stream is **not** that one.

/// Seeded counter. `state` is the whole stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterRng {
    state: u64,
}

impl CounterRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn state(&self) -> u64 {
        self.state
    }

    /// Replace the counter. The next [`Self::next_u64`] continues from `state`.
    pub fn set_state(&mut self, state: u64) {
        self.state = state;
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream_and_no_global() {
        let mut a = CounterRng::new(0x1234_5678_9ABC_DEF0);
        let mut b = CounterRng::new(0x1234_5678_9ABC_DEF0);
        let mut c = CounterRng::new(0x1234_5678_9ABC_DEF0);
        let first = a.next_u64();
        assert_eq!(first, b.next_u64());
        assert_eq!(c.next_u64(), first);
        let second = a.next_u64();
        assert_ne!(second, first);
        assert_eq!(b.next_u64(), second);
        let mut other = CounterRng::new(1);
        assert_ne!(other.next_u64(), first);
        let saved = a.state();
        let n = a.next_u64();
        a.set_state(saved);
        assert_eq!(a.next_u64(), n);
    }

    #[test]
    fn reference_splitmix64_values_and_counter_wraparound() {
        let mut r = CounterRng::new(0);
        assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
        assert_eq!(r.next_u64(), 0x6E789E6AA1B965F4);
        assert_eq!(r.next_u64(), 0x06C45D188009454F);
        let mut top = CounterRng::new(u64::MAX);
        assert_eq!(top.next_u64(), 0xE4D971771B652C20);
        assert_eq!(top.state(), 0x9E3779B97F4A7C14);
    }

    #[test]
    fn instances_do_not_share_counter_state() {
        let mut a = CounterRng::new(42);
        let b = CounterRng::new(42);
        let mut cloned = b.clone();
        let mut replay = CounterRng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), replay.next_u64());
        }
        assert_eq!(b.state(), 42, "advancing one rng changed another");
        assert_eq!(cloned.state(), 42, "a clone shares state with its source");
        let mut from_seed = CounterRng::new(42);
        assert_eq!(cloned.next_u64(), from_seed.next_u64());
        assert_eq!(b.state(), 42);
        assert_eq!(a.state(), replay.state());
        let mut other = CounterRng::new(42);
        other.set_state(7);
        assert_eq!(b.state(), 42, "set_state on one instance wrote another");
        assert_eq!(other.state(), 7);
        assert_ne!(other.next_u64(), from_seed.next_u64());
    }
}
