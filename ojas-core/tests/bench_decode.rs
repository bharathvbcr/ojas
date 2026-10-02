//! Host decode benchmark: `Tensor::to_f32_vec` against the push-loop decoder
//! it replaced, kept here as the reference oracle.
//!
//! Run the timing (release, one thread, interleaved A/B, min-of-N):
//!
//! ```text
//! cargo test -p ojas-core --release --test bench_decode -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The parity test is not ignored and runs with the normal suite.

use ojas_core::{Budget, DType, Scratch, Tensor};
use std::hint::black_box;
use std::time::{Duration, Instant};

/// The decoder `Tensor::to_f32_vec` used before it was rewritten: a
/// `with_capacity` + `push` loop over 4-byte chunks. The per-push capacity
/// check keeps LLVM from turning it into a straight copy.
fn reference_decode_f32(bytes: &[u8]) -> Vec<f32> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "f32 window is not a multiple of 4 bytes");
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        out.push(f32::from_ne_bytes(*chunk));
    }
    out
}

/// Same loop for `Tensor::to_u32_vec`.
fn reference_decode_u32(bytes: &[u8]) -> Vec<u32> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty(), "u32 window is not a multiple of 4 bytes");
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        out.push(u32::from_ne_bytes(*chunk));
    }
    out
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

const SPECIAL_BITS: [u32; 10] = [
    0x8000_0000,
    0x0000_0001,
    0x007F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0001,
    0xFFC0_1234,
    0x7F80_0001,
    0x7FBF_FFFF,
    0xFF80_0001,
];

fn random_bytes(n_words: usize, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64(seed);
    let mut bytes = Vec::with_capacity(n_words * 4);
    for i in 0..n_words {
        let word = if i % 11 == 3 {
            SPECIAL_BITS[(rng.next() as usize) % SPECIAL_BITS.len()]
        } else {
            rng.next() as u32
        };
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    bytes
}

fn raw_tensor(bytes: &[u8], shape: &[usize], dtype: DType, budget: &Budget) -> Tensor {
    let mut scratch = Scratch::<u8>::try_alloc(bytes.len(), budget).unwrap();
    scratch.as_mut_slice().copy_from_slice(bytes);
    Tensor::from_scratch(scratch, shape, dtype).unwrap()
}

#[test]
fn reference_and_to_f32_vec_agree_bit_for_bit() {
    let budget = Budget::new(1 << 26);
    let counts = [
        0usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65, 255, 256, 257, 4099, 393_216,
    ];
    for (k, &n) in counts.iter().enumerate() {
        let bytes = random_bytes(n, 0xB17_5EED ^ k as u64);
        let f = raw_tensor(&bytes, &[n], DType::F32, &budget);
        let want: Vec<u32> = reference_decode_f32(f.contiguous_bytes().unwrap())
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let got: Vec<u32> = f
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, want, "f32 n={n}");

        let u = raw_tensor(&bytes, &[n], DType::U32, &budget);
        let want = reference_decode_u32(u.contiguous_bytes().unwrap());
        assert_eq!(u.to_u32_vec().unwrap(), want, "u32 n={n}");
    }
    // A window at a non-zero byte offset inside a larger allocation.
    let bytes = random_bytes(1031, 0x0FF5E7);
    let f = raw_tensor(&bytes, &[1031], DType::F32, &budget);
    let window = f.narrow(4 * 7, &[1000], &[1]).unwrap();
    let want: Vec<u32> = reference_decode_f32(window.contiguous_bytes().unwrap())
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let got: Vec<u32> = window
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(got, want);
}

fn time<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let start = Instant::now();
    let out = black_box(f());
    (start.elapsed(), out)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

#[test]
#[ignore = "timing benchmark; run with --ignored --nocapture --test-threads=1"]
fn bench_to_f32_vec_against_push_loop() {
    const ROUNDS: usize = 300;
    let budget = Budget::new(1 << 28);
    // 768x768 and 512x768: the weight and activation of the linear at
    // 512x768x768.
    for (label, rows, cols) in [("768x768", 768usize, 768usize), ("512x768", 512, 768)] {
        let n = rows * cols;
        let bytes = random_bytes(n, 0xBE7C_0000 + n as u64);
        let t = raw_tensor(&bytes, &[rows, cols], DType::F32, &budget);
        let window = t.contiguous_bytes().unwrap();

        // Both sides agree before anything is timed.
        let a: Vec<u32> = reference_decode_f32(window)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let b: Vec<u32> = t
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(a, b);

        for _ in 0..10 {
            drop(black_box(reference_decode_f32(black_box(window))));
            drop(black_box(t.to_f32_vec().unwrap()));
        }

        let mut push_loop = Vec::with_capacity(ROUNDS);
        let mut public = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            // Alternate which side goes first so neither always runs on a
            // freshly released allocation.
            if round % 2 == 0 {
                let (d, out) = time(|| reference_decode_f32(black_box(window)));
                push_loop.push(d);
                drop(out);
                let (d, out) = time(|| t.to_f32_vec().unwrap());
                public.push(d);
                drop(out);
            } else {
                let (d, out) = time(|| t.to_f32_vec().unwrap());
                public.push(d);
                drop(out);
                let (d, out) = time(|| reference_decode_f32(black_box(window)));
                push_loop.push(d);
                drop(out);
            }
        }
        let a_min = *push_loop.iter().min().unwrap();
        let b_min = *public.iter().min().unwrap();
        let a_med = median(push_loop);
        let b_med = median(public);
        println!(
            "{label} ({n} f32, {ROUNDS} interleaved rounds): \
             push-loop reference min {:.4} ms med {:.4} ms | \
             Tensor::to_f32_vec min {:.4} ms med {:.4} ms | \
             min ratio {:.2}x",
            a_min.as_secs_f64() * 1e3,
            a_med.as_secs_f64() * 1e3,
            b_min.as_secs_f64() * 1e3,
            b_med.as_secs_f64() * 1e3,
            a_min.as_secs_f64() / b_min.as_secs_f64(),
        );
    }
}
